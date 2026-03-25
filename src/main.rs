use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Gather stats on moderated (slop) PRs in rust-lang/rust")]
struct Cli {
    /// Data directory for cached results
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search GitHub for moderated PRs and cache the list
    Discover,
    /// Fetch detailed data for each discovered PR
    Fetch,
    /// Analyze cached data and print stats
    Analyze,
    /// Delete all cached data
    Clean,
    /// Publish report.html as a GitHub gist
    Publish,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// The top-level index of discovered PRs.
#[derive(Serialize, Deserialize, Default)]
struct PrIndex {
    /// PR number -> basic discovery info
    prs: BTreeMap<u64, PrSummary>,
}

#[derive(Serialize, Deserialize, Clone)]
struct PrSummary {
    number: u64,
    title: String,
    author: String,
    created_at: String,
    closed_at: Option<String>,
}

/// Full cached data for a single PR.
#[derive(Serialize, Deserialize)]
struct PrData {
    number: u64,
    title: String,
    author: String,
    created_at: String,
    closed_at: Option<String>,
    additions: u64,
    deletions: u64,
    /// Comments on the PR (subset: those from known moderators or containing key phrases)
    moderation_comments: Vec<ModerationComment>,
    /// What kind of moderation action was taken
    moderation_tier: ModerationTier,
    /// Stats about the PR author
    author_stats: AuthorStats,
}

#[derive(Serialize, Deserialize)]
struct ModerationComment {
    author: String,
    body: String,
    created_at: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[serde(rename_all = "snake_case")]
enum ModerationTier {
    Ban,
    Warning,
    SoftClose,
    Unknown,
}

#[derive(Serialize, Deserialize)]
struct AuthorStats {
    /// When the GitHub account was created
    account_created: Option<String>,
    /// Total PR count across all of GitHub
    total_prs: Option<u64>,
    /// PRs opened in the last 7 days (from the PR's created_at) across all of GitHub
    recent_prs: Option<u64>,
    /// Distinct repos with PRs in the last 7 days
    recent_distinct_repos: Option<u64>,
    /// List of recent repos for inspection
    recent_repos: Vec<String>,
    /// Total PRs against rust-lang/rust
    rust_prs: Option<u64>,
}

impl Default for AuthorStats {
    fn default() -> Self {
        AuthorStats {
            account_created: None,
            total_prs: None,
            recent_prs: None,
            recent_distinct_repos: None,
            recent_repos: Vec::new(),
            rust_prs: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GitHub helpers (via `gh` CLI)
// ---------------------------------------------------------------------------

fn gh_api(endpoint: &str) -> Result<serde_json::Value> {
    let output = Command::new("gh")
        .args(["api", endpoint])
        .output()
        .context("failed to run `gh api`")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("gh api {endpoint} failed: {stderr}");
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn gh_graphql(query: &str) -> Result<serde_json::Value> {
    let output = Command::new("gh")
        .args(["api", "graphql", "-f", &format!("query={query}")])
        .output()
        .context("failed to run `gh api graphql`")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("gh api graphql failed: {stderr}");
    }
    let val: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    if let Some(errors) = val.get("errors") {
        bail!("GraphQL errors: {errors}");
    }
    Ok(val)
}

/// Run a GitHub search and return issue count + edges.
/// `after` is the cursor for pagination.
fn search_prs(
    query: &str,
    after: Option<&str>,
) -> Result<(u64, Vec<serde_json::Value>, Option<String>)> {
    let after_arg = match after {
        Some(cursor) => format!(r#", after: "{cursor}""#),
        None => String::new(),
    };
    let gql = format!(
        r#"{{
  search(query: "{query}", type: ISSUE, first: 50{after_arg}) {{
    issueCount
    pageInfo {{ hasNextPage endCursor }}
    edges {{
      node {{
        ... on PullRequest {{
          number
          title
          author {{ login }}
          createdAt
          closedAt
        }}
      }}
    }}
  }}
}}"#
    );
    let val = gh_graphql(&gql)?;
    let search = &val["data"]["search"];
    let count = search["issueCount"].as_u64().unwrap_or(0);
    let edges = search["edges"].as_array().cloned().unwrap_or_default();
    let next_cursor = if search["pageInfo"]["hasNextPage"].as_bool() == Some(true) {
        search["pageInfo"]["endCursor"]
            .as_str()
            .map(|s| s.to_string())
    } else {
        None
    };
    Ok((count, edges, next_cursor))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn discover(data_dir: &Path) -> Result<()> {
    fs::create_dir_all(data_dir)?;
    let index_path = data_dir.join("prs.json");

    let mut index: PrIndex = if index_path.exists() {
        let contents = fs::read_to_string(&index_path)?;
        serde_json::from_str(&contents)?
    } else {
        PrIndex::default()
    };

    let existing_count = index.prs.len();

    // Multiple search strategies to maximize coverage.
    // Date-filtered to 2025+ to avoid false positives from older PRs
    // that happen to mention "moderation team", "ban", etc. in unrelated contexts.
    let queries = [
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"moderation team\""#,
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"moderation warning\""#,
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"banning you\""#,
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"vibecoded\""#,
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"LLM spam\""#,
        r#"repo:rust-lang/rust is:pr is:closed created:>=2025-01-01 \"LLM\" \"without reviewing\""#,
    ];

    for query in &queries {
        eprintln!("Searching: {query}");
        let mut cursor: Option<String> = None;
        loop {
            let (count, edges, next) = search_prs(query, cursor.as_deref())?;
            if cursor.is_none() {
                eprintln!("  -> {count} results");
            }
            for edge in &edges {
                let node = &edge["node"];
                let number = match node["number"].as_u64() {
                    Some(n) => n,
                    None => continue,
                };
                if index.prs.contains_key(&number) {
                    continue;
                }
                let summary = PrSummary {
                    number,
                    title: node["title"].as_str().unwrap_or("").to_string(),
                    author: node["author"]["login"].as_str().unwrap_or("").to_string(),
                    created_at: node["createdAt"].as_str().unwrap_or("").to_string(),
                    closed_at: node["closedAt"].as_str().map(|s| s.to_string()),
                };
                eprintln!("  Found PR #{number}: {}", summary.title);
                index.prs.insert(number, summary);
            }
            match next {
                Some(c) => cursor = Some(c),
                None => break,
            }
        }
    }

    let new_count = index.prs.len();
    eprintln!(
        "Discovery complete: {new_count} PRs total ({} new)",
        new_count - existing_count
    );

    let json = serde_json::to_string_pretty(&index)?;
    fs::write(&index_path, json)?;
    Ok(())
}

fn fetch(data_dir: &Path) -> Result<()> {
    let index_path = data_dir.join("prs.json");
    if !index_path.exists() {
        bail!("No prs.json found. Run `discover` first.");
    }
    let index: PrIndex = serde_json::from_str(&fs::read_to_string(&index_path)?)?;

    let pr_dir = data_dir.join("pr");
    fs::create_dir_all(&pr_dir)?;

    let total = index.prs.len();
    for (i, (number, summary)) in index.prs.iter().enumerate() {
        let pr_path = pr_dir.join(format!("{number}.json"));
        if pr_path.exists() {
            eprintln!("[{}/{}] PR #{number}: cached, skipping", i + 1, total);
            continue;
        }
        eprintln!("[{}/{}] Fetching PR #{number}...", i + 1, total);

        let pr_data = fetch_pr(*number, summary)?;

        let json = serde_json::to_string_pretty(&pr_data)?;
        fs::write(&pr_path, json)?;
    }

    eprintln!("Fetch complete.");
    Ok(())
}

fn fetch_pr(number: u64, summary: &PrSummary) -> Result<PrData> {
    // 1. Get PR details (additions/deletions) via REST
    let pr_rest = gh_api(&format!("repos/rust-lang/rust/pulls/{number}"))?;
    let additions = pr_rest["additions"].as_u64().unwrap_or(0);
    let deletions = pr_rest["deletions"].as_u64().unwrap_or(0);

    // 2. Get all comments via REST (paginated)
    let comments = fetch_all_comments(number)?;

    // 3. Extract moderation comments
    let moderation_comments = extract_moderation_comments(&comments);
    let moderation_tier = classify_moderation(&moderation_comments);

    // 4. Fetch author stats
    let author_stats = fetch_author_stats(&summary.author, &summary.created_at)?;

    Ok(PrData {
        number,
        title: summary.title.clone(),
        author: summary.author.clone(),
        created_at: summary.created_at.clone(),
        closed_at: summary.closed_at.clone(),
        additions,
        deletions,
        moderation_comments,
        moderation_tier,
        author_stats,
    })
}

fn fetch_all_comments(number: u64) -> Result<Vec<serde_json::Value>> {
    let mut all = Vec::new();
    let mut page = 1u32;
    loop {
        let val = gh_api(&format!(
            "repos/rust-lang/rust/issues/{number}/comments?per_page=100&page={page}"
        ))?;
        let arr = val.as_array().cloned().unwrap_or_default();
        if arr.is_empty() {
            break;
        }
        all.extend(arr);
        page += 1;
    }
    Ok(all)
}

fn extract_moderation_comments(comments: &[serde_json::Value]) -> Vec<ModerationComment> {
    let mod_phrases = [
        "moderation team",
        "moderation warning",
        "banning you",
        "vibecoded",
        "LLM spam",
        "LLM without reviewing",
        "used an LLM",
    ];
    let known_moderators = ["oli-obk", "jieyouxu"];

    comments
        .iter()
        .filter(|c| {
            let body = c["body"].as_str().unwrap_or("");
            let author = c["user"]["login"].as_str().unwrap_or("");
            let body_lower = body.to_lowercase();
            // Include if from a known moderator and mentions moderation-related terms,
            // or if the comment contains moderation phrases
            mod_phrases
                .iter()
                .any(|p| body_lower.contains(&p.to_lowercase()))
                || (known_moderators.contains(&author)
                    && (body_lower.contains("moderat")
                        || body_lower.contains("ban")
                        || body_lower.contains("llm")))
        })
        .map(|c| ModerationComment {
            author: c["user"]["login"].as_str().unwrap_or("").to_string(),
            body: c["body"].as_str().unwrap_or("").to_string(),
            created_at: c["created_at"].as_str().unwrap_or("").to_string(),
        })
        .collect()
}

fn classify_moderation(comments: &[ModerationComment]) -> ModerationTier {
    for c in comments {
        let lower = c.body.to_lowercase();
        if lower.contains("banning you") || lower.contains("thus banning") {
            return ModerationTier::Ban;
        }
    }
    for c in comments {
        let lower = c.body.to_lowercase();
        if lower.contains("moderation warning") {
            return ModerationTier::Warning;
        }
    }
    for c in comments {
        let lower = c.body.to_lowercase();
        if lower.contains("vibecoded") || lower.contains("moderation team") || lower.contains("llm")
        {
            return ModerationTier::SoftClose;
        }
    }
    ModerationTier::Unknown
}

fn fetch_author_stats(author: &str, pr_created_at: &str) -> Result<AuthorStats> {
    if author.is_empty() {
        return Ok(AuthorStats::default());
    }

    // 1. Account creation date
    let user = gh_api(&format!("users/{author}"));
    let account_created = user
        .as_ref()
        .ok()
        .and_then(|u| u["created_at"].as_str().map(|s| s.to_string()));

    // 2. Total PRs across GitHub
    let total_prs = count_search(&format!("author:{author} is:pr"))?;

    // 3. PRs against rust-lang/rust
    let rust_prs = count_search(&format!("author:{author} repo:rust-lang/rust is:pr"))?;

    // 4. Recent PRs (7 days before the moderated PR was opened)
    let (recent_prs, recent_distinct_repos, recent_repos) =
        fetch_recent_pr_stats(author, pr_created_at)?;

    Ok(AuthorStats {
        account_created,
        total_prs: Some(total_prs),
        recent_prs: Some(recent_prs),
        recent_distinct_repos: Some(recent_distinct_repos),
        recent_repos,
        rust_prs: Some(rust_prs),
    })
}

fn count_search(query: &str) -> Result<u64> {
    let gql = format!(r#"{{ search(query: "{query}", type: ISSUE, first: 1) {{ issueCount }} }}"#);
    let val = gh_graphql(&gql)?;
    Ok(val["data"]["search"]["issueCount"].as_u64().unwrap_or(0))
}

fn fetch_recent_pr_stats(author: &str, pr_created_at: &str) -> Result<(u64, u64, Vec<String>)> {
    // Parse date, subtract 7 days
    let date_part = &pr_created_at[..10]; // "2026-03-24"
    let since = subtract_days(date_part, 7)?;

    let query = format!("author:{author} is:pr created:>={since}");
    let gql = format!(
        r#"{{
  search(query: "{query}", type: ISSUE, first: 100) {{
    issueCount
    edges {{
      node {{
        ... on PullRequest {{
          repository {{ nameWithOwner }}
        }}
      }}
    }}
  }}
}}"#
    );
    let val = gh_graphql(&gql)?;
    let search = &val["data"]["search"];
    let count = search["issueCount"].as_u64().unwrap_or(0);
    let edges = search["edges"].as_array().cloned().unwrap_or_default();

    let mut repos: Vec<String> = edges
        .iter()
        .filter_map(|e| {
            e["node"]["repository"]["nameWithOwner"]
                .as_str()
                .map(|s| s.to_string())
        })
        .collect();
    repos.sort();
    repos.dedup();
    let distinct = repos.len() as u64;

    Ok((count, distinct, repos))
}

/// Subtract `days` from a "YYYY-MM-DD" date string. Simple implementation.
fn subtract_days(date: &str, days: u64) -> Result<String> {
    let parts: Vec<u32> = date
        .split('-')
        .map(|p| p.parse())
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        bail!("invalid date: {date}");
    }
    let (y, m, d) = (parts[0], parts[1], parts[2]);

    // Convert to a rough day count, subtract, convert back.
    // Good enough for 7-day windows.
    let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut total_days = d;
    let mut remaining = days as u32;
    let mut cm = m;
    let mut cy = y;

    while remaining > 0 {
        if remaining < total_days {
            total_days -= remaining;
            remaining = 0;
        } else {
            remaining -= total_days;
            cm -= 1;
            if cm == 0 {
                cm = 12;
                cy -= 1;
            }
            total_days = days_in_month[cm as usize];
            // leap year feb
            if cm == 2 && (cy % 4 == 0 && (cy % 100 != 0 || cy % 400 == 0)) {
                total_days = 29;
            }
        }
    }

    Ok(format!("{cy:04}-{cm:02}-{total_days:02}"))
}

fn load_prs(data_dir: &Path) -> Result<Vec<PrData>> {
    let pr_dir = data_dir.join("pr");
    if !pr_dir.exists() {
        bail!("No PR data found. Run `fetch` first.");
    }

    let mut prs: Vec<PrData> = Vec::new();
    for entry in fs::read_dir(&pr_dir)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "json") {
            let contents = fs::read_to_string(entry.path())?;
            let pr: PrData = serde_json::from_str(&contents)?;
            prs.push(pr);
        }
    }

    prs.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    Ok(prs)
}

/// Build a stable anonymization mapping: author -> "Author A", "Author B", etc.
fn build_anon_map(prs: &[PrData]) -> BTreeMap<String, String> {
    let mut seen: Vec<String> = Vec::new();
    for pr in prs {
        if !seen.contains(&pr.author) {
            seen.push(pr.author.clone());
        }
    }
    seen.iter()
        .enumerate()
        .map(|(i, name)| {
            let label = if i < 26 {
                format!("Author {}", (b'A' + i as u8) as char)
            } else {
                format!("Author {}", i + 1)
            };
            (name.clone(), label)
        })
        .collect()
}

fn tier_str(tier: ModerationTier) -> &'static str {
    match tier {
        ModerationTier::Ban => "Ban",
        ModerationTier::Warning => "Warning",
        ModerationTier::SoftClose => "Soft Close",
        ModerationTier::Unknown => "Unknown",
    }
}

fn tier_color(tier: ModerationTier) -> &'static str {
    match tier {
        ModerationTier::Ban => "#e74c3c",
        ModerationTier::Warning => "#f39c12",
        ModerationTier::SoftClose => "#3498db",
        ModerationTier::Unknown => "#95a5a6",
    }
}

/// Compute days between two "YYYY-MM-DD" date strings. Approximate but fine for display.
fn days_between(earlier: &str, later: &str) -> Option<i64> {
    fn to_days(date: &str) -> Option<i64> {
        let parts: Vec<i64> = date.split('-').filter_map(|p| p.parse().ok()).collect();
        if parts.len() != 3 {
            return None;
        }
        // Rough days since epoch — good enough for differences
        Some(parts[0] * 365 + parts[1] * 30 + parts[2])
    }
    Some(to_days(later)? - to_days(earlier)?)
}

fn analyze(data_dir: &Path) -> Result<()> {
    let prs = load_prs(data_dir)?;
    let anon = build_anon_map(&prs);

    let report_path = data_dir.join("report.html");

    // --- Build chart data ---

    // 1. Timeline by week, stacked by tier
    let mut weeks: BTreeMap<String, [u64; 4]> = BTreeMap::new();
    for pr in &prs {
        if pr.created_at.len() >= 10 {
            let week = week_of(&pr.created_at[..10]);
            let counts = weeks.entry(week).or_insert([0; 4]);
            match pr.moderation_tier {
                ModerationTier::Ban => counts[0] += 1,
                ModerationTier::Warning => counts[1] += 1,
                ModerationTier::SoftClose => counts[2] += 1,
                ModerationTier::Unknown => counts[3] += 1,
            }
        }
    }
    let week_labels: Vec<&str> = weeks.keys().map(|s| s.as_str()).collect();
    let week_bans: Vec<u64> = weeks.values().map(|c| c[0]).collect();
    let week_warnings: Vec<u64> = weeks.values().map(|c| c[1]).collect();
    let week_soft: Vec<u64> = weeks.values().map(|c| c[2]).collect();

    // 2. Per-PR data for size chart, rust PRs chart, account age chart (all tier-colored)
    let pr_labels: Vec<String> = prs.iter().map(|pr| anon[&pr.author].clone()).collect();
    let pr_colors: Vec<&str> = prs
        .iter()
        .map(|pr| tier_color(pr.moderation_tier))
        .collect();
    let size_totals: Vec<u64> = prs.iter().map(|pr| pr.additions + pr.deletions).collect();
    let rust_pr_counts: Vec<u64> = prs
        .iter()
        .map(|pr| pr.author_stats.rust_prs.unwrap_or(0))
        .collect();
    let account_ages: Vec<i64> = prs
        .iter()
        .map(|pr| {
            pr.author_stats
                .account_created
                .as_deref()
                .and_then(|created| {
                    if created.len() >= 10 && pr.created_at.len() >= 10 {
                        days_between(&created[..10], &pr.created_at[..10])
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        })
        .collect();

    // 3. Author activity scatter — deduplicate by author, color by tier
    let mut author_scatter: BTreeMap<String, (u64, u64, ModerationTier)> = BTreeMap::new();
    for pr in &prs {
        let label = anon[&pr.author].clone();
        let recent_prs = pr.author_stats.recent_prs.unwrap_or(0);
        let recent_repos = pr.author_stats.recent_distinct_repos.unwrap_or(0);
        author_scatter.insert(label, (recent_prs, recent_repos, pr.moderation_tier));
    }
    // Group scatter points by tier so the legend shows tiers, not individual authors
    let mut scatter_by_tier: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for (label, (prs_count, repos, tier)) in &author_scatter {
        let tier_name = tier_str(*tier);
        scatter_by_tier
            .entry(tier_name)
            .or_default()
            .push(serde_json::json!({
                "x": prs_count,
                "y": repos,
                "r": (*prs_count as f64).sqrt().mul_add(2.0, 0.0).max(5.0).min(40.0),
                "label": label,
            }));
    }

    // 4. Summary table data
    let table_rows: Vec<String> = prs
        .iter()
        .map(|pr| {
            let s = &pr.author_stats;
            let account_created = s
                .account_created
                .as_deref()
                .map(|d| &d[..10])
                .unwrap_or("?");
            let age_days = pr
                .author_stats
                .account_created
                .as_deref()
                .and_then(|created| {
                    if created.len() >= 10 && pr.created_at.len() >= 10 {
                        days_between(&created[..10], &pr.created_at[..10])
                    } else {
                        None
                    }
                });
            let age_str = match age_days {
                Some(d) if d < 30 => format!("{d}d"),
                Some(d) if d < 365 => format!("{}mo", d / 30),
                Some(d) => format!("{}y", d / 365),
                None => "?".to_string(),
            };
            format!(
                r#"["{}", "{}", "{}", {}, {}, {}, {}, {}, {}, "{}", "{}", "{}"]"#,
                anon[&pr.author],
                &pr.created_at[..10],
                tier_str(pr.moderation_tier),
                pr.additions,
                pr.deletions,
                s.total_prs.unwrap_or(0),
                s.recent_prs.unwrap_or(0),
                s.recent_distinct_repos.unwrap_or(0),
                s.rust_prs.unwrap_or(0),
                account_created,
                age_str,
                tier_color(pr.moderation_tier),
            )
        })
        .collect();

    // --- Generate HTML ---
    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moderation Action Stats — rust-lang/rust</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 2px solid #e74c3c; padding-bottom: 0.5rem; }}
  h2 {{ margin-top: 2.5rem; color: #555; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  canvas {{ max-height: 400px; }}
  .two-charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .two-charts .chart-container {{ margin: 0; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #34495e; color: white; padding: 0.75rem; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #eee; font-size: 0.85rem; }}
  tr:hover td {{ background: #f7f7f7; }}
  .tier {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: white; font-size: 0.8rem; font-weight: 600; }}
  .summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }}
  .stat-card .number {{ font-size: 2rem; font-weight: 700; }}
  .stat-card .label {{ color: #777; font-size: 0.85rem; margin-top: 0.25rem; }}
  .legend-hint {{ font-size: 0.85rem; color: #888; margin-top: -0.5rem; }}
</style>
</head>
<body>
<h1>Moderation Action Stats</h1>
<p>Moderated PRs in <code>rust-lang/rust</code>. Data anonymized. Colors: <span style="color:#e74c3c;font-weight:600">Ban</span> · <span style="color:#f39c12;font-weight:600">Warning</span> · <span style="color:#3498db;font-weight:600">Soft Close</span></p>

<div class="summary">
  <div class="stat-card"><div class="number">{total}</div><div class="label">Total PRs</div></div>
  <div class="stat-card"><div class="number" style="color:#e74c3c">{bans}</div><div class="label">Bans</div></div>
  <div class="stat-card"><div class="number" style="color:#f39c12">{warnings}</div><div class="label">Warnings</div></div>
  <div class="stat-card"><div class="number" style="color:#3498db">{soft_closes}</div><div class="label">Soft Closes</div></div>
</div>

<h2>Moderation Actions Over Time</h2>
<div class="chart-container"><canvas id="timelineChart"></canvas></div>

<h2>PR Size (lines changed)</h2>
<div class="chart-container"><canvas id="sizeChart"></canvas></div>

<div class="two-charts">
  <div>
    <h2>Prior rust-lang/rust PRs</h2>
    <div class="chart-container"><canvas id="rustPrChart"></canvas></div>
  </div>
  <div>
    <h2>Account Age at PR Time</h2>
    <div class="chart-container"><canvas id="ageChart"></canvas></div>
  </div>
</div>

<h2>Author Spray Pattern (7 days before moderated PR)</h2>
<p class="legend-hint">X = PRs opened across all of GitHub, Y = distinct repos. Bubble size = PR count.</p>
<div class="chart-container"><canvas id="scatterChart"></canvas></div>

<h2>Details</h2>
<table>
<thead><tr><th>Author</th><th>Date</th><th>Action</th><th>+</th><th>-</th><th>Total PRs</th><th>7d PRs</th><th>7d Repos</th><th>Rust PRs</th><th>Acct Age</th></tr></thead>
<tbody id="tableBody"></tbody>
</table>

<script>
const TIER_COLORS = {{ Ban: '#e74c3c', Warning: '#f39c12', 'Soft Close': '#3498db', Unknown: '#95a5a6' }};

// --- Timeline ---
new Chart(document.getElementById('timelineChart'), {{
  type: 'bar',
  data: {{
    labels: {week_labels_json},
    datasets: [
      {{ label: 'Bans', data: {week_bans_json}, backgroundColor: '#e74c3c' }},
      {{ label: 'Warnings', data: {week_warnings_json}, backgroundColor: '#f39c12' }},
      {{ label: 'Soft Closes', data: {week_soft_json}, backgroundColor: '#3498db' }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, beginAtZero: true, ticks: {{ stepSize: 1 }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// --- PR Size ---
new Chart(document.getElementById('sizeChart'), {{
  type: 'bar',
  data: {{
    labels: {pr_labels_json},
    datasets: [{{ data: {size_totals_json}, backgroundColor: {pr_colors_json} }}]
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ type: 'logarithmic' }} }},
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{ label: (ctx) => ctx.raw.toLocaleString() + ' lines' }}
    }} }}
  }}
}});

// --- Prior Rust PRs ---
new Chart(document.getElementById('rustPrChart'), {{
  type: 'bar',
  data: {{
    labels: {pr_labels_json},
    datasets: [{{ data: {rust_pr_json}, backgroundColor: {pr_colors_json} }}]
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// --- Account Age ---
new Chart(document.getElementById('ageChart'), {{
  type: 'bar',
  data: {{
    labels: {pr_labels_json},
    datasets: [{{ data: {age_json}, backgroundColor: {pr_colors_json} }}]
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Days' }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// --- Scatter ---
const scatterByTier = {scatter_by_tier_json};
new Chart(document.getElementById('scatterChart'), {{
  type: 'bubble',
  data: {{
    datasets: Object.entries(scatterByTier).map(([tier, points]) => ({{
      label: tier,
      data: points,
      backgroundColor: TIER_COLORS[tier] + '99',
      borderColor: TIER_COLORS[tier],
    }}))
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'PRs in 7-day window' }}, type: 'logarithmic' }},
      y: {{ title: {{ display: true, text: 'Distinct repos' }}, beginAtZero: true }}
    }},
    plugins: {{ tooltip: {{
      callbacks: {{ label: (ctx) => ctx.raw.label + ': ' + ctx.raw.x + ' PRs, ' + ctx.raw.y + ' repos' }}
    }} }}
  }}
}});

// --- Table ---
const rows = {table_rows_json};
const tbody = document.getElementById('tableBody');
rows.forEach(r => {{
  const tr = document.createElement('tr');
  const tierColor = r[11];
  tr.innerHTML = `
    <td>${{r[0]}}</td>
    <td>${{r[1]}}</td>
    <td><span class="tier" style="background:${{tierColor}}">${{r[2]}}</span></td>
    <td>${{r[3].toLocaleString()}}</td>
    <td>${{r[4].toLocaleString()}}</td>
    <td>${{r[5]}}</td>
    <td>${{r[6]}}</td>
    <td>${{r[7]}}</td>
    <td>${{r[8]}}</td>
    <td>${{r[10]}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"##,
        total = prs.len(),
        bans = prs
            .iter()
            .filter(|p| matches!(p.moderation_tier, ModerationTier::Ban))
            .count(),
        warnings = prs
            .iter()
            .filter(|p| matches!(p.moderation_tier, ModerationTier::Warning))
            .count(),
        soft_closes = prs
            .iter()
            .filter(|p| matches!(p.moderation_tier, ModerationTier::SoftClose))
            .count(),
        week_labels_json = serde_json::to_string(&week_labels)?,
        week_bans_json = serde_json::to_string(&week_bans)?,
        week_warnings_json = serde_json::to_string(&week_warnings)?,
        week_soft_json = serde_json::to_string(&week_soft)?,
        pr_labels_json = serde_json::to_string(&pr_labels)?,
        pr_colors_json = serde_json::to_string(&pr_colors)?,
        size_totals_json = serde_json::to_string(&size_totals)?,
        rust_pr_json = serde_json::to_string(&rust_pr_counts)?,
        age_json = serde_json::to_string(&account_ages)?,
        scatter_by_tier_json = serde_json::to_string(&scatter_by_tier)?,
        table_rows_json = format!("[{}]", table_rows.join(",")),
    );

    fs::write(&report_path, html)?;
    eprintln!("Report written to {}", report_path.display());
    Ok(())
}

/// Return ISO week string like "2026-W12" for a "YYYY-MM-DD" date.
fn week_of(date: &str) -> String {
    // Simple: bucket by 7-day periods from the start of the year
    let parts: Vec<u32> = date.split('-').filter_map(|p| p.parse().ok()).collect();
    if parts.len() != 3 {
        return date.to_string();
    }
    let (y, m, d) = (parts[0], parts[1], parts[2]);
    let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut day_of_year = d;
    for i in 1..m {
        day_of_year += days_in_month[i as usize];
        if i == 2 && (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)) {
            day_of_year += 1;
        }
    }
    let week = (day_of_year - 1) / 7 + 1;
    format!("{y}-W{week:02}")
}

fn clean(data_dir: &Path) -> Result<()> {
    if data_dir.exists() {
        fs::remove_dir_all(data_dir)?;
        eprintln!("Removed {}", data_dir.display());
    } else {
        eprintln!("Nothing to clean.");
    }
    Ok(())
}

fn publish(data_dir: &Path) -> Result<()> {
    let report_path = data_dir.join("report.html");
    if !report_path.exists() {
        bail!("No report.html found. Run `analyze` first.");
    }

    eprintln!("Creating private gist...");
    let output = Command::new("gh")
        .args([
            "gist",
            "create",
            "--desc",
            "Moderation Action Stats — rust-lang/rust (anonymized)",
            report_path.to_str().unwrap(),
        ])
        .output()
        .context("failed to run `gh gist create`")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("gh gist create failed: {stderr}");
    }

    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    eprintln!("Gist created: {url}");
    println!("{url}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Cmd::Discover => discover(&cli.data_dir),
        Cmd::Fetch => fetch(&cli.data_dir),
        Cmd::Analyze => analyze(&cli.data_dir),
        Cmd::Clean => clean(&cli.data_dir),
        Cmd::Publish => publish(&cli.data_dir),
    }
}

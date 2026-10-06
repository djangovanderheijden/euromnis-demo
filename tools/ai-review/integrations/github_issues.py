#!/usr/bin/env python3
"""Create a GitHub issue for every reviewed commit that has findings.

Reads the JSON report written by `review.py --json`, from a file or stdin:

    python3 review.py "$BEFORE..$AFTER" --json report.json
    python3 integrations/github_issues.py report.json --assign-authors

Environment (GitHub Actions sets the last two; pass the token with `env: GITHUB_TOKEN: ${{ github.token }}`):
    GITHUB_TOKEN        a token that may create issues (workflow permission "issues: write")
    GITHUB_REPOSITORY   owner/name
    GITHUB_API_URL      default https://api.github.com

Standard library only. This script does not import review.py: copy it, change it, make it yours.
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request


def github(method, url, token, body=None):
    """Call the GitHub API and return the decoded JSON response."""
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json", "User-Agent": "ai-review"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def describe(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    return str(getattr(exc, "reason", exc))


def find_login(repo_url, token, sha):
    """The GitHub account the commit is linked to (through the author's email address), if any."""
    try:
        commit = github("GET", f"{repo_url}/commits/{sha}", token)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  ! could not look up the author of {sha[:8]}: {describe(exc)}", file=sys.stderr)
        return None
    return (commit.get("author") or {}).get("login")


def create_issue(url, token, payload):
    """Create the issue; if GitHub rejects the assignee, create it unassigned."""
    try:
        return github("POST", url, token, payload)
    except urllib.error.HTTPError as exc:
        if "assignees" not in payload or exc.code >= 500:
            raise
        print(f"  ! GitHub rejected the issue with an assignee ({describe(exc)}); trying unassigned", file=sys.stderr)
        return github("POST", url, token, {key: value for key, value in payload.items() if key != "assignees"})


def commit_reviews_with_findings(report):
    for review in report.get("reviews", []):
        if review.get("status") != "findings":
            continue
        if review.get("kind") != "commit":
            print("  – skipping the working-tree review: issues are only created for commits", file=sys.stderr)
            continue
        yield review


def issue_title(review):
    return f"AI review: {review['sha'][:8]} — {review['subject'][:80]}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create GitHub issues from an AI review report.")
    parser.add_argument("report", nargs="?", default="-", help="report JSON file (default: read from stdin)")
    parser.add_argument("--assign-authors", action="store_true", help="assign each issue to the commit's author when found")
    parser.add_argument("--dry-run", action="store_true", help="print the issues instead of creating them")
    args = parser.parse_args(argv)

    report = json.load(sys.stdin) if args.report == "-" else json.loads(pathlib.Path(args.report).read_text(encoding="utf-8"))
    labels = ["ai-review", f"ai-review-model: {report.get('model', 'unknown')}"[:50]]  # GitHub allows 50 characters
    reviews = list(commit_reviews_with_findings(report))

    if args.dry_run:
        for review in reviews:
            print(f"=== {issue_title(review)}\nlabels: {', '.join(labels)}")
            if args.assign_authors:
                print("assignee: the GitHub account linked to the commit (looked up when not a dry run)")
            print(f"\n{review['markdown']}\n")
        print(f"{len(reviews)} issue(s) would be created.", file=sys.stderr)
        return 0

    missing = [name for name in ("GITHUB_TOKEN", "GITHUB_REPOSITORY") if not os.environ.get(name)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    token = os.environ["GITHUB_TOKEN"]
    repo_url = f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com').rstrip('/')}/repos/{os.environ['GITHUB_REPOSITORY']}"

    failures = 0
    for review in reviews:
        payload = {"title": issue_title(review), "body": review["markdown"], "labels": labels}
        if args.assign_authors:
            login = find_login(repo_url, token, review["sha"])
            if login:
                payload["assignees"] = [login]
        try:
            issue = create_issue(f"{repo_url}/issues", token, payload)
            print(f"  ✓ {issue.get('html_url', issue.get('number'))}", file=sys.stderr)
        except (urllib.error.URLError, OSError) as exc:
            failures += 1
            print(f"  ✗ could not create the issue for {review['sha'][:8]}: {describe(exc)}", file=sys.stderr)
    print(f"{len(reviews) - failures} issue(s) created.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create a GitLab issue for every reviewed commit that has findings.

Reads the JSON report written by `review.py --json`, from a file or stdin:

    python3 review.py "$CI_COMMIT_BEFORE_SHA..$CI_COMMIT_SHA" --json report.json
    python3 integrations/gitlab_issues.py report.json --assign-authors

Environment (GitLab CI sets the first two):
    CI_API_V4_URL      e.g. https://gitlab.example.com/api/v4
    CI_PROJECT_ID      the project's numeric ID
    GITLAB_API_TOKEN   a project access token with the "api" scope

Standard library only. This script does not import review.py: copy it, change it, make it yours.
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request


def gitlab(method, url, token, body=None):
    """Call the GitLab API and return the decoded JSON response."""
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"PRIVATE-TOKEN": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def describe(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    return str(getattr(exc, "reason", exc))


def find_user_id(api_url, token, email, name):
    """The GitLab user ID of a commit author: search by email, then by name; only a single match counts."""
    for query in (email, name):
        if not query:
            continue
        try:
            users = gitlab("GET", f"{api_url}/users?" + urllib.parse.urlencode({"search": query}), token)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ! could not look up GitLab user {query!r}: {describe(exc)}", file=sys.stderr)
            continue
        if len(users) == 1:
            return users[0]["id"]
    return None


def create_issue(url, token, payload):
    """Create the issue; if GitLab rejects the assignee, create it unassigned."""
    try:
        return gitlab("POST", url, token, payload)
    except urllib.error.HTTPError as exc:
        if "assignee_ids" not in payload or exc.code >= 500:
            raise
        print(f"  ! GitLab rejected the issue with an assignee ({describe(exc)}); trying unassigned", file=sys.stderr)
        return gitlab("POST", url, token, {key: value for key, value in payload.items() if key != "assignee_ids"})


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
    parser = argparse.ArgumentParser(description="Create GitLab issues from an AI review report.")
    parser.add_argument("report", nargs="?", default="-", help="report JSON file (default: read from stdin)")
    parser.add_argument("--assign-authors", action="store_true", help="assign each issue to the commit's author when found")
    parser.add_argument("--dry-run", action="store_true", help="print the issues instead of creating them")
    args = parser.parse_args(argv)

    report = json.load(sys.stdin) if args.report == "-" else json.loads(pathlib.Path(args.report).read_text(encoding="utf-8"))
    labels = f"ai-review,ai-review-model::{report.get('model', 'unknown')}"
    reviews = list(commit_reviews_with_findings(report))

    if args.dry_run:
        for review in reviews:
            print(f"=== {issue_title(review)}\nlabels: {labels}")
            if args.assign_authors:
                print(f"assignee: the GitLab user for {review['author_email']} (looked up when not a dry run)")
            print(f"\n{review['markdown']}\n")
        print(f"{len(reviews)} issue(s) would be created.", file=sys.stderr)
        return 0

    missing = [name for name in ("CI_API_V4_URL", "CI_PROJECT_ID", "GITLAB_API_TOKEN") if not os.environ.get(name)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    api_url = os.environ["CI_API_V4_URL"].rstrip("/")
    token = os.environ["GITLAB_API_TOKEN"]
    issues_url = f"{api_url}/projects/{urllib.parse.quote(os.environ['CI_PROJECT_ID'], safe='')}/issues"

    failures = 0
    for review in reviews:
        payload = {"title": issue_title(review), "description": review["markdown"], "labels": labels}
        if args.assign_authors:
            user_id = find_user_id(api_url, token, review["author_email"], review["author_name"])
            if user_id:
                payload["assignee_ids"] = [user_id]
        try:
            issue = create_issue(issues_url, token, payload)
            print(f"  ✓ {issue.get('web_url', issue.get('iid'))}", file=sys.stderr)
        except (urllib.error.URLError, OSError) as exc:
            failures += 1
            print(f"  ✗ could not create the issue for {review['sha'][:8]}: {describe(exc)}", file=sys.stderr)
    print(f"{len(reviews) - failures} issue(s) created.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

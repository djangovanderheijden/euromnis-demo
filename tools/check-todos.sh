#!/usr/bin/env bash

set -e

echo "Checking .omh changes for TODO comments..."

if [ "$#" -eq 2 ]; then
    # CI mode: compare two commits
    DIFF_CMD=(git diff --unified=0 "$1" "$2" -- '*.omh')
else
    # Local pre-commit mode: inspect staged changes
    DIFF_CMD=(git diff --cached --unified=0 -- '*.omh')
fi

matches="$(
    "${DIFF_CMD[@]}" |
    grep '^+' |
    grep -v '^+++' |
    grep -E '#[[:space:]]*TODO' || true
)"

if [ -n "$matches" ]; then
    echo
    echo "ERROR: TODO comments found in changed lines:"
    echo "$matches"
    echo
    echo "Please remove them before continuing."
    exit 1
fi

echo "No TODO comments found in changed lines."
exit 0
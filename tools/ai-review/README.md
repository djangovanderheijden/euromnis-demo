# AI Review

Let an LLM review your git changes. It reads the diff, can look around your code with read-only tools (`read_file`, `grep_search`, `list_directory`), and reports concrete findings: bugs, security issues, missed edge cases, dead code.

- One Python file, standard library only. Nothing to install.
- Works with [Ollama](https://ollama.com) and any OpenAI-compatible API ([OpenRouter](https://openrouter.ai), vLLM, LM Studio, …).
- Reviews your uncommitted changes, a commit, or every commit in a range.
- Writes a JSON report; small scripts turn it into GitLab or GitHub issues.

## Quick start

You need Python 3.12+, git, and a model.

**With OpenRouter**: create a `.env` file next to `review.py`:

```sh
OPENROUTER_API_KEY=sk-or-...
AI_REVIEW_MODEL=claude-haiku-openrouter
```

**With Ollama** running on your machine, pull a model with tool-calling support and point the default profile in `models.json` at it (it ships with `qwen3.6:27b-coding-mxfp8`).

Then, from anywhere inside the repository you want to review:

```sh
python3 path/to/review.py        # review your uncommitted changes
python3 path/to/review.py -v     # the same, watching the model think and read your code
```

## What it reviews

| Command | Reviews |
|---|---|
| `python3 review.py` | your uncommitted changes, including new files |
| `python3 review.py HEAD` | one commit |
| `python3 review.py main~5..main` | each commit in the range, oldest first (merge commits are skipped) |
| `python3 review.py a1b2c3d e4f5a6b` | several commits |

Each commit is reviewed on its own. When the model reads files during a commit review, it sees them exactly as they were in that commit.

## Options

```
--model NAME          model profile from models.json (default: $AI_REVIEW_MODEL, else "default")
-v, --verbose         stream the model's reasoning and answer live, show tool results and token usage
--json PATH           write the JSON report ("-" = stdout)
--markdown PATH       write a Markdown report
--fail-on-findings    exit with code 1 when anything was found
--time-limit SECONDS  time for tool calls per review before the model must answer (0 = no limit)
--prompt PATH         use another prompt file
--config PATH         use another models file
--repo PATH           review a repository other than the current directory
```

Exit codes: `0` finished · `1` something was found and `--fail-on-findings` was given · `2` nothing could be reviewed (a configuration problem, or every review failed).

Progress goes to stderr, so `--json -` can be piped into other tools.

## Configuration

### `models.json`

Each profile says which API to talk to and what to send:

```json
"claude-haiku-openrouter": {
  "api": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "${OPENROUTER_API_KEY}",
  "request": {
    "model": "~anthropic/claude-haiku-latest",
    "reasoning": { "enabled": true }
  }
}
```

- `api`: `"ollama"` (Ollama's own API) or `"openai"` (any OpenAI-compatible `/chat/completions` endpoint).
- `base_url`: the Ollama server (`http://localhost:11434`), or the API base including `/v1`.
- `api_key`: sent as a Bearer token when not empty.
- `request`: sent to the API exactly as written, so any parameter the API supports (temperature, Ollama `options`, reasoning settings, …) goes here.
- `${VAR}` takes a value from the environment, and `${VAR:-fallback}` uses `fallback` when the variable is not set. Only the profile you use needs its variables.

The `settings` block holds options for all profiles: `time_limit`, `max_diff_chars` (longer diffs are cut off), `max_tool_output_chars`, `request_timeout` (seconds without any data from the API), `request_retries`, and `prompt`.

### `.env`

`review.py` reads a `.env` file next to itself if there is one (see `.env.example`). Variables that are already set in your environment win. Keep secrets and private URLs here: `.env` is ignored by git.

## The prompt

`prompt.txt` has two sections: `---SYSTEM---` with the reviewer's instructions and `---USER---` with the template for each change, using the placeholders `{sha}`, `{author_name}`, `{author_email}`, `{timestamp}`, `{message}`, `{changed_files}` and `{diff}`.

The shipped prompt is tuned for [Omnis Studio](https://www.omnis.net), the language we use. The system section is where you describe your own stack, its conventions, and the mistakes you don't want reported.

Whatever you change, keep the output format section: the script expects findings as blocks of `file:`, `line_range:` and `description:`, or `NO_FINDINGS`.

## The JSON report

`--json report.json` writes one entry per review:

```json
{
  "model": "claude-haiku-openrouter",
  "reviews": [
    {
      "kind": "commit",
      "sha": "a1b2c3d4…",
      "subject": "Fix rounding in invoice totals",
      "author_name": "Jane Doe",
      "author_email": "jane@example.com",
      "status": "findings",
      "findings": [{ "file": "src/invoice.py", "line_range": "42-58", "description": "…" }],
      "markdown": "## AI code review findings …",
      "tool_calls": [{ "name": "read_file", "args": { "path": "src/invoice.py" } }],
      "tokens": { "input": 12034, "output": 811 }
    }
  ]
}
```

`status` is `findings`, `clean`, `skipped` (nothing to review, e.g. a merge commit) or `error` (see `error`). `markdown` is a ready-to-post issue body for reviews with findings. The report also records the model, request parameters, settings and timings. The API key is never written.

## CI integration

The integration scripts read the JSON report and create one issue per commit with findings. Both support `--dry-run` (print the issues instead of creating them) and `--assign-authors`.

### GitHub Actions

Copy `integrations/github-workflow.example.yml` to `.github/workflows/ai-review.yml`. On every push to the default branch it reviews the pushed commits, opens an issue per commit with findings (`integrations/github_issues.py`), and adds the Markdown report to the job summary. It needs an `OPENROUTER_API_KEY` repository secret (or another reachable model) and the `issues: write` permission.

### GitLab CI

Add the job from `integrations/gitlab-ci.example.yml` to your `.gitlab-ci.yml`. On every push to the default branch it reviews the pushed commits and opens an issue per commit with findings (`integrations/gitlab_issues.py`). It needs a `GITLAB_API_TOKEN` (project access token with the `api` scope) and the variables for your model.

### Anything else

Use the JSON report or the exit code. A git `pre-push` hook that blocks a push when the model finds something (and lets it through when the review itself can't run):

```sh
#!/bin/sh
# .git/hooks/pre-push: review the commits you are about to push
zero=0000000000000000000000000000000000000000
while read local_ref local_sha remote_ref remote_sha; do
  [ "$local_sha" = "$zero" ] && continue    # deleting a branch: nothing to review
  [ "$remote_sha" = "$zero" ] && continue   # new branch: nothing to compare with
  python3 tools/ai-review/review.py "$remote_sha..$local_sha" --fail-on-findings
  case $? in
    1) exit 1 ;;                                                 # findings: stop the push
    2) echo "AI review could not run; pushing anyway." >&2 ;;   # e.g. the model is unreachable
  esac
done
```

## How it works

1. **Collect:** for each change, git provides the diff, the list of changed files and the commit details.
2. **Ask:** the prompt, with the diff filled in, goes to the model together with three read-only tools.
3. **Investigate:** the model may call tools to read files, search the code or list directories. Calls run against the working tree, or against the commit's own snapshot in git. Repeated calls are refused, and after the time limit the model has to answer.
4. **Report:** the answer is parsed into findings and shown in the terminal, and optionally written to JSON and Markdown.

## License

MIT, see [LICENSE](LICENSE).

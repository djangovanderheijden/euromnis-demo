#!/usr/bin/env python3
"""AI review: let an LLM review your git changes, with read-only access to your code.

Reviews the uncommitted changes in the working tree (default), or one or more
commits / commit ranges. Works with Ollama and any OpenAI-compatible API.
Python standard library only.

    python3 review.py                   # uncommitted changes in the working tree
    python3 review.py HEAD              # the last commit
    python3 review.py main~5..main -v   # every commit in a range, streaming the model live

See README.md for configuration, the JSON report and CI integration.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import posixpath
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Configuration: models.json, .env and ${VAR} references
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "prompt": "prompt.txt",
    "time_limit": 600,
    "max_diff_chars": 30000,
    "max_tool_output_chars": 10000,
    "request_timeout": 400,
    "request_retries": 1,
}


class ConfigError(Exception):
    """Something the user has to fix before anything can be reviewed (exit code 2)."""


@dataclass
class ModelConfig:
    name: str
    api: str          # "ollama" or "openai"
    base_url: str
    api_key: str
    request: dict     # sent to the API as-is, plus messages, tools and stream
    settings: dict

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.base_url).netloc or self.base_url


def load_dotenv(path: pathlib.Path) -> list[str]:
    """Load KEY=VALUE lines from a .env file; variables that are already set win.

    Returns the names of the variables that were loaded.
    """
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value):
    """Expand ${VAR} and ${VAR:-fallback} in every string of a JSON value (shell semantics)."""
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match) -> str:
        name, fallback = match.group(1), match.group(2)
        current = os.environ.get(name)
        if fallback is not None and not current:
            return fallback
        if current is None:
            raise ConfigError(f"Environment variable {name} is not set (it is used in models.json)")
        return current

    return _ENV_REFERENCE.sub(substitute, value)


def load_config(config_path: pathlib.Path, model_name: str | None = None) -> ModelConfig:
    """Load a model profile: --model, else $AI_REVIEW_MODEL, else "default" from the file."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid JSON: {exc}") from None

    models = data.get("models", {})
    name = model_name or os.environ.get("AI_REVIEW_MODEL") or data.get("default")
    if not name:
        raise ConfigError('No model selected: use --model, set AI_REVIEW_MODEL or set "default" in models.json')
    if name not in models:
        raise ConfigError(f"Unknown model '{name}'. Available: {', '.join(models) or '(none)'}")

    entry = expand_env(models[name])  # only the selected model: others may use variables that are not set
    api = entry.get("api", "ollama")
    if api not in ("ollama", "openai"):
        raise ConfigError(f"Model '{name}': \"api\" must be \"ollama\" or \"openai\", not {api!r}")
    base_url = (entry.get("base_url") or "").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(f"Model '{name}': base_url must start with http:// or https:// (got {base_url!r})")
    request = entry.get("request") or {}
    if not request.get("model"):
        raise ConfigError(f"Model '{name}' has no request.model")

    settings = {**DEFAULT_SETTINGS, **data.get("settings", {})}
    settings["prompt"] = str((config_path.parent / settings["prompt"]).resolve())
    return ModelConfig(name, api, base_url, entry.get("api_key") or "", request, settings)


# ---------------------------------------------------------------------------
# Git: what to review
# ---------------------------------------------------------------------------

class GitError(Exception):
    """A git command failed."""


def run_git(repo: pathlib.Path, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    # Pin settings that change the output we parse, whatever the user's own git config says.
    return subprocess.run(["git", "-c", "core.quotePath=false", "-c", "log.showSignature=false", *args], cwd=repo, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def git(repo: pathlib.Path, *args: str, check: bool = True) -> str:
    """Run a git command in `repo` and return its output."""
    result = run_git(repo, *args)
    if check and result.returncode != 0:
        raise GitError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout


def find_repo_root(path: pathlib.Path) -> pathlib.Path:
    if shutil.which("git") is None:
        raise ConfigError("git was not found on PATH")
    try:
        result = run_git(path, "rev-parse", "--show-toplevel")
    except OSError as exc:
        raise ConfigError(f"Cannot open {path}: {exc.strerror}") from None
    if result.returncode != 0:
        detail = (result.stderr.strip().splitlines() or ["git rev-parse failed"])[0]
        raise ConfigError(f"Not inside a git repository: {path.resolve()} ({detail})")
    return pathlib.Path(result.stdout.strip()).resolve()


@dataclass
class Change:
    """One thing to review: the uncommitted changes, or a single commit."""
    kind: str                 # "working-tree" or "commit"
    sha: str | None           # None for the working tree
    subject: str
    author_name: str
    author_email: str
    date: str
    diff: str
    changed_files: list[str]

    @property
    def label(self) -> str:
        return self.sha[:8] if self.sha else "working tree"


def _paths(output: str) -> list[str]:
    return [path for path in output.split("\0") if path]


def working_tree_change(repo: pathlib.Path, max_diff_chars: int = 0) -> Change | None:
    """All uncommitted changes, including new files that are not ignored. None if there are none.

    New files are always listed, but their contents are only added while the diff is within
    max_diff_chars (0 = no limit): the prompt would cut them off anyway.
    """
    has_commits = run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
    base = "HEAD" if has_commits else git(repo, "hash-object", "-t", "tree", "/dev/null").strip()  # the empty tree
    diff = git(repo, "diff", base, "--no-color", "--no-ext-diff")
    files = _paths(git(repo, "diff", base, "--name-only", "-z", "--no-ext-diff"))
    for path in _paths(git(repo, "ls-files", "--others", "--exclude-standard", "-z")):
        if not max_diff_chars or len(diff) <= max_diff_chars:
            diff += git(repo, "diff", "--no-index", "--no-color", "--no-ext-diff", "--", "/dev/null", path, check=False)
        files.append(path)
    if not files:
        return None
    return Change(
        kind="working-tree",
        sha=None,
        subject="Uncommitted changes",
        author_name=git(repo, "config", "user.name", check=False).strip() or "Unknown",
        author_email=git(repo, "config", "user.email", check=False).strip(),
        date=datetime.now().astimezone().isoformat(timespec="seconds"),
        diff=diff,
        changed_files=files,
    )


def commit_change(repo: pathlib.Path, sha: str) -> Change:
    """A single commit. Merge commits get an empty diff (and are skipped by the review)."""
    sha, name, email, date, subject = git(repo, "show", "-s", "--format=%H%x00%an%x00%ae%x00%aI%x00%s", sha).rstrip("\n").split("\0")
    shallow = pathlib.Path(repo, git(repo, "rev-parse", "--git-path", "shallow").strip())
    if shallow.is_file() and sha in shallow.read_text().split():
        raise ConfigError(f"Commit {sha[:8]} is at the edge of a shallow clone: its parent was not fetched, so its "
                          "changes can't be shown. Fetch more history (e.g. fetch-depth: 0 in GitHub Actions, "
                          "GIT_DEPTH: 0 in GitLab CI).")
    return Change(
        kind="commit",
        sha=sha,
        subject=subject,
        author_name=name,
        author_email=email,
        date=date,
        diff=git(repo, "diff-tree", "-p", "--root", "--no-commit-id", "--no-color", "--no-ext-diff", sha),
        changed_files=_paths(git(repo, "diff-tree", "-r", "--root", "--no-commit-id", "--name-only", "-z", sha)),
    )


def expand_target(repo: pathlib.Path, target: str) -> list[str]:
    """A commit becomes [sha]; a range (anything with "..") becomes its commits, oldest first, without merges."""
    if target.startswith("-"):
        raise ConfigError(f"Not a commit or commit range: {target}")
    try:
        if ".." in target:
            return git(repo, "rev-list", "--reverse", "--no-merges", target).split()
        return [git(repo, "rev-parse", "--verify", "--quiet", f"{target}^{{commit}}").strip()]
    except GitError:
        raise ConfigError(f"Not a commit or commit range: {target}") from None


def collect_changes(repo: pathlib.Path, targets: list[str], console, max_diff_chars: int = 0) -> list[Change]:
    """Turn the command-line targets into changes to review (the working tree when there are none)."""
    if not targets:
        change = working_tree_change(repo, max_diff_chars)
        return [change] if change else []
    shas: list[str] = []
    for target in targets:
        found = expand_target(repo, target)
        if not found:
            console.warn(f"No commits in {target}")
        shas += [sha for sha in found if sha not in shas]
    return [commit_change(repo, sha) for sha in shas]


def diff_stats(diff: str) -> tuple[int, int]:
    """Number of added and removed lines in a unified diff."""
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def truncate(text: str, limit: int, hint: str = "") -> str:
    if limit and len(text) > limit:
        return text[:limit] + f"\n... [truncated at {limit} characters{hint}]"
    return text


# ---------------------------------------------------------------------------
# Tools: read-only access to the code for the model
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """A tool call that failed; the message is returned to the model."""


def clean_path(path: str | None) -> str:
    """Normalize a repository-relative path from the model; reject paths that leave the repository."""
    cleaned = posixpath.normpath(str(path or "").strip().replace("\\", "/").lstrip("/") or ".")
    if cleaned == ".." or cleaned.startswith("../"):
        raise ToolError(f"path escapes the repository: {path}")
    return cleaned


def _git_grep(repo: pathlib.Path, args: list[str]) -> str:
    try:
        result = run_git(repo, "grep", "-n", "-I", "--no-color", "-G", *args, timeout=30)
    except subprocess.TimeoutExpired:
        raise ToolError("search timed out") from None
    if result.returncode == 1:
        return ""  # no matches
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "search failed")
    return result.stdout


class WorkingTreeFiles:
    """The files on disk (for reviews of uncommitted changes)."""

    def __init__(self, repo: pathlib.Path):
        self.root = repo.resolve()

    def _resolve(self, path) -> pathlib.Path:
        target = (self.root / clean_path(path)).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(f"path escapes the repository: {path}")
        if target.relative_to(self.root).parts[:1] == (".git",):
            raise ToolError("the .git directory is not part of the code under review")
        return target

    def read(self, path) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"'{path}' is not a file or does not exist")
        if run_git(self.root, "check-ignore", "-q", "--", target.relative_to(self.root).as_posix()).returncode == 0:
            raise ToolError(f"'{path}' is ignored by git, so it is not part of the code under review")  # e.g. .env secrets
        return target.read_text(encoding="utf-8", errors="replace")

    def list(self, path) -> list[str]:
        target = self._resolve(path)
        if not target.is_dir():
            raise ToolError(f"'{path}' is not a directory or does not exist")
        return [entry.name + ("/" if entry.is_dir() else "") for entry in sorted(target.iterdir()) if entry.name != ".git"]

    def grep(self, pattern: str, path) -> str:
        return _git_grep(self.root, ["--untracked", "-e", pattern, "--", clean_path(path)])


class CommitFiles:
    """The files exactly as they were in a commit, read from git objects (no checkout needed)."""

    def __init__(self, repo: pathlib.Path, sha: str):
        self.repo, self.sha = repo, sha

    def _object(self, path) -> tuple[str, str]:
        cleaned = clean_path(path)
        spec = f"{self.sha}:" if cleaned == "." else f"{self.sha}:{cleaned}"
        return spec, git(self.repo, "cat-file", "-t", spec, check=False).strip()

    def read(self, path) -> str:
        spec, kind = self._object(path)
        if kind != "blob":
            raise ToolError(f"'{path}' is not a file in commit {self.sha[:8]}")
        return git(self.repo, "cat-file", "blob", spec)

    def list(self, path) -> list[str]:
        spec, kind = self._object(path)
        if kind != "tree":
            raise ToolError(f"'{path}' is not a directory in commit {self.sha[:8]}")
        entries = []
        for item in _paths(git(self.repo, "ls-tree", "-z", spec)):
            meta, _, name = item.partition("\t")
            entries.append(name + ("/" if meta.split()[1] == "tree" else ""))
        return sorted(entries)

    def grep(self, pattern: str, path) -> str:
        output = _git_grep(self.repo, ["-e", pattern, self.sha, "--", clean_path(path)])
        prefix = f"{self.sha}:"
        return "".join(line.removeprefix(prefix) for line in output.splitlines(keepends=True))


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the repository. Use this to inspect source code that is referenced in the diff or that you need for additional context. Use offset and line_count to read large files in sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the repository root."},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from. Omit to start from the beginning."},
                    "line_count": {"type": "integer", "description": "Number of lines to return. Omit to read all remaining lines."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search the codebase for a text pattern. Returns matching lines with file paths and line numbers. Use this to find usages, definitions, or references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search for (grep basic regex)."},
                    "path": {"type": "string", "description": "Optional subdirectory to scope the search to (relative to repo root)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a given path in the repository. Use this to explore the project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to the repository root. Omit or leave empty for the root."},
                },
                "required": [],
            },
        },
    },
]


def _read_file(files, path, offset, line_count, max_chars: int) -> str:
    if not path:
        raise ToolError("path is required")
    lines = files.read(path).splitlines(keepends=True)
    start = max(int(offset) - 1, 0) if offset else 0
    end = start + int(line_count) if line_count else len(lines)
    chunk = "".join(lines[start:end])
    if not chunk:
        return f"(nothing to show: the file has {len(lines)} lines)"
    if len(chunk) > max_chars:
        return chunk[:max_chars] + f"\n... [truncated at {max_chars} chars — use offset/line_count to read in sections]"
    if end < len(lines):
        chunk += f"\n... [showing lines {start + 1}–{end} of {len(lines)} — use offset/line_count to read more]"
    return chunk


def run_tool(files, name: str, args, max_chars: int) -> str:
    """Run one tool call for the model. Problems are returned as "Error: ..." text, never raised."""
    try:
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        if name == "read_file":
            return _read_file(files, args.get("path"), args.get("offset"), args.get("line_count"), max_chars)
        if name == "grep_search":
            if not args.get("pattern"):
                raise ToolError("pattern is required")
            output = files.grep(str(args["pattern"]), args.get("path"))
            return truncate(output.rstrip("\n"), max_chars) if output else "No matches found."
        if name == "list_directory":
            entries = files.list(args.get("path"))
            text = "\n".join(entries[:200]) or "(empty directory)"
            if len(entries) > 200:
                text += f"\n... [{len(entries) - 200} more entries]"
            return text
        raise ToolError(f"unknown tool '{name}'")
    except Exception as exc:  # whatever the model sends, it gets an answer instead of ending the review
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Console: human-readable progress on stderr (stdout is kept free for --json -)
# ---------------------------------------------------------------------------

class Console:
    STYLES = {"bold": "1", "dim": "2", "italic": "3", "red": "31", "green": "32",
              "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36"}
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, verbose: bool = False, stream=None, color: bool | None = None):
        self.out = stream or sys.stderr
        self.verbose = verbose
        if color is None:
            color = bool(os.environ.get("FORCE_COLOR")) or (not os.environ.get("NO_COLOR") and self.out.isatty())
        self.color = color
        self.width = min(shutil.get_terminal_size((100, 24)).columns, 100)
        self.live_status = self.out.isatty() and not verbose  # verbose streams the model instead
        self._lock = threading.RLock()
        self._status = None          # [label, start time, streamed chunks] while a request runs
        self._status_drawn = False
        self._spinner = None
        self._stream_kind = None     # "reasoning" or "content" while model output is streaming
        self._mid_line = False

    def style(self, text: str, *names: str) -> str:
        if not self.color or not names:
            return text
        return f"\033[{';'.join(self.STYLES[name] for name in names)}m{text}\033[0m"

    def line(self, text: str = "", indent: int = 0) -> None:
        with self._lock:
            self._clear_status()
            self._end_stream()
            self.out.write(" " * indent + text + "\n")
            self.out.flush()

    def detail(self, text: str, indent: int = 2) -> None:
        """A dim line that is only shown in verbose mode."""
        if self.verbose:
            self.line(self.style(text, "dim"), indent)

    def warn(self, text: str) -> None:
        self.line(self.style(f"! {text}", "yellow"), 2)

    def error(self, text: str) -> None:
        self.line(self.style("Error: ", "red", "bold") + text)

    def wrapped(self, text: str, indent: int = 4) -> None:
        """Print text wrapped to the terminal width, keeping each line's own indentation."""
        for raw in text.splitlines() or [""]:
            lead = raw[: len(raw) - len(raw.lstrip())]
            width = max(self.width - indent - len(lead), 20)
            for part in textwrap.wrap(raw.strip(), width, break_long_words=False, break_on_hyphens=False) or [""]:
                self.line(lead + part, indent)

    # -- live model output ----------------------------------------------------

    def model_output(self, kind: str, text: str) -> None:
        """Streamed model text: printed live in verbose mode, counted on the status line otherwise."""
        if not text:
            return
        with self._lock:
            if not self.verbose:
                if self._status:
                    self._status[2] += 1
                return
            self._clear_status()
            if kind != self._stream_kind:
                self._end_stream()
                label = "thinking" if kind == "reasoning" else "answer"
                self.out.write("    " + self.style(f"╭ {label}", "dim") + "\n")
                self._stream_kind = kind
            styles = ("dim", "italic") if kind == "reasoning" else ()
            for index, part in enumerate(text.split("\n")):
                if index and self._mid_line:
                    self.out.write("\n")
                    self._mid_line = False
                if part:
                    if not self._mid_line:
                        self.out.write("    " + self.style("│ ", "dim"))
                        self._mid_line = True
                    self.out.write(self.style(part, *styles))
            self.out.flush()

    def _end_stream(self) -> None:
        if self._mid_line:
            self.out.write("\n")
            self._mid_line = False
        self._stream_kind = None

    # -- status line ("⠹ thinking · 23s · 412 tokens") --------------------------

    def start_status(self, label: str) -> None:
        if not self.live_status:
            return
        with self._lock:
            self._status = [label, time.monotonic(), 0]
            if self._spinner is None:
                self._spinner = threading.Thread(target=self._spin, daemon=True)
                self._spinner.start()

    def stop_status(self) -> None:
        with self._lock:
            self._clear_status()
            self._status = None

    def _spin(self) -> None:
        frame = 0
        while True:
            time.sleep(0.1)
            with self._lock:
                if self._status is None:
                    continue
                label, started, chunks = self._status
                text = f"{self.SPINNER[frame % len(self.SPINNER)]} {label} · {time.monotonic() - started:.0f}s"
                if chunks:
                    text += f" · {chunks} tokens"
                self.out.write("\r\033[K  " + self.style(text, "dim"))
                self.out.flush()
                self._status_drawn = True
                frame += 1

    def _clear_status(self) -> None:
        if self._status_drawn:
            self.out.write("\r\033[K")
            self._status_drawn = False


# ---------------------------------------------------------------------------
# LLM client: streaming chat for Ollama and OpenAI-compatible APIs
# ---------------------------------------------------------------------------
#
# Messages are kept in one internal shape and converted per API:
#   {"role": "system" | "user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}],
#    "reasoning": str, "reasoning_details": list}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}

class LLMError(Exception):
    """The model API failed (after retries)."""


class _Retryable(Exception):
    """A failure worth retrying: connection problems, timeouts, HTTP 429/5xx, cut-off streams."""


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    reasoning_details: list = field(default_factory=list)  # OpenAI-compatible APIs (e.g. OpenRouter)
    tool_calls: list = field(default_factory=list)         # [{"id", "name", "arguments": dict}]
    input_tokens: int | None = None
    output_tokens: int | None = None


def chat(model: ModelConfig, messages: list[dict], tools: list | None = None, allow_tool_calls: bool = True,
         on_text=None, on_retry=None) -> Reply:
    """Send the conversation and stream the reply.

    on_text(kind, text) receives "reasoning" and "content" text as it arrives;
    on_retry(message) is called before a retry.
    """
    on_text = on_text or (lambda kind, text: None)
    attempts = 1 + max(int(model.settings["request_retries"]), 0)
    send = _ollama_chat if model.api == "ollama" else _openai_chat
    for attempt in range(1, attempts + 1):
        try:
            return send(model, messages, tools, allow_tool_calls, on_text)
        except _Retryable as exc:
            if attempt == attempts:
                raise LLMError(f"{exc} (after {attempts} attempt{'s' if attempts > 1 else ''})") from None
            if on_retry:
                on_retry(f"{exc}; retrying ({attempt + 1}/{attempts})")
    raise AssertionError("unreachable")


def _auth_headers(model: ModelConfig) -> dict:
    return {"Authorization": f"Bearer {model.api_key}"} if model.api_key else {}


def _post_stream(url: str, body: dict, headers: dict, timeout: float):
    """POST JSON and yield the response line by line. `timeout` is the longest allowed silence."""
    try:
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json", **headers})
    except ValueError as exc:
        raise LLMError(f"Invalid base_url: {url} ({exc})") from None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        with exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
        if exc.code == 429 or exc.code >= 500:
            raise _Retryable(f"HTTP {exc.code}: {detail}") from None
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise _Retryable(f"cannot reach {url}: {getattr(exc, 'reason', exc)}") from None
    with response:
        try:
            for raw in response:
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
        except (http.client.HTTPException, OSError) as exc:
            raise _Retryable(f"connection lost while streaming: {exc}") from None


def _parse_chunk(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise _Retryable(f"invalid JSON in the stream: {text[:200]}") from None


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _ollama_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "assistant":
            item = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("reasoning"):
                item["thinking"] = message["reasoning"]
            if message.get("tool_calls"):
                item["tool_calls"] = [{"function": {"name": call["name"], "arguments": call["arguments"]}}
                                      for call in message["tool_calls"]]
            converted.append(item)
        elif message["role"] == "tool":
            converted.append({"role": "tool", "tool_name": message["name"], "content": message["content"]})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return converted


def _ollama_chat(model, messages, tools, allow_tool_calls, on_text) -> Reply:
    body = {**model.request, "messages": _ollama_messages(messages), "stream": True}
    if tools and allow_tool_calls:
        body["tools"] = tools
    reply, done = Reply(), False
    for line in _post_stream(f"{model.base_url}/api/chat", body, _auth_headers(model), model.settings["request_timeout"]):
        if not line.strip():
            continue
        chunk = _parse_chunk(line)
        if chunk.get("error"):
            raise LLMError(f"Ollama error: {chunk['error']}")
        message = chunk.get("message") or {}
        if message.get("thinking"):
            reply.reasoning += message["thinking"]
            on_text("reasoning", message["thinking"])
        if message.get("content"):
            reply.content += message["content"]
            on_text("content", message["content"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            reply.tool_calls.append({"id": call.get("id") or f"call_{len(reply.tool_calls) + 1}",
                                     "name": function.get("name") or "",
                                     "arguments": _parse_arguments(function.get("arguments"))})
        if chunk.get("done"):
            reply.input_tokens, reply.output_tokens = chunk.get("prompt_eval_count"), chunk.get("eval_count")
            done = True
    if not done:
        raise _Retryable("the stream ended before the reply was complete")
    return reply


def _openai_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "assistant":
            item = {"role": "assistant", "content": message.get("content") or (None if message.get("tool_calls") else "")}
            if message.get("tool_calls"):
                item["tool_calls"] = [{"id": call["id"], "type": "function",
                                       "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
                                      for call in message["tool_calls"]]
            if message.get("reasoning_details"):
                item["reasoning_details"] = message["reasoning_details"]  # keeps reasoning models going across tool calls
            converted.append(item)
        elif message["role"] == "tool":
            converted.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return converted


def _merge_reasoning_detail(merged: dict, item: dict) -> None:
    """Streamed reasoning_details arrive as fragments sharing an index; text pieces are concatenated.

    Per OpenRouter's reasoning-tokens docs; the merged items must be passed back unmodified.
    """
    target = merged.setdefault(item.get("index", 0), {})
    for key, value in item.items():
        if key in ("text", "summary", "data") and isinstance(value, str) and isinstance(target.get(key), str):
            target[key] += value
        elif value is not None:
            target[key] = value


def _openai_chat(model, messages, tools, allow_tool_calls, on_text) -> Reply:
    body = {**model.request, "messages": _openai_messages(messages), "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
        if not allow_tool_calls:
            body["tool_choice"] = "none"  # keep tools defined (some providers require it) but forbid calls
    reply, done = Reply(), False
    calls: dict[int, dict] = {}
    details: dict[int, dict] = {}
    for line in _post_stream(f"{model.base_url}/chat/completions", body, _auth_headers(model), model.settings["request_timeout"]):
        if not line.startswith("data:"):
            continue  # blank lines, ": comments" and other SSE fields
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = _parse_chunk(data)
        if chunk.get("error"):
            error = chunk["error"]
            raise LLMError(f"API error: {error.get('message', error) if isinstance(error, dict) else error}")
        if chunk.get("usage"):
            reply.input_tokens = chunk["usage"].get("prompt_tokens")
            reply.output_tokens = chunk["usage"].get("completion_tokens")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if not reasoning:  # some providers only stream the structured reasoning_details
                reasoning = "".join(item.get("text") or item.get("summary") or ""
                                    for item in delta.get("reasoning_details") or [] if isinstance(item, dict))
            if reasoning:
                reply.reasoning += reasoning
                on_text("reasoning", reasoning)
            if delta.get("content"):
                reply.content += delta["content"]
                on_text("content", delta["content"])
            for item in delta.get("reasoning_details") or []:
                _merge_reasoning_detail(details, item)
            for piece in delta.get("tool_calls") or []:
                call = calls.setdefault(piece.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                function = piece.get("function") or {}
                call["id"] = call["id"] or piece.get("id") or ""
                call["name"] = call["name"] or function.get("name") or ""
                call["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                done = True
    if not done:
        raise _Retryable("the stream ended before the reply was complete")
    reply.tool_calls = [{"id": call["id"] or f"call_{index}", "name": call["name"],
                         "arguments": _parse_arguments(call["arguments"])}
                        for index, call in sorted(calls.items())]
    reply.reasoning_details = [details[index] for index in sorted(details)]
    return reply


# ---------------------------------------------------------------------------
# The review: prompt, tool-calling loop and the model's answer
# ---------------------------------------------------------------------------

TIME_LIMIT_MESSAGE = ("Time limit reached. Do not call any more tools. "
                      "Write your findings now based on what you have reviewed so far.")


@dataclass
class Review:
    change: Change
    status: str = "clean"                                 # "findings", "clean", "skipped" or "error"
    findings: list = field(default_factory=list)          # [{"file", "line_range", "description"}]
    error: str | None = None
    duration_s: float = 0.0
    tool_calls: list = field(default_factory=list)        # [{"name", "args"}]
    input_tokens: int | None = None
    output_tokens: int | None = None


def load_prompt(path) -> tuple[str, str]:
    """Read the prompt file: an optional ---SYSTEM--- section, then a ---USER--- section."""
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read prompt file {path}: {exc.strerror}") from None
    if "---USER---" not in text:
        raise ConfigError(f"Prompt file {path} has no ---USER--- section")
    system, _, user = text.partition("---USER---")
    return system.replace("---SYSTEM---", "").strip(), user.strip()


def _file_list(paths: list[str], limit: int = 200) -> str:
    text = "\n".join(f"- {path}" for path in paths[:limit])
    if len(paths) > limit:
        text += f"\n- ... and {len(paths) - limit} more files"
    return text


def build_messages(prompt: tuple[str, str], change: Change, max_diff_chars: int) -> list[dict]:
    system, user = prompt
    values = {
        "sha": change.sha or "(uncommitted)",
        "author_name": change.author_name,
        "author_email": change.author_email,
        "timestamp": change.date,
        "message": change.subject,
        "changed_files": _file_list(change.changed_files),
        "diff": truncate(change.diff, max_diff_chars, " — the rest of the diff is not shown"),
    }
    # One pass of plain replacement, not str.format: other braces in a prompt (or in the diff) are left alone.
    placeholders = re.compile(r"\{(" + "|".join(values) + r")\}")
    user = placeholders.sub(lambda match: values[match.group(1)], user)
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def parse_findings(text: str) -> list[dict] | None:
    """Parse the answer: blocks of 'file:' / 'line_range:' / 'description:', or NO_FINDINGS.

    Returns the findings ([] for NO_FINDINGS), or None when the answer can't be parsed.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()  # reasoning some servers leave in the answer
    if not text:
        return None
    if re.search(r"^\s*NO_FINDINGS\s*$", text, re.MULTILINE | re.IGNORECASE):
        return []
    findings = []
    for block in re.split(r"\n(?=file:\s)", text):
        block = block.strip()
        if not block.startswith("file:"):
            continue
        finding, description, in_description = {}, [], False
        for line in block.split("\n"):
            if in_description:
                description.append(line)
            elif line.startswith("file:"):
                finding["file"] = line[len("file:"):].strip()
            elif line.startswith("line_range:"):
                finding["line_range"] = line[len("line_range:"):].strip()
            elif line.startswith("description:"):
                rest = line[len("description:"):].strip()
                if rest:
                    description.append(rest)
                in_description = True
        if finding.get("file") and "\n".join(description).strip():
            findings.append({"file": finding["file"], "line_range": finding.get("line_range") or None,
                             "description": "\n".join(description).strip()})
    return findings or None


def review_change(repo: pathlib.Path, change: Change, model: ModelConfig, prompt, console: Console) -> Review:
    """Review one change: let the model read the diff (and the code), then parse its findings."""
    review = Review(change)
    if not change.diff.strip():
        review.status = "skipped"
        return review
    files = WorkingTreeFiles(repo) if change.sha is None else CommitFiles(repo, change.sha)
    messages = build_messages(prompt, change, model.settings["max_diff_chars"])
    console.detail(f"prompt: {sum(len(m['content']) for m in messages):,} chars (diff: {len(change.diff):,} chars)")
    started = time.monotonic()
    try:
        answer = _tool_loop(messages, files, model, console, review)
    except LLMError as exc:
        review.status, review.error = "error", str(exc)
    else:
        findings = parse_findings(answer)
        if findings is None:
            review.status = "error"
            review.error = "Could not parse the model's answer: " + (answer.strip()[:300] or "(empty answer)")
        else:
            review.findings, review.status = findings, ("findings" if findings else "clean")
    review.duration_s = time.monotonic() - started
    return review


def _tool_loop(messages: list[dict], files, model: ModelConfig, console: Console, review: Review) -> str:
    """Let the model call tools until it answers. Returns the answer text."""
    seen: set[tuple[str, str]] = set()
    started = time.monotonic()
    while True:
        reply = _ask(model, messages, console, review)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            key = (call["name"], json.dumps(call["arguments"], sort_keys=True))
            repeated = key in seen
            seen.add(key)
            show_tool_call(console, call, repeated)
            if repeated:
                result = (f"Error: you already called {call['name']} with these exact arguments. "
                          "Use the result from the earlier call instead of repeating it.")
            else:
                result = run_tool(files, call["name"], call["arguments"], model.settings["max_tool_output_chars"])
                show_tool_result(console, result)
            review.tool_calls.append({"name": call["name"], "args": call["arguments"]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
        time_limit = model.settings["time_limit"]
        if time_limit and time.monotonic() - started >= time_limit:
            return _finish_after_time_limit(messages, model, console, review)


def _finish_after_time_limit(messages, model, console, review) -> str:
    console.warn(f"Time limit ({model.settings['time_limit']}s) reached; asking for the findings")
    messages.append({"role": "user", "content": TIME_LIMIT_MESSAGE})
    for _ in range(3):
        reply = _ask(model, messages, console, review, allow_tool_calls=False)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                             "content": "Error: no tools available."})
        messages.append({"role": "user", "content": "Write only your findings. Do not call tools."})
    return reply.content


def _ask(model, messages, console, review, allow_tool_calls=True) -> Reply:
    """One model request: live output, token accounting, and the reply added to the conversation."""
    console.start_status("thinking")
    try:
        reply = chat(model, messages, TOOL_DEFINITIONS, allow_tool_calls,
                     on_text=console.model_output, on_retry=console.warn)
    finally:
        console.stop_status()
    if reply.input_tokens is not None:
        review.input_tokens = (review.input_tokens or 0) + reply.input_tokens
    if reply.output_tokens is not None:
        review.output_tokens = (review.output_tokens or 0) + reply.output_tokens
    if reply.input_tokens is not None or reply.output_tokens is not None:
        console.detail(f"tokens: {reply.input_tokens or 0:,} in · {reply.output_tokens or 0:,} out", 4)
    messages.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls,
                     "reasoning": reply.reasoning, "reasoning_details": reply.reasoning_details})
    return reply


def describe_tool_call(name: str, args: dict) -> str:
    """A short human-readable summary of a tool call, e.g. 'app/calc.py  lines 1–80'."""
    try:
        if name == "read_file":
            text = str(args.get("path", ""))
            offset, count = args.get("offset"), args.get("line_count")
            if offset and count:
                text += f"  lines {int(offset)}–{int(offset) + int(count) - 1}"
            elif offset:
                text += f"  from line {int(offset)}"
            elif count:
                text += f"  first {int(count)} lines"
            return text
        if name == "grep_search":
            return f'"{args.get("pattern", "")}"' + (f"  in {args['path']}" if args.get("path") else "")
        if name == "list_directory":
            return str(args.get("path") or ".")
    except Exception:
        pass
    return json.dumps(args, default=str)


def show_tool_call(console: Console, call: dict, repeated: bool = False) -> None:
    name = str(call["name"] or "")
    text = console.style("⚙ ", "cyan") + f"{name:<15}" + describe_tool_call(name, call["arguments"])
    if repeated:
        text += console.style("  (repeated call, refused)", "yellow")
    console.line(text, 2)


def show_tool_result(console: Console, result: str) -> None:
    lines = result.splitlines()
    if result.startswith("Error:"):
        console.line(console.style("↳ " + lines[0], "yellow"), 4)
    elif console.verbose:
        for text in lines[:3]:
            console.line(console.style("│ " + text[: max(console.width - 8, 20)], "dim"), 4)
        more = f"{len(lines) - 3} more lines · " if len(lines) > 3 else ""
        console.line(console.style(f"└ {more}{len(result):,} chars", "dim"), 4)


# ---------------------------------------------------------------------------
# Reports: JSON for tools and integrations, Markdown for people
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def summary_text(reviews: list[Review], duration_s: float) -> str:
    counts = Counter(review.status for review in reviews)
    total = sum(len(review.findings) for review in reviews)
    parts = [f"{len(reviews)} reviewed", f"{counts['findings']} with findings ({total})", f"{counts['clean']} clean"]
    if counts["skipped"]:
        parts.append(f"{counts['skipped']} skipped")
    parts += [f"{counts['error']} error{'' if counts['error'] == 1 else 's'}", format_duration(duration_s)]
    return " · ".join(parts)


def review_markdown(review: Review, model: ModelConfig, heading: str = "AI code review findings") -> str:
    """A ready-to-post Markdown body for one review with findings (used for issues and reports)."""
    change = review.change
    lines = [f"## {heading}", ""]
    if change.sha:
        lines.append(f"**Commit:** {change.sha}  ")  # GitLab and GitHub turn a full SHA into a link
    lines += [
        f"**Author:** {change.author_name} ({change.author_email})  ",
        f"**Date:** {change.date}  ",
        f"**Message:** {change.subject}",
        "", "---", "",
    ]
    for number, finding in enumerate(review.findings, 1):
        location = f"`{finding['file']}`" + (f" line {finding['line_range']}" if finding["line_range"] else "")
        lines += [f"### {number}. {location}", "", finding["description"], ""]
    lines += [
        "---",
        "*Generated automatically by AI review.*",
        "",
        "<details>",
        "<summary>Review details</summary>",
        "",
        f"- **Model:** {model.name} (`{model.request['model']}` via {model.api})",
        f"- **Duration:** {review.duration_s:.1f}s",
    ]
    if review.input_tokens is not None:
        lines.append(f"- **Tokens:** {review.input_tokens:,} in · {review.output_tokens or 0:,} out")
    lines.append(f"- **Tool calls:** {len(review.tool_calls)}")
    for call in review.tool_calls:
        arguments = ", ".join(f"{key}={value!r}" for key, value in call["args"].items())
        lines.append(f"  - `{call['name']}({arguments})`")
    lines += ["", "**Request parameters**", "", "```json", json.dumps(model.request, indent=2), "```", "", "</details>"]
    return "\n".join(lines)


def review_record(review: Review, model: ModelConfig) -> dict:
    change = review.change
    tokens = None
    if review.input_tokens is not None or review.output_tokens is not None:
        tokens = {"input": review.input_tokens, "output": review.output_tokens}
    return {
        "kind": change.kind,
        "sha": change.sha,
        "subject": change.subject,
        "author_name": change.author_name,
        "author_email": change.author_email,
        "date": change.date,
        "changed_files": change.changed_files,
        "status": review.status,
        "findings": review.findings,
        "error": review.error,
        "duration_s": round(review.duration_s, 1),
        "tool_calls": review.tool_calls,
        "tokens": tokens,
        "markdown": review_markdown(review, model) if review.status == "findings" else None,
    }


def build_report(model: ModelConfig, reviews: list[Review], started_at: str, duration_s: float) -> dict:
    return {
        "model": model.name,
        "api": model.api,
        "base_url": model.base_url,
        "request": model.request,      # never contains the API key, which lives outside "request"
        "settings": model.settings,
        "started_at": started_at,
        "duration_s": round(duration_s, 1),
        "reviews": [review_record(review, model) for review in reviews],
    }


def markdown_report(model: ModelConfig, reviews: list[Review], duration_s: float) -> str:
    lines = ["# AI review", "", f"{summary_text(reviews, duration_s)} · model `{model.name}`", ""]
    for review in reviews:
        if review.status == "findings":
            lines += [review_markdown(review, model, f"{review.change.label} · {review.change.subject}"), ""]
    others = [review for review in reviews if review.status != "findings"]
    if others:
        lines += ["## Other reviews", ""]
        for review in others:
            note = {"clean": "no findings", "skipped": "skipped (nothing to review)"}.get(review.status, f"error: {review.error}")
            lines.append(f"- `{review.change.label}` {review.change.subject}: {note}")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

EXAMPLES = """\
examples:
  python3 review.py                    review your uncommitted changes
  python3 review.py HEAD               review the last commit
  python3 review.py main~5..main -v    review each commit in a range, streaming the model live
  python3 review.py HEAD --json -      print the JSON report to stdout

exit codes: 0 finished · 1 findings (with --fail-on-findings) · 2 nothing could be reviewed
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="review.py", description="Review git changes with an LLM that can read your code.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", metavar="TARGET",
                        help="a commit (SHA, HEAD~2, tag, ...) or range (main~5..main); default: uncommitted changes")
    parser.add_argument("--model", help='model profile from models.json (default: $AI_REVIEW_MODEL, else "default")')
    parser.add_argument("--config", type=pathlib.Path, default=SCRIPT_DIR / "models.json",
                        help="model profiles file (default: models.json next to this script)")
    parser.add_argument("--prompt", type=pathlib.Path, help="prompt file (default: prompt.txt next to the config)")
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."),
                        help="repository to review (default: the current directory)")
    parser.add_argument("--time-limit", type=int, metavar="SECONDS",
                        help="per-review time for tool calls before the model must answer; 0 = no limit")
    parser.add_argument("--json", dest="json_path", metavar="PATH", help='write the JSON report to PATH ("-" = stdout)')
    parser.add_argument("--markdown", dest="markdown_path", metavar="PATH", help="write a Markdown report to PATH")
    parser.add_argument("--fail-on-findings", action="store_true", help="exit with code 1 when anything was found")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream the model's reasoning and answer live, show tool results and token usage")
    return parser.parse_args(argv)


def show_header(console: Console, model: ModelConfig) -> None:
    console.line(console.style("◆ AI review", "bold", "magenta")
                 + console.style(f" · {model.name} ({model.api} @ {model.host})", "dim"))


def show_settings(console: Console, model: ModelConfig, repo: pathlib.Path, loaded_env: list[str]) -> None:
    console.detail(f"repo      {repo}")
    console.detail(f"endpoint  {model.base_url} ({model.api})")
    console.detail(f"request   {json.dumps(model.request)}")
    console.detail("settings  " + " · ".join(f"{key}={value}" for key, value in model.settings.items() if key != "prompt"))
    console.detail(f"prompt    {model.settings['prompt']}")
    if loaded_env:
        console.detail(f".env      {', '.join(loaded_env)}")


def show_review_start(console: Console, index: int, total: int, change: Change) -> None:
    added, removed = diff_stats(change.diff)
    files = len(change.changed_files)
    console.line()
    console.line(f"[{index}/{total}] " + console.style(change.label, "yellow", "bold") + f"  {change.subject}"
                 + console.style(f" · {change.author_name}", "dim"))
    console.line(console.style(f"› {files} file{'' if files == 1 else 's'} changed, +{added} −{removed}", "dim"), 2)


def show_review_result(console: Console, review: Review) -> None:
    stats = format_duration(review.duration_s)
    if review.tool_calls:
        stats += f" · {len(review.tool_calls)} tool call{'' if len(review.tool_calls) == 1 else 's'}"
    if console.verbose and review.input_tokens is not None:
        stats += f" · {review.input_tokens:,} tokens in, {review.output_tokens or 0:,} out"
    if review.status == "findings":
        count = len(review.findings)
        console.line(console.style(f"✗ {count} finding{'' if count == 1 else 's'}", "red", "bold")
                     + console.style(f" · {stats}", "dim"), 2)
        for finding in review.findings:
            console.line()
            location = finding["file"] + (f":{finding['line_range']}" if finding["line_range"] else "")
            console.line(console.style(location, "bold"), 4)
            console.wrapped(finding["description"], 4)
    elif review.status == "clean":
        console.line(console.style("✓ No findings", "green", "bold") + console.style(f" · {stats}", "dim"), 2)
    elif review.status == "skipped":
        console.line(console.style("– Skipped: nothing to review in this commit (merge commit?)", "dim"), 2)
    else:
        console.line(console.style("! Review failed: ", "red", "bold") + (review.error or "unknown error"), 2)


def write_output(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.flush()
    else:
        pathlib.Path(path).write_text(text, encoding="utf-8")


def run(args: argparse.Namespace, console: Console) -> int:
    started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    loaded_env = load_dotenv(SCRIPT_DIR / ".env")
    model = load_config(args.config, args.model)
    if args.time_limit is not None:
        model.settings["time_limit"] = args.time_limit
    if args.prompt:
        model.settings["prompt"] = str(args.prompt.resolve())
    prompt = load_prompt(model.settings["prompt"])
    repo = find_repo_root(args.repo)
    for path in (args.json_path, args.markdown_path):  # fail now, not after minutes of reviewing
        if path and path != "-" and not pathlib.Path(path).resolve().parent.is_dir():
            raise ConfigError(f"Cannot write {path}: the directory {pathlib.Path(path).resolve().parent} does not exist")

    show_header(console, model)
    show_settings(console, model, repo, loaded_env)
    changes = collect_changes(repo, args.targets, console, model.settings["max_diff_chars"])
    if args.targets:
        console.line(f"{len(changes)} commit{'' if len(changes) == 1 else 's'} from {' '.join(args.targets)}", 2)
    else:
        console.line(f"Uncommitted changes in {repo.name}", 2)

    reviews = []
    for index, change in enumerate(changes, 1):
        show_review_start(console, index, len(changes), change)
        try:
            review = review_change(repo, change, model, prompt, console)
        except Exception as exc:  # one broken review should not stop the others
            console.detail(traceback.format_exc().rstrip(), 4)
            review = Review(change, status="error", error=f"{type(exc).__name__}: {exc}")
        show_review_result(console, review)
        reviews.append(review)

    duration = time.monotonic() - started
    console.line()
    if reviews:
        console.line(console.style("━━ ", "magenta") + summary_text(reviews, duration))
    else:
        console.line("Nothing to review.", 2)
    if args.json_path:
        write_output(args.json_path, json.dumps(build_report(model, reviews, started_at, duration), indent=2, ensure_ascii=False) + "\n")
    if args.markdown_path:
        write_output(args.markdown_path, markdown_report(model, reviews, duration))

    if reviews and all(review.status == "error" for review in reviews):
        return 2
    if args.fail_on_findings and any(review.status == "findings" for review in reviews):
        return 1
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.stderr.reconfigure(errors="backslashreplace")
    console = Console(verbose=args.verbose)
    try:
        return run(args, console)
    except (ConfigError, GitError) as exc:
        console.error(str(exc))
        return 2
    except KeyboardInterrupt:
        console.stop_status()
        console.error("Interrupted.")
        return 130
    except Exception as exc:  # anything unexpected; never exit 1, which means "findings"
        console.stop_status()
        console.error(f"{type(exc).__name__}: {exc}")
        console.detail(traceback.format_exc().rstrip())
        return 2


if __name__ == "__main__":
    sys.exit(main())

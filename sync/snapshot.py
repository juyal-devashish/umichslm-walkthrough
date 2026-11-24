#!/usr/bin/env python3
"""Weekly snapshot sync for the walkthrough repository.

Reads an upstream checkout, takes the newest commit at least N days old, copies
the allowlisted text files through redaction, runs fail-closed gates, and
rewrites the managed files in the target. Prints counts and rule names only,
never matched text, because the run logs of a public repo are public.

Private terms and name replacements come from the environment so they never
appear in this repository:
  SYNC_PRIVATE_TERMS         comma-separated words that must not appear anywhere
  SYNC_NAME_REPLACEMENTS     "Full Name=replacement;Other Name=replacement"
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXT = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".gitignore", ""}
MAX_BYTES = 1_000_000
VERBATIM = {"LICENSE"}  # carries the upstream copyright notice the MIT licence requires
OWNED_PREFIXES = ("sync/", ".github/", "about/")
OWNED_FILES = {"README.md", "CHANGELOG.md"}
BANNER = ("<!-- sync:banner -->", "<!-- /sync:banner -->")

# Ordered redactions: (pattern, replacement)
PATH_RULES = [
    (re.compile(r"/scratch/[^\s\"'`\\):,;]+"), "/path/to/data"),
    (re.compile(r"/nfs/[^\s\"'`\\):,;]+"), "/path/to/data"),
    (re.compile(r"/home/[A-Za-z0-9._-]+"), "/home/user"),
    (re.compile(r"\bcse\d{3}[a-z]\d{3}\w*", re.I), "<account>"),
    (re.compile(r"(--account=)\S+"), r"\1<account>"),
    (re.compile(r"(--mail-user=)\S+"), r"\1user@example.com"),
    (re.compile(r"\b(?!user@example\.com)[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|edu|org|net|io|ai|gov|us)\b"), "user@example.com"),
]
# Fail-closed gates: (rule name, pattern). Names are safe to print; matches are not.
GATES = [
    ("secret-hf-token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("secret-openai-style-key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("secret-github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("secret-aws-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("secret-private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("cluster-path", re.compile(r"/scratch/|/nfs/")),
    ("home-path", re.compile(r"/home/(?!user\b|[%$])")),
    ("email-address", re.compile(r"(?<![\w.])[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|edu|org|net|io|ai|gov|us)\b")),
    ("institutional-host", re.compile(r"\.umich\.edu")),
    ("course-account", re.compile(r"cse\d{3}[a-z]\d{3}\w*", re.I)),
]


class GateError(RuntimeError):
    pass


@dataclass
class Rules:
    includes: list[str] = field(default_factory=list)
    existing_dirs: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "Rules":
        r = cls()
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("!"):
                r.excludes.append(line[1:])
            elif line.startswith("@existing "):
                r.existing_dirs.append(line.split(None, 1)[1].strip())
            else:
                r.includes.append(line)
        return r

    def _glob(self, path: str, pat: str) -> bool:
        if fnmatch.fnmatchcase(path, pat):
            return True
        return pat.startswith("**/") and fnmatch.fnmatchcase(path, pat[3:])

    def allows(self, path: str, existing: set[str]) -> bool:
        if any(self._glob(path, p) for p in self.excludes):
            return False
        for inc in self.includes:
            if inc.endswith("/") and path.startswith(inc):
                return True
            if path == inc:
                return True
        return any(path.startswith(d) and path in existing for d in self.existing_dirs)

    def managed_by(self, path: str) -> bool:
        """Paths the sync may delete: anything the allowlist could ever publish."""
        if path in OWNED_FILES or path.startswith(OWNED_PREFIXES):
            return False
        return any(path.startswith(d) for d in self.includes if d.endswith("/")) \
            or path in self.includes \
            or any(path.startswith(d) for d in self.existing_dirs)


def private_terms() -> list[re.Pattern]:
    terms = [t.strip() for t in os.environ.get("SYNC_PRIVATE_TERMS", "").split(",") if t.strip()]
    return [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms]


def name_replacements() -> list[tuple[re.Pattern, str]]:
    out = []
    for pair in os.environ.get("SYNC_NAME_REPLACEMENTS", "").split(";"):
        if "=" in pair:
            name, repl = pair.split("=", 1)
            out.append((re.compile(rf"\b{re.escape(name.strip())}\b"), repl.strip()))
    return out


def redact(text: str) -> str:
    for pat, repl in name_replacements() + PATH_RULES:
        text = pat.sub(repl, text)
    return text


def scan_text(path: str, text: str) -> list[str]:
    """Return 'path:line rule' findings, never the matched content."""
    hits = []
    extra = [("private-term", p) for p in private_terms()]
    for n, line in enumerate(text.splitlines(), 1):
        for rule, pat in GATES + extra:
            if pat.search(line):
                hits.append(f"{path}:{n} {rule}")
    return hits


def run(*args: str, cwd: Path | None = None, binary: bool = False):
    p = subprocess.run(args, cwd=cwd, check=True, capture_output=True)
    return p.stdout if binary else p.stdout.decode().strip()


def pick_commit(upstream: Path, lag_days: int, now: dt.datetime) -> tuple[str, str]:
    cutoff = (now - dt.timedelta(days=lag_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = run("git", "rev-list", "-1", f"--before={cutoff}", "HEAD", cwd=upstream)
    if not sha:
        raise GateError("no upstream commit is old enough")
    date = run("git", "show", "-s", "--format=%cs", sha, cwd=upstream)
    return sha, date


def export_tree(upstream: Path, sha: str) -> dict[str, bytes]:
    blob = run("git", "archive", sha, cwd=upstream, binary=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for m in tar:
            if m.isfile():
                files[m.name] = tar.extractfile(m).read()
    return files


def build_snapshot(upstream_files: dict[str, bytes], rules: Rules, existing: set[str]) -> tuple[dict[str, str], list[str]]:
    out, skipped = {}, []
    for path in sorted(upstream_files):
        if not rules.allows(path, existing):
            continue
        data = upstream_files[path]
        ext = Path(path).suffix.lower()
        if ext not in TEXT_EXT or len(data) > MAX_BYTES or b"\0" in data:
            skipped.append(path)
            continue
        text = data.decode("utf-8", errors="replace")
        out[path] = text if path in VERBATIM else redact(text)
    return out, skipped


def gate(snapshot: dict[str, str]) -> None:
    hits = [h for p, t in snapshot.items() if p not in VERBATIM for h in scan_text(p, t)]
    if hits:
        print(f"GATE FAILED: {len(hits)} finding(s); nothing was written", file=sys.stderr)
        for h in hits[:50]:
            print("  " + h, file=sys.stderr)
        raise GateError(f"{len(hits)} gate findings")


def tracked_files(target: Path) -> set[str]:
    out = run("git", "ls-files", "-z", cwd=target)
    return {p for p in out.split("\0") if p}


def apply_snapshot(target: Path, snapshot: dict[str, str], rules: Rules, current: set[str]):
    added = [p for p in snapshot if p not in current]
    removed = [p for p in current if rules.managed_by(p) and p not in snapshot]
    modified = []
    for p, text in snapshot.items():
        f = target / p
        if f.exists():
            if f.read_text(encoding="utf-8") != text:
                modified.append(p)
            else:
                continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    for p in removed:
        (target / p).unlink(missing_ok=True)
    return sorted(added), sorted(modified), sorted(removed)


def area(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "(root)"


def summarize_counts(added, modified, removed) -> str:
    def tally(paths):
        c: dict[str, int] = {}
        for p in paths:
            c[area(p)] = c.get(area(p), 0) + 1
        return ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    parts = [f"{label}: {tally(ps)}" for label, ps in (("added", added), ("modified", modified), ("removed", removed)) if ps]
    return "; ".join(parts)


def stamp_banner(readme: Path, sha: str, date: str, synced: str) -> None:
    text = readme.read_text(encoding="utf-8")
    body = (f"> **Snapshot, not the live codebase.** This is a periodically refreshed, reviewed snapshot "
            f"of the project, taken from upstream commit `{sha[:7]}` ({date}). Active development stays in a "
            f"private repository, so recent work will not appear here.  \n"
            f"> Last synced: {synced}")
    pat = re.compile(re.escape(BANNER[0]) + r".*?" + re.escape(BANNER[1]), re.S)
    new = f"{BANNER[0]}\n{body}\n{BANNER[1]}"
    if not pat.search(text):
        raise GateError("README.md is missing the sync banner markers")
    readme.write_text(pat.sub(lambda _: new, text), encoding="utf-8")


def prepend_changelog(path: Path, entry: str) -> None:
    head = "# Changelog\n\nEach entry is one snapshot. Revert the matching pull request to roll one back.\n\n"
    old = path.read_text(encoding="utf-8") if path.exists() else head
    if not old.startswith("# Changelog"):
        old = head + old
    marker = old.index("\n\n", old.index("Each entry")) + 2 if "Each entry" in old else len(head)
    path.write_text(old[:marker] + entry + "\n" + old[marker:], encoding="utf-8")


def cmd_sync(a) -> int:
    target, upstream = Path(a.target).resolve(), Path(a.upstream).resolve()
    now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    rules = Rules.parse((target / "sync/allowlist.txt").read_text())
    state_file = target / "sync/STATE.json"
    prev = json.loads(state_file.read_text()) if state_file.exists() else {}

    sha, date = pick_commit(upstream, a.lag_days, now)
    if sha == prev.get("upstream_sha"):
        print("up to date: no new snapshot commit")
        return emit(a, changed=False, sha=sha)

    current = tracked_files(target)
    files = export_tree(upstream, sha)
    snapshot, skipped = build_snapshot(files, rules, current)
    unpublished = sorted(p for p in files if p.startswith(tuple(rules.existing_dirs)) and p not in current and not rules.allows(p, current))
    gate(snapshot)

    added, modified, removed = apply_snapshot(target, snapshot, rules, current)
    print(f"snapshot {sha[:7]} ({date}): +{len(added)} ~{len(modified)} -{len(removed)}; "
          f"{len(skipped)} non-text skipped; {len(unpublished)} new upstream files awaiting allowlist")
    if not (added or modified or removed):
        state_file.write_text(json.dumps({**prev, "upstream_sha": sha}, indent=2) + "\n")
        return emit(a, changed=False, sha=sha)
    if a.dry_run:
        print("dry run: files written to the working tree; STATE, banner, changelog untouched")
        return emit(a, changed=False, sha=sha)

    synced = now.strftime("%Y-%m-%d")
    state_file.write_text(json.dumps({"upstream_sha": sha, "upstream_date": date, "synced_at": synced}, indent=2) + "\n")
    stamp_banner(target / "README.md", sha, date, synced)
    note = ""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from summarize import weekly_note  # sibling module
        note = weekly_note(target)
    entry = f"## {synced} — snapshot of upstream `{sha[:7]}` ({date})\n\n{summarize_counts(added, modified, removed)}\n" + (f"\n{note}\n" if note else "")
    prepend_changelog(target / "CHANGELOG.md", entry)
    return emit(a, changed=True, sha=sha)


def emit(a, changed: bool, sha: str) -> int:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\nsha={sha[:7]}\n")
    return 0


def cmd_scan(a) -> int:
    target = Path(a.target).resolve()
    hits = []
    for p in tracked_files(target):
        if p in VERBATIM or p.startswith((".git/", "sync/", ".github/")) or Path(p).suffix.lower() not in TEXT_EXT | {".ipynb"}:
            continue
        hits += scan_text(p, (target / p).read_text(encoding="utf-8", errors="replace"))
    if hits:
        print(f"SCAN FAILED: {len(hits)} finding(s)", file=sys.stderr)
        for h in hits[:50]:
            print("  " + h, file=sys.stderr)
        return 1
    print("scan clean")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--upstream", required=True)
    s.add_argument("--target", default=".")
    s.add_argument("--lag-days", type=int, default=14)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--now", help="ISO timestamp with offset, for tests")
    c = sub.add_parser("scan")
    c.add_argument("--target", default=".")
    a = ap.parse_args(argv)
    try:
        return cmd_sync(a) if a.cmd == "sync" else cmd_scan(a)
    except GateError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

# Sync tooling

A daily GitHub Actions job (`.github/workflows/sync.yml`) refreshes this repository from a private source.

**Flow.** Pick the newest source commit at least 14 days old, copy allowlisted text files, apply redactions, run the gates, then open a pull request and squash-merge it. Reverting that pull request rolls the snapshot back.

**Safety.** `allowlist.txt` is the only thing that decides what can be published. The gates fail closed: on any finding the job writes nothing and opens an issue naming rules and file lines, never matched text. `gitleaks` runs on the result as well.

## One-time setup (repository settings)

| Kind | Name | Purpose |
| --- | --- | --- |
| Secret | `UPSTREAM_READ_TOKEN` | Fine-grained token, read-only on the source repository's contents |
| Variable | `UPSTREAM_REPO` | `owner/name` of the source repository |
| Variable | `SYNC_PRIVATE_TERMS` | Comma-separated words that must never appear in published files |
| Variable | `SYNC_NAME_REPLACEMENTS` | `Name=replacement;Name2=replacement` applied before the gates |
| Secret (optional) | `ANTHROPIC_API_KEY` | Enables a short generated change note per snapshot |

These live in settings, not in the repository, so the names they contain stay out of the published files.

## Local run

```bash
python3 sync/test_sync.py
python3 sync/snapshot.py sync --upstream /path/to/source --target . --dry-run
```

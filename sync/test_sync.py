"""Deterministic tests: temp git repos, fixed clock, no network."""
import datetime as dt
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import snapshot as S  # noqa: E402

os.environ["SYNC_PRIVATE_TERMS"] = "Zork"
os.environ["SYNC_NAME_REPLACEMENTS"] = "Zork Quux=the project lead"
UTC = dt.timezone.utc
RULES = S.Rules.parse("src/\n@existing tests/\nLICENSE\n!**/*.log\n")


def git(cwd, *a, env=None):
    e = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
         "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", **(env or {})}
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=e).stdout.strip()


def commit(repo, files, when):
    for p, t in files.items():
        f = Path(repo) / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t)
    git(repo, "add", "-A")
    d = when.isoformat()
    git(repo, "commit", "-qm", "c", env={"GIT_AUTHOR_DATE": d, "GIT_COMMITTER_DATE": d})


class Redaction(unittest.TestCase):
    def test_paths_emails_names(self):
        t = "p=/scratch/cls_root/x/y/z --account=cse123a456 a@b.edu Zork Quux wrote /home/bob/x"
        r = S.redact(t)
        self.assertEqual(S.scan_text("f", r), [])
        self.assertIn("the project lead", r)

    def test_matmul_is_not_an_email(self):
        self.assertEqual(S.redact("y = a@b.T"), "y = a@b.T")
        self.assertEqual(S.scan_text("f", "y = a@b.T"), [])

    def test_gate_reports_rule_not_content(self):
        hits = S.scan_text("f.py", "token = hf_" + "a" * 34)
        self.assertEqual(hits, ["f.py:1 secret-hf-token"])
        self.assertNotIn("aaaa", " ".join(hits))

    def test_private_term_blocks(self):
        self.assertEqual(S.scan_text("f", "by Zork"), ["f:1 private-term"])


class Allowlist(unittest.TestCase):
    def test_rules(self):
        ex = {"tests/a.py"}
        self.assertTrue(RULES.allows("src/m.py", ex))
        self.assertTrue(RULES.allows("tests/a.py", ex))
        self.assertFalse(RULES.allows("tests/new.py", ex))
        self.assertFalse(RULES.allows("RL_PLAN.md", ex))
        self.assertFalse(RULES.allows("src/run.log", ex))

    def test_owned_paths_never_managed(self):
        for p in ("README.md", "CHANGELOG.md", "sync/sync.py", ".github/workflows/x.yml", "about/t.md"):
            self.assertFalse(RULES.managed_by(p))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.up, self.wt = root / "up", root / "wt"
        for r in (self.up, self.wt):
            r.mkdir()
            git(r, "init", "-q")
        self.now = dt.datetime(2026, 12, 1, tzinfo=UTC)
        commit(self.up, {"src/a.py": "x = 1  # /scratch/c/u/d\n", "LICENSE": "MIT\nCopyright Zork\n",
                         "SECRET_PLAN.md": "no"}, self.now - dt.timedelta(days=30))
        commit(self.up, {"src/b.py": "y = 2\n"}, self.now - dt.timedelta(days=3))  # inside the lag window
        (self.wt / "sync").mkdir()
        (self.wt / "sync/allowlist.txt").write_text("src/\nLICENSE\n")
        (self.wt / "README.md").write_text("# t\n<!-- sync:banner -->\nold\n<!-- /sync:banner -->\n")
        commit(self.wt, {"src/stale.py": "gone\n"}, self.now - dt.timedelta(days=60))

    def tearDown(self):
        self.tmp.cleanup()

    def run_sync(self, *extra):
        return S.main(["sync", "--upstream", str(self.up), "--target", str(self.wt), "--now", self.now.isoformat(), *extra])

    def test_lag_redaction_ownership_and_idempotence(self):
        self.assertEqual(self.run_sync(), 0)
        w = self.wt
        self.assertTrue((w / "src/a.py").exists())
        self.assertFalse((w / "src/b.py").exists(), "commit inside the lag window must not publish")
        self.assertFalse((w / "src/stale.py").exists(), "managed files absent upstream are removed")
        self.assertFalse((w / "SECRET_PLAN.md").exists())
        self.assertIn("/path/to/data", (w / "src/a.py").read_text())
        self.assertIn("Copyright Zork", (w / "LICENSE").read_text(), "LICENSE is verbatim")
        self.assertIn("Last synced: 2026-12-01", (w / "README.md").read_text())
        self.assertIn("snapshot of upstream", (w / "CHANGELOG.md").read_text())
        git(w, "add", "-A"); git(w, "commit", "-qm", "s")
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(git(w, "status", "--porcelain"), "", "second run changes nothing")

    def test_gate_failure_writes_nothing(self):
        commit(self.up, {"src/c.py": "k='hf_" + "b" * 34 + "'\n"}, self.now - dt.timedelta(days=20))
        self.assertEqual(self.run_sync(), 2)
        self.assertEqual(git(self.wt, "status", "--porcelain"), "")

    def test_scan_ignores_license_only(self):
        (self.wt / "LICENSE").write_text("Copyright Zork\n")
        (self.wt / "src").mkdir(exist_ok=True)
        (self.wt / "src/z.py").write_text("# Zork\n")
        git(self.wt, "add", "-A")
        self.assertEqual(S.main(["scan", "--target", str(self.wt)]), 1)


if __name__ == "__main__":
    unittest.main()

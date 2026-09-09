"""Source selection and checkout preservation, using temporary local Git repos."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dev", ROOT / "scripts/dev.py")
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


class DevTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "upstream"
        self.repo.mkdir()
        self.command("init", "--quiet", "--initial-branch=main")
        self.command("config", "user.name", "Synthetic test")
        self.command("config", "user.email", "test@example.com")
        (self.repo / "content.txt").write_text("first\n")
        for name, directory in dev.OSS_PACKAGES.items():
            path = self.repo / directory
            path.mkdir()
            (path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.2.0"\n')
        self.commit = self.commit_change("Initial test source")
        self.cache = self.root / "cache"

    def command(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit_change(self, message):
        self.command("add", ".")
        self.command("commit", "--quiet", "-m", message)
        return self.command("rev-parse", "HEAD")

    def choice(self, ref=None):
        return {"kind": "git", "repository": str(self.repo), "ref": ref or self.commit}

    def test_default_is_pinned_and_release_rejects_overrides(self):
        selected = dev.selection(ROOT, {})
        self.assertRegex(selected["ref"], r"^[0-9a-f]{40}$")
        for env in [{"MEMORYLAYER_OSS_REF": "feature/work"}, {"MEMORYLAYER_OSS_PATH": str(self.repo)}]:
            with self.assertRaisesRegex(ValueError, "Release installs"):
                dev.selection(ROOT, env, release=True)

    def test_conflicting_and_invalid_selectors_fail(self):
        for env in [
            {"MEMORYLAYER_OSS_REF": "main", "MEMORYLAYER_OSS_PATH": str(self.repo)},
            {"MEMORYLAYER_OSS_REF": "--upload-pack=invalid"},
            {"MEMORYLAYER_OSS_REF": "main\nother"},
            {"MEMORYLAYER_OSS_PATH": str(self.root / "missing")},
        ]:
            with self.assertRaises(ValueError):
                dev.selection(ROOT, env)

    def test_checked_in_default_must_be_immutable(self):
        (self.root / "oss-source.toml").write_text('repository = "https://github.com/scitrera/memorylayer.git"\ncommit = "main"\n')
        with self.assertRaisesRegex(ValueError, "40-character"):
            dev.selection(self.root, {})

    def test_pinned_snapshot_reused_without_fetch(self):
        first = dev.prepare_source(self.choice(), self.cache)
        with patch.object(dev, "run", side_effect=AssertionError("Unexpected fetch")):
            second = dev.prepare_source(self.choice(), self.cache)
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(second["commit"], self.commit)
        self.assertFalse(second["dirty"])

    def test_branch_moves_to_new_snapshot_and_preserves_old_install(self):
        first = dev.prepare_source(self.choice("main"), self.cache)
        (self.repo / "content.txt").write_text("second\n")
        commit = self.commit_change("Move test branch")
        second = dev.prepare_source(self.choice("main"), self.cache)
        self.assertEqual(second["commit"], commit)
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual((first["path"] / "content.txt").read_text(), "first\n")
        self.assertEqual((second["path"] / "content.txt").read_text(), "second\n")
        self.assertEqual(self.command("branch", "--show-current"), "main")

    def test_edited_managed_snapshot_is_never_reset(self):
        source = dev.prepare_source(self.choice(), self.cache)
        path = source["path"] / "content.txt"
        path.write_text("work in progress\n")
        with self.assertRaisesRegex(ValueError, "was edited"):
            dev.prepare_source(self.choice(), self.cache)
        self.assertEqual(path.read_text(), "work in progress\n")

    def test_dirty_local_checkout_is_used_without_fetch_or_checkout(self):
        path = self.repo / "content.txt"
        path.write_text("local work\n")
        before = self.command("status", "--porcelain")
        choice = dev.selection(ROOT, {"MEMORYLAYER_OSS_PATH": str(self.repo)})
        with patch.object(dev, "run", side_effect=AssertionError("Unexpected mutation")):
            source = dev.prepare_source(choice, self.cache)
        self.assertEqual(source["path"], self.repo)
        self.assertTrue(source["dirty"])
        self.assertEqual(source["commit"], self.commit)
        self.assertEqual(before, self.command("status", "--porcelain"))
        self.assertEqual(self.command("branch", "--show-current"), "main")

    def test_all_python_packages_use_one_checkout_and_core_once(self):
        projects = dev.python_projects(ROOT, self.repo, ["enterprise", "embed"])
        self.assertEqual(projects[:3], [self.repo / path for path in dev.OSS_PACKAGES.values()])
        self.assertEqual(len(projects), 5)
        command = dev.install_command("python", ROOT, projects, editable=True, dev=True)
        self.assertEqual(command.count("--editable"), 5)
        self.assertEqual(command.count(str(self.repo / "memorylayer-core-python")), 1)
        self.assertIn(str(ROOT / "memorylayer-enterprise") + "[dev]", command)
        self.assertNotIn("--no-deps", command)

    def test_image_install_has_no_editables_or_test_extras(self):
        projects = dev.python_projects(ROOT, self.repo, ["enterprise"])
        command = dev.install_command("python", ROOT, projects, editable=False, dev=False)
        self.assertNotIn("--editable", command)
        self.assertFalse(any("[dev]" in arg for arg in command))
        self.assertIn(str(self.repo / "memorylayer-server-rpg-python"), command)

    def test_missing_or_wrong_oss_package_fails_before_install(self):
        path = self.repo / "memorylayer-server-rpg-python/pyproject.toml"
        path.write_text('[project]\nname = "unrelated-package"\n')
        with self.assertRaisesRegex(ValueError, "Incorrect OSS package"):
            dev.python_projects(ROOT, self.repo, ["enterprise"])

    def test_admin_link_does_not_rewrite_release_metadata(self):
        sdk = self.repo / "memorylayer-sdk-typescript"
        sdk.mkdir()
        (sdk / "package.json").write_text(json.dumps({"name": "@scitrera/memorylayer-sdk"}))
        with patch.object(dev, "run") as run:
            dev.install_admin(ROOT, self.repo)
        command = run.call_args_list[-1].args[0]
        for flag in ["--no-save", "--package-lock=false", "--install-links=false", str(sdk)]:
            self.assertIn(flag, command)


if __name__ == "__main__":
    unittest.main()

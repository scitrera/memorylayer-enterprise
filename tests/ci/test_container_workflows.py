# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Container release contracts and nightly retry behavior; no network calls."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


updates = load_script("postgres_updates")
release = load_script("check_release")


def manifest(digit="a", arches=("amd64", "arm64")):
    return {"digest": "sha256:" + digit * 64,
            "manifests": [{"platform": {"os": "linux", "architecture": arch}} for arch in arches]}


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ["postgres-container", "scripts", ".github"]:
            shutil.copytree(ROOT / directory, self.root / directory, ignore=shutil.ignore_patterns("__pycache__"))
        for name in [".dockerignore", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]:
            shutil.copyfile(ROOT / name, self.root / name)
        self.config = yaml.safe_load((ROOT / "versions.yaml").read_text())

    def plan(self, variant="standard", force=False, upstream=None, marker=None):
        values = [upstream or manifest(), marker or updates.RegistryError("not found")]
        with patch.object(updates, "inspect_image", side_effect=values):
            return updates.plan_variant(self.root, variant, self.config, force=force)

    def test_first_build_and_unchanged_success(self):
        self.assertTrue(self.plan()["changed"])
        self.assertFalse(self.plan(marker=manifest())["changed"])

    def test_local_dockerfile_change_invalidates_success(self):
        before = self.plan()["fingerprint"]
        path = self.root / "postgres-container/Dockerfile"
        path.write_text(path.read_text() + "\n# changed recipe\n")
        self.assertNotEqual(before, self.plan()["fingerprint"])

    def test_upstream_digest_change_invalidates_success(self):
        self.assertNotEqual(self.plan()["fingerprint"], self.plan(upstream=manifest("b"))["fingerprint"])

    def test_build_uses_resolved_digest_and_variant_specific_source(self):
        self.assertEqual(self.plan()["base_image"], "pgvector/pgvector:pg17@sha256:" + "a" * 64)
        self.assertEqual(self.plan("cnpg")["base_image"], "ghcr.io/cloudnative-pg/postgresql:17-system-bookworm@sha256:" + "a" * 64)

    def test_forced_build_does_not_consult_success_marker(self):
        with patch.object(updates, "inspect_image", return_value=manifest()) as inspect:
            plan = updates.plan_variant(self.root, "standard", self.config, force=True)
        self.assertTrue(plan["changed"])
        self.assertEqual(inspect.call_count, 1)

    def test_partial_platform_success_marker_is_rebuilt(self):
        self.assertTrue(self.plan(marker=manifest(arches=("amd64",)))["changed"])

    def test_missing_upstream_platform_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "amd64/arm64"):
            self.plan(upstream=manifest(arches=("amd64",)))

    def test_upstream_lookup_failure_is_not_treated_as_unchanged(self):
        with patch.object(updates, "inspect_image", side_effect=updates.RegistryError("offline")):
            with self.assertRaises(updates.RegistryError):
                updates.plan_variant(self.root, "standard", self.config)

    def test_failed_build_marker_lookup_is_retried(self):
        self.assertTrue(self.plan(marker=updates.RegistryError("registry unavailable"))["changed"])

    def test_marker_is_written_after_both_rolling_tags_verify(self):
        plan = self.plan()
        with patch.object(updates.subprocess, "run") as run, patch.object(updates, "inspect_image", return_value=manifest("d")):
            updates.publish(plan, "sha256:" + "b" * 64, "sha256:" + "c" * 64)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn(plan["image"] + ":17", commands[0])
        self.assertIn(plan["image"] + ":nightly", commands[0])
        self.assertIn(plan["image"] + ":inputs-" + plan["fingerprint"], commands[1])
        self.assertNotIn(plan["image"] + ":" + plan["version"], str(commands))

    def test_failed_promotion_does_not_record_success(self):
        plan = self.plan()
        with patch.object(updates.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "create")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                updates.publish(plan, "sha256:" + "b" * 64, "sha256:" + "c" * 64)
        self.assertEqual(run.call_count, 1)

    def test_incomplete_promoted_manifest_does_not_record_success(self):
        plan = self.plan()
        with patch.object(updates.subprocess, "run") as run, patch.object(updates, "inspect_image", return_value=manifest(arches=("amd64",))):
            with self.assertRaises(updates.RegistryError):
                updates.publish(plan, "sha256:" + "b" * 64, "sha256:" + "c" * 64)
        self.assertEqual(run.call_count, 1)

    def test_unpinned_or_unexpected_build_target_is_refused(self):
        plan = self.plan()
        plan["base_image"] = "pgvector/pgvector:pg17"
        with self.assertRaises(ValueError):
            updates.validate_plan(plan)


class WorkflowTests(unittest.TestCase):
    def workflow(self, name):
        # BaseLoader keeps GitHub's "on" key as a string (YAML 1.2 semantics).
        return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)

    def test_all_four_release_images_have_two_native_jobs(self):
        workflow = self.workflow("build-docker.yml")
        for image in ["memorylayer-enterprise", "memorylayer-data-connectors", "memorylayer-postgres", "memorylayer-postgres-cnpg"]:
            amd = f"build-{image}-linux-amd64"
            arm = f"build-{image}-linux-arm64"
            self.assertEqual(workflow["jobs"][amd]["runs-on"], "ubuntu-24.04")
            self.assertEqual(workflow["jobs"][arm]["runs-on"], "ubuntu-24.04-arm")
            self.assertEqual(set(workflow["jobs"][f"merge-{image}"]["needs"]), {amd, arm})
        self.assertNotIn("setup-qemu", json.dumps(workflow))

    def test_pull_requests_have_no_registry_write_path(self):
        publishing = self.workflow("build-docker.yml")
        self.assertNotIn("pull_request", publishing["on"])
        pr = self.workflow("container-check.yml")
        self.assertEqual(pr["permissions"], {"contents": "read"})
        for step in pr["jobs"]["build"]["steps"]:
            self.assertNotIn("login-action", step.get("uses", ""))
            if "build-push-action" in step.get("uses", ""):
                self.assertEqual(step["with"]["push"], "false")

    def test_no_package_registry_publication_is_generated(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            self.assertNotIn("pypa/gh-action-pypi-publish", path.read_text())
            self.assertNotIn("npm publish", path.read_text())
        config = yaml.safe_load((ROOT / "versions.yaml").read_text())
        self.assertEqual(set(config["ci"]["only_workflows"]), {"version-check", "test-python", "test-npm", "build-docker"})

    def test_postgres_license_metadata_cannot_inherit_repository_agpl(self):
        workflow = self.workflow("build-docker.yml")
        config = yaml.safe_load((ROOT / "versions.yaml").read_text())
        self.assertIn("build-docker", config["ci"]["skip_workflows"])
        for job_name, job in workflow["jobs"].items():
            for step in job["steps"]:
                if "docker/metadata-action" not in step.get("uses", ""):
                    continue
                expected = "" if "memorylayer-postgres" in job_name else "AGPL-3.0-only"
                self.assertIn("org.opencontainers.image.licenses=" + expected + "\n", step["with"]["labels"])
            if job_name.startswith("build-memorylayer-postgres"):
                build = next(step for step in job["steps"] if "docker/build-push-action" in step.get("uses", ""))
                self.assertEqual(build["with"]["labels"], "${{ steps.meta.outputs.labels }}")
        action = yaml.load((ROOT / ".github/actions/build-postgres/action.yml").read_text(), Loader=yaml.BaseLoader)
        for step in action["runs"]["steps"]:
            if "docker/build-push-action" in step.get("uses", ""):
                self.assertIn("org.opencontainers.image.licenses=\n", step["with"]["labels"])
        for name in ["Dockerfile", "Dockerfile.cnpg"]:
            source = (ROOT / "postgres-container" / name).read_text()
            self.assertIn('LABEL org.opencontainers.image.licenses=""', source)
            self.assertIn("COPY postgres-container/LICENSE postgres-container/NOTICE", source)
            self.assertNotIn("COPY LICENSE NOTICE", source)
            self.assertNotIn("AGPL-3.0-only", source)

    def test_container_version_matches_the_root_release_tag_source(self):
        config = yaml.safe_load((ROOT / "versions.yaml").read_text())
        for image in config["docker"]["images"].values():
            self.assertEqual(image["version_from"], "memorylayer-extensions")

    def test_nightly_merge_requires_both_native_smoke_tests(self):
        flow = self.workflow("postgres-image.yml")
        self.assertEqual(set(flow["jobs"]["merge"]["needs"]), {"amd64", "arm64"})
        action = yaml.load((ROOT / ".github/actions/build-postgres/action.yml").read_text(), Loader=yaml.BaseLoader)
        steps = action["runs"]["steps"]
        smoke = next(i for i, step in enumerate(steps) if "smoke-postgres.sh" in step.get("run", ""))
        push = next(i for i, step in enumerate(steps) if step.get("id") == "push")
        self.assertLess(smoke, push)

    def test_releases_pin_one_base_for_both_postgres_stages(self):
        for name in ["Dockerfile", "Dockerfile.cnpg"]:
            source = (ROOT / "postgres-container" / name).read_text()
            self.assertRegex(source, r"ARG BASE_IMAGE=\S+@sha256:[0-9a-f]{64}")
            self.assertEqual(source.count("FROM ${BASE_IMAGE}"), 2)

    def test_release_context_rejects_wrong_tag_and_nondefault_dispatch(self):
        release.check_context("push", "tag", "v0.0.1", "main", "0.0.1")
        release.check_context("pull_request", "branch", "feature", "main", "0.0.1")
        with self.assertRaises(ValueError):
            release.check_context("push", "tag", "v0.0.2", "main", "0.0.1")
        with self.assertRaises(ValueError):
            release.check_context("workflow_dispatch", "branch", "feature", "main", "0.0.1")

    def test_private_staging_allows_tests_but_blocks_publication(self):
        for event in ["push", "pull_request"]:
            release.check_context(event, "branch", "main", "main", "0.0.1", repository_private=True)
        for event, ref_type, ref in [("push", "tag", "v0.0.1"), ("workflow_dispatch", "branch", "main")]:
            with self.subTest(event=event):
                with self.assertRaisesRegex(ValueError, "repository is private"):
                    release.check_context(event, ref_type, ref, "main", "0.0.1", repository_private=True)
                release.check_context(event, ref_type, ref, "main", "0.0.1", repository_private=False)

    def test_nightly_publication_is_disabled_for_private_staging(self):
        flow = self.workflow("postgres-updates.yml")
        self.assertIn("github.event.repository.private == false", flow["jobs"]["detect"]["if"])
        for job in ["standard", "cnpg"]:
            self.assertEqual(flow["jobs"][job]["needs"], "detect")


if __name__ == "__main__":
    unittest.main()

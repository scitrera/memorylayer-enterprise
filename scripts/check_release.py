#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Validate the shared container release tag and manual dispatch branch."""
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def check_context(event: str, ref_type: str, ref: str, default_branch: str, version: str,
                  *, repository_private: bool = False) -> None:
    if repository_private and (ref_type == "tag" or event == "workflow_dispatch"):
        raise ValueError("Container publication is disabled while the repository is private")
    if ref_type == "tag" and ref != f"v{version}":
        raise ValueError(f"Release tag must be v{version}, got {ref}")
    if event == "workflow_dispatch" and (ref_type != "branch" or ref != default_branch):
        raise ValueError("Manual container publication must run from the default branch")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    version = str(yaml.safe_load((root / "versions.yaml").read_text())["memorylayer-extensions"])
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    check_context(os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF_TYPE"],
                  os.environ["GITHUB_REF_NAME"], event["repository"]["default_branch"], version,
                  repository_private=event["repository"].get("private", True))
    subprocess.run([sys.executable, str(root / "scripts/dev.py"), "--release", "--check"], check=True)

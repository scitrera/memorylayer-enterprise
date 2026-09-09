#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Install extensions alongside one Git revision or an editable OSS checkout.

Requires Python 3.12+, Git, and pip. The admin target also requires Node/npm.
Release defaults come from oss-source.toml; development can select either
MEMORYLAYER_OSS_REF or MEMORYLAYER_OSS_PATH without editing tracked metadata.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
COMPONENTS = {
    "enterprise": "memorylayer-enterprise",
    "embed": "memorylayer-embed-server-enterprise",
    "connectors": "memorylayer-data-connectors",
    "admin": "memorylayer-admin",
}
OSS_PACKAGES = {
    "memorylayer-server": "memorylayer-core-python",
    "memorylayer-server-rpg": "memorylayer-server-rpg-python",
    "memorylayer-embed-server": "memorylayer-embed-server",
}


def run(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def selection(root: Path, env: dict, *, release: bool = False) -> dict:
    config = tomllib.loads((root / "oss-source.toml").read_text())
    if config.get("repository") != "https://github.com/scitrera/memorylayer.git":
        raise ValueError("oss-source.toml must select the public MemoryLayer repository")
    if not COMMIT.fullmatch(config.get("commit", "")):
        raise ValueError("oss-source.toml must pin a full 40-character commit")
    ref, local = env.get("MEMORYLAYER_OSS_REF", ""), env.get("MEMORYLAYER_OSS_PATH", "")
    if ref and local:
        raise ValueError("Set only one of MEMORYLAYER_OSS_REF and MEMORYLAYER_OSS_PATH")
    if release and (ref or local):
        raise ValueError("Release installs require the checked-in pin; unset OSS development overrides")
    if ref and (ref.startswith("-") or any(c.isspace() or ord(c) < 32 for c in ref)):
        raise ValueError("Invalid MEMORYLAYER_OSS_REF")
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"OSS checkout does not exist: {path}")
        return {"kind": "local", "path": path, "repository": config["repository"]}
    return {"kind": "git", "ref": ref or config["commit"], "repository": config["repository"]}


def validate_snapshot(path: Path, repository: str, commit: str) -> None:
    if path.is_symlink() or not (path / ".git").is_dir():
        raise ValueError(f"Invalid managed OSS snapshot: {path}")
    if git(path, "remote", "get-url", "origin") != repository or git(path, "rev-parse", "HEAD") != commit:
        raise ValueError(f"Managed OSS snapshot does not match its source: {path}")
    if git(path, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"Managed OSS snapshot was edited: {path}; use MEMORYLAYER_OSS_PATH to keep developing those edits")


def prepare_source(choice: dict, cache: Path) -> dict:
    """Fetch branch refs afresh; preserve existing checkouts and dirty local work."""
    if choice["kind"] == "local":
        path = choice["path"]
        try:
            # Do not misattribute an extracted source directory to a parent repo.
            if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path:
                raise ValueError("Not a repository root")
            commit = git(path, "rev-parse", "HEAD")
            dirty = bool(git(path, "status", "--porcelain", "--untracked-files=normal"))
        except (subprocess.CalledProcessError, ValueError):
            commit, dirty = None, None
        return {**choice, "commit": commit, "dirty": dirty}
    cache.mkdir(parents=True, exist_ok=True)
    repository, ref = choice["repository"], choice["ref"]
    if COMMIT.fullmatch(ref) and (cache / ref).exists():
        validate_snapshot(cache / ref, repository, ref)
        return {**choice, "path": cache / ref, "commit": ref, "dirty": False}
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=cache) as temp:
        checkout = Path(temp) / "source"
        run(["git", "init", "--quiet", str(checkout)])
        git(checkout, "remote", "add", "origin", repository)
        # No shell expansion; a branch is resolved once for every package.
        run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", ref])
        git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        commit = git(checkout, "rev-parse", "HEAD")
        if not COMMIT.fullmatch(commit) or (COMMIT.fullmatch(ref) and ref != commit):
            raise ValueError("Fetched OSS revision did not match the requested commit")
        destination = cache / commit
        if not destination.exists():
            try:
                checkout.rename(destination)
            except OSError:
                # Another installer may have completed the identical snapshot.
                if not destination.exists():
                    raise
        validate_snapshot(destination, repository, commit)
        return {**choice, "path": destination, "commit": commit, "dirty": False}


def python_projects(root: Path, source: Path | None, components: list[str]) -> list[Path]:
    names = []
    if "enterprise" in components or "embed" in components:
        names.append("memorylayer-server")
    if "enterprise" in components:
        names.append("memorylayer-server-rpg")
    if "embed" in components:
        names.append("memorylayer-embed-server")
    projects = []
    for name in names:
        path = source / OSS_PACKAGES[name]
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"OSS package escapes selected checkout: {name}")
        metadata = tomllib.loads((path / "pyproject.toml").read_text())["project"]
        if metadata["name"] != name:
            raise ValueError(f"Incorrect OSS package at {path}")
        projects.append(path)
    projects.extend(root / COMPONENTS[c] for c in components if c != "admin")
    return projects


def install_command(python: str, root: Path, projects: list[Path], *, editable: bool, dev: bool) -> list[str]:
    command = [python, "-m", "pip", "install", "--upgrade"]
    for path in projects:
        if editable:
            command.append("--editable")
        # Dev extras belong to the requested extensions, not every OSS project.
        extra = "[dev]" if dev and path.parent == root else ""
        command.append(str(path) + extra)
    return command


def install_admin(root: Path, source: Path) -> None:
    sdk = source / "memorylayer-sdk-typescript"
    if json.loads((sdk / "package.json").read_text())["name"] != "@scitrera/memorylayer-sdk":
        raise ValueError("Selected OSS checkout does not contain the TypeScript SDK")
    run(["npm", "ci"], cwd=sdk)
    run(["npm", "run", "build"], cwd=sdk)
    admin = root / COMPONENTS["admin"]
    run(["npm", "ci"], cwd=admin)
    # Link for co-development without rewriting package.json or the release lock.
    run(["npm", "install", "--no-save", "--package-lock=false", "--install-links=false", str(sdk)], cwd=admin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", choices=list(COMPONENTS))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--python", help="Install into an existing Python interpreter with pip")
    target.add_argument("--venv", type=Path, default=ROOT / ".venv", help="Virtual environment to create/reuse")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".dev/oss")
    parser.add_argument("--no-dev", action="store_true", help="Omit extension test/development extras")
    parser.add_argument("--non-editable", action="store_true", help="Install wheels suitable for an image")
    parser.add_argument("--release", action="store_true", help="Refuse development overrides")
    parser.add_argument("--check", action="store_true", help="Validate source selection without cloning/installing")
    parser.add_argument("--record", type=Path, help="Write provenance after a successful installation")
    args = parser.parse_args()
    components = list(dict.fromkeys(args.components or ["enterprise"]))
    choice = selection(ROOT, os.environ, release=args.release)
    if args.check:
        print(json.dumps(choice, default=str, indent=2))
        return
    source = None
    if any(c != "connectors" for c in components):
        source = prepare_source(choice, args.cache_dir.resolve())
    projects = python_projects(ROOT, source["path"] if source else None, components)
    if projects:
        python = args.python
        if not python:
            environment = args.venv.resolve()
            python_path = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python_path.exists():
                venv.EnvBuilder(with_pip=True).create(environment)
            python = str(python_path)
        run(install_command(python, ROOT, projects, editable=not args.non_editable, dev=not args.no_dev))
        run([python, "-m", "pip", "check"])
    if "admin" in components:
        install_admin(ROOT, source["path"])
    if source:
        record = {key: value for key, value in source.items() if key != "path"}
        if source["kind"] == "local":
            record["path"] = str(source["path"])
        record["components"] = components
        record_path = args.record or ROOT / ".dev/oss-source.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"OSS source: {source['kind']} {source['commit'] or source['path']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

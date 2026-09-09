#!/usr/bin/env python3
"""Plan PostgreSQL rebuilds from source hashes and upstream OCI index digests.

An inputs-<hash> registry tag records a successfully tested, two-platform build.
It is written after both rolling tags are promoted, so failed builds retry on
the next run. No GitHub cache, repository write, or external state branch is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "standard": ("memorylayer-postgres", "postgres-container/Dockerfile"),
    "cnpg": ("memorylayer-postgres-cnpg", "postgres-container/Dockerfile.cnpg"),
}
PLATFORMS = {"linux/amd64", "linux/arm64"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RegistryError(RuntimeError):
    pass


def inspect_image(reference: str) -> dict:
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", reference, "--format", "{{json .Manifest}}"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RegistryError(f"Cannot inspect {reference}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid manifest response for {reference}") from exc


def platforms(manifest: dict) -> set[str]:
    return {
        f"{p.get('os')}/{p.get('architecture')}"
        for item in manifest.get("manifests", [])
        if (p := item.get("platform", {})).get("os") == "linux"
    }


def fingerprint(root: Path, descriptor: dict, version: str, base_image: str) -> str:
    paths = list((root / "postgres-container").rglob("*"))
    paths += [root / p for p in (
        ".dockerignore", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
        "scripts/postgres_updates.py", "scripts/smoke-postgres.sh",
        ".github/workflows/postgres-updates.yml", ".github/workflows/postgres-image.yml",
        ".github/actions/build-postgres/action.yml",
    )]
    inputs = {"schema": 1, "image": descriptor, "version": version, "base_image": base_image,
              "platforms": sorted(PLATFORMS), "files": {}}
    for path in sorted(paths):
        if path.is_symlink():
            raise ValueError(f"Symlink in build inputs: {path}")
        if path.is_dir():
            continue
        inputs["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_variant(root: Path, variant: str, config: dict, *, force: bool = False) -> dict:
    name, dockerfile = VARIANTS[variant]
    descriptor = config["docker"]["images"][name]
    if descriptor["context"] != "." or descriptor["dockerfile"] != dockerfile or descriptor.get("build_args"):
        raise ValueError("Update the nightly planner when changing PostgreSQL contexts or build arguments")
    version = str(config[descriptor["version_from"]])
    match = re.search(r"^ARG BASE_IMAGE=(\S+)$", (root / dockerfile).read_text(), re.M)
    if not match:
        raise ValueError(f"Missing BASE_IMAGE default in {dockerfile}")
    # Release builds use the checked-in digest. Nightly builds track that tag.
    tracking = match[1].split("@", 1)[0]
    manifest = inspect_image(tracking)  # An upstream failure must fail the check.
    digest = manifest.get("digest", "")
    if not DIGEST.fullmatch(digest) or not PLATFORMS <= platforms(manifest):
        raise ValueError(f"{tracking} must resolve to an amd64/arm64 manifest list with a digest")
    base_image = f"{tracking}@{digest}"
    value = fingerprint(root, descriptor, version, base_image)
    image = f"ghcr.io/{config['docker']['ghcr']}/{name}"
    record = {"variant": variant, "image": image, "dockerfile": dockerfile,
              "version": version, "base_image": base_image, "fingerprint": value,
              "changed": True, "reason": "forced" if force else "no successful build for these inputs"}
    validate_plan(record)
    if not force:
        try:
            marker = inspect_image(f"{image}:inputs-{value}")
        except (RegistryError, subprocess.TimeoutExpired) as exc:
            print(f"{name}: success marker absent/unavailable; rebuild required ({exc})", file=sys.stderr)
        else:
            if PLATFORMS <= platforms(marker) and DIGEST.fullmatch(marker.get("digest", "")):
                record.update(changed=False, reason="already built and tested for both platforms")
    return record


def validate_plan(plan: dict) -> None:
    variant = plan.get("variant")
    if variant not in VARIANTS:
        raise ValueError("Unknown PostgreSQL variant")
    name, dockerfile = VARIANTS[variant]
    if plan.get("image") != f"ghcr.io/scitrera/{name}" or plan.get("dockerfile") != dockerfile:
        raise ValueError("Unexpected image or Dockerfile")
    if not re.fullmatch(r"[0-9a-f]{64}", plan.get("fingerprint", "")):
        raise ValueError("Invalid input fingerprint")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", plan.get("version", "")):
        raise ValueError("Invalid image version")
    expected = "pgvector/pgvector:pg17" if variant == "standard" else "ghcr.io/cloudnative-pg/postgresql:17-system-bookworm"
    base, separator, digest = plan.get("base_image", "").partition("@")
    if base != expected or not separator or not DIGEST.fullmatch(digest):
        raise ValueError("Unexpected or unpinned upstream image")


def publish(plan: dict, amd64: str, arm64: str) -> None:
    validate_plan(plan)
    if not DIGEST.fullmatch(amd64) or not DIGEST.fullmatch(arm64):
        raise ValueError("Both tested platform digests are required")
    image = plan["image"]
    # The caller's two successful native jobs have already smoke-tested these.
    subprocess.run([
        "docker", "buildx", "imagetools", "create", "--tag", f"{image}:17",
        "--tag", f"{image}:nightly", f"{image}@{amd64}", f"{image}@{arm64}",
    ], check=True, timeout=120)
    nightly = inspect_image(f"{image}:nightly")
    rolling = inspect_image(f"{image}:17")
    digest = nightly.get("digest", "")
    if (not DIGEST.fullmatch(digest) or digest != rolling.get("digest")
            or not PLATFORMS <= platforms(nightly) or not PLATFORMS <= platforms(rolling)):
        raise RegistryError("Rolling tags did not both resolve to the complete multi-platform image")
    # Write the success marker LAST. A failed promotion cannot suppress retries.
    subprocess.run([
        "docker", "buildx", "imagetools", "create", "--tag", f"{image}:inputs-{plan['fingerprint']}",
        f"{image}@{digest}",
    ], check=True, timeout=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--force", action="store_true")
    plan_parser.add_argument("--output", type=Path)
    for name in ("validate", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
        if name == "publish":
            p.add_argument("--amd64", required=True)
            p.add_argument("--arm64", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        config = yaml.safe_load((ROOT / "versions.yaml").read_text())
        plans = {v: plan_variant(ROOT, v, config, force=args.force) for v in VARIANTS}
        print(json.dumps(plans, indent=2))
        if args.output:
            args.output.write_text(json.dumps(plans, indent=2) + "\n")
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as stream:
                for name, plan in plans.items():
                    stream.write(f"{name}={json.dumps(plan, separators=(',', ':'))}\n")
    elif args.command == "validate":
        validate_plan(json.loads(args.plan))
    else:
        publish(json.loads(args.plan), args.amd64, args.arm64)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a minimal, reviewable image context from explicitly selected packages."""
import argparse
import email
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def stage_sources(package: Path, destination: Path):
    """Build only declared package sources/metadata, never a checkout's stale build tree."""
    destination.mkdir(parents=True)
    for name in ('src', 'pyproject.toml', 'setup.py', 'setup.cfg', 'MANIFEST.in',
                 'README.md', 'README.rst', 'LICENSE', 'LICENSE.md', 'NOTICE', 'NOTICE.md'):
        source = package / name
        if source.is_dir():
            shutil.copytree(source, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
        elif source.is_file():
            shutil.copy2(source, destination / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['combined', 'embedding', 'transcription'], default='combined')
    parser.add_argument('--oss', type=Path, required=True)
    parser.add_argument('--sparkrun', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh-lock', action='store_true', help='Resolve new dependency pins deliberately')
    args = parser.parse_args()
    deploy = Path(__file__).resolve().parent
    profile_dir = deploy if args.profile == "combined" else deploy.parent / "l4-v2" / args.profile
    import yaml
    # Fail locally on malformed recipes before building/uploading an image.
    for recipe_path in (profile_dir / "recipes").glob("*.yaml"):
        recipe = yaml.safe_load(recipe_path.read_text())
        if not isinstance(recipe.get("command"), str) or not recipe.get("model_revision"):
            raise SystemExit(f"Invalid pinned recipe: {recipe_path.name}")
    if args.output.exists():
        raise SystemExit('Choose a new empty output directory; existing contexts are never overwritten')
    args.output.mkdir(parents=True)
    wheels = args.output / 'wheels'
    wheels.mkdir()
    packages = [args.oss / 'memorylayer-core-python', args.oss / 'memorylayer-embed-server', deploy.parent.parent, args.sparkrun]
    manifest = {'profile': 'qwen-colmodern-unlimited-l4-v1' if args.profile == 'combined' else args.profile + '-l4-v2', 'packages': [], 'wheels': {}}
    for package in packages:
        commit = subprocess.check_output(['git', '-C', str(package), 'rev-parse', 'HEAD'], text=True).strip()
        epoch = subprocess.check_output(['git', '-C', str(package), 'show', '-s', '--format=%ct', 'HEAD'], text=True).strip()
        with tempfile.TemporaryDirectory(prefix='memorylayer-wheel-') as temporary:
            staged = Path(temporary) / package.name
            stage_sources(package, staged)
            subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(wheels.resolve()), str(staged)], check=True,
                           env={**os.environ, 'SOURCE_DATE_EPOCH': epoch})
        manifest['packages'].append({'name': package.name, 'base_commit': commit})
    local_names = set()
    messages = []
    for wheel in sorted(wheels.glob('*.whl')):
        manifest['wheels'][wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as archive:
            metadata = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            message = email.message_from_bytes(archive.read(metadata))
            messages.append(message)
            local_names.add(message['Name'].lower().replace('_', '-'))
    from packaging.requirements import Requirement
    requirements = {'vllm==0.25.0', 'ninja==1.13.0', 'torch==2.11.0', 'torchvision==0.26.0'}
    for message in messages:
        for value in message.get_all('Requires-Dist', []):
            req = Requirement(value)
            if req.name.lower().replace('_', '-') in local_names:
                continue
            if req.marker and not req.marker.evaluate({'extra': '', 'platform_machine': 'x86_64', 'python_version': '3.12'}):
                continue
            requirements.add(value)
    (args.output / 'requirements.in').write_text('\n'.join(sorted(requirements)) + '\n')
    for lock_name, inputs in [('requirements.lock', args.output / 'requirements.in'),
                              ('controller.lock', args.output / 'controller.in')]:
        if lock_name == 'controller.lock':
            sparkrun_meta = next(m for m in messages if m['Name'] == 'sparkrun')
            inputs.write_text('\n'.join(v for v in sparkrun_meta.get_all('Requires-Dist', [])
                                       if not Requirement(v).marker or Requirement(v).marker.evaluate({'extra': ''})) + '\n')
        if not args.refresh_lock and (deploy / lock_name).exists():
            shutil.copy2(deploy / lock_name, args.output / lock_name)
        else:
            subprocess.run(['uv', 'pip', 'compile', str(inputs), '--python-version', '3.12',
                            '--python-platform', 'x86_64-manylinux_2_28', '--generate-hashes', '--no-annotate',
                            '--output-file', str(args.output / lock_name)], check=True, stdout=subprocess.DEVNULL)
    manifest['requirements_sha256'] = hashlib.sha256((args.output / 'requirements.lock').read_bytes()).hexdigest()
    for name in ('recipes', 'prepare.py', 'serve.py', 'profile.env', 'sparkrun.yaml', 'Dockerfile', 'smoke.py'):
        source = (profile_dir if name in {'recipes', 'profile.env'} else deploy) / name
        if name == 'Dockerfile' and (profile_dir / name).exists():
            source = profile_dir / name
        if name == 'smoke.py' and args.profile != 'combined':
            source = deploy.parent / 'l4-v2' / 'smoke.py'
        if source.is_dir():
            shutil.copytree(source, args.output / name)
        else:
            shutil.copy2(source, args.output / name)
    if (profile_dir / 'ner.json').exists():
        for name in ('ner.lock', 'ner_worker.py'):
            shutil.copy2(deploy.parent / 'l4-v2' / name, args.output / name)
        shutil.copy2(profile_dir / 'ner.json', args.output / 'ner.json')
    manifest['deployment_files'] = {
        str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob('*'))
        if path.is_file() and path.suffix != '.whl' and path.name not in {'requirements.in', 'controller.in'}
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / '.dockerignore').write_text('requirements.in\ncontroller.in\nqualification*.json\n')
    print(args.output)


if __name__ == '__main__':
    main()

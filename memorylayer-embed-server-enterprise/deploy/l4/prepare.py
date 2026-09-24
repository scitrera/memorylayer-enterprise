# SPDX-License-Identifier: AGPL-3.0-only
"""Independent native-environment and model-cache preparation; never launches inference."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = os.environ.get('HF_HOME', '/cache/huggingface')


def prepare(phase):
    from sparkrun import api
    from sparkrun.application import initialize
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig
    from sparkrun.core.recipe import Recipe

    sctx = initialize(config_path=ROOT / 'sparkrun.yaml')
    model_phase = phase == 'models'
    cluster = ClusterDefinition('embed-preparation', ['localhost'], cache_dir=CACHE,
        distribution=ClusterDistributionConfig.from_dict({'model': {'enabled': model_phase}}))
    reports = []
    for path in sorted((ROOT / 'recipes').glob('*.yaml')):
        recipe = Recipe.load(path)
        recipe.builder_config['verify_only'] = model_phase
        options = api.BuildOptions(recipe=recipe, cluster=cluster, hosts=('localhost',), executor='local',
            cache_dir=CACHE, local_cache_dir=CACHE, transfer_mode='local', sync_tuning=False, trust=True)
        plan = api.plan_build(options, sctx=sctx)
        result = api.build(options, plan=plan, sctx=sctx)
        if result.dry_run or not result.environment_file:
            raise RuntimeError('Preparation did not produce a native environment')
        if model_phase:
            from huggingface_hub import snapshot_download
            # Prove the exact revision is present, with network disabled here.
            snapshot = Path(snapshot_download(recipe.model, revision=recipe.model_revision,
                                              cache_dir=str(Path(CACHE) / "hub"), local_files_only=True))
            weights = [p for p in snapshot.glob('*.safetensors') if p.is_file()]
            required = [snapshot / 'config.json']
            for index in snapshot.glob('*.safetensors.index.json'):
                required += [snapshot / name for name in set(json.loads(index.read_text())['weight_map'].values())]
            broken = [p for p in snapshot.rglob('*') if p.is_symlink() and not p.exists()]
            if not weights or broken or any(not path.is_file() for path in required):
                raise RuntimeError(f'Incomplete model cache: {recipe.model}')
        reports.append(asdict(result))
    ner_spec = ROOT / 'ner.json'
    if model_phase and ner_spec.exists():
        spec = json.loads(ner_spec.read_text())
        snapshot_download(spec['model'], revision=spec['revision'], cache_dir=str(Path(CACHE) / 'hub'))
        snapshot_download(spec['model'], revision=spec['revision'], cache_dir=str(Path(CACHE) / 'hub'), local_files_only=True)
        reports.append({'ner': spec})
    output = Path(CACHE) / 'prepared.json' if model_phase else ROOT / 'environment-build.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, default=str, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['environment', 'models'])
    prepare(parser.parse_args().phase)

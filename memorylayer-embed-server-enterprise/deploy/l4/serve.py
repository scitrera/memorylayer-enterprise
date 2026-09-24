# SPDX-License-Identifier: AGPL-3.0-only
"""Container entrypoint shared by Modal, Compose and Kubernetes."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    for line in (ROOT / 'profile.env').read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key, value)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['UV_OFFLINE'] = '1'
    os.environ['MEMORYLAYER_EMBED_SPARKRUN_CONFIG'] = str(ROOT / 'sparkrun.yaml')
    # Import only after setting offline mode: HF reads these at import time.
    import yaml
    from huggingface_hub import snapshot_download

    for path in (ROOT / 'recipes').glob('*.yaml'):
        recipe = yaml.safe_load(path.read_text())
        snapshot_download(recipe['model'], revision=recipe['model_revision'],
                          cache_dir=os.environ['HF_HUB_CACHE'], local_files_only=True)
    if (ROOT / 'ner.json').exists():
        import json
        spec = json.loads((ROOT / 'ner.json').read_text())
        snapshot_download(spec['model'], revision=spec['revision'], cache_dir=os.environ['HF_HUB_CACHE'], local_files_only=True)
    os.execv(sys.executable, [sys.executable, '-m', 'memorylayer_embed_server.cli', 'serve'])


if __name__ == '__main__':
    main()

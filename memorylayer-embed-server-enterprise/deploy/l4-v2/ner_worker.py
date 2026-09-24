# SPDX-License-Identifier: AGPL-3.0-only
"""Single-owner offline GLiNER2 worker. JSON lines over private subprocess pipes."""

import contextlib
import json
import os
import sys
from pathlib import Path


def main():
    protocol = sys.stdout
    # Libraries print diagnostics; reserve stdout exclusively for framed replies.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from gliner2 import GLiNER2
        from huggingface_hub import snapshot_download

        spec = json.loads(Path("/opt/memorylayer/ner.json").read_text())
        snapshot = snapshot_download(spec["model"], revision=spec["revision"], local_files_only=True, cache_dir=os.environ["HF_HUB_CACHE"])
        model = GLiNER2.from_pretrained(snapshot, map_location="cpu")
        model = model.to(device=os.environ.get("MEMORYLAYER_EMBED_GLINER2_DEVICE", "cuda"), dtype=torch.float32).eval()
        with torch.inference_mode():
            model.batch_extract_entities(["Alice works at Scitrera in Chicago."], ["person", "organization", "location"])
    print(json.dumps({"ready": True}), file=protocol, flush=True)
    for line in sys.stdin:
        try:
            if len(line) > 100_000:
                raise ValueError("NER request too large")
            data = json.loads(line)
            texts, labels = data["texts"], data["labels"]
            if not 1 <= len(texts) <= 8 or not 1 <= len(labels) <= 32 or sum(len(t.encode()) for t in texts) > 8192:
                raise ValueError("NER batch exceeds limits")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                schema = model.create_schema().entities(labels).build()
                for text in texts:
                    # Same schema and punctuation convention as GLiNER's collator.
                    text = text if text.endswith((".", "!", "?")) else text + "."
                    encoded = model.processor.transform_and_format(text, schema)
                    if len(encoded.input_ids) > 512:
                        raise ValueError("NER text plus label schema exceeds 512 tokens; chunk explicitly")
                result = model.batch_extract_entities(texts, labels, batch_size=8, max_len=None)
            response = {"results": [item["entities"] for item in result]}
        except ValueError as exc:
            response = {"invalid": str(exc)}
        except Exception:
            # Keep source text and model internals out of client error replies.
            import traceback

            traceback.print_exc(file=sys.stderr)
            response = {"error": "NER worker inference failed"}
        print(json.dumps(response), file=protocol, flush=True)


if __name__ == "__main__":
    main()

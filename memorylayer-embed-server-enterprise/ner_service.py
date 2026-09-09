"""DECOMMISSIONED — the standalone GLiNER2 NER service has been productionized.

This file used to run a separate FastAPI app on :61055 inside the bf-ml-embed
container (the experiment). The GLiNER2 typed-NER endpoint is now a first-class
route on the MAIN embed server:

    POST http://<embed-server>:61051/v1/ner
    {"texts": [...], "labels": [...] | null}
      -> {"results": [{"entities": {label: [spans]}}], "model": "..."}

It is served by the enterprise embed-server package (so it survives restarts and
is baked into the image), not a hand-launched process:

    router:   src/memorylayer_embed_server_enterprise/api/v1/ner.py   (NERAPIPlugin)
    service:  src/memorylayer_embed_server_enterprise/services/ner.py (GLiNER2NERService)
    wiring:   src/memorylayer_embed_server_enterprise/lifecycle/ner.py (preload via async_ready)
    config:   src/memorylayer_embed_server_enterprise/config.py
              MEMORYLAYER_EMBED_GLINER2_ENABLED  (default False — opt in)
              MEMORYLAYER_EMBED_GLINER2_MODEL    (default fastino/gliner2-base-v1)
              MEMORYLAYER_EMBED_GLINER2_LABELS   (default CSV)

The app-side GLiNER2ExtractionService now defaults its NER URL to the main embed
server (MEMORYLAYER_EMBED_SERVER_URL) instead of :61055.

Do NOT run this module. It intentionally exits immediately.
"""
import sys

if __name__ == "__main__":
    sys.exit(
        "ner_service.py is decommissioned. The /v1/ner endpoint now lives on the "
        "main embed server (:61051) via the memorylayer-embed-server-enterprise "
        "package. See this file's docstring."
    )

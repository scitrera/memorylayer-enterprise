#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

# This service has an environment-driven entrypoint; --help starts the server.
set -euo pipefail
image="${1:?Usage: smoke-connectors.sh IMAGE}"
docker run --rm --network none --entrypoint python "$image" -m pip check
container=$(docker run --detach --network none "$image")
trap 'docker rm --force "$container" >/dev/null 2>&1 || true' EXIT
for attempt in {1..30}; do
    if docker exec "$container" python -c 'import json, urllib.request; data=json.load(urllib.request.urlopen("http://127.0.0.1:8100/healthz", timeout=2)); assert data["status"] == "ok"' 2>/dev/null; then
        echo 'Connector container started and /healthz returned ok.'
        exit 0
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
        break
    fi
    sleep 1
done
docker logs "$container"
echo 'Connector container did not become healthy' >&2
exit 1

#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

# Test the installed extensions in a disposable, network-isolated container.
set -euo pipefail
image="${1:?Usage: smoke-postgres.sh IMAGE}"
license_label=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.licenses"}}' "$image")
test -z "$license_label"
docker run --rm --network none --user postgres --entrypoint /bin/bash "$image" -ceu '
  export PATH="/usr/lib/postgresql/17/bin:$PATH"
  test -s /usr/share/licenses/memorylayer/LICENSE
  test -s /usr/share/licenses/memorylayer/NOTICE
  grep -q "Apache License" /usr/share/licenses/memorylayer/LICENSE
  if grep -q "GNU AFFERO GENERAL PUBLIC LICENSE" /usr/share/licenses/memorylayer/LICENSE; then
    echo "PostgreSQL image must not carry the root AGPL license as its license" >&2
    exit 1
  fi
  for component in age pg_textsearch; do
    test -s "/usr/share/licenses/memorylayer/extensions/$component/LICENSE"
    test -s "/usr/share/licenses/memorylayer/extensions/$component/SOURCE"
  done
  work=$(mktemp -d)
  trap '\''pg_ctl -D "$work/data" -m immediate stop >/dev/null 2>&1 || true'\'' EXIT
  initdb -D "$work/data" --auth=trust >/dev/null
  pg_ctl -D "$work/data" -l "$work/server.log" \
    -o "-k $work -c listen_addresses= -c shared_preload_libraries=pg_textsearch" -w start
  if [ -f /docker-entrypoint-initdb.d/init.sql ]; then
    psql -h "$work" -U postgres -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/init.sql
  else
    psql -h "$work" -U postgres -v ON_ERROR_STOP=1 \
      -c "CREATE EXTENSION vector; CREATE EXTENSION age; CREATE EXTENSION pg_textsearch; LOAD '\''age'\'';"
  fi
  psql -h "$work" -U postgres -v ON_ERROR_STOP=1 \
    -c "SELECT extname, extversion FROM pg_extension ORDER BY extname;" \
    -c "CREATE TABLE smoke_vectors (embedding vector(3)); INSERT INTO smoke_vectors VALUES ('\''[1,2,3]'\'');" \
    -c "CREATE TABLE smoke_text (body text); CREATE INDEX smoke_bm25 ON smoke_text USING bm25 (body) WITH (text_config='\''english'\'');"
  pg_ctl -D "$work/data" -m fast -w stop
'

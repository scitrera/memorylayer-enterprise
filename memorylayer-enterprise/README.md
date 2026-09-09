# MemoryLayer Enterprise

AGPLv3 extensions to the Apache-2.0 MemoryLayer server: PostgreSQL/pgvector and
Apache AGE storage, graph retrieval, document ingestion and transcription,
datasets, background jobs, tenant services, and hot/cold memory tiering.

## Development

Requires Python 3.12+, the matching MemoryLayer 0.2.0 core and RPG packages,
PostgreSQL with the required extensions, and configured embedding services.
See [dependency availability and service setup](../docs/DEVELOPMENT.md) before
installing this initial source snapshot.

```sh
cd .. # repository root
python3 scripts/dev.py enterprise
.venv/bin/python -m pytest memorylayer-enterprise/tests/unit -q
.venv/bin/memorylayer-enterprise --help
```

The installed CLI is `memorylayer-enterprise`; background workers use
`memorylayer-worker`. Python imports retain the `memorylayer_saas` namespace.
Configuration keys include `MEMORYLAYER_POSTGRESQL_URL`,
`MEMORYLAYER_CACHE_REDIS_URL`, and `MEMORYLAYER_EMBED_SERVER_URL`.
Supply database and Aether credentials through your runtime environment.
Select `MEMORYLAYER_OSS_REF` or `MEMORYLAYER_OSS_PATH` at install time for paired
OSS/enterprise development; the linked development guide covers both modes.

Migration source is included in `migrations/` and auxiliary SQL in
`src/memorylayer_saas/storage/migrations/`. For wheel deployments, copy the
versioned migrations and Alembic configuration from the matching source
release and set the migration path overrides documented in the development guide.

## License

Copyright (c) 2026 Scitrera. Licensed under [AGPL-3.0-only](LICENSE).
The separate MemoryLayer core and RPG dependencies retain Apache-2.0.

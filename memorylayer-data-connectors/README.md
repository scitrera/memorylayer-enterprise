# MemoryLayer Data Connectors

Data connectors, VFS catalog, blob storage integration, and access-controlled
upload/download URLs for MemoryLayer. Connectors cover manual upload, local
files, S3, Google Drive, Dropbox, Slack, Discord, Teams, GitHub, and web pages.
New or changed content can enqueue document ingestion work through Aether.

## Layout

- `src/data_connectors/server/`: FastAPI service and Aether integration.
- `src/data_connectors/connectors/`: Connector implementations.
- `src/data_connectors/vfs/`: File catalog and blob backends.
- `src/data_connectors/services/`: Synchronization, URL minting, and garbage collection.
- `src/data_connectors/messages/`: Request and response models.
- `src/data_connectors/db/migrations/`: Alembic migrations.

## Development

Requires Python 3.12+, PostgreSQL, and the services used by your configured
connectors. See [development requirements](../docs/DEVELOPMENT.md).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests/unit -q
.venv/bin/data-connectors
```

The entrypoint starts the server and reads `DC_HOST`/`DC_PORT` from the
environment; it does not implement command-line flags.

Integration tests require a separately configured service stack. The included
fixtures are synthetic text and a PDF reproducible with
`python3 tools/build_test_pdf.py`. Use only synthetic documents and test-only
credentials when running or extending these tests.

## License

Copyright (c) 2026 Scitrera. Licensed under [AGPL-3.0-only](LICENSE).

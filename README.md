# MemoryLayer Enterprise

AGPLv3 extensions for [MemoryLayer](https://github.com/scitrera/memorylayer).
Public repository: [scitrera/memorylayer-enterprise](https://github.com/scitrera/memorylayer-enterprise).

| Component | Purpose |
| --- | --- |
| [Enterprise server](memorylayer-enterprise/README.md) | PostgreSQL and Apache AGE storage, graph retrieval, document ingestion, datasets, background work, memory tiering, and tenant services |
| [Data connectors](memorylayer-data-connectors/README.md) | External data connectors, VFS catalog, blob storage integration, and access-controlled file URLs |
| [Embedding extensions](memorylayer-embed-server-enterprise/README.md) | Visual tokenization and GLiNER2 named-entity extraction plugins |
| [Admin dashboard](memorylayer-admin/README.md) | Web interface for memories, documents, datasets, jobs, workspaces, and administration |

## Development status

Core, RPG, and the embedding base come from one reviewed Apache-2.0 OSS commit
in [oss-source.toml](oss-source.toml). To start combined development from this
repository's root:

```sh
python3 scripts/dev.py enterprise
# Or select a branch/tag, or an existing checkout with your local edits:
MEMORYLAYER_OSS_REF=main python3 scripts/dev.py enterprise
MEMORYLAYER_OSS_PATH=../memorylayer python3 scripts/dev.py enterprise
```

The installer creates `.venv` and installs the requested Python projects in
editable mode. Set only one OSS override at a time. See
[development and release requirements](docs/DEVELOPMENT.md) for embedding,
dashboard SDK linking, source provenance, and remaining integration checks.

Python components require Python 3.12 or later; the dashboard uses Node.js 22
or later. Package names and Python import paths retain their existing names.
The initial extension versions remain 0.0.1 until the release version is chosen.

[Container builds and releases](docs/CONTAINERS.md) use Scitrera repo-tools
for the enterprise service, data connectors, and both PostgreSQL variants.
GitHub Actions builds each image on native amd64 and arm64 runners, with
nightly PostgreSQL checks for local build-input and upstream-image changes.
No PyPI or npm publication is configured.

The enterprise server uses PostgreSQL with pgvector, Apache AGE, and
pg_textsearch. Aether provides messaging and authentication integrations.
[MemoryLayer Storage](https://github.com/scitrera/memorylayer-storage) provides
the separately released blob gateway and filesystem services.

## Contributing

Contributions require acceptance of the
[MemoryLayer Enterprise Contributor License Agreement](CLA.md) before merge.
Scitrera LLC is the Project Owner. See [CONTRIBUTING.md](CONTRIBUTING.md) for
manual acceptance, contribution requirements, and relevant checks.

## License

Copyright (c) 2026 Scitrera.

The Scitrera extensions in this repository are licensed under the GNU Affero
General Public License, version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE)
and [NOTICE](NOTICE). This software is provided without warranty.

The existing MemoryLayer OSS core and SDKs retain **Apache-2.0**. They are
separate dependencies and are not relicensed by this repository. MemoryLayer
Storage retains its own licensing: AGPL-3.0-only for the storage services and
Apache-2.0 for its Python client. Third-party code retains its original terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).

When distributing binaries or operating a modified network service, provide
the corresponding source as required by the license. Release source must
match the deployed revision and include the relevant build/install scripts.
The public source location is this repository; image source/revision labels
and the enterprise image's OSS source record identify the corresponding inputs.

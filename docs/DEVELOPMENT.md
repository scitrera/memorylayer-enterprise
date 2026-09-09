# Development and release requirements

Before submitting changes, read [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[CLA](../CLA.md). Maintainers verify CLA acceptance manually before merge.

## Shared OSS source

The target is [scitrera/memorylayer-enterprise](https://github.com/scitrera/memorylayer-enterprise).
[`oss-source.toml`](../oss-source.toml) selects one full Git commit from
[MemoryLayer OSS](https://github.com/scitrera/memorylayer). The initial pin is
`6ee99813f287ff7c20fe85f427032558f1bd56d4`, the public v0.2.0 release. Core, RPG,
and the embedding base are installed together from that checkout as needed.
Their existing Apache-2.0 licenses remain unchanged.

The supported Python install path is `scripts/dev.py`, also used by enterprise
CI and its container. Package metadata declares compatible versions (`>=0.2.0`);
the installer supplies explicit source projects to pip in one resolution.
It checks dependencies normally, so a branch whose RPG package requires an
incompatible core fails visibly. A bare `pip install .` does not select the
shared source and may use PyPI or fail for the source-only embedding base.

## Combined development

Run these commands from this repository's root with Python 3.12+, Git, and pip:

```sh
# Reviewed OSS commit, editable core + RPG + enterprise, dev dependencies:
python3 scripts/dev.py enterprise
.venv/bin/python -m pytest memorylayer-enterprise/tests/unit -q

# Resolve a branch/tag to one commit for both OSS packages:
MEMORYLAYER_OSS_REF=main python3 scripts/dev.py enterprise

# Use your existing OSS checkout, including uncommitted work:
MEMORYLAYER_OSS_PATH=../memorylayer python3 scripts/dev.py enterprise

# Optional components, alone or in the same environment:
python3 scripts/dev.py enterprise connectors
python3 scripts/dev.py embed
```

Set **only one** of `MEMORYLAYER_OSS_REF` and `MEMORYLAYER_OSS_PATH`. Unset both
to return to the reviewed pin, then rerun the installer. Source selection occurs
during installation, not server startup. Restart the relevant Python process
after editing code; editable installs import directly from the selected paths.
When changing dependency metadata, rerun the installer.

Git refs are fetched into commit-specific snapshots under `.dev/oss/`. Each
branch invocation resolves its current commit; previous snapshots stay intact
for environments still using them. Edited managed snapshots are preserved and
refused for automatic reuse; point `MEMORYLAYER_OSS_PATH` at one to explicitly
use those edits. Local checkout mode does not fetch, switch branches, reset,
or commit that checkout.

The default environment is `.venv`; `--venv /path/to/venv` selects another,
and `--python /path/to/python` installs into an existing interpreter with pip.
`--no-dev` omits test extras and `--non-editable` installs wheels. Source records
in `.dev/oss-source.json` include the resolved commit and local dirty state.
`.dev`, environments, credentials, and downloaded source are ignored by Git
and Docker and remain outside the reviewed export.

## Dashboard and SDK development

For the dashboard, use Node.js 22+ (CI uses 24), run `npm ci`, then
`npm run typecheck` and `npm run build`. Its committed lockfile resolves the
published SDK. Both checks passed with a fresh public-registry installation;
the previous `fs/promises` bundle failure is resolved with this SDK release.
No application-side filesystem shim was added. Run `npm run dev` on port 3200.

To work on the TypeScript SDK and dashboard together, select the same source
override and run:

```sh
MEMORYLAYER_OSS_PATH=../memorylayer python3 scripts/dev.py admin
npm --prefix memorylayer-admin run dev
```

The helper installs/builds the SDK in the selected checkout and links it into
the dashboard. It preserves both tracked package metadata and the release
lockfile. Rebuild the SDK after TypeScript edits; rerun `npm ci` in the dashboard
to restore its published SDK. The normal dashboard CI still validates npm SDK
0.2.0 from its release lock. Python client 0.2.1 remains available to callers;
these extensions do not directly depend on that client.

Enterprise's 1,362 unit tests passed in a fresh environment using the pinned
Git core/RPG packages. All 79 embedding unit tests passed against the tagged base
and published core with existing ML libraries; the complete ML dependency set
was resolver-checked, not freshly installed or qualified on a GPU.

## Services and migrations

Enterprise defaults to PostgreSQL, Redis-compatible caching, and remote
embedding. Configure `MEMORYLAYER_POSTGRESQL_URL`, `MEMORYLAYER_CACHE_REDIS_URL`,
and `MEMORYLAYER_EMBED_SERVER_URL` for your own services. Aether-backed features
also require an Aether deployment and credentials supplied at runtime.

The enterprise source distribution includes `migrations/` and `alembic.ini`.
For a wheel installation, deploy these files from the matching source release
and set `MEMORYLAYER_ALEMBIC_DIR` and `MEMORYLAYER_ALEMBIC_CONFIG` to their absolute
paths. Wheel-only automatic discovery of these files is not yet supported.
For Alembic CLI use, enterprise `migrations/env.py` reads `DATABASE_URL`;
the running service reads `MEMORYLAYER_POSTGRESQL_URL`.

Data connectors use `DC_POSTGRESQL_URL` and optional `DC_DATABASE_URL` for the
Alembic CLI. Their migrations are included under `src/data_connectors/db/`.
Run schema migrations with one owner before starting multiple replicas.
Verify upgrades and startup against disposable databases before release.

Database image source is in `postgres-container/`. Both recipes pin base-image
digests and extension source commits, and retain extension notices. The
enterprise container ships its matching Alembic source and configures the
required path overrides. See [container builds and releases](CONTAINERS.md)
for the native amd64/arm64 CI, nightly upstream checks, and smoke-test scope.

## Storage and inference

MemoryLayer Storage is a separate dependency. The enterprise `blobgw` backend
requires its Apache-2.0 Python client, currently distributed from the storage
Git repository rather than PyPI. Follow the storage repository's tagged
installation instructions. Data connectors have a native HTTP fallback when
that client is unavailable. No storage source is copied into this repository.

Model weights, caches, customer documents, connector tokens, database exports,
and inference results are not source artifacts. Supply them outside the
repository. Review each downloaded model's own license independently.

The NER optional dependency set has an existing Transformers version conflict:
the development embedding image installs GLiNER packages with `--no-deps`.
Establish and test a supported dependency set before describing the NER extra
as a normal pip installation path.

## Remaining release work

- Choose the first shared container release version for `memorylayer-enterprise`.
  The image policy is four container images with no PyPI/npm publication.
- Run hosted builds on both architectures. All four containers have passed
  local arm64 build/smoke checks; enterprise checks cover its CLI, AGE/RPG import,
  Python dependency consistency, and bundled migrations, not full service startup.
- Supply public deployment/Compose examples if needed; private sibling-checkout
  development launchers are not included here.
- Run database upgrades, full-stack connector ingestion, dashboard runtime, and
  GPU/model integration checks. Pin and qualify the remaining ML/transitive
  Python dependency sets for deployed images.
- Generate a resolved dependency inventory, vulnerability results, and notices
  for any distributed wheels or container images, including the GPU base image.
- Keep the deployed source revision available at the public target. The admin
  sidebar links to source/license information; image metadata and the OSS
  source record identify build inputs.
- Run the prepared repo-tools and container CI in the public repository, rerun
  secret and private-data review on the final tracked tree, and create a new
  Git history from reviewed files only.

This source snapshot does not establish that those release gates have passed.

# Container builds and releases

The target [scitrera/memorylayer-enterprise](https://github.com/scitrera/memorylayer-enterprise)
builds these four images in GHCR:

| Image under `ghcr.io/scitrera/` | Recipe | Contents |
| --- | --- | --- |
| `memorylayer-enterprise` | `docker/Dockerfile.enterprise` | Enterprise service and worker, document tools, matching Alembic source |
| `memorylayer-data-connectors` | `docker/Dockerfile.data-connectors` | Connector service and packaged database migrations |
| `memorylayer-postgres` | `postgres-container/Dockerfile` | PostgreSQL 17, pgvector, Apache AGE, pg_textsearch; standard PostgreSQL entrypoint |
| `memorylayer-postgres-cnpg` | `postgres-container/Dockerfile.cnpg` | The same extensions on the CloudNativePG operand image |

All recipes use the repository root as their build context. Each hosted image
build uses `ubuntu-24.04` for `linux/amd64` and `ubuntu-24.04-arm` for
`linux/arm64`, then combines the platform digests into one image index. These
are [native GitHub runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners);
the workflows do not install QEMU. The design follows Docker's
[build distribution across runners](https://docs.docker.com/build/ci/github-actions/multi-platform/).

Embedding extensions and the dashboard remain included as source components.
This configuration publishes no Python or npm packages and builds no separate
embedding or dashboard image.

## repo-tools configuration

[`versions.yaml`](../versions.yaml) uses `scitrera-repo-tools==0.1.29`, following
the [MemoryLayer Storage](https://github.com/scitrera/memorylayer-storage)
repository's approach. It manages package version metadata and generates
`version-check.yml`, `test-python.yml`, `test-npm.yml`, and `build-docker.yml`:

```sh
uvx --from scitrera-repo-tools==0.1.29 sync-versions --check
uvx --from scitrera-repo-tools==0.1.29 generate-ci-gha
# After editing versions.yaml, regenerate the managed workflows:
uvx --from scitrera-repo-tools==0.1.29 generate-ci-gha --force
```

The workflow allowlist excludes registry package publication. In this
repo-tools version, an empty `publish_projects` list selects all projects;
it must not be used to disable publication.

The native publishing template in 0.1.29 pushes platform digests even when
configured to build on pull requests. Therefore `build_on_pr` is disabled and
the separately maintained `container-check.yml` builds and smoke-tests all
four images on both native architectures with read-only permissions and no
registry login or push. It also tests the nightly planner and workflow rules.

The dashboard workflow uses `npm ci` with the committed lockfile, builds the
production bundle, and checks TypeScript. It does not publish an npm package.

## Private staging

While the repository is private, PostgreSQL refresh jobs are skipped and
numbered/manual container publication fails its prerequisite check before any
image is pushed. Branch and PR tests still run. Use the **Native container
checks** workflow's manual dispatch to build and smoke-test all four images
on both architectures without registry publication.

Making the repository public enables the normal publication triggers described
below, including the next nightly PostgreSQL check. Visibility is changed
separately by a repository administrator after staging review.

## Numbered releases

Enterprise CI and the Dockerfile use `scripts/dev.py` to install core and RPG
from the same commit in `oss-source.toml`. CI passes `--release`, which refuses
development source overrides. The checked-in source must be a full commit SHA.
Publishing workflows never pass a Docker OSS-ref override. To advance the OSS
baseline, update this one pin, test the paired sources, then release the images.
The image records the OSS repository and commit at
`/usr/share/licenses/memorylayer/oss-source.json`, independently of the enterprise
revision recorded by OCI labels.

For a development image using an OSS branch or tag:

```sh
docker build -f docker/Dockerfile.enterprise \
  --build-arg MEMORYLAYER_OSS_REF=main -t memorylayer-enterprise:dev .
```

The ref resolves once for both OSS packages and its commit is recorded in the
image. Use the [editable host install](DEVELOPMENT.md#combined-development)
for uncommitted local changes. Core/RPG package metadata remains compatible with
the selected source; the build does not bypass dependency checking.

The `memorylayer-extensions` value in `versions.yaml` is the shared image
release version, initially `0.0.1`. A matching root tag such as `v0.0.1`
triggers the generated container workflow. Package versions remain tracked
for source metadata; they do not independently trigger image or PyPI releases.
All four images use the shared image version, avoiding mixed image versions
on one release tag.

The generated workflow runs enterprise and connector unit tests before any
image publishing jobs. A mismatched release tag fails the prerequisite checks.
Manual dispatch is restricted by those checks to the default branch and uses
the checked-in version. Generated image metadata supplies version, major/minor,
and commit tags; `latest` is a moving tag. Use exact image digests for deployed
revisions. Re-running a release can replace version tags, so allocate a new
version for changed release contents.

Numbered PostgreSQL releases use the base digests checked into the Dockerfiles.
To promote an upstream refresh to a numbered release, review and update these
pins and the shared version first. Nightly checks do not commit Dockerfile
changes or change numbered image tags.

## PostgreSQL refreshes

`postgres-updates.yml` checks at **06:17 UTC nightly**, on relevant changes to
`main`, and on manual dispatch. Publishing is limited to the Scitrera
repository's default branch. The tracked upstream tags are:

- Standard: `pgvector/pgvector:pg17`
- CloudNativePG: `ghcr.io/cloudnative-pg/postgresql:17-system-bookworm`

The planner resolves each tag once to an OCI index digest and requires both
architectures. It hashes that resolved base together with the local PostgreSQL
recipes, extension build script, initialization SQL, license files, relevant
automation, and image configuration/version. Both variants share the local
PostgreSQL source hash; editing either recipe can rebuild both. An upstream
digest change rebuilds its affected variant.

For each variant, a complete `inputs-<sha256>` registry tag records a previously
successful build for those inputs. Matching inputs skip the build. Missing or
incomplete success records cause a rebuild; an upstream lookup failure fails
the check. No repository writes, state branch, or persistent Actions cache is
required for change detection.

Each native job passes the same resolved `BASE_IMAGE` into both Dockerfile
stages, starts a disposable database, and exercises vector, AGE, and BM25
extension setup before pushing its platform digest. Only after both jobs
succeed does the merge job promote `:17` and `:nightly`, verify both platforms,
and write the success record last. A failed build or promotion remains eligible
for retry. Manual dispatch's `force` option rebuilds despite a matching record.
Deleting success-record tags also causes rebuilds.

This refreshes the source **container**. Apache AGE and pg_textsearch are pinned
to explicit Git commits and updated through reviewed Dockerfile changes.
The initial pins provide AGE 1.7.0 and pg_textsearch `1.5.0-dev`; the latter
captures the previously used upstream main revision and is not a stable release
claim. Pinning the base and extension source does not freeze apt or Python
package repositories, so builds are not guaranteed byte-reproducible.

### CloudNativePG base and initialization

The candidate uses `17-system-bookworm`, preserving the system image's Barman
tools. The legacy `:17` base tracks Bullseye and failed the local build against
expired package metadata. The upstream documents `system` as deprecated;
moving to its `standard` flavor and Barman backup plugin is separate migration
work. See the [upstream image flavors and OS policy](https://github.com/cloudnative-pg/postgres-containers).
For existing databases, qualify the OS/collation change before changing an
operand image; this source preparation does not perform that migration.

The CNPG recipe preserves the upstream user and entrypoint. Configure
`postgresql.shared_preload_libraries: [pg_textsearch]` and create `vector`,
`age`, and `pg_textsearch` using the operator's
`bootstrap.initdb.postInitSQL`. The standard image instead supplies init SQL
for newly initialized data directories and a preload default. Init SQL does
not upgrade an existing database; follow the extension upgrade procedures.

## Local validation and current limits

From the repository root, for example:

```sh
docker build -f postgres-container/Dockerfile -t memorylayer-postgres:check .
bash scripts/smoke-postgres.sh memorylayer-postgres:check
docker build -f postgres-container/Dockerfile.cnpg -t memorylayer-postgres-cnpg:check .
bash scripts/smoke-postgres.sh memorylayer-postgres-cnpg:check
docker build -f docker/Dockerfile.data-connectors -t memorylayer-data-connectors:check .
bash scripts/smoke-connectors.sh memorylayer-data-connectors:check
uv run --no-project --with scitrera-repo-tools==0.1.29 python -m unittest discover -s tests/ci -v
# Read-only upstream resolution and build plan, without publishing:
uv run --no-project --with scitrera-repo-tools==0.1.29 python scripts/postgres_updates.py plan --force
```

On 2026-09-08 both PostgreSQL images and the connector image built and passed
smoke checks locally on native arm64, using public dependencies. Connector
checks cover dependency consistency and `/healthz` with in-memory defaults,
not database/Aether integration. Registry promotion remains untested.

On 2026-09-09 the enterprise image also built on native arm64 with core and RPG
0.2.0 installed from the pinned public Git checkout. Its CLI,
`pip check`, AGE/RPG import, and bundled migrations passed smoke checks, and
1,362 unit tests passed in a fresh public-dependency environment. This clears
the missing-package blocker; full service/database integration remains
unqualified. See [development requirements](DEVELOPMENT.md) for the
client versions and the embedding server's pinned public source dependency.

Private staging on 2026-09-09 passed all eight native amd64/arm64 image build
and smoke jobs, plus 33 automation tests, in the
[hosted container checks](https://github.com/scitrera/memorylayer-enterprise/actions/runs/34388837564)
for commit `e319fb1aec17df0a5f7219ec9a2d87109f7f4fe5`.
[Hosted Python CI](https://github.com/scitrera/memorylayer-enterprise/actions/runs/34388783662)
passed 1,362 enterprise and 253 connector unit tests;
[dashboard CI](https://github.com/scitrera/memorylayer-enterprise/actions/runs/34388783760)
passed its production build and typecheck. These checks published no images.
PostgreSQL refresh publication was skipped while the repository was private.
Database upgrades, full service/Aether integration, browser/backend behavior,
GPU/model qualification, and actual registry promotion remain separate checks.

Service images install from their included source, run as non-root users,
carry root license notices and a `pip inspect` inventory, and require runtime
credentials through the operator's environment. The enterprise recipe also
ships the matching migration files with configured path overrides. PostgreSQL
images retain the extension sources' LICENSE/NOTICE files and source commits.
Complete dependency/license and vulnerability review against final image
digests before distribution. OCI source/revision labels identify the public
checkout when built by the workflows; keep that revision and its build scripts
publicly available with each distributed image.

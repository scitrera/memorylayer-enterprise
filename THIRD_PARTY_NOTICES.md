# Third-party notices

Scitrera's extensions are AGPL-3.0-only. Dependencies and upstream material
retain their original licenses. This source snapshot does not vendor the
MemoryLayer core, SDKs, storage services, Aether, model weights, or installed
Python/JavaScript dependency trees.

The PostgreSQL image collection is excluded from the blanket AGPL grant.
Scitrera's recipes and build glue in `postgres-container/` are Apache-2.0;
see that directory's [LICENSE](postgres-container/LICENSE) and
[NOTICE](postgres-container/NOTICE). Each assembled image component retains
its own terms. The recipes copy their own notices and extension notices into
the images, without copying the root AGPL license as an image-wide grant.

## Dashboard components

The dashboard identifies shadcn/ui as its component source. Retain the
[shadcn/ui MIT license](third_party/licenses/shadcn-ui/LICENSE) with the adapted
components in `memorylayer-admin/src/components/ui/`. Upstream:
<https://github.com/shadcn-ui/ui>. The copied license credits shadcn (2023).
Those adapted source files carry MIT SPDX identifiers and the upstream
copyright attribution; the first-party application files carry AGPL-3.0-only.

## Separately supplied Scitrera dependencies

| Dependency | Existing license |
| --- | --- |
| MemoryLayer core, RPG, embedding server, TypeScript SDK | Apache-2.0 |
| Aether Python client | Apache-2.0 |
| Scitrera messaging spec | Apache-2.0 |
| Scitrera rt-data | BSD-3-Clause |
| MemoryLayer Storage services | AGPL-3.0-only |
| MemoryLayer Storage Python blobgw client | Apache-2.0 |

Sources: [MemoryLayer](https://github.com/scitrera/memorylayer),
[Aether](https://github.com/scitrera/aether),
[messaging spec](https://github.com/scitrera/ecosystem-messaging-spec),
[rt-data](https://github.com/scitrera/scitrera-rt-data-python), and
[MemoryLayer Storage](https://github.com/scitrera/memorylayer-storage).

## Distribution inventory

This file is a source attribution record, not a complete resolved dependency
inventory. Before shipping wheels, JavaScript bundles, or images, inventory the
actual resolved packages and retain their applicable licenses and notices.
Include PostgreSQL/pgvector, Apache AGE, pg_textsearch, system tools, and all
dependencies inherited from the embedding base image. Downloaded model weights
have separate terms. Do not apply this repository's license to those materials.

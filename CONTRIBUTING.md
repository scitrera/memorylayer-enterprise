# Contributing to MemoryLayer Enterprise

Contributions to this repository require acceptance of the
[MemoryLayer Enterprise Contributor License Agreement](CLA.md) before merge.
The Project Owner is Scitrera LLC.

You retain ownership of your contributions. The CLA grants Scitrera copyright
and patent rights, including commercial and proprietary licensing rights, and
includes an open-source availability commitment. Read the full agreement;
this summary does not replace its terms.

## Accepting the CLA

This repository uses manual acceptance. No electronic CLA service or automated
CLA check is configured.

1. Read [CLA.md](CLA.md), version 1.0, including its treatment of prior and
   future contributions and any authorization required from your employer.
2. Ask a maintainer on your pull request for Scitrera's designated private
   acceptance channel. Complete the signature section and send the signed
   agreement through that channel. Do not post signed forms, email addresses,
   or other private acceptance details in public issues, PRs, or this repository.
3. Wait for a maintainer to confirm that acceptance covers the contribution
   and contributor identity before the contribution is merged. Existing
   acceptance can be reused if it covers the work; a CLA for another project
   does not establish acceptance of this agreement.

A PR checkbox or a commit `Signed-off-by` line is not CLA acceptance. Each
contributor whose work is included must be covered by an applicable individual
or authorized entity acceptance. A submitter cannot accept for other authors
without authority to do so.

General discussion, feature requests, and bug reports that do not intentionally
submit copyrightable material for inclusion are outside the CLA's definition
of a Contribution.

## Preparing a contribution

- Submit work you authored or are authorized to contribute, and identify
  third-party material with its source and license.
- Preserve copyright, SPDX, attribution, and license notices. This repository
  uses AGPL-3.0-only; the separate MemoryLayer OSS core and SDKs remain
  Apache-2.0 and follow their own contribution process. Storage also has its
  own repository and contribution policy.
- Add language-appropriate `SPDX-FileCopyrightText` and
  `SPDX-License-Identifier: AGPL-3.0-only` headers to first-party application
  source, tests, scripts, SQL migrations, and source templates. Preserve
  shebangs and Docker syntax directives as the first line. The adapted
  shadcn/ui files retain MIT and their upstream attribution. PostgreSQL
  container files are excluded from these AGPL headers; their build glue is
  Apache-2.0 and the images retain each component's terms as described in
  [postgres-container/NOTICE](postgres-container/NOTICE). Do not add comments
  to JSON, lockfiles, fixtures, or generated files that do not support them.
- Use synthetic fixtures. Keep customer documents, credentials, private
  configuration, database exports, and model data outside the repository.
- Follow the [development guide](docs/DEVELOPMENT.md), run checks relevant to
  your changes, and describe their results in the PR. For container or CI
  changes, also follow [container validation](docs/CONTAINERS.md).

## Maintainer verification

Before merging, verify acceptance for every contributor represented by the
work, including co-authors. Keep signed agreements and acceptance records in
Scitrera's private records, outside Git history and release exports. Record the
agreement version and exact text (or its hash), contributor identity, acceptance
date, contribution coverage, and entity authorization where applicable.

Confirm only that CLA coverage has been verified in the PR review; do not
publish the private record. If coverage or authority is unclear, resolve it
privately before merging. Passing CI or completing the PR template does not
perform this verification.

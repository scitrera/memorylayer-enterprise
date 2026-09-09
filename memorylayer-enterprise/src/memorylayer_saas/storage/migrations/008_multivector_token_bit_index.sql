-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- The optional binary-quantized (bit) candidate index is now created and
-- managed by 009 (alongside the type conversion + halfvec HNSW), so that index
-- creation always happens after the token column is guaranteed to be halfvec.
-- Creating/dropping it here would either run against a still-vector column on an
-- existing DB, or cause a rebuild-every-startup. Intentionally a no-op.
SELECT 1;

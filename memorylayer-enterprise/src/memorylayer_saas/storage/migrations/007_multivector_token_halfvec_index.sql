-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- Superseded by 006: the token column is now stored as halfvec directly, so the
-- dedicated halfvec *expression* index (built during the 1a experiment, when the
-- column was still float32 vector) is redundant with 006's direct halfvec HNSW.
-- Drop it if present. Idempotent; no-op on fresh DBs.
DROP INDEX IF EXISTS mmt_embedding_hnsw_half;

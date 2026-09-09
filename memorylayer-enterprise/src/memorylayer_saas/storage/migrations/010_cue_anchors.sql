-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- Cue-anchor storage for the Memora-inspired cue retrieval channel.
--
-- A cue anchor is a short "[entity] + [aspect]" semantic key generated per
-- memory (ExtractionService.generate_cue_anchors), embedded, and stored here.
-- At recall the cue embeddings are searched by vector similarity and
-- dereferenced back to their parent memory for the RRF cue fusion arm
-- (MemoryService._fuse_cue_results). Ships DARK: rows are only written when the
-- cue channel (MEMORYLAYER_CUE_CHANNEL_ENABLED) is on.
--
-- Idempotent (CREATE ... IF NOT EXISTS); runs every startup. The ORM
-- (Base.metadata.create_all -> CueAnchorModel) is the AUTHORITATIVE creator and
-- runs BEFORE this file, using the configured embedding dimension
-- (MEMORYLAYER_EMBEDDING_DIMENSIONS). The statements below are a defensive
-- fallback for DBs not built via create_all; the vector dimension here is the
-- default (1536) and the IF NOT EXISTS guards make it a no-op when the ORM
-- already created the table/indexes. The HNSW index mirrors idx_fragments_embedding
-- (m=16, ef_construction=64, vector_cosine_ops).
CREATE TABLE IF NOT EXISTS cue_anchors (
    id             text PRIMARY KEY,
    workspace_id   text NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    memory_id      text NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    cue            text NOT NULL,
    normalized_cue text NOT NULL,
    embedding      vector(1536),
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cue_anchors_workspace ON cue_anchors (workspace_id);
CREATE INDEX IF NOT EXISTS idx_cue_anchors_memory ON cue_anchors (memory_id);
CREATE INDEX IF NOT EXISTS idx_cue_anchors_normalized ON cue_anchors (normalized_cue);

-- HNSW cosine index over the cue embeddings (stage-1 ANN for the cue arm).
CREATE INDEX IF NOT EXISTS idx_cue_anchors_embedding
    ON cue_anchors USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

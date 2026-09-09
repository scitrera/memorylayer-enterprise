-- ColBERT-style token-flatten index for indexed late-interaction (MaxSim).
-- The brute-force max_sim path (search_memories_multivector) seq-scans every
-- memory's token array and does not scale. This satellite table stores one row
-- per document token-vector with an HNSW index, enabling two-stage retrieval:
-- (1) ANN over query tokens prunes to a small candidate set, (2) exact max_sim
-- reranks only those (see search_memories_multivector_indexed). Idempotent.
--
-- Token vectors are stored as halfvec(128) (16-bit): measured equal-quality to
-- float32 (experiment 1a/1b) at half the bytes, with faster HNSW distance.
-- Token dim is fixed at 128 (ColPali / ModernVBERT late-interaction encoder).
CREATE TABLE IF NOT EXISTS memory_multivector_tokens (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    memory_id    text NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    workspace_id text NOT NULL,
    token_index  int  NOT NULL,
    embedding    halfvec(128) NOT NULL
);

CREATE INDEX IF NOT EXISTS mmt_memory_idx ON memory_multivector_tokens (memory_id);
CREATE INDEX IF NOT EXISTS mmt_workspace_idx ON memory_multivector_tokens (workspace_id);

-- The HNSW candidate index (halfvec_cosine_ops) is created by 009, which also
-- handles converting pre-halfvec DBs. Keeping it there (not here) avoids trying
-- to build a halfvec index on a column that is still float32 vector on an
-- existing DB before the 009 type-conversion runs.

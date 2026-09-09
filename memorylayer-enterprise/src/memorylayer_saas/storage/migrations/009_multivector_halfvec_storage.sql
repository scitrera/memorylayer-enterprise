-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- Production halfvec storage migration for the multivector pipeline, and the
-- authoritative owner of the token HNSW candidate indexes.
--
-- Converts pre-halfvec DBs in place: memory_multivector_tokens.embedding and
-- memories.multivector from float32 vector -> 16-bit halfvec (measured
-- equal-quality at half the bytes; experiment 1a/1b). The conversion is guarded
-- by the current column type, so it is a no-op on fresh DBs that 006 + the ORM
-- already created as halfvec. The CREATE INDEX IF NOT EXISTS statements after
-- the DO block ensure the production indexes exist exactly once (they run after
-- the column is guaranteed halfvec, and are no-ops once the indexes exist).
-- Idempotent; runs every startup.
--
-- NOTE: ALTER COLUMN TYPE rewrites the table — on large existing token tables
-- this is a maintenance-window operation.
DO $$
BEGIN
    -- memory_multivector_tokens.embedding: vector(128) -> halfvec(128).
    -- Drop the float32/legacy candidate indexes first; 009 recreates them as
    -- halfvec below.
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_multivector_tokens'
          AND column_name = 'embedding'
          AND udt_name = 'vector'
    ) THEN
        DROP INDEX IF EXISTS mmt_embedding_hnsw;
        DROP INDEX IF EXISTS mmt_embedding_hnsw_half;
        DROP INDEX IF EXISTS mmt_embedding_hnsw_bit;
        ALTER TABLE memory_multivector_tokens
            ALTER COLUMN embedding TYPE halfvec(128) USING embedding::halfvec(128);
    END IF;

    -- memories.multivector: vector(128)[] -> halfvec(128)[].
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memories'
          AND column_name = 'multivector'
          AND udt_name = '_vector'
    ) THEN
        ALTER TABLE memories
            ALTER COLUMN multivector TYPE halfvec(128)[] USING multivector::halfvec(128)[];
    END IF;
END $$;

-- Stage-1 candidate index: HNSW over the halfvec token vectors (cosine).
CREATE INDEX IF NOT EXISTS mmt_embedding_hnsw
    ON memory_multivector_tokens USING hnsw (embedding halfvec_cosine_ops);

-- Optional binary-quantized (bit) candidate index for the most aggressive
-- compression tier (ann_precision='bit'). binary_quantize operates on vector,
-- so cast halfvec->vector; keep this expression identical to the ann_distance
-- in search_memories_multivector_indexed.
CREATE INDEX IF NOT EXISTS mmt_embedding_hnsw_bit
    ON memory_multivector_tokens
    USING hnsw ((binary_quantize(embedding::vector(128))::bit(128)) bit_hamming_ops);

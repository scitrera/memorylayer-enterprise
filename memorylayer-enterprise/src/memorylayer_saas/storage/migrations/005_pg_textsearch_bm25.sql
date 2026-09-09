-- pg_textsearch BM25 full-text search for the hybrid-retrieval keyword arm.
-- Replaces the unindexed to_tsvector / ts_rank_cd path (which underperformed
-- and ran a sequential scan) with an indexed BM25 ranking. Idempotent: runs on
-- every startup via PostgreSQLBackend._run_migrations.

CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- Searchable text = content + folded-in metadata aliases, mirroring the OSS
-- SQLite FTS arm so alias-only queries still retrieve the canonical memory.
-- A STORED generated column gives the BM25 index a stable, indexable text
-- column that auto-maintains on insert/update (note: the DB column for the
-- ORM `meta` attribute is named "metadata").
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS fts_content text
    GENERATED ALWAYS AS (content || ' ' || coalesce(metadata ->> 'aliases', '')) STORED;

-- BM25 index (Block-Max WAND accelerated top-k). text_config='english' matches
-- the prior to_tsvector('english', ...) tokenizer/stemmer behavior.
CREATE INDEX IF NOT EXISTS memories_fts_bm25
    ON memories USING bm25 (fts_content) WITH (text_config = 'english');

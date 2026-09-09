-- Link cue anchors to the canonical Entity they name (cue retrieval channel).
--
-- Adds cue_anchors.entity_id, a nullable FK onto entities(id). Populated at
-- ingest when the cue's entity string resolves against the entity registry
-- (else NULL). This lets the cue arm and the entity-anchored arm share one
-- entity vocabulary. ON DELETE SET NULL so a cue survives its entity being
-- merged/removed. Ships DARK (only written when the cue channel is on).
--
-- Idempotent (ADD COLUMN / CREATE INDEX IF NOT EXISTS); runs every startup. The
-- ORM (Base.metadata.create_all -> CueAnchorModel) is the AUTHORITATIVE creator
-- and runs BEFORE this file; the statements below are a defensive fallback for
-- DBs not built via create_all, and the IF NOT EXISTS guards make them a no-op
-- when the ORM already added the column/index.
ALTER TABLE cue_anchors
    ADD COLUMN IF NOT EXISTS entity_id text
    REFERENCES entities (id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_cue_anchors_entity ON cue_anchors (entity_id);

-- Mirror of alembic migration 015_add_thread_scope.py for the runtime
-- _run_migrations() path.  PostgreSQLBackend.connect() runs Base.metadata
-- create_all (which only creates missing tables, never alters existing
-- ones) followed by every *.sql file in this directory.  Alembic exists
-- in proprietary/memorylayer-enterprise/migrations/versions/ but is NOT
-- invoked at runtime — it's the production/ops-managed path.
--
-- For dev and any deployment that bootstraps via ``connect()``,
-- schema-altering migrations need a SQL twin here so existing tables
-- (created by an earlier create_all when the ORM didn't yet declare
-- ``scope``) get the new column without operator intervention.
--
-- Adds a nullable ``scope`` text column to ``chat_threads`` so the
-- application can separate web threads (scope='web') from Office add-in
-- threads (scope='office').  Existing rows get NULL (treated as 'web').
-- IF NOT EXISTS / IF EXISTS guards make this safe to re-run on every
-- startup — _run_migrations() is not state-tracked.

ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS scope TEXT;

CREATE INDEX IF NOT EXISTS idx_chat_threads_scope ON chat_threads (workspace_id, scope);

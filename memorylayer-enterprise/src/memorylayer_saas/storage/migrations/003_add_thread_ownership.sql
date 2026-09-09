-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- Mirror of alembic migration 016_add_chat_thread_ownership.py for the
-- runtime _run_migrations() path. PostgreSQLBackend.connect() runs
-- Base.metadata create_all (which only creates missing tables, never
-- alters existing ones) followed by every *.sql file in this directory.
-- Alembic exists in proprietary/memorylayer-enterprise/migrations/versions/
-- but is NOT invoked at runtime — it's the production/ops-managed path.
--
-- For dev and any deployment that bootstraps via ``connect()``,
-- schema-altering migrations need a SQL twin here so existing tables
-- (created by an earlier create_all when the ORM didn't yet declare
-- ``ownership``) get the new column without operator intervention.
--
-- Adds a non-null ``ownership`` text column to ``chat_threads`` with a
-- server-side default of ``'user'`` so the application can separate
-- user-owned threads (the user-session-scoped right rail) from
-- workspace-shared threads. Existing rows backfill to ``'user'`` via the
-- column default — matches the "leave legacy per-workspace _default
-- threads alone" decision. IF NOT EXISTS guards make this safe to re-run
-- on every startup — _run_migrations() is not state-tracked.

ALTER TABLE chat_threads ADD COLUMN IF NOT EXISTS ownership VARCHAR(16) NOT NULL DEFAULT 'user';

CREATE INDEX IF NOT EXISTS idx_chat_threads_tenant_user_ownership
    ON chat_threads (tenant_id, user_id, ownership)
    WHERE user_id IS NOT NULL;

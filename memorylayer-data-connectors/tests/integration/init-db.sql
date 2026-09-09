-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- Create the aether database alongside the default memorylayer database.
-- This runs as a docker-entrypoint-initdb.d script in the postgres container.
CREATE DATABASE aether WITH OWNER memorylayer;

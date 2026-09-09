-- Create the aether database alongside the default memorylayer database.
-- This runs as a docker-entrypoint-initdb.d script in the postgres container.
CREATE DATABASE aether WITH OWNER memorylayer;

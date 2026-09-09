-- Enable extensions in the default database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
LOAD 'age';

-- Set search_path at database level (applies to all connections to this db)
ALTER DATABASE :"DBNAME" SET search_path = ag_catalog, "$user", public;

-- Also set for the current user (whoever POSTGRES_USER is)
DO
$$
    BEGIN
        EXECUTE format('ALTER ROLE %I SET search_path = ag_catalog, "$user", public', current_user);
    END
$$;

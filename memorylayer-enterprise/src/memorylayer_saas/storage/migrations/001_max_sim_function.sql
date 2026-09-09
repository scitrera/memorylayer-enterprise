-- SPDX-FileCopyrightText: 2026 Scitrera LLC
-- SPDX-License-Identifier: AGPL-3.0-only

-- MaxSim function for vector arrays using cosine distance
-- Used for ColPali/ColBERT late interaction retrieval

-- vector[] version uses cosine distance <=>
CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[]) RETURNS double precision AS
$$
WITH queries AS (SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query) as q),
     documents AS (SELECT unnest(document) AS document),
     similarities AS (SELECT query_number, 1 - (document <=> query) AS similarity
                      FROM queries
                               CROSS JOIN documents),
     max_similarities AS (SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number)
SELECT SUM(max_similarity)
FROM max_similarities
$$ LANGUAGE SQL;

-- halfvec[] version for reduced precision storage
CREATE OR REPLACE FUNCTION max_sim(document halfvec[], query halfvec[]) RETURNS double precision AS
$$
WITH queries AS (SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query) as q),
     documents AS (SELECT unnest(document) AS document),
     similarities AS (SELECT query_number, 1 - (document <=> query) AS similarity
                      FROM queries
                               CROSS JOIN documents),
     max_similarities AS (SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number)
SELECT SUM(max_similarity)
FROM max_similarities
$$ LANGUAGE SQL;

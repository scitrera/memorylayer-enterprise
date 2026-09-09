# syntax=docker/dockerfile:1
# Repository-root context. Nightly builds resolve the tracking tag to a digest
# once and pass BASE_IMAGE to BOTH stages. Releases use this checked-in pin.
ARG BASE_IMAGE=pgvector/pgvector:pg17@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f
FROM ${BASE_IMAGE} AS builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates flex bison postgresql-server-dev-17 && \
    rm -rf /var/lib/apt/lists/*
ARG APACHE_AGE_REF=b7143d72f60db63d2d92081c5571a85df2ae8031
ARG PG_TEXTSEARCH_REF=f940210f9a7761529fab1cf85fc10e5d70699f0b
COPY postgres-container/build-extensions.sh /build-extensions.sh
RUN sh /build-extensions.sh "$APACHE_AGE_REF" "$PG_TEXTSEARCH_REF"

FROM ${BASE_IMAGE}
COPY --from=builder /extension-root/usr/lib/postgresql/17/lib/ /usr/lib/postgresql/17/lib/
COPY --from=builder /extension-root/usr/share/postgresql/17/extension/ /usr/share/postgresql/17/extension/
COPY --from=builder /extension-notices/ /usr/share/licenses/memorylayer/extensions/
COPY postgres-container/LICENSE postgres-container/NOTICE /usr/share/licenses/memorylayer/
# Clear inherited single-license metadata; components retain their own terms.
LABEL org.opencontainers.image.licenses=""
LABEL org.opencontainers.image.source="https://github.com/scitrera/memorylayer-enterprise"
# Loaded by initdb; the smoke test also sets this explicitly for both variants.
RUN echo "shared_preload_libraries = 'pg_textsearch'" >> /usr/share/postgresql/postgresql.conf.sample
COPY postgres-container/init.sql /docker-entrypoint-initdb.d/init.sql

#!/bin/sh
set -eu
age_ref="${1:?Apache AGE commit is required}"
text_ref="${2:?pg_textsearch commit is required}"
pg_config=/usr/lib/postgresql/17/bin/pg_config

build_extension() {
    name="$1"; repository="$2"; revision="$3"
    case "$revision" in *[!0-9a-f]*|'') echo 'Expected a full commit SHA' >&2; exit 1;; esac
    test "${#revision}" -eq 40
    source_dir="/tmp/extension-$name"
    git init "$source_dir"
    git -C "$source_dir" remote add origin "$repository"
    git -C "$source_dir" fetch --depth 1 origin "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
    test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
    make -C "$source_dir" -j2 PG_CONFIG="$pg_config"
    make -C "$source_dir" install DESTDIR=/extension-root PG_CONFIG="$pg_config"
    mkdir -p "/extension-notices/$name"
    cp "$source_dir/LICENSE" "/extension-notices/$name/LICENSE"
    if [ -f "$source_dir/NOTICE" ]; then
        cp "$source_dir/NOTICE" "/extension-notices/$name/NOTICE"
    fi
    printf '%s\n%s\n' "$repository" "$revision" > "/extension-notices/$name/SOURCE"
    rm -rf "$source_dir"
}

build_extension age https://github.com/apache/age.git "$age_ref"
build_extension pg_textsearch https://github.com/timescale/pg_textsearch.git "$text_ref"

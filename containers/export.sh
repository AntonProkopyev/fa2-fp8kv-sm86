#!/bin/sh
set -eu
identity=$(cat /artifacts/ARTIFACT_ID)
case "$identity" in ''|*[!0-9a-f]*) echo 'Invalid artifact identity' >&2; exit 1 ;; esac
[ "${#identity}" -eq 64 ] || exit 1
destination="/export/$identity"
mkdir -p /export
if [ -d "$destination" ]; then
    (cd "$destination" && sha256sum -c /artifacts/SHA256SUMS)
    exit 0
fi
staging=$(mktemp -d /export/.fa2-XXXXXX)
trap 'rm -rf "$staging"' EXIT
cp -a /artifacts/. "$staging/"
(cd "$staging" && sha256sum -c SHA256SUMS)
if ! mv -T "$staging" "$destination"; then
    # Another initializer may have published the same immutable directory.
    (cd "$destination" && sha256sum -c /artifacts/SHA256SUMS)
fi

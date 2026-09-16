#!/usr/bin/env bash
set -euo pipefail

RESULTS="${RESULTS:-/home/colligo/sglang-wiki/repos/sglang-omni/results/cosmos3-super/2gpu}"
BUNDLE="${RESULTS}/cosmos3-super-2gpu-artifacts.tar.gz"
FILE_LIST="$(mktemp /tmp/cosmos3-super-artifacts.XXXXXX)"
trap 'rm -f "${FILE_LIST}"' EXIT

cd "${RESULTS}"

# Native media request directories duplicate the stable, named output files.
find . -type f \
  ! -path './generation-quality/native-media/*' \
  ! -path './__pycache__/*' \
  ! -name 'cosmos3-super-2gpu-artifacts.tar.gz' \
  ! -name 'BUNDLE_SHA256' \
  ! -name 'SHA256SUMS' \
  -printf '%P\n' | LC_ALL=C sort > "${FILE_LIST}"

xargs -d '\n' sha256sum < "${FILE_LIST}" > SHA256SUMS
printf '%s\n' SHA256SUMS >> "${FILE_LIST}"
LC_ALL=C sort -o "${FILE_LIST}" "${FILE_LIST}"

tar --create --gzip --file "${BUNDLE}" --files-from "${FILE_LIST}"
sha256sum "$(basename "${BUNDLE}")" > BUNDLE_SHA256

echo "Wrote ${BUNDLE}"
cat BUNDLE_SHA256

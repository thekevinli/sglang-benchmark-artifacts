#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/colligo/sglang-wiki/envs/run/bin/python}"
SGLANG_SHA="${SGLANG_SHA:-4e9e407d3720045d59cae185c05f33649f4e544e}"
SGLANG_SRC="${SGLANG_SRC:-/mnt/localssd/src/sglang-${SGLANG_SHA}}"

mkdir -p "$(dirname "${SGLANG_SRC}")"
if [[ ! -d "${SGLANG_SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/sgl-project/sglang.git "${SGLANG_SRC}"
fi
git -C "${SGLANG_SRC}" fetch --depth 1 origin "${SGLANG_SHA}"
git -C "${SGLANG_SRC}" checkout --detach "${SGLANG_SHA}"

SGLANG_BUILD_RUST_EXTS=none uv pip install \
  --python "${PYTHON_BIN}" --no-deps --editable "${SGLANG_SRC}/python"
uv pip install --python "${PYTHON_BIN}" sglang-kernel==0.4.7

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import sglang

print("sglang", importlib.metadata.version("sglang"), sglang.__file__)
print("sglang-kernel", importlib.metadata.version("sglang-kernel"))
print("flashinfer-python", importlib.metadata.version("flashinfer-python"))
PY

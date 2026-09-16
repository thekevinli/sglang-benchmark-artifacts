#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Export a Hugging Face token with access to nvidia/Cosmos3-Super}"

readonly MODEL_ID="nvidia/Cosmos3-Super"
readonly MODEL_REVISION="fe77b66696d645f663b8f27e942b3b43e4629e23"
readonly MODEL_DIR="${COSMOS3_SUPER_MODEL_DIR:-/mnt/localssd/models/Cosmos3-Super-${MODEL_REVISION}}"
readonly HF_BIN="${COSMOS3_SUPER_HF_BIN:-/home/colligo/sglang-wiki/envs/run/bin/hf}"

mkdir -p "${MODEL_DIR}"

exec "${HF_BIN}" download \
  "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_DIR}"

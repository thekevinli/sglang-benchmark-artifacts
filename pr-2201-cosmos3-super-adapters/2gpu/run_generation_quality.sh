#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/colligo/sglang-wiki/repos/sglang-omni}"
PYTHON_BIN="${PYTHON_BIN:-/home/colligo/sglang-wiki/envs/run/bin/python}"
RESULTS="${RESULTS:-${ROOT}/results/cosmos3-super/2gpu}"

export COSMOS3_SUPER_RUN_GPU=1
export COSMOS3_SUPER_MODEL_PATH="${COSMOS3_SUPER_MODEL_PATH:-/mnt/localssd/models/Cosmos3-Super-fe77b66696d645f663b8f27e942b3b43e4629e23}"
export COSMOS3_SUPER_GPU_IDS="${COSMOS3_SUPER_GPU_IDS:-0,1}"
export COSMOS3_SUPER_CHECKPOINT_REVISION="${COSMOS3_SUPER_CHECKPOINT_REVISION:-fe77b66696d645f663b8f27e942b3b43e4629e23}"
export COSMOS3_SUPER_NATIVE_REVISION="${COSMOS3_SUPER_NATIVE_REVISION:-git:sgl-project/sglang@4e9e407d3720045d59cae185c05f33649f4e544e;sglang-kernel==0.4.7;flashinfer-python==0.6.18}"
export COSMOS3_SUPER_STARTUP_TIMEOUT="${COSMOS3_SUPER_STARTUP_TIMEOUT:-1800}"
export COSMOS3_SUPER_REQUEST_TIMEOUT="${COSMOS3_SUPER_REQUEST_TIMEOUT:-7200}"
export COSMOS3_SUPER_NATIVE_OVERRIDES="${COSMOS3_SUPER_NATIVE_OVERRIDES:-${RESULTS}/native-overrides-layerwise.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULTS}/generation-quality"
cd "${ROOT}"

# Defaults to the three PR examples. Pass any subset of: t2v i2v t2vs i2vs.
if [[ "$#" -eq 0 ]]; then
  set -- t2v i2v t2vs
fi

"${PYTHON_BIN}" "${RESULTS}/run_generation_quality.py" "$@" \
  2>&1 | tee "${RESULTS}/generation-quality-run.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/colligo/sglang-wiki/repos/sglang-omni}"
PYTHON_BIN="${PYTHON_BIN:-/home/colligo/sglang-wiki/envs/run/bin/python}"
RESULTS="${RESULTS:-${ROOT}/results/cosmos3-super/2gpu}"
MODEL_PATH="${COSMOS3_SUPER_MODEL_PATH:-/mnt/localssd/models/Cosmos3-Super-fe77b66696d645f663b8f27e942b3b43e4629e23}"
TAG="${CAMPAIGN_TAG:-reproduction}"
MODE="${1:-all}"

export COSMOS3_SUPER_RUN_GPU=1
export COSMOS3_SUPER_MODEL_PATH="${MODEL_PATH}"
export COSMOS3_SUPER_GPU_IDS="${COSMOS3_SUPER_GPU_IDS:-0,1}"
export COSMOS3_SUPER_CHECKPOINT_REVISION="${COSMOS3_SUPER_CHECKPOINT_REVISION:-fe77b66696d645f663b8f27e942b3b43e4629e23}"
export COSMOS3_SUPER_NATIVE_REVISION="${COSMOS3_SUPER_NATIVE_REVISION:-git:sgl-project/sglang@4e9e407d3720045d59cae185c05f33649f4e544e;sglang-kernel==0.4.7;flashinfer-python==0.6.18}"
export COSMOS3_SUPER_STARTUP_TIMEOUT="${COSMOS3_SUPER_STARTUP_TIMEOUT:-1800}"
export COSMOS3_SUPER_REQUEST_TIMEOUT="${COSMOS3_SUPER_REQUEST_TIMEOUT:-1800}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULTS}"
cd "${ROOT}"

run_generation() {
  export COSMOS3_SUPER_NATIVE_OVERRIDES="${RESULTS}/native-overrides-layerwise.json"
  "${PYTHON_BIN}" -m pytest -vv -s \
    tests/integration/cosmos3/test_super_gpu.py::test_super_generation \
    --junitxml="${RESULTS}/${TAG}-generation.xml" \
    --basetemp="${RESULTS}/${TAG}-generation-artifacts" --durations=20 \
    2>&1 | tee "${RESULTS}/${TAG}-generation.log"
  unset COSMOS3_SUPER_NATIVE_OVERRIDES
}

run_reasoner() {
  "${PYTHON_BIN}" -m pytest -vv -s \
    tests/integration/cosmos3/test_super_gpu.py::test_super_reasoner \
    --junitxml="${RESULTS}/${TAG}-reasoner.xml" \
    --basetemp="${RESULTS}/${TAG}-reasoner-artifacts" --durations=20 \
    2>&1 | tee "${RESULTS}/${TAG}-reasoner.log"
}

run_lifecycle() {
  "${PYTHON_BIN}" -m pytest -vv -s \
    tests/integration/cosmos3/test_super_gpu.py::test_super_reasoner_lifecycle \
    --junitxml="${RESULTS}/${TAG}-lifecycle.xml" \
    --basetemp="${RESULTS}/${TAG}-lifecycle-artifacts" --durations=20 \
    2>&1 | tee "${RESULTS}/${TAG}-lifecycle.log"
}

case "${MODE}" in
  generation) run_generation ;;
  reasoner) run_reasoner ;;
  lifecycle) run_lifecycle ;;
  all) run_generation; run_reasoner; run_lifecycle ;;
  *) echo "usage: $0 [all|generation|reasoner|lifecycle]" >&2; exit 2 ;;
esac

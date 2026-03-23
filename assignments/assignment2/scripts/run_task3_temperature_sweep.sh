#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

MODEL_NAME="${ASR_MODEL_NAME:-./models/facebook-wav2vec2-base-100h}"
DEVICE="${ASR_DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${ASR_MANIFEST:-data/librispeech_test_other/manifest.csv}"
TEMPERATURES="${TEMPERATURES:-0.5 0.8 1.0 1.2 1.5 2.0}"
LIMIT="${LIMIT:-}"

OUT_DIR="outputs/task3_temperature_sweep"
mkdir -p "${OUT_DIR}"

cmd=(
    "${PYTHON_BIN}" sweep_decoder.py
    --model-name "${MODEL_NAME}"
    --manifest "${MANIFEST}"
    --device "${DEVICE}"
    --output-dir "${OUT_DIR}"
    temperature
    --temperatures ${TEMPERATURES}
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"

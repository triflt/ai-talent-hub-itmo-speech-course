#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

MODEL_NAME="${ASR_MODEL_NAME:-./models/facebook-wav2vec2-base-100h}"
DEVICE="${ASR_DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${ASR_MANIFEST:-data/librispeech_test_other/manifest.csv}"
BEAM_WIDTHS="${BEAM_WIDTHS:-1 3 10 50}"
LIMIT="${LIMIT:-}"

OUT_DIR="outputs/task2_beam_sweep"
mkdir -p "${OUT_DIR}"

cmd=(
    "${PYTHON_BIN}" sweep_decoder.py
    --model-name "${MODEL_NAME}"
    --manifest "${MANIFEST}"
    --device "${DEVICE}"
    --output-dir "${OUT_DIR}"
    beam
    --beam-widths ${BEAM_WIDTHS}
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"

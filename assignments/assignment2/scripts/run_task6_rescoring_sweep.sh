#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

MODEL_NAME="${ASR_MODEL_NAME:-./models/facebook-wav2vec2-base-100h}"
DEVICE="${ASR_DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${ASR_MANIFEST:-data/librispeech_test_other/manifest.csv}"
LM_MODEL_PATH="${LM_MODEL_PATH:-lm/3-gram.pruned.1e-7.arpa.gz}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"
ALPHAS="${ALPHAS:-0.01 0.05 0.1 0.5}"
BETAS="${BETAS:-0.0 0.5 1.0}"
LIMIT="${LIMIT:-}"

OUT_DIR="outputs/task6_rescoring"
mkdir -p "${OUT_DIR}"

cmd=(
    "${PYTHON_BIN}" sweep_decoder.py
    --model-name "${MODEL_NAME}"
    --manifest "${MANIFEST}"
    --lm-model-path "${LM_MODEL_PATH}"
    --device "${DEVICE}"
    --output-dir "${OUT_DIR}"
    beam_lm_rescore
    --beam-width "${BEAM_WIDTH}"
    --alphas ${ALPHAS}
    --betas ${BETAS}
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"

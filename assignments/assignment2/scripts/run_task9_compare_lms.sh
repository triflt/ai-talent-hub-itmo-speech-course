#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

MODEL_NAME="${ASR_MODEL_NAME:-./models/facebook-wav2vec2-base-100h}"
DEVICE="${ASR_DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"
ALPHA="${ALPHA:-0.1}"
BETA="${BETA:-0.5}"
LIMIT="${LIMIT:-}"

GENERAL_LM_NAME="${GENERAL_LM_NAME:-librispeech_3gram}"
GENERAL_LM_PATH="${GENERAL_LM_PATH:-lm/3-gram.pruned.1e-7.arpa.gz}"
FIN_LM_NAME="${FIN_LM_NAME:-financial_3gram}"
FIN_LM_PATH="${FIN_LM_PATH:-lm/financial-3gram.binary}"
OUT_DIR="outputs/task9_compare_lms"

cmd=(
    "${PYTHON_BIN}" run_task9_compare_lms.py
    --model-name "${MODEL_NAME}"
    --device "${DEVICE}"
    --beam-width "${BEAM_WIDTH}"
    --alpha "${ALPHA}"
    --beta "${BETA}"
    --output-dir "${OUT_DIR}"
    --lm "${GENERAL_LM_NAME}" "${GENERAL_LM_PATH}"
    --lm "${FIN_LM_NAME}" "${FIN_LM_PATH}"
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"

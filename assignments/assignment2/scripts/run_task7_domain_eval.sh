#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

MODEL_NAME="${ASR_MODEL_NAME:-./models/facebook-wav2vec2-base-100h}"
DEVICE="${ASR_DEVICE:-mps}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LM_MODEL_PATH="${LM_MODEL_PATH:-lm/3-gram.pruned.1e-7.arpa.gz}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"
ALPHA="${ALPHA:-0.1}"
BETA="${BETA:-0.5}"
LIMIT="${LIMIT:-}"
OUT_DIR="outputs/task7_domain_eval"

cmd=(
    "${PYTHON_BIN}" run_task7_eval.py
    --model-name "${MODEL_NAME}"
    --lm-model-path "${LM_MODEL_PATH}"
    --device "${DEVICE}"
    --beam-width "${BEAM_WIDTH}"
    --alpha "${ALPHA}"
    --beta "${BETA}"
    --output-dir "${OUT_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"

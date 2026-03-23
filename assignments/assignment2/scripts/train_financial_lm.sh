#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

KENLM_BUILD_DIR="${KENLM_BUILD_DIR:-/tmp/kenlm_build/build}"
LMPLZ_BIN="${LMPLZ_BIN:-${KENLM_BUILD_DIR}/bin/lmplz}"
BUILD_BINARY_BIN="${BUILD_BINARY_BIN:-${KENLM_BUILD_DIR}/bin/build_binary}"
CORPUS_PATH="${CORPUS_PATH:-data/earnings22_train/corpus.txt}"
ORDER="${ORDER:-3}"
OUT_PREFIX="${OUT_PREFIX:-lm/financial-${ORDER}gram}"

ARPA_PATH="${OUT_PREFIX}.arpa"
ARPA_GZ_PATH="${OUT_PREFIX}.arpa.gz"
BINARY_PATH="${OUT_PREFIX}.binary"

if [[ ! -x "${LMPLZ_BIN}" ]]; then
    echo "Missing lmplz: ${LMPLZ_BIN}" >&2
    exit 1
fi

if [[ ! -x "${BUILD_BINARY_BIN}" ]]; then
    echo "Missing build_binary: ${BUILD_BINARY_BIN}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT_PREFIX}")"

"${LMPLZ_BIN}" -o "${ORDER}" --discount_fallback < "${CORPUS_PATH}" > "${ARPA_PATH}"
gzip -c "${ARPA_PATH}" > "${ARPA_GZ_PATH}"
"${BUILD_BINARY_BIN}" "${ARPA_PATH}" "${BINARY_PATH}"

echo "Saved: ${ARPA_PATH}"
echo "Saved: ${ARPA_GZ_PATH}"
echo "Saved: ${BINARY_PATH}"

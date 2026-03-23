#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ASSIGNMENT_DIR}"

RUN_LM_SWEEPS="${RUN_LM_SWEEPS:-0}"
RUN_TASK2="${RUN_TASK2:-1}"
RUN_TASK3="${RUN_TASK3:-1}"
RUN_TASK4="${RUN_TASK4:-${RUN_LM_SWEEPS}}"
RUN_TASK6="${RUN_TASK6:-${RUN_LM_SWEEPS}}"

if [[ "${RUN_TASK2}" == "1" ]]; then
    bash "${SCRIPT_DIR}/run_task2_beam_sweep.sh"
else
    echo "Skipped Task 2 beam sweep."
fi

if [[ "${RUN_TASK3}" == "1" ]]; then
    bash "${SCRIPT_DIR}/run_task3_temperature_sweep.sh"
else
    echo "Skipped Task 3 temperature sweep."
fi

if [[ "${RUN_TASK4}" == "1" ]]; then
    bash "${SCRIPT_DIR}/run_task4_shallow_fusion_sweep.sh"
else
    echo "Skipped Task 4 shallow-fusion sweep."
fi

if [[ "${RUN_TASK6}" == "1" ]]; then
    bash "${SCRIPT_DIR}/run_task6_rescoring_sweep.sh"
else
    echo "Skipped Task 6 rescoring sweep."
fi

#!/usr/bin/env bash
# Evaluate a fixed checkpoint (the frozen teacher or the initial student) under the common
# protocol for the three evaluation seeds (11, 22, 33): same serialisation, decoding, stop
# tokens and answer checker as the distilled students (Appendix B.2; supplementary E02-E05).
#
#   scripts/evaluate_reference.sh configs/qwen3_4b/vdopd.yaml frozen_teacher Qwen/Qwen3-8B
#   scripts/evaluate_reference.sh configs/qwen3_4b/vdopd.yaml initial_student
#   scripts/evaluate_reference.sh configs/qwen3_4b/vdopd.yaml frozen_teacher Qwen/Qwen3-8B 11
#
# Arguments: <config> <name> [<model id or checkpoint folder>] [<seed> ...]
# Without a model the initial student of the config is evaluated.
# Output:   results/<group>/<name>/seed<seed>.json (input of scripts/aggregate_results.py
#           and scripts/supplementary_tables.py)
# Extra `--set key=value` overrides can be passed through the VDOPD_OVERRIDES variable.
set -euo pipefail

config="$1"; name="$2"; shift 2
model=()
if [ $# -gt 0 ] && ! [[ "$1" =~ ^[0-9]+$ ]]; then model=(--student "$1"); shift; fi
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(11 22 33)
group="$(basename "$(dirname "$config")")"
overrides=()
for item in ${VDOPD_OVERRIDES:-}; do overrides+=(--set "$item"); done

for seed in "${seeds[@]}"; do
  vdopd eval --config "$config" --set "evaluation.decoding.seed=${seed}" \
    ${model[@]+"${model[@]}"} --output "results/${group}/${name}/seed${seed}.json" \
    ${overrides[@]+"${overrides[@]}"}
done

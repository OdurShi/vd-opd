#!/usr/bin/env bash
# Train and evaluate one configuration for the paper's three seeds (11, 22, 33).
#
#   scripts/run_experiment.sh configs/qwen3_1p7b/vdopd.yaml            # all three seeds
#   scripts/run_experiment.sh configs/qwen3_1p7b/eopd_full.yaml 11     # a single seed
#
# Outputs:  runs/<group>/<method>/seed<seed>/        training logs and checkpoints
#           results/<group>/<method>/seed<seed>.json evaluation (input of aggregate_results.py)
# Extra `--set key=value` overrides can be passed through the VDOPD_OVERRIDES variable.
set -euo pipefail

config="$1"; shift
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(11 22 33)
method="$(basename "${config%.yaml}")"
group="$(basename "$(dirname "$config")")"
overrides=()
for item in ${VDOPD_OVERRIDES:-}; do overrides+=(--set "$item"); done

for seed in "${seeds[@]}"; do
  run_dir="runs/${group}/${method}/seed${seed}"
  vdopd train --config "$config" --set "train.seed=${seed}" --set "train.output_dir=${run_dir}" \
    ${overrides[@]+"${overrides[@]}"}
  vdopd eval --config "$config" --set "evaluation.decoding.seed=${seed}" \
    --student "${run_dir}/final/student" --output "results/${group}/${method}/seed${seed}.json" \
    ${overrides[@]+"${overrides[@]}"}
done

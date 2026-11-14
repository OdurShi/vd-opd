# Reduced-scale example outputs

Raw outputs of the configurations in `configs/reduced_scale/`: teacher
`HuggingFaceTB/SmolLM2-360M-Instruct`, student `HuggingFaceTB/SmolLM2-135M-Instruct`,
synthetic integer addition/subtraction, 30 rollout batches of 32 prompts, seed 11,
evaluation on 60 held-out problems with 4 samples each (Avg@4 / Pass@4) under the canonical
prompt and the four held-out wrappers.

| Path | Content |
|---|---|
| `<method>/seed11.json` | evaluation file written by `vdopd eval --output` |
| `training/<method>/manifest.json` | resolved configuration and software versions of the training run |
| `training/<method>/metrics.jsonl` | per-iteration training metrics (loss, `N_k`, retained mass, entropies, FLOPs) |

Commands:

```bash
vdopd train --config configs/reduced_scale/gkd_rkl.yaml
vdopd train --config configs/reduced_scale/vdopd.yaml
vdopd eval --config configs/reduced_scale/vdopd.yaml \
    --output results/reduced_scale/initial_student/seed11.json
vdopd eval --config configs/reduced_scale/vdopd.yaml \
    --student HuggingFaceTB/SmolLM2-360M-Instruct \
    --output results/reduced_scale/frozen_teacher/seed11.json
vdopd eval --config configs/reduced_scale/gkd_rkl.yaml \
    --student runs/reduced_scale/gkd_rkl/seed11/final/student \
    --output results/reduced_scale/gkd_rkl/seed11.json
vdopd eval --config configs/reduced_scale/vdopd.yaml \
    --student runs/reduced_scale/vdopd/seed11/final/student \
    --output results/reduced_scale/vdopd/seed11.json
```

The tables below are the output of
`python scripts/aggregate_results.py --results-dir results/reduced_scale --metric <metric>`
(percent, one seed).

## `avg_at_k`

| method | seeds | synthetic_arithmetic | mean |
|---|---|---|---|
| frozen_teacher | 1 | 51.2 | 51.2 |
| gkd_rkl | 1 | 34.6 | 34.6 |
| initial_student | 1 | 3.8 | 3.8 |
| vdopd | 1 | 32.1 | 32.1 |

## `pass_at_k`

| method | seeds | synthetic_arithmetic | mean |
|---|---|---|---|
| frozen_teacher | 1 | 78.3 | 78.3 |
| gkd_rkl | 1 | 55.0 | 55.0 |
| initial_student | 1 | 10.0 | 10.0 |
| vdopd | 1 | 61.7 | 61.7 |

## `mean_view`

| method | seeds | synthetic_arithmetic | mean |
|---|---|---|---|
| frozen_teacher | 1 | 55.5 | 55.5 |
| gkd_rkl | 1 | 28.1 | 28.1 |
| initial_student | 1 | 4.9 | 4.9 |
| vdopd | 1 | 24.9 | 24.9 |

## `worst_view`

| method | seeds | synthetic_arithmetic | mean |
|---|---|---|---|
| frozen_teacher | 1 | 42.9 | 42.9 |
| gkd_rkl | 1 | 16.7 | 16.7 |
| initial_student | 1 | 0.8 | 0.8 |
| vdopd | 1 | 14.2 | 14.2 |

## `worst_template`

| method | seeds | synthetic_arithmetic | mean |
|---|---|---|---|
| frozen_teacher | 1 | 49.6 | 49.6 |
| gkd_rkl | 1 | 26.2 | 26.2 |
| initial_student | 1 | 3.8 | 3.8 |
| vdopd | 1 | 22.1 | 22.1 |

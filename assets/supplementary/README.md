# Figures of the supplementary experiments

Drawn by [`scripts/plot_supplementary.py`](../../scripts/plot_supplementary.py) from the
tables in [`results/supplementary/`](../../results/supplementary/README.md); the tables
themselves are rendered in [docs/supplementary_results.md](../../docs/supplementary_results.md).

| File | Content | Table |
|---|---|---|
| `e01_gain_by_benchmark.png` | Gain of VD-OPD over EOPD(full) in Avg@8 for three students and six benchmarks, with 95% confidence intervals | E01 |
| `e02_e04_macro_by_method.png` | Macro Avg@8 over the six benchmarks for every method and student (mean of the per-benchmark entries); the vertical line is the frozen teacher | E02-E04 |
| `e07_target_weight.png` | Target and weight ablation, macro Avg@8, mean ± SD over three seeds | E07 |
| `e10_view_pool.png` | EOPD(2V) and VD-OPD under three training view pools, ordered by the mean agreement Z of the pool | E10 |
| `e14_heldout_views.png` | MeanView, WorstView and canonical accuracy on three families of held-out prompt views | E14 |
| `e17_local_updates.png` | Effect of one matched update in the low-Z and high-Z quartiles, with 95% confidence intervals | E17 |

To redraw after editing a table: `python scripts/plot_supplementary.py`.

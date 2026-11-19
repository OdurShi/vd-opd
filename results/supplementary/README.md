# Supplementary experiment tables (E01-E18)

Supplementary results reported by the authors of the paper: seed variation and confidence
intervals for the main tables, compute- and view-matched controls, further ablations,
prompt-robustness tests and an analysis of the cross-view agreement `Z`. The rendered tables,
with the setup and the key findings of each experiment, are in
[docs/supplementary_results.md](../../docs/supplementary_results.md).

## Layout

| Folder | Experiments |
|---|---|
| `01_main_results/` | E01 three-seed per-benchmark results with 95% CIs; E02-E04 complete Tables 6-8 (Avg@8 and Pass@8, all methods); E05 teacher vs. students on MATH500 |
| `02_compute_controls/` | E06 controls matched in total FLOPs and in teacher views and updates (Table 2) |
| `03_ablations/` | E07 target x weight (Table 4); E08 two vs. four teacher views; E09 gradient agreement of the view pairs; E10 training view pool |
| `04_baselines/` | E11 additional baseline tuning and reruns |
| `05_robustness/` | E12-E13 unseen wrappers (macro and per benchmark); E14 student-side augmentation; E15 held-out structural changes and paraphrases |
| `06_agreement_analysis/` | E16 teacher diagnostics by `Z`; E17 matched local updates by `Z` stratum; E18 two-token example |

`index.yaml` lists the experiments (title, the table of the paper they extend, setup, key
findings) and says how the CSV columns are displayed.

## File format

One CSV file per experiment, UTF-8, one header row. Composite cells are split into one
column per number:

| Displayed as | Columns |
|---|---|
| `mean ± SD` | `<name>_mean`, `<name>_sd` |
| `value [low, high]` (95% CI) | `<name>`, `<name>_ci_low`, `<name>_ci_high` |
| `a / b` | two columns, e.g. `math500_avg8`, `math500_pass8` |

Accuracy is in percent, gains in percentage points, FLOPs in units of 1e18. A blank cell
means "not applicable" (for example the gain of the reference row).

## Reading the tables

```python
from vdopd.supplementary import load, load_index

table = load("E07")  # one experiment
print(table.to_markdown())
row = table.row(weight="state-specific Z")
print(table.number(row, "macro_avg8_mean"))

for experiment in load_index():  # all of them
    print(experiment.id, experiment.title, len(experiment.rows))
```

```bash
python scripts/render_supplementary.py --show E07    # print one table
python scripts/plot_supplementary.py                 # figures in assets/supplementary/
```

## Updating the numbers

1. Edit the CSV file (keep the header). For a new experiment add a CSV file and an entry in
   `index.yaml`.
2. `python scripts/render_supplementary.py` rewrites `docs/supplementary_results.md` and
   prints every disagreement between tables that report the same quantity (for example a
   macro average that no longer equals the mean of the per-benchmark scores, or a confidence
   interval that does not contain its point estimate).
3. `python scripts/plot_supplementary.py` redraws the figures.
4. `pytest tests/test_supplementary_results.py` runs the same checks.

Tables in the same column layout can be produced from evaluation files of new runs with
`scripts/aggregate_results.py`.

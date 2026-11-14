# Paper figures

Figures of the paper "View-Decomposed On-Policy Distillation for Robust Reasoning in Large
Language Models" (under review), cropped from the paper PDF without their captions by
`scripts/extract_paper_figures.py`. Captions are given as printed in the paper.

| File | Figure | Caption |
|---|---|---|
| `fig01.png` | Figure 1 | From teacher imitation to consensus supervision. Left: matching a single teacher view can transfer prompt-sensitive predictions. Middle: task-preserving prompt views can disagree at the same student prefix, which entropy alone does not reveal. Right: VD-OPD learns what both views support and reduces supervision when they disagree. |
| `fig02.png` | Figure 2 | Teacher entropy and cross-view variation across 5,000 paired predictions. (a) Entropy–variation landscape, colored by pairs per hexagon. (b) JS divergence within entropy quartiles. Boxes span the interquartile range, lines indicate medians, whiskers mark the 5th–95th percentile range, and dots show values beyond the whiskers. |
| `fig03.png` | Figure 3 | VD-OPD evaluates both teacher views at the same student prefix. The indices i, t identify a response and a position within it; N_k counts valid response positions in iteration k. Square roots, products ⊙, and logarithms act elementwise. Teacher predictions are detached throughout the student update. |
| `fig04.png` | Figure 4 | Generalization across student sizes and model families on MATH500. Bars show zero-shot gains over the initial student in percentage points; dashed lines show frozen-teacher gains in the corresponding metric color. Dark bars highlight VD-OPD, with gain annotations listing Avg@8 / Pass@8. FKL/RKL/JSD denote GKD variants, OPD denotes sampled-token OPD, and EOPD uses full-vocabulary FKL; method definitions follow Table 1. Absolute scores are in Tables 6–8. |
| `fig05.png` | Figure 5 | Accuracy versus teacher-scoring computation. Qwen3-8B → Qwen3-1.7B-Base; zero-shot Avg@8. Points and error bars show mean ±1 sample SD over seeds 11, 22, and 33. The horizontal axis measures teacher-only FLOPs, distinct from total FLOPs in Table 2. |
| `fig06.png` | Figure 6 | Consensus mass and utility. Hexagons: log token density. Solid/band: median and 10th–90th percentiles. Dashed: token-weighted response correctness. Binning and smoothing follow Appendix B.7. |

Regenerate with:

```bash
pip install -e ".[figures]"
python scripts/extract_paper_figures.py /path/to/paper.pdf
```

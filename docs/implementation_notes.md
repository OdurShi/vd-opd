# Implementation notes

This document maps the paper "View-Decomposed On-Policy Distillation for Robust Reasoning in
Large Language Models" (under review) to the code, lists every detail the paper leaves open
together with the choice made here, and states what has and has not been executed.

Equation, table, algorithm and appendix numbers refer to the paper.

## 1. Paper-to-code map

### Method (Section 4, Appendix A)

| Paper | Code |
|---|---|
| Eq. 2, teacher views at the same student prefix | `vdopd/scoring.py`: `pack_prompt_and_response`, `response_hidden_states`, `iter_log_prob_blocks`; `vdopd/trainer.py`: `Trainer._teacher_hidden` |
| Table 5, App. A.1: wrappers, two newlines, unchanged suffix | `vdopd/views.py`: `TRAIN_WRAPPERS`, `apply_wrapper` |
| App. A.1: validity checks, uniform choice, identity fallback | `vdopd/views.py`: `fits_context`, `preserves_content_and_roles`, `select_views` |
| App. A.1: response-token alignment across prompt lengths | `vdopd/scoring.py`: `pack_prompt_and_response` (`gather_index`) |
| App. A.1/B.4: fixed chat serialisation, Qwen3 non-thinking block | `vdopd/chat.py`: `HFChatCodec` (`enable_thinking: false`) |
| Eq. 3, square-root coordinates `u+`, `u-` | `vdopd/consensus.py`: `view_decomposition` |
| Eq. 4-5, consensus `c` and mass `Z` (formed as in App. A.2) | `vdopd/consensus.py`: `geometric_consensus`, `bhattacharyya_affinity`, `squared_hellinger` |
| Eq. 6, generalised KL barycentre | `vdopd/consensus.py`: `generalized_kl`; checked in `tests/test_consensus.py` |
| Eq. 7, loss with unweighted denominator `N_k` | `vdopd/objectives.py`: `ConsensusObjective("geometric", "mass")`; reduction in `Trainer.accumulate_gradients` |
| Eq. 8, loss decomposition and logit gradient | `tests/test_objectives.py::test_loss_decomposition_and_logit_gradient` |
| App. A.2: full-vocabulary rollouts at temperature one | `vdopd/models.py`: `CausalLM.generate`; `Trainer.collect_rollouts` |
| App. A.2: blockwise evaluation, common denominator, masking | `Trainer.accumulate_gradients` |
| App. A.2: skip rules (empty, non-finite, zero mass) | `Trainer.accumulate_gradients` (`status`), `Trainer.apply_update` |
| Algorithm 1 | `Trainer.train_step` = `collect_rollouts` (lines 2-4) + `accumulate_gradients` (5-11) + `apply_update` (12-13) |
| App. B.4: AdamW, warmup, cosine decay, clipping | `vdopd/optim.py` |
| App. B.4: independent random streams | `vdopd/seeding.py` |

### Baselines, controls and ablations (Section 5, Appendix B.3, B.5, B.6)

| Paper | Code (`objective.name`) |
|---|---|
| GKD-FKL / GKD-RKL (full) / GKD-JSD | `ForwardKL` (`gkd_fkl`), `ReverseKL` (`gkd_rkl`), `GeneralizedJSD` (`gkd_jsd`) |
| Sampled-token OPD | `SampledTokenOPD` (`sampled_opd`) |
| EOPD (top-16) / EOPD (full) | `EOPD` (`eopd_top16`, `eopd_full`) |
| Eq. 13, EOPD-2V | `TwoViewAverage` (`eopd_2v`) |
| Table 4(a): arithmetic / weighted arithmetic / geometric / VD-OPD | `ConsensusObjective` presets `arithmetic`, `weighted_arithmetic`, `geometric`, `vdopd` |
| Table 4(b): shuffled weight, batch-mean weight | presets `shuffled_weight`, `batch_mean_weight`; `assign_weights` |
| App. B.5: FLOPs components and budgets (Table 2, Figure 5) | `vdopd/flops.py`; `TrainConfig.max_total_flops`, `max_teacher_flops` |
| ToDi-OP, TSD-KD, Veto, CROP | hooks only: `UNSPECIFIED_BASELINES`, `register_objective` (Section 3 below) |

### Evaluation and diagnostics (Appendix B.1, B.2, B.7, B.8)

| Paper | Code |
|---|---|
| App. B.1: deduplication, overlap removal, stratified 500/500 split, seed 42 | `vdopd/data/math_train.py`, `scripts/make_math_splits.py`, index files in `data/math_train/splits/` |
| App. B.2.1: benchmarks | `vdopd/data/benchmarks.py`, files in `data/benchmarks/`, `scripts/download_data.py` |
| App. B.2.1: decoding protocol | `vdopd/eval/evaluator.py`: `EvalConfig`, `sample_correctness` |
| Eq. 10, Avg@8 and Pass@8 | `vdopd/eval/metrics.py`: `avg_at_k`, `pass_at_k` |
| App. B.2.2: four held-out wrappers | `vdopd/views.py`: `EVAL_WRAPPERS` |
| Eq. 11-12, MeanView / WorstView / WorstTemplate | `vdopd/eval/metrics.py`: `view_robustness` |
| Eq. 14, seed mean and sample SD | `vdopd/eval/metrics.py`: `seed_mean_and_sd`; `scripts/teacher_flops_curve.py` |
| Eq. 1, Figure 2 | `vdopd/consensus.py`: `js_divergence`, `entropy`; `vdopd/diagnostics.py`: `paired_view_statistics`, `entropy_quartile_summary`; `scripts/preliminary_study.py` |
| Eq. 9, Figure 6, Table 12, App. B.7 | `vdopd/diagnostics.py`: `consensus_fit_loss`, `position_statistics`, `matched_branch_updates`, `binned_utility_summary`; `scripts/local_update_diagnostic.py` |
| Eq. 15-16, candidate distributions | `vdopd/diagnostics.py`: `candidate_logits`; `vdopd/theory.py`: `softmax` |
| Eq. 17-20, 22 | `vdopd/theory.py`: `smoothed_reference`, `wrapper_shift`, `cross_fit_margins`, `rescale_mean_shift` |

### Theory (Section 4.4, Appendix D)

`vdopd/theory.py` holds float64 NumPy implementations; `tests/test_theory.py` checks the
statements numerically.

| Statement | Check |
|---|---|
| App. D.1 identities and two-token example | `tests/test_consensus.py` |
| Proposition 2 (Eq. 23) and Eq. 24 | `tests/test_consensus.py` |
| Eq. 25, Hessian `Z[diag(q) - qq^T]` and `||grad||_1 <= 2Z` | `tests/test_objectives.py` |
| Theorem 1 (`E E(r) - E(p0) = C - G`), Eq. 53 | `test_theorem1_exact_improvement_identity`, `test_normalised_geometric_pool_is_softmax_of_average_logits` |
| Lemma 3 (tilted-variance integral, bounds) | `test_lemma3_tilted_variance_representation_and_bounds` |
| Lemma 4 (range-variance envelopes, sharpness, `Psi`) | `test_lemma4_range_variance_envelopes` |
| Corollary 5 (Eq. 57, centred noise) | `test_corollary5_centered_noise_cannot_improve_the_teacher` |
| Corollary 6 (`phi_-`, `phi_+`, Eq. 65-66) | `test_corollary6_distribution_sensitive_bounds` |
| Proposition 7 (local Fisher expansion) | `test_proposition7_local_fisher_expansion` |
| Theorem 8 (binary minimax risk) | `test_theorem8_binary_minimax_risk` (grid search over estimates) |
| Proposition 9 (Eq. 95, 98, 99) | `test_proposition9_agreement_is_not_a_correctness_certificate` |

The unrestricted minimax statement of Proposition 9 (Eq. 96) is not checked numerically.

## 2. How the core pieces work

**Alignment.** For a prompt of `P` tokens and a response `y_0 .. y_{L-1}`, the model input is
`prompt + response` and the distribution of `y_t` is the output at input position
`P - 1 + t`. `pack_prompt_and_response` right-pads the batch, so causal masking makes every
real position independent of padding and of the other sequences. The two teacher views get
the same response ids behind prompts of different length; their `gather_index` differs,
their per-position order does not. `tests/test_scoring.py` verifies this with a stub model
whose output is a known function of the prefix, against incremental decoding of a real
(tiny) transformer, and under different batch compositions.

**Blockwise loss.** Per microbatch the student backbone is run once with gradients and the
teacher once per view without. For each block of `position_block_size` (32) response
positions the output heads produce full-vocabulary logits; the block's hidden states are
detached, the block loss (already divided by `N_k`) is back-propagated to the head
parameters and to the detached hidden states, and the logits are freed. After the last block
the collected hidden-state gradient is back-propagated through the student backbone once.
The result equals ordinary autograd on the whole batch
(`tests/test_trainer.py::test_blockwise_gradient_equals_direct_autograd`) and does not depend
on block or microbatch size.

**Detachment.** `geometric_consensus` detaches both teacher views; every objective treats
teacher log-probabilities and the weights as constants (checked for all objectives in
`test_teacher_receives_no_gradient_and_weight_is_not_differentiated`). The trainer also sets
`requires_grad=False` on the teacher.

## 3. Unspecified details and choices

Items are ordered roughly by how much they could affect results.

1. **Baseline objectives are cited, not stated.** The paper gives no formula for
   sampled-token OPD, EOPD, GKD-JSD, ToDi, TSD-KD, Veto or CROP.
   - *Sampled-token OPD*: implemented as the score-function surrogate
     `sg[log q(y) - log p(y)] * log q(y)` on the sampled token; its expected gradient over
     `y ~ q` is the gradient of the per-token reverse KL (tested exactly). This matches the
     paper's statement that it differs from full-vocabulary RKL.
   - *EOPD*: `l = l_OPD + lambda * 1[H(p) > tau] * FKL(p, q)` with `tau = 0.8` nats and
     `lambda = 1.0`, i.e. the forward KL is *added* at high-entropy positions ("supplements
     reverse KL", "each term preserves the original sampled-token OPD component and
     teacher-entropy-gated FKL component"). The teacher entropy is computed on the full
     vocabulary. For the top-16 variant the forward KL is the truncated sum over the
     teacher's 16 most likely tokens without renormalisation;
     `eopd_renormalize_top_k: true` renormalises both distributions on that support instead.
   - *GKD-JSD*: the generalised JSD of GKD,
     `beta KL(p || m) + (1 - beta) KL(q || m)`, `m = beta p + (1 - beta) q`, `beta = 0.9`.
   - *ToDi-OP, TSD-KD, Veto, CROP*: **not implemented.** The paper names their ingredients
     but not their objectives, and CROP additionally needs externally validated
     paraphrase/counterfactual triplets. `build_objective` raises `NotImplementedError` for
     `todi`, `tsd_kd`, `veto`, `crop`; implementations written from the original
     publications can be plugged in with `vdopd.objectives.register_objective`.
   - The separate trajectory protocols (SeqKD, off-policy soft KD, SKD, DistiLLM-2) and the
     GRPO reference are outside the shared on-policy loop and are not implemented.
2. **Prompt wording.** The task instruction is not printed. `DEFAULT_PROMPT_TEMPLATE`
   appends "Please reason step by step, and put your final answer within \boxed{}." to the
   question, for training and evaluation alike (`data.prompt_template`).
3. **Answer extraction and equivalence.** The frozen symbolic-equivalence rules are not
   published. `vdopd/eval/answers.py` takes the last `\boxed{...}`, applies a MATH-style
   normalisation, then exact rational, elementwise tuple/interval, numeric and SymPy
   comparison. Scores can differ slightly from the paper's evaluator. Numeric tolerance is
   `rel_tol = 1e-9` for every benchmark (Minerva Math is sometimes scored with a looser
   tolerance; change `evaluation.decoding.rel_tol` if desired).
4. **Hub identifiers and dataset files.** The paper names models and benchmarks but not
   where they are hosted. The configs use `Qwen/Qwen3-8B`, `Qwen/Qwen3-1.7B-Base`,
   `Qwen/Qwen3-4B-Base`, `meta-llama/Llama-3.1-8B-Instruct`,
   `meta-llama/Llama-3.2-3B-Instruct`; model revisions are not pinned (set `*_revision`).
   The datasets are public Hub mirrors whose sizes match the paper (500, 40, 30, 30, 272,
   674 problems; GSM8K test 1,319); their revisions are pinned in
   `vdopd/data/benchmarks.py` and `vdopd/data/math_train.py`. Files whose licence permits
   redistribution are stored in `data/` (MATH training split, MATH500, AIME25, the
   OlympiadBench subset, GSM8K test); AMC23, AIME24 and Minerva Math carry no licence on
   their dataset cards and are fetched by `scripts/download_data.py`. Loaders use the local
   file when it exists and the Hub otherwise. All 500 MATH500 problems and solutions were
   checked to be verbatim entries of the MIT-licensed MATH test split. `data/README.md`
   lists sources, licences, revisions and SHA-256 digests.
5. **Stop tokens.** `<|im_end|>` (Qwen3) or `<|eot_id|>` (Llama) plus the tokenizer's EOS
   end a response (`model.stop_tokens`).
6. **FLOPs accounting.** The paper reports recorded FLOPs without a counting rule.
   `vdopd/flops.py` uses `2 x parameters x tokens` for a forward pass, `4 x` for the
   backward pass and one extra forward pass under gradient checkpointing; the input
   embedding table is excluded unless tied to the output head. The paper's absolute budgets
   (4.7e18 total, 1.101333e18 teacher-only) are therefore not transferable: to reproduce the
   compute controls, take the budget from the FLOPs recorded by *your* EOPD (full) run
   (`flops_total`, `flops_teacher_scoring` in `metrics.jsonl`). An identity-view fallback is
   still scored (and charged) as two teacher passes.
7. **Learning-rate schedule.** Warmup lasts `ceil(0.03 J)` iterations and is linear from
   `lr / W` to `lr`; cosine decay then goes to `min_lr_ratio * lr = 0`. The schedule is a
   function of the iteration index, so a skipped batch changes neither the optimizer state
   nor the rest of the schedule.
8. **Iteration budget.** `J = ceil(passes * n / batch_size)`; each pass is a fresh seeded
   permutation and batches may straddle pass boundaries, so every batch has exactly 128
   prompts. Runs stopped by a FLOPs budget keep the schedule of the full `J`.
   For Figure 5 budgets above the default, `train.passes` must be raised as well; how the
   paper schedules that additional training is not stated.
9. **Microbatches and caching.** The microbatch size (8) is not given; it does not change
   the objective. Teacher hidden states of the response positions are cached for one
   iteration (`train.teacher_cache_device`); the shuffled and batch-mean controls need all
   masses before the loss and add one extra pass over the teacher head.
10. **Skip rules.** "Non-finite" is detected on the summed loss and the global gradient
    norm. The zero-mass rule applies to objectives that weight by the mass (VD-OPD, weighted
    arithmetic, shuffled, batch-mean). For the unit-weight geometric target a position with
    `Z = 0` (impossible with finite logits) contributes zero, because `r = c / Z` is
    undefined there.
11. **Serialisation check.** "Serialized messages preserve the original content and roles"
    is implemented on both levels: message roles/contents, and the rendered text, which must
    equal the original rendering with only the prefix and two newlines inserted in front of
    the final user content. A wrapper failing the check is treated as invalid.
12. **Base student and chat template.** The teacher's tokenizer and template serialise
    prompts for both models (Qwen3 base checkpoints need not ship a chat template).
    For the Llama pair the template's date line is fixed with `date_string`, an assumption
    made to keep the serialisation constant.
13. **Random streams.** Seeds are derived statelessly from `(run seed, stream, indices)`
    with `numpy.random.SeedSequence`; streams are prompt order, rollouts, view selection,
    weight permutation and evaluation. Sampling uses the framework generator
    (`torch.manual_seed`) immediately before each generation call.
14. **Data preparation.** Duplicates and overlaps are detected by exact match after
    lower-casing and whitespace normalisation; overlaps are checked against the six
    mathematics benchmarks. On the pinned data this removes one duplicate
    (`math_train/algebra/959`) and no overlapping problem, leaving 7,499 problems.
    Stratified allocation uses the largest-remainder method per (subject, level) stratum
    and gives 500 diagnostic, 500 development and 6,499 optimisation problems (153 rollout
    batches for three passes at 128 prompts). The 250/250 diagnostic split is a seeded
    permutation of problem ids. The partition is stored as index files in
    `data/math_train/splits/` (`scripts/make_math_splits.py`) and is used whenever the
    configured split parameters equal the stored ones; otherwise it is recomputed. The
    paper's frozen example ids are not available, so the partition follows the described
    procedure but need not coincide with the paper's.
15. **Evaluation details.** Each view uses its own seeds, shared across checkpoints (paired
    streams). If a prompt plus the response cap exceeds the context limit, the cap is
    reduced for that batch. Cross-domain evaluations (GPQA-Diamond, MMLU-Pro) are not
    implemented; `extract_choice` provides option-letter extraction only.
16. **Preliminary study.** "Two instruction wrappers" is read as Instruction 1 and
    Instruction 2 of Table 5; the 5,000 pairs are a seeded uniform sample of response
    positions.
17. **Local-update diagnostic.** A branch update uses the configured learning rate without
    warmup. The three-bin moving average of Figure 6 averages the non-empty bins inside a
    centred window (truncated at the boundaries); empty bins stay empty.
18. **Target-improvement diagnostic (App. B.8).** The estimators are implemented and tested,
    but there is no end-to-end script: the paper does not say how the three incorrect
    candidates or the canonical answer strings are produced, nor whether EOS is appended.
19. **Decoding defaults of checkpoints.** Pretrained checkpoints carry their own sampling
    defaults (for example a temperature below one or a top-k limit) that
    `transformers.generate` merges into a request. `CausalLM` replaces the wrapped model's
    `generation_config` with a neutral one, so rollouts are drawn from the full vocabulary
    at temperature one and evaluation uses exactly the configured temperature and top-p.
20. **Software versions.** Developed and tested with Python 3.11, PyTorch 2.15 and
    Transformers 5.23 on macOS (CPU and MPS); float64 reductions are done on the CPU because
    MPS has no float64. Other versions are untested.
21. **Hardware.** The paper trains on eight A100 GPUs with sharded optimizer states. This
    code runs on one device (the teacher may be placed on another with
    `model.teacher_device`); distributed training is not implemented.

## 4. Observations about the paper

These are points a reader may want to know; none changes the method.

- **Batch-mean control and Adam.** Batch-mean weighting multiplies the whole batch gradient
  by the scalar `Z_bar_k`. Adam is invariant to a constant gradient scale (up to epsilon),
  and gradient-norm clipping at 1.0 removes scale as well, so this control can differ from
  the unit-weight geometric row only through the variation of `Z_bar_k` across iterations
  inside Adam's moment estimates. In a single branch update from a fresh optimizer state the
  two are identical; `scripts/local_update_diagnostic.py --toy` reproduces this.
- **Preliminary study versus training views.** Figure 2 pairs two wrapped prompts, whereas
  training pairs the original prompt with one wrapper.
- **Number of updates.** With roughly 6,500 optimisation prompts, three passes and 128
  prompts per rollout batch, a default run takes about 150 optimizer steps.
- **Compute figures.** Tables 2 and 11 and Figure 5 cannot be recomputed from the text
  because the FLOPs counting rule is not given (item 6).

## 5. Record of runs

All runs below were made on one Apple M-series laptop with Python 3.11, PyTorch 2.15 and
Transformers 5.23.

**Tests and lint.** `pytest` (offline, CPU, tiny models built in code) and `ruff check .`.
The tests cover hand-computed consensus targets and masses, the limiting cases (identical
views reduce to full-weight single-view distillation; disjoint supports give zero loss and
gradient), gradient checks (Eq. 8 and 25; no gradient reaches the teacher or the weight),
cross-view position alignment, block/microbatch/padding invariance, the skip rules of
Algorithm 1, answer checking, Avg@k and the view metrics, the shipped data and its
partition, and numerical checks of Theorem 1, Lemmas 3-4, Corollaries 5-6, Propositions 7
and 9 and Theorem 8. The suite also passes under Python 3.10.

**Quick start.** `vdopd demo` and `python examples/quickstart.py` give identical,
deterministic output (shown in the README): toy teacher 81.2, initial student 8.1,
distilled student 38.8 Avg@4. Two runs used 133 s and 165 s of CPU time (198 s and 413 s
wall-clock on a machine shared with other jobs).

**Diagnostics on the toy pair.** `python scripts/local_update_diagnostic.py --toy` and
`python scripts/preliminary_study.py --toy --num-pairs 2000` run to completion; pre-update
token means in that run were H(p1) = 0.043, H(p2) = 0.042, H(q) = 0.364, Z = 0.996.

**Data.** `python scripts/download_data.py --update-manifest` fetched all eight dataset
files at the pinned revisions and `python scripts/make_math_splits.py` wrote the partition
(item 14).

**Reduced-scale run with real models.** Teacher `HuggingFaceTB/SmolLM2-360M-Instruct`,
student `HuggingFaceTB/SmolLM2-135M-Instruct`, synthetic integer addition/subtraction with
three fixed exemplar turns, 30 rollout batches of 32 prompts, response cap 16 tokens,
learning rate 1e-5, float32 on Apple MPS, one seed (11); evaluation on 60 held-out problems
with 4 samples each under the canonical prompt and the four held-out wrappers; about 13
minutes in total. This setting differs from the paper's in models, data, scale and number
of seeds. Raw outputs are in `results/reduced_scale/`.

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
python scripts/aggregate_results.py --results-dir results/reduced_scale
```

| Model | Avg@4 | Pass@4 | MeanView | WorstView | WorstTemplate |
|---|---|---|---|---|---|
| Frozen teacher (360M) | 51.2 | 78.3 | 55.5 | 42.9 | 49.6 |
| Initial student (135M) | 3.8 | 10.0 | 4.9 | 0.8 | 3.8 |
| GKD-RKL (full) | 34.6 | 55.0 | 28.1 | 16.7 | 26.2 |
| VD-OPD | 32.1 | 61.7 | 24.9 | 14.2 | 22.1 |

In this run VD-OPD was 2.5 points below full-vocabulary reverse-KL distillation in Avg@4
and 2.5-4.2 points below it on the unseen-wrapper metrics, and 6.7 points above it in
Pass@4. With one seed and 240 completions per cell these differences are within sampling
noise. The two teacher views almost always agreed (mean retained mass Z = 0.992 over the
run), so the consensus weighting had little to act on. Both objectives raised the student
from 3.8 to 32-35 Avg@4. The estimated teacher-scoring cost was 1.84e14 FLOPs for VD-OPD
(two views) and 0.89e14 for GKD-RKL (one view). The run exercises the real-model code path
end to end; it does not test the paper's claims.

**Scope of the runs.** No run used the paper's models (Qwen3-8B, Qwen3-1.7B/4B-Base, the
Llama pair), BF16, gradient checkpointing or CUDA; the corresponding configs have been
parsed and validated only. The six mathematics benchmarks and GSM8K have been loaded but
not evaluated. `scripts/run_experiment.sh` has been syntax-checked;
`scripts/aggregate_results.py` and `scripts/teacher_flops_curve.py` have been run on the
reduced-scale outputs. The numbers reported in the paper (README, "Results") are quoted
from the paper and were not regenerated with this code.

**Components outside the shared loop.** ToDi-OP, TSD-KD, Veto and CROP are registered
hooks without an implementation (item 1); the trajectory protocols of Table 11 (SeqKD,
off-policy KD, SKD, DistiLLM-2, GRPO), the cross-domain evaluations of Table 3
(GPQA-Diamond, MMLU-Pro) and an end-to-end script for Tables 13-15 are not included
(items 1, 15, 18). FLOPs are estimated analytically (item 6).

## 6. Paper figures

`assets/paper_figures/fig01.png` to `fig06.png` are the six numbered figures of the paper,
cropped from the PDF at 200 dpi without captions, page headers or margin line numbers by
`scripts/extract_paper_figures.py` (crop rectangles are stored in the script).

## 7. Supplementary experiments (E01-E18)

The supplementary experiments extend Tables 1, 2, 4, 6-8 and 10 of the paper and add
analyses of the cross-view agreement `Z`. Their reported tables are in
`results/supplementary/` and `docs/supplementary_results.md`. This section maps each
experiment to the code that runs it, lists what the experiment descriptions leave open
together with the choice made here, and states what has been executed.

### 7.1 Experiment-to-code map

`<group>` is `qwen3_1p7b`, `qwen3_4b` or `llama3p2_3b`. `tables` stands for
`python scripts/supplementary_tables.py --results-dir results --runs-dir runs --output-dir <dir>`,
which writes CSV files with the file and column names of the reported tables.

| ID | Experiment | Code | Configs | Command |
|---|---|---|---|---|
| E01 | Three-seed Avg@8 per benchmark, gain with 95% CI | `vdopd/eval/bootstrap.py`: `paired_question_bootstrap`, `seed_summary`; `vdopd/eval/tables.py`: `table_e01` | `configs/<group>/{eopd_full,vdopd}.yaml` | `scripts/run_experiment.sh <config>`; `tables --tables E01` |
| E02-E04 | Complete Avg@8 / Pass@8 tables of the three students | `vdopd/eval/evaluator.py`: `evaluate`; `vdopd/eval/metrics.py`: `avg_at_k`, `pass_at_k`; `tables.py` | all method files in `configs/<group>/` | `scripts/run_experiment.sh <config>`; `scripts/evaluate_reference.sh <config> frozen_teacher <teacher>` and `... initial_student`; `tables --tables E02 E03 E04` |
| E05 | Teacher and students under one protocol | `EvalConfig` defaults; `pipeline.py`: `evaluate_student`, `check_generation_suffix` | `configs/qwen3_4b/` (`model.expected_generation_suffix` in the base file) | `scripts/evaluate_reference.sh configs/qwen3_4b/vdopd.yaml frozen_teacher Qwen/Qwen3-8B`; `tables --tables E05` |
| E06 | Compute- and view-matched controls | `objectives.py`: `TwoViewAverage`; `flops.py`: `budget_reached`; `bootstrap.py`: `stratified_paired_bootstrap` | `eopd_full`, `eopd_2v`, `vdopd`, `vdopd_budget` | `scripts/run_experiment.sh`; `tables --tables E06` |
| E07 | Target x weight factorial | `objectives.py`: `ConsensusObjective`, `assign_weights` | `ablation_arithmetic`, `ablation_weighted_arithmetic`, `ablation_geometric`, `ablation_shuffled_weight`, `ablation_batch_mean_weight`, `vdopd` | `scripts/run_experiment.sh`; `tables --tables E07` |
| E08 | Two versus four teacher views | `consensus.py`: `geometric_consensus(*views)`; `ConsensusObjective(num_views=K)`; `views.py`: `select_views(num_views=K)`; `Trainer._teacher_hidden` | `vdopd`, `vdopd_k4`, `vdopd_k4_budget` | `scripts/run_experiment.sh`; `tables --tables E08` |
| E09 | Gradient agreement of the three view pairs | `diagnostics.py`: `pair_gradient_cosines` | `vdopd` | `python scripts/pair_gradient_cosine.py --config configs/qwen3_1p7b/vdopd.yaml --student <checkpoint>/student --csv <file>` (`--toy` on CPU) |
| E10 | Training view pools | `view_pools.py` (file schema, `load_view_file`, `group_views`, `training_view_pool`); `views.py`: `RewrittenView`; `Trainer(prompt_views=...)`, `Trainer.summary` (mean `Z`) | `eopd_2v_paraphrase`, `vdopd_paraphrase`, `eopd_2v_lowquality`, `vdopd_lowquality` | `VDOPD_OVERRIDES="views.pool_file=<views.jsonl>" scripts/run_experiment.sh <config>`; `tables --tables E10` |
| E11 | Baseline tuning | `tuning.py`: `TuningConfig`, `eopd_grid`, `select_best`; `pipeline.py`: `run_tuning_trial` | `eopd_top16_tuned`, `eopd_full_tuned`, `eopd_2v_tuned`; grid in the `tuning:` section | `python scripts/tune_eopd.py --config configs/qwen3_1p7b/eopd_full.yaml`; `scripts/run_experiment.sh configs/qwen3_1p7b/eopd_full_tuned.yaml`; `python scripts/aggregate_results.py --results-dir results/qwen3_1p7b --sd` |
| E12-E13 | Unseen-wrapper robustness, macro and per benchmark | `metrics.py`: `view_robustness`, `sensitivity_gaps`, `greedy_answer_flip_rate`; `evaluator.py`: `greedy_answers` | `eopd_full`, `eopd_2v`, `eopd_full_studentaug`, `eopd_2v_studentaug`, `vdopd` | `VDOPD_OVERRIDES="evaluation.greedy_answers=true" scripts/run_experiment.sh <config>`; `tables --tables E12 E13` |
| E14 | Student-side augmentation across view families | `trainer.py`: `ViewConfig.student_wrap_probability`; `select_views(student_wrap_probability=...)`; `views.py`: `ViewFamily` | `eopd_full_studentaug`, `eopd_2v_studentaug` | as E15; `tables --tables E14` |
| E15 | Held-out structural changes and paraphrases | `views.py`: `TemplateView`, `STRUCTURAL_VIEWS`; `view_pools.py`: `paraphrase_family`; `metrics.py`: `parse_failure_rate`; `pipeline.py`: `evaluation_families` | any method file | `VDOPD_OVERRIDES="evaluation.view_families=[prefix,structural,paraphrase] evaluation.paraphrase_file=<views.jsonl> evaluation.greedy_answers=true" scripts/run_experiment.sh <config>`; `tables --tables E15` |
| E16 | Teacher diagnostics by `Z` quartile | `diagnostics.py`: `build_analysis_states`, `state_agreement`, `hellinger_tail_split`, `sample_final_answers`, `pair_outcomes`, `agreement_summary` | `vdopd` | `python scripts/agreement_diagnostic.py --config configs/qwen3_1p7b/vdopd.yaml --student <checkpoint>/student --csv <file>` (`--toy` on CPU) |
| E17 | Matched local updates by `Z` stratum | `diagnostics.py`: `matched_branch_updates`, `candidate_distributions`, `oracle_kl_decrease`, `shared_wrong_answers`, `stratum_means`; `bootstrap.py`: `clustered_mean_bootstrap` | `vdopd` | `python scripts/local_update_diagnostic.py --config configs/qwen3_1p7b/vdopd.yaml --supplementary --checkpoint <checkpoint> --candidates <candidates.jsonl> --csv <file>` (`--toy --supplementary` on CPU) |
| E18 | Two-token example | `theory.py`: `two_token_example`, `geometric_pool` | - | `python examples/two_token_example.py` |

Tests: `tests/test_bootstrap.py` (E01, E06-E08, E10, E17), `tests/test_multi_view.py` (E06,
E07, E08, E18), `tests/test_view_pools.py` (E10, E14, structural views),
`tests/test_robustness_eval.py` (E02-E05, E12-E15), `tests/test_agreement_diagnostics.py`
(E09, E16, E17), `tests/test_result_tables.py` (table layout), `tests/test_tuning.py` (E11).

### 7.2 Unspecified details and choices

22. **Bootstrap (E01, E06-E08, E10, E17).** The descriptions fix the resampling unit (the
    question, with its eight samples and paired seeds) but not the rest. Here: 10,000
    resamples, percentile interval, generator seed 0 (`--resamples`, `--bootstrap-seed`).
    A question's score is its mean over samples and seeds; a resample draws questions with
    replacement and uses the paired difference of the two methods. For macro averages
    questions are resampled within each benchmark and the benchmark means are averaged
    without weights. The reported gain is the difference of the seed means; the interval
    uses the seeds both methods have. Evaluation files store, per view, the number of
    correct completions of every question (`questions.correct_counts`); files without them
    give gains without intervals.
23. **Common evaluation protocol (E02-E05).** "Three evaluations" is read as the three
    evaluation seeds 11, 22, 33: a distilled student is evaluated once per training seed
    with the paired evaluation seed, a fixed checkpoint (teacher, initial student) once per
    evaluation seed (`scripts/evaluate_reference.sh`). All models pass through
    `evaluate_student`, i.e. the teacher's chat template, `model.stop_tokens`, the decoding
    section and `vdopd/eval/answers.py`. The Qwen3 base config requires every serialised
    prompt to end with `<think>\n\n</think>\n\n` (`model.expected_generation_suffix`); this
    string was written from the published Qwen3 chat template and has not been compared
    with the tokenizer files, which were not available offline. Set it to `null` if it does
    not match.
24. **Budget-matched runs (E06, E08).** The total-FLOPs budget is checked between
    iterations, so a budgeted run ends at the first iteration boundary at which the budget
    is spent and exceeds it by less than one iteration; its learning-rate schedule is that
    of the full run (item 8). The budgets in `vdopd_budget.yaml` (4.7e18) and
    `vdopd_k4_budget.yaml` (5.8e18) are the reported totals in the paper's FLOPs
    accounting; with this repository's analytic estimate (item 6) set them to the
    `flops_total` of your own EOPD (full) and default VD-OPD runs.
25. **K teacher views (E08).** `K = 4` is the original prompt plus all three wrappers of
    Table 5. For a general `K`, all valid alternatives are used if there are at most
    `K - 1`; otherwise `K - 1` are drawn uniformly without replacement and kept in pool
    order. Missing views fall back to the identity transformation (an alternative that does
    not fit the teacher context is replaced by the original prompt). The zero-mass skip rule
    and the logged mass use `Z_K`. For `K = 2` the selection draws exactly as before.
26. **Pair gradients (E09).** "The same fixed states" are the response positions of one
    rollout batch of the diagnostic update problems (`--batch 0`) at the given student
    checkpoint. For each of the three wrappers the gradient of the VD-OPD objective
    (Equation 7, before clipping, all trainable parameters) is accumulated with every
    response paired with that wrapper; the mean gradient is the arithmetic mean of the three.
    The number of states and the checkpoint used for the reported values are not stated.
27. **View-pool files (E10, E15).** The delivery format of externally produced views is not
    specified; `vdopd/view_pools.py` defines one (JSON Lines with `problem_id`, `view`,
    `question`, optional `pool`, `validated`). A record holds a rewritten *question*, which
    is placed in the same prompt template as the original, so the instruction and answer
    format stay verbatim. Validation is limited to what can be checked mechanically: the
    schema, unknown problems, duplicates, a view equal to the original question, the
    `validated` flag (`views.require_validated`), and at use time identical roles, earlier
    turns and assistant prefix of the serialised prompt (`preserves_content_and_roles`).
    Whether a paraphrase preserves conditions, variables and answer is the external
    validator's statement. The "lower-quality, task-preserving" pool is a second `pool`
    label of the same format. The second teacher view of a response is sampled uniformly
    from its question's views; a question without views uses the identity fallback (logged
    as `identity_view_fraction`). No view file is distributed; the configs point to
    `data/view_pools/math_train.jsonl` as a placeholder.
28. **Mean Z of a view pool (E10).** Not defined in the description. `Trainer.summary()`
    reports the token mean of `Z` over all response positions scored during the run
    (`summary.json`, also logged); the table takes it from the VD-OPD run of each pool.
29. **Baseline tuning (E11).** The grid values are not stated. The default grid is
    `tau in {0.6, 0.8, 1.0}` nats and `alpha in {0.8, 1.0, 1.2}` (`tuning.*`), centred on
    the defaults and containing the three selected pairs. Each trial is one training run
    with the seed of the config, scored by Avg@8 on the development split under the common
    decoding protocol with the canonical prompt; ties go to the lower total training FLOPs
    (Appendix B.4). One of the 500 development problems has an empty boxed answer in its
    reference solution and is skipped. "Further evaluation" is read as three-seed runs of
    the `*_tuned.yaml` configs. ToDi-OP, TSD-KD and Veto remain registration hooks (item 1);
    the rerun settings named for them are: ToDi-OP with its published vocabulary-wise
    mixing, TSD-KD with JSD parameter 0.9 and top-10 candidates, Veto with initial
    `beta = 0.8` and linear decay.
30. **Structural views (E15).** The description names the kinds of change ("instruction
    relocation and layout/delimiter changes") but not the four views. The defaults are two
    views that place a solving instruction after the problem
    (`{content}\n\nPlease solve this problem.`,
    `{content}\n\nDetermine the answer to the problem above.`) and two that change the
    delimiter of a layout heading (`Question: {content}`, `---\n{content}\n---`). They are
    this repository's choice; replace them with `evaluation.structural_views`.
31. **Paraphrase family (E14, E15).** Read from `evaluation.paraphrase_file` in the format
    of item 27; every evaluated question needs four views with the same four names, and
    view `m` of every question forms "template" `m` for WorstTemplate.
32. **Greedy answer flips and parse failures (E12, E15).** Neither is defined. A flip is a
    (question, view) pair whose greedy final answer is not equivalent to the greedy final
    answer under the canonical prompt, as a fraction of all pairs of the family
    (`evaluation.flip_mode: question` counts questions with at least one flip instead); two
    missing answers count as equal. Parse failures are the fraction of *sampled*
    completions under the family's views without an extractable final answer. Canonical -
    Mean and Mean - Worst are differences of the macro averages.
33. **StudentAug (E14).** The wrapped prompt of a response is its sampled alternative
    teacher view (one of the three training wrappers, uniform), and an independent draw with
    probability 0.5 (`views.student_wrap_probability`, random stream `student_views`)
    decides whether the student is rolled out and scored on it or on the original prompt.
    The teacher views are selected as without augmentation. In the two-view variant the
    teacher scores the original and the wrapped prompt. Which prompt the one-view variant's
    teacher scores is not stated; by default it is the prompt the student was given
    (`views.student_aug_teacher_view: student`; `original` is the alternative). A wrapped
    prompt that does not fit the student context is not used.
34. **Analysis states (E16, E17).** For each measurement question the student generates
    four independent responses with the rollout settings and each is cut at one uniformly
    sampled position (`--prefixes-per-question`); each question gets one alternative teacher
    view, sampled uniformly from the valid training wrappers. Quartiles are four equal-sized
    rank groups of `Z` over the states. The reported strata hold 243 and 251 states, which
    an equal split of 1,000 states does not produce; the rule behind those sizes is not
    stated.
35. **Teacher diagnostics (E16).** The tail statistic of a stratum is its summed squared
    Hellinger distance outside the union of both views' top-10 tokens divided by its summed
    squared Hellinger distance (a share of the stratum's total, not a mean of per-state
    shares). A continuation pair is one teacher continuation of the prefix under each view,
    decoded with the evaluation protocol and the same random seeds; four pairs per state
    (`--continuations`). Two incorrect continuations are "same wrong" only if both have a
    final answer and the answers are equivalent; a pair with a missing answer is "different
    wrong".
36. **Matched local updates (E17).** The candidate answers of the measurement problems are
    not published (item 18); they are read from `--candidates` (final-answer strings, the
    correct one first) and scored as given, without EOS. A state is eligible for the
    shared-wrong statistic if at least one of its teacher continuation pairs is "same
    wrong"; its shared wrong answer is the one occurring in most such pairs. The student's
    probability of that answer is its candidate probability (Equation 16) over the
    problem's candidates, to which the shared wrong answer is added if it is not equivalent
    to one of them. The correctness change is the change in accuracy of
    `evaluation.decoding.num_samples` student continuations per state with paired seeds.
    Intervals resample questions with all their states of the stratum. The "number of
    effective update positions" is read as the `N_k` response positions of the shared
    rollout batch. With `--checkpoint` a branch starts from a saved training checkpoint
    (student weights and optimizer state) and uses the scheduled learning rate of its
    iteration; with `--student` it uses the configured rate without warmup (item 17). Which
    checkpoint is "middle" is not stated; `train.checkpoint_every` controls what is saved.
37. **Tables.** GPU-hours are not recorded by this code and stay empty; FLOPs columns use
    the analytic estimate of item 6. E11 is not produced by `supplementary_tables.py`
    (`tune_eopd.py` writes `grid.json`; seed statistics come from `aggregate_results.py`).
    E09, E16, E17 and E18 are written by their own scripts (`--csv`).

### 7.3 Changes to existing code paths

- `geometric_consensus`, `normalized_consensus` and `arithmetic_mean` accept more than two
  views; the two-view case uses the same operations as before.
- `PositionBlock.student_log_probs` caches the student log-softmax of a block. EOPD-2V
  evaluated it once per term and the trainer once more for the logged entropy; all three
  now share one evaluation. For EOPD-2V this changes floating-point rounding (the two terms
  back-propagate through one node), not the objective. A five-iteration run on the tiny
  test models gave identical losses, gradient norms and final weights before and after
  these changes for the other twelve objectives; for EOPD-2V the first loss is identical
  and later losses agree to five to six significant digits.
- The consensus objectives return the mass they compute (`BlockLoss.mass`), so the trainer
  no longer forms the target a second time for logging.
- `matched_branch_updates` runs the frozen teacher once for all branches and statistics
  passes (`Trainer.teacher_hidden_states`) instead of once per branch and pass.
- `pack_prompt_and_response` builds the alignment index per response with tensor
  operations instead of a Python loop over tokens; `candidate_logits` averages token
  log-probabilities without a Python loop.
- `theory.consensus_mass` forms the product of probabilities as a mean of logarithms; the
  previous `sqrt(p0 * pT)` underflowed for products below the float64 range.
- `position_statistics` scores the student on the prompt it was given
  (`ViewPair.student_prompt_ids`), which equals the original prompt unless StudentAug is on.

### 7.4 Record of runs

The code of this section was exercised as follows; no supplementary experiment was run with
the paper's models, and none of the reported numbers was regenerated.

- **Unit tests** (offline, CPU, tiny random models and stub models; Python 3.11 only): all
  functions listed in 7.1, including the equality of blockwise and direct gradients for
  four views, the FLOPs of four views, the budget stop, view selection from a file pool,
  StudentAug, evaluation over three view families with greedy flips and parse failures,
  the cached teacher states, and the table layout on synthetic evaluation files.
- **Toy pair** (`--toy`, the synthetic arithmetic task of the quick start):
  `scripts/pair_gradient_cosine.py`, `scripts/agreement_diagnostic.py` (160 states from 40
  questions) and `scripts/local_update_diagnostic.py --supplementary` (also from a saved toy
  checkpoint with `--checkpoint`) run to completion and write CSV files with the columns of
  the reported tables. The toy candidates of E17 are the integer answer and three near
  misses, a scaffold of the toy task only.
- **Closed form**: `examples/two_token_example.py` reproduces the E18 table exactly.
- **Not executed**: training or evaluation with Qwen3 or Llama models for any of E01-E17;
  `scripts/tune_eopd.py` beyond `--dry-run` (its trial function was run with a tiny model
  and the synthetic test split); `scripts/evaluate_reference.sh` (its command lines were
  checked with a stub in place of `vdopd`);
  training with a file view pool or evaluation on a paraphrase family beyond the
  hand-written three-question fixture `tests/fixtures/view_pool.jsonl`;
  `scripts/supplementary_tables.py` on real evaluation files (it was run on synthetic files
  and on the reduced-scale results, which lack per-question records). The view checks were
  also run against one real ChatML chat template available offline (SmolLM2 tokenizer).

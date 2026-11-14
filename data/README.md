# Data

Datasets used by the experiments of "View-Decomposed On-Policy Distillation for Robust
Reasoning in Large Language Models" (under review). The loaders in `vdopd/data/` read the
files in this directory and fall back to the Hugging Face Hub only when a file is absent.
The directory can be changed with `--data-dir`, `data.data_dir` in a config, or the
`VDOPD_DATA_DIR` environment variable.

All files are JSON lines with the original fields of their source; every Hub revision is
pinned in the code and recorded in `manifest.json` together with the SHA-256 of each file.

## Layout

```text
data/
├── manifest.json                    source, revision, licence, size and SHA-256 of every file
├── licenses/                        upstream licence texts
├── math_train/
│   ├── train.jsonl                  MATH training split, 7,500 problems
│   └── splits/                      Appendix B.1 partition as index files (one id per line)
│       ├── deduplicated.txt         7,499 problems kept after cleaning
│       ├── optimization.txt         6,499 problems used for distillation
│       ├── development.txt          500 problems for configuration selection
│       ├── diagnostic.txt           500 problems for mechanism diagnostics
│       ├── diagnostic_update.txt    250 of the diagnostic problems (branch updates, App. B.7)
│       ├── diagnostic_measurement.txt  250 of the diagnostic problems (measurement, App. B.7)
│       └── split_info.json          parameters and sizes
└── benchmarks/
    ├── math500.jsonl                distributed
    ├── aime25.jsonl                 distributed
    ├── olympiadbench.jsonl          distributed
    ├── gsm8k.jsonl                  distributed
    ├── amc23.jsonl                  fetched by scripts/download_data.py
    ├── aime24.jsonl                 fetched by scripts/download_data.py
    └── minerva.jsonl                fetched by scripts/download_data.py
```

## Datasets

| Dataset | Used for | Source (Hugging Face Hub) | Revision | Licence | Examples | File |
|---|---|---|---|---|---|---|
| MATH, training split | training prompts of every distillation run (Tables 1-4, 6-8), development and diagnostic sets (Figures 2 and 6, Table 12) | [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math), seven subject configs, split `train`; upstream <https://github.com/hendrycks/math> | `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT (`licenses/MATH-LICENSE.txt`) | 7,500 | `math_train/train.jsonl` |
| MATH500 | primary benchmark (Tables 1-3, 6-8, 10, 11; Figures 4, 5) | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), split `test` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | MIT: all 500 problems and solutions are verbatim entries of the MATH test split (`licenses/MATH-LICENSE.txt`); the subset was selected in PRM800K (MIT, `licenses/PRM800K-LICENSE.txt`) | 500 | `benchmarks/math500.jsonl` |
| AIME25 | primary benchmark | [`math-ai/aime25`](https://huggingface.co/datasets/math-ai/aime25), split `test` | `563bb8404243c5f09de6ec262f2db674fe5bce9b` | Apache-2.0 as stated on the dataset card (`licenses/Apache-2.0.txt`) | 30 | `benchmarks/aime25.jsonl` |
| OlympiadBench, English text-only open-ended mathematics subset | primary benchmark | [`Hothan/OlympiadBench`](https://huggingface.co/datasets/Hothan/OlympiadBench), config `OE_TO_maths_en_COMP`, split `train`; upstream <https://github.com/OpenBMB/OlympiadBench> | `91184b52131e7fc9455fef848035173aea8cc01a` | Apache-2.0 on the dataset card (`licenses/Apache-2.0.txt`), MIT in the upstream repository (`licenses/OlympiadBench-LICENSE.txt`) | 674 | `benchmarks/olympiadbench.jsonl` |
| GSM8K, official test split | supplementary evaluation (Table 9) | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), config `main`, split `test`; upstream <https://github.com/openai/grade-school-math> | `740312add88f781978c0658806c59bc2815b9866` | MIT (`licenses/GSM8K-LICENSE.txt`) | 1,319 | `benchmarks/gsm8k.jsonl` |
| AMC23 | primary benchmark | [`math-ai/amc23`](https://huggingface.co/datasets/math-ai/amc23), split `test` | `80815d37005feb82cd7f8fbc6901d5d3eff43057` | no licence stated on the dataset card | 40 | `benchmarks/amc23.jsonl` (download) |
| AIME24 | primary benchmark | [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), split `train` | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` | no licence stated on the dataset card | 30 | `benchmarks/aime24.jsonl` (download) |
| Minerva Math | primary benchmark | [`math-ai/minervamath`](https://huggingface.co/datasets/math-ai/minervamath), split `test` | `ee46ddc498933b1977577953250ca5c66be64f96` | no licence stated on the dataset card | 272 | `benchmarks/minerva.jsonl` (download) |

The three datasets without a stated licence are not distributed here. Fetch them into the
same layout with

```bash
pip install -e ".[data]"
python scripts/download_data.py            # fetches every file that is missing
```

The script downloads the pinned revisions, writes the JSON-lines files and compares their
SHA-256 with `manifest.json`. `python scripts/download_data.py --force` fetches all eight
files again.

The synthetic integer-arithmetic task of the quick start and of `configs/reduced_scale/`
needs no file: `vdopd/data/synthetic.py` enumerates the problems (`a+b` and `a-b` for all
operands in a range) and splits them with a seeded permutation.

## MATH training partition (Appendix B.1)

`math_train/splits/` is produced by `python scripts/make_math_splits.py`:

1. duplicates are removed by exact match after lower-casing and whitespace normalisation
   (one problem: `math_train/algebra/959` repeats `math_train/algebra/925`);
2. problems that also occur in one of the six mathematics benchmarks are removed (none);
3. 500 diagnostic and 500 development problems are reserved, stratified by subject and
   difficulty level with split seed 42 (largest-remainder allocation per stratum); the
   remaining 6,499 problems form the optimisation set;
4. the diagnostic problems are divided into 250 update and 250 measurement problems with
   the same seed (Appendix B.7).

Problem ids have the form `math_train/<subject config>/<row index in the Hub split>`. The
procedure is deterministic; `tests/test_shipped_data.py` checks the index files against a
recomputation. The paper does not list its example ids, so the index files are the result
of applying the described procedure in this repository.

## Record formats

| File | Fields |
|---|---|
| `math_train/train.jsonl` | `id`, `problem`, `level`, `type`, `solution` |
| `benchmarks/math500.jsonl` | `problem`, `solution`, `answer`, `subject`, `level`, `unique_id` |
| `benchmarks/aime25.jsonl` | `problem`, `answer`, `id` |
| `benchmarks/olympiadbench.jsonl` | `id`, `question`, `solution`, `final_answer`, `context`, `image_1` ... `image_9`, `modality`, `difficulty`, `is_multiple_answer`, `unit`, `answer_type`, `error`, `question_type`, `subfield`, `subject`, `language` |
| `benchmarks/gsm8k.jsonl` | `question`, `answer` (reference answer after `####`) |
| `benchmarks/amc23.jsonl` | `id`, `question`, `answer`, `url` |
| `benchmarks/aime24.jsonl` | `id`, `problem`, `solution`, `answer`, `url`, `year` |
| `benchmarks/minerva.jsonl` | `question`, `answer` |

Reference answers of the MATH training problems (the last boxed expression of each
solution) are used for development scoring only; distillation reads the question text.

## SHA-256

Distributed files:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `math_train/train.jsonl` | 6,794,360 | `7a08cf61a5843ccbb9c1be1bc43642b3dc38242e01f10a513a0bff9b746f1f61` |
| `math_train/splits/deduplicated.txt` | 215,576 | `70463a98357698f5c32f373bf0c7b14adb506fc13aa2c0d4d9c8798a705ad5bb` |
| `math_train/splits/optimization.txt` | 186,846 | `81b2be4e146237e4960cdf34748d7cf4165909aec8e4548d502cfd05c2ac69e8` |
| `math_train/splits/development.txt` | 14,355 | `8a273030de4194e749d6dd6254837e8083e70c600f7a92ecce237f81c4c312ab` |
| `math_train/splits/diagnostic.txt` | 14,375 | `8cc7eda9aaa0f26b4e88d2807b4e70c70950f280b77f1f3a1de3463474bbc13a` |
| `math_train/splits/diagnostic_update.txt` | 7,322 | `5980f66825b4cd8f33430a19034a8773f87072f47ae1ce180edf8f282ae5ae38` |
| `math_train/splits/diagnostic_measurement.txt` | 7,053 | `b6a5abc659c2f8abf9ec2fb95ebf46400efbacc8a01130ab250b44309a2ebda1` |
| `math_train/splits/split_info.json` | 371 | `2ab54b094ba70c7631f9409697789bfa97e3464894d9cf467963a4aeeca524dc` |
| `benchmarks/math500.jsonl` | 446,564 | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| `benchmarks/aime25.jsonl` | 15,873 | `c19e99ec39a417daed2ffbd5710819bc6d4cca775929268e47a96cdcff5c68ea` |
| `benchmarks/olympiadbench.jsonl` | 1,676,052 | `d5f737f0ed8dd9d1974002d9f1ab9fce7cfa6eb254e36abb7afc9cef14a67bcd` |
| `benchmarks/gsm8k.jsonl` | 748,933 | `752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac` |

Files written by `scripts/download_data.py` (for verification after download):

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `benchmarks/amc23.jsonl` | 16,218 | `bc4407852feca67f6e76175d9acb487caba92aeb491689fb0475e0cc1570eac9` |
| `benchmarks/aime24.jsonl` | 146,688 | `3b791bec81242339a0b5b2d881805e6a1c75df58d5f84278127607285cc19dae` |
| `benchmarks/minerva.jsonl` | 130,303 | `e04ec2bc1d8f80a0091b4b1989f1ffb366f3583f5bae381546a066d4ccc26e3d` |

The same digests are stored in `manifest.json`; the test-suite compares every file that is
present against it.

## Licences and attribution

- MATH (training split and the MATH500 subset): MIT License, Copyright (c) 2021 Dan
  Hendrycks; see `licenses/MATH-LICENSE.txt`. Reference: Hendrycks et al., "Measuring
  Mathematical Problem Solving With the MATH Dataset", 2021.
- PRM800K (selection of the 500 MATH500 problems): MIT License, Copyright (c) 2023 OpenAI;
  see `licenses/PRM800K-LICENSE.txt`.
- GSM8K: MIT License, Copyright (c) 2021 OpenAI; see `licenses/GSM8K-LICENSE.txt`.
  Reference: Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021.
- OlympiadBench: Apache-2.0 on the dataset card, MIT License, Copyright (c) 2024 OpenBMB in
  the upstream repository; see `licenses/Apache-2.0.txt` and
  `licenses/OlympiadBench-LICENSE.txt`. Reference: He et al., "OlympiadBench: A Challenging
  Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems",
  2024.
- AIME25: Apache-2.0 as stated on the `math-ai/aime25` dataset card; see
  `licenses/Apache-2.0.txt`.
- AMC23, AIME24 and Minerva Math are obtained from their Hub mirrors by the download
  script and are subject to the terms of their sources (Minerva Math: Lewkowycz et al.,
  "Solving Quantitative Reasoning Problems with Language Models", 2022).

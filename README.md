# TagCover: predicting AI benchmark outcomes from few labels

Code of **TagCover**, our submission to the Predictive AI Evaluation Competition (PAIEC) at NeurIPS 2026,
with the scripts that reproduce its local results in the technical report

> Toan Nguyen-Mau and Van-Nam Huynh. *TagCover: Hierarchical Bayesian Shrinkage with Tag Offsets and LLM
> Difficulty Ratings for Predicting AI Benchmark Outcomes.* Technical report, PAIEC 2026 (NeurIPS 2026 workshop).

Codabench submission selected for the final evaluation: **1017632**.

## The method in short

`predict([subject, item], labeled)` returns the probability that the subject solves the item, from the labels
acquired so far in the draw.

1. **Pair success rate.** A three-level logistic-normal model (draw, benchmark, subject-benchmark pair), solved
   exactly by message passing on an 81-point grid of logits.
2. **Item corrections**, fitted on the residuals of the labelled items of the same benchmark:
   offsets of the metadata tags (`item_features`), slopes on two difficulty ratings that
   Qwen3-4B-Instruct-2507 generates from the item text (minutes and difficulty), and an offset per item from
   the labels of other subjects on it. The corrected probability is mixed with the pair rate (weight 0.45).
3. **Label acquisition.** A pair's labels are spread evenly over its tag groups.

The LLM stage is optional at scoring time: without a GPU, or when the model cannot load, the predictor answers
as the same model without the ratings. All settings are in [submission/config.json](submission/config.json).

## Layout

```
submission/                the archive: model.py, labeling.py, paiec_core.py, config.json, models.txt, prediction_config.json
reference/BetaShrinkage/   the Beta shrinkage reference of the report
paiec/                     local benchmarks (data.py); metric and local streaming evaluator (evaluation.py)
scripts/                   download_data, build_benchmark, evaluate, compute_ratings, make_submission, smoke_test
features/                  Qwen3-4B ratings of the 11,532 public item texts (SHA-1 keys and five numbers, no text)
```

## Setup

Python 3.10 or newer:

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # synthetic data, a few seconds, no download and no GPU
```

The LLM rater itself (only for `scripts/compute_ratings.py`, or to rate new items) also needs
`pip install torch "transformers>=4.51" accelerate`.

## Reproduce the local results

The training data is the public [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
(gated: request access once, then `huggingface-cli login`).

```bash
python scripts/download_data.py       # revision bc8204d8, about 1.1 GB
python scripts/build_benchmark.py     # paiec_sim, paiec_local, paiec_simfresh, paiec_localfresh
python scripts/evaluate.py --dataset paiec_sim                                  # TagCover
python scripts/evaluate.py --dataset paiec_sim --model reference/BetaShrinkage  # reference
python scripts/evaluate.py --dataset paiec_sim --set "features=[]"              # without the LLM stage
```

`paiec_sim` holds 300 platform-like draws (sub-benchmarks, 7 per draw, pairs cut to 115 items) and
`paiec_local` 100 draws over the public benchmarks as they are; the two `fresh` benchmarks are more draws of
the same two kinds from another seed. The ratings of the public items are read from `features/`. A run of
TagCover takes a few minutes on one CPU core. Each run prints the Brier ALC (lower is better) and writes one
row per evaluated pair to `output/`; `--compare <file>` adds the paired difference to another run.

Brier ALC of the runs in Table 2 of the report (blend = 0.529 sim + 0.471 local):

| Row of Table 2 | `evaluate.py` options | sim | local | blend | fresh sim | fresh local |
|---|---|---|---|---|---|---|
| BetaShrinkage reference | `--model reference/BetaShrinkage` | 0.1636 | 0.2164 | 0.1885 | 0.1662 | 0.2172 |
| Pair rate, m0 = 0, draw_sd = 0 | `--set tag_sd=0 --set item_sd=0 --set shrink=0 --set "features=[]" --set feat_sd=0 --set acquire=random --set prior_mean=0 --set draw_sd=0` | 0.1595 | 0.2178 | 0.1870 | 0.1622 | 0.2186 |
| Pair rate, m0 = -0.5, draw_sd = 0.5 | `--set tag_sd=0 --set item_sd=0 --set shrink=0 --set "features=[]" --set feat_sd=0 --set acquire=random` | 0.1585 | 0.2160 | 0.1856 | 0.1610 | 0.2167 |
| + tags, items, shrink | `--set "features=[]" --set feat_sd=0 --set acquire=random` | 0.1572 | 0.2149 | 0.1844 | 0.1596 | 0.2157 |
| + LLM ratings | `--set acquire=random` | 0.1567 | 0.2108 | 0.1822 | 0.1591 | 0.2120 |
| **+ tag-balanced acquisition = TagCover** | (none) | **0.1573** | **0.2094** | **0.1818** | **0.1594** | **0.2127** |
| without the LLM stage | `--set "features=[]"` | 0.1577 | 0.2133 | 0.1839 | 0.1600 | 0.2163 |
| without tags | `--set tag_sd=0` | 0.1583 | 0.2094 | 0.1823 | 0.1603 | 0.2123 |
| without item offsets | `--set item_sd=0` | 0.1574 | 0.2100 | 0.1822 | 0.1596 | 0.2131 |
| without shrink | `--set shrink=0` | 0.1584 | 0.2091 | 0.1823 | 0.1605 | 0.2131 |
| without draw adaptation | `--set draw_sd=0` | 0.1572 | 0.2090 | 0.1816 | 0.1593 | 0.2123 |
| without prior slope | `--set feat_prior=0` | 0.1574 | 0.2113 | 0.1828 | 0.1597 | 0.2145 |
| prior level m0 = 0 | `--set prior_mean=0` | 0.1580 | 0.2115 | 0.1832 | 0.1604 | 0.2149 |

The datasets are `paiec_sim`, `paiec_local`, `paiec_simfresh` and `paiec_localfresh` (`--dataset`). A constant
0.5 scores 0.25 on every benchmark.

## Build the submission archive

```bash
python scripts/make_submission.py     # -> output/TagCover.zip, model.py at the root
```

The archive we uploaded (`TagCover_c6acf076__proxy0.1812.zip`) carried a larger shared core with code paths
of other configurations that TagCover never runs. `submission/paiec_core.py` keeps only the paths TagCover
runs; on the four local benchmarks it gives the same per-pair scores as the uploaded archive, digit for digit.

## LLM ratings

At scoring time the archive rates each item live with Qwen/Qwen3-4B-Instruct-2507 (greedy decoding, 24 new
tokens, the first 2,000 characters of the item text, one third from the head and two thirds from the tail).
The prompt is `RATER_PROMPT` in `submission/paiec_core.py`. For local scoring, `features/` holds the ratings of
all public items, made with the same code; `python scripts/compute_ratings.py` rebuilds them (a GPU is
recommended for 11,532 texts).

## License

The code is released under the MIT license. The data is not included; its own terms apply to it.

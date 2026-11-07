# LLMDifficulty: predicting AI benchmark outcomes from few labels

Code of **LLMDifficulty**, our submission to the Predictive AI Evaluation Competition (PAIEC) at NeurIPS 2026,
with the scripts that reproduce its local results in the technical report

> Toan Nguyen-Mau and Van-Nam Huynh. *LLMDifficulty: Hierarchical Bayesian Shrinkage with LLM Difficulty Ratings
> and Tag Offsets for Predicting AI Benchmark Outcomes.* Technical report, PAIEC 2026 (NeurIPS 2026 workshop).

Codabench submission selected for the final evaluation: **1020707** (archive `LLMDifficulty_56538240__proxy0.1800.zip`).

## The method in short

`predict([subject, item], labeled)` returns the probability that the subject solves the item, from the labels
acquired so far in the draw.

1. **Pair success rate.** A three-level logistic-normal model (draw, benchmark, subject-benchmark pair), solved
   exactly by message passing on an 81-point grid of logits.
2. **Item corrections**, fitted on the residuals of the labelled items of the same benchmark. The main one is a
   slope on two difficulty ratings that Qwen3-4B-Instruct-2507 generates from the item text (minutes and
   difficulty), around a fixed prior slope ("rated harder = solved less") that the labels can move only a little.
   Offsets of the metadata tags (`item_features`) and an offset per item from the labels of other subjects on it
   add smaller corrections. The corrected probability is mixed with the pair rate (weight 0.15).
3. **Labels.** There is no `labeling.py`: each pair receives the labels of the first candidates of its random
   stream, as the platform does by default.

The LLM stage is optional at scoring time: without a GPU, or when the model cannot load, the predictor answers
as the same model without the ratings. All settings are in [submission/config.json](submission/config.json).

## Layout

```
submission/                the archive: model.py, paiec_core.py, config.json, models.txt, prediction_config.json
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
python scripts/evaluate.py --dataset paiec_sim                                  # LLMDifficulty
python scripts/evaluate.py --dataset paiec_sim --model reference/BetaShrinkage  # reference
python scripts/evaluate.py --dataset paiec_sim --set "features=[]"              # without the LLM stage
python scripts/evaluate.py --dataset paiec_sim --set acquire=tagcover           # with tag-balanced labels
```

`paiec_sim` holds 300 platform-like draws (sub-benchmarks, 7 per draw, pairs cut to 115 items) and
`paiec_local` 100 draws over the public benchmarks as they are; the two `fresh` benchmarks are more draws of
the same two kinds from another seed. The ratings of the public items are read from `features/`. A run of
LLMDifficulty takes a few minutes on one CPU core. Each run prints the Brier ALC (lower is better) and writes one
row per evaluated pair to `output/`; `--compare <file>` adds the paired difference to another run.

Brier ALC of the runs in Table 2 of the report (blend = 0.524 sim + 0.476 local):

| Row of Table 2 | `evaluate.py` options | sim | local | blend | fresh sim | fresh local |
|---|---|---|---|---|---|---|
| BetaShrinkage reference | `--model reference/BetaShrinkage` | 0.1636 | 0.2164 | 0.1888 | 0.1662 | 0.2172 |
| Pair rate, m0 = 0, draw_sd = 0 | `--set tag_sd=0 --set item_sd=0 --set shrink=0 --set "features=[]" --set feat_sd=0 --set prior_mean=0 --set draw_sd=0` | 0.1608 | 0.2171 | 0.1876 | 0.1635 | 0.2178 |
| Pair rate, m0 = -0.5, draw_sd = 0.25 | `--set tag_sd=0 --set item_sd=0 --set shrink=0 --set "features=[]" --set feat_sd=0` | 0.1598 | 0.2146 | 0.1859 | 0.1621 | 0.2153 |
| + tags, items, shrink (= without the LLM stage) | `--set "features=[]"` | 0.1574 | 0.2139 | 0.1843 | 0.1598 | 0.2148 |
| **+ LLM ratings = LLMDifficulty** | (none) | **0.1572** | **0.2082** | **0.1815** | **0.1595** | **0.2096** |
| without tags | `--set tag_sd=0` | 0.1593 | 0.2076 | 0.1823 | 0.1614 | 0.2089 |
| without item offsets | `--set item_sd=0` | 0.1575 | 0.2087 | 0.1819 | 0.1598 | 0.2101 |
| without shrink | `--set shrink=0` | 0.1574 | 0.2081 | 0.1816 | 0.1597 | 0.2096 |
| without draw adaptation | `--set draw_sd=0` | 0.1572 | 0.2081 | 0.1815 | 0.1595 | 0.2096 |
| without prior slope | `--set feat_prior=0` | 0.1572 | 0.2127 | 0.1836 | 0.1595 | 0.2137 |
| prior level m0 = 0 | `--set prior_mean=0` | 0.1579 | 0.2109 | 0.1831 | 0.1604 | 0.2122 |
| with tag-balanced labels | `--set acquire=tagcover` | 0.1578 | 0.2067 | 0.1811 | 0.1598 | 0.2102 |
| TagCover archive (our earlier finalist) | `--set acquire=tagcover --set bench_sd=1.5 --set draw_sd=0.5 --set feat_prior=0.1 --set feat_sd=0.3 --set item_sd=4.0 --set shrink=0.45` | 0.1573 | 0.2094 | 0.1821 | 0.1594 | 0.2127 |

The datasets are `paiec_sim`, `paiec_local`, `paiec_simfresh` and `paiec_localfresh` (`--dataset`). A constant
0.5 scores 0.25 on every benchmark.

## Build the submission archive

```bash
python scripts/make_submission.py     # -> output/LLMDifficulty.zip, model.py at the root
```

The archive we uploaded (`LLMDifficulty_56538240__proxy0.1800.zip`) carried a larger shared core with code paths
of other configurations that LLMDifficulty never runs. `submission/paiec_core.py` keeps only the paths that
LLMDifficulty and the ablations above run; on the four local benchmarks it gives the same per-pair scores as the
uploaded archive, digit for digit.

## LLM ratings

At scoring time the archive rates each item live with Qwen/Qwen3-4B-Instruct-2507 (greedy decoding, 24 new
tokens, the first 2,000 characters of the item text, one third from the head and two thirds from the tail).
The prompt is `RATER_PROMPT` in `submission/paiec_core.py`. For local scoring, `features/` holds the ratings of
all public items, made with the same code; `python scripts/compute_ratings.py` rebuilds them (a GPU is
recommended for 11,532 texts).

## Earlier version

The first release of this repository (commit 87b3029) held TagCover (`TagCover_c6acf076__proxy0.1812.zip`), the
archive of the first version of the report. It is the last row of the table above.

## License

The code is released under the MIT license. The data is not included; its own terms apply to it.

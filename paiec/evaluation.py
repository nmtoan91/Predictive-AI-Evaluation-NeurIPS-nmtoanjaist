"""The contest metric and a local stand-in for the streaming evaluator.

Metric: per subject-benchmark pair,

    Brier ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31

with B_k the Brier score on the pair's evaluation items when k labels of its
acquisition pool have been acquired; pairs are averaged with equal weight.
Lower is better; a constant 0.5 scores exactly 0.25.

`run_episode` replays one formative evaluation as the starting kit's README and
its public client (tools/streaming_ingestion.py) describe it:

  * acquisition candidates arrive one at a time in a fixed random order; True
    acquires one recorded response, False skips the item for good;
  * one trajectory with nested prefixes, evaluated at budgets 0/1/3/7/15/31; all
    pairs finish a checkpoint before any pair gets more labels;
  * `labeled` is all evidence acquired so far, across the pairs of the draw;
  * every response run of an evaluation item is scored with the one
    probability predicted for that item;
  * without an acquisition function the client samples uniformly (hash-seeded).
"""

import copy
import hashlib
import inspect
import json
import math

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MAX_LABELS = BUDGETS[-1]


def brier_alc(briers):
    """Weighted area under one pair's learning curve; `briers` follows BUDGETS."""
    return sum(w * b for w, b in zip(WEIGHTS, briers))


def validate_probability(value):
    """The organizers' check: a finite real number in [0, 1], and not a bool."""
    if isinstance(value, bool):
        raise ValueError("predict() must return a finite probability, got a bool")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"predict() must return a finite probability in [0, 1], got {value!r}")
    return number


class AcquisitionHook:
    """Calls a streaming acquisition function the way the organizers' client does:
    acquisition_function(input, prediction=None, labeled=None, context=None) -> bool,
    with any of the three keywords left out, each argument a deep copy. Without a
    function the client samples uniformly ("random")."""

    def __init__(self, function):
        self.function, self.mode, self.keywords = function, "random", ()
        if function is not None:
            parameters = inspect.signature(function).parameters
            self.keywords = tuple(name for name in ("prediction", "labeled", "context") if name in parameters)
            self.mode = "stream"

    @property
    def needs_prediction(self):
        return "prediction" in self.keywords

    def __call__(self, input, prediction=None, labeled=None, context=None):
        values = {"prediction": prediction, "labeled": labeled, "context": context}
        result = self.function(copy.deepcopy(input), **{key: copy.deepcopy(values[key]) for key in self.keywords})
        if type(result) is not bool:
            raise ValueError("acquisition_function must return a bool")
        return result


def _default_random(context, current_input):
    """The client's default policy: uniform without replacement, hash-seeded."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


def run_episode(episode, predict, hook):
    """Replay one formative evaluation with predict(input, labeled) and an AcquisitionHook.

    Returns one row per pair: `briers` (per budget), `alc`, and `labels` (how many
    labels the pair held at each checkpoint).
    """
    pairs = episode["pairs"]
    labeled = []
    state = [{"cursor": 0, "acquired": 0} for _ in pairs]
    rows = [{"episode": episode["episode"], "subject_id": pair["subject_id"],
             "benchmark_id": pair["benchmark_id"], "source_benchmark": pair["source_benchmark"],
             "source_subject": pair["source_subject"], "n_eval": len(pair["evaluation"]),
             "n_rows": sum(len(ys) for _, ys in pair["evaluation"]),
             "base_rate": sum(sum(ys) for _, ys in pair["evaluation"])
             / sum(len(ys) for _, ys in pair["evaluation"]),
             "briers": [], "labels": []} for pair in pairs]

    for budget in BUDGETS:
        for pair, progress in zip(pairs, state):
            pool = pair["acquisition"]
            while progress["acquired"] < budget and progress["cursor"] < len(pool):
                item, label = pool[progress["cursor"]]
                current_input = [pair["subject"], item]
                context = {"subject_id": pair["subject_id"], "benchmark_id": pair["benchmark_id"],
                           "labels_acquired": progress["acquired"],
                           "labels_remaining": MAX_LABELS - progress["acquired"],
                           "max_labels": MAX_LABELS,
                           "items_remaining": len(pool) - progress["cursor"]}
                if hook.mode == "random":
                    query = _default_random(context, current_input)
                else:
                    probability = (validate_probability(predict(current_input, labeled))
                                   if hook.needs_prediction else None)
                    query = hook(current_input, probability, labeled, context)
                progress["cursor"] += 1
                if query:
                    labeled.append([current_input, label])
                    progress["acquired"] += 1
        for pair, progress, row in zip(pairs, state, rows):
            cache, predictions = {}, []
            for item, _ in pair["evaluation"]:
                # the platform also calls once per distinct input
                if id(item) not in cache:
                    cache[id(item)] = validate_probability(predict([pair["subject"], item], labeled))
                predictions.append(cache[id(item)])
            flat_p = [p for p, (_, ys) in zip(predictions, pair["evaluation"]) for _ in ys]
            flat_y = [y for _, ys in pair["evaluation"] for y in ys]
            row["briers"].append(sum((p - y) ** 2 for p, y in zip(flat_p, flat_y)) / len(flat_y))
            row["labels"].append(progress["acquired"])
    for row in rows:
        row["alc"] = brier_alc(row["briers"])
    return rows


def summarize(rows):
    """Run-level numbers from per-pair rows.

    One upload is scored as the unweighted mean over the pairs of one draw, that
    is one episode. `score` is therefore the mean of the per-episode means, and
    `episode_sd` their spread: how far a single upload can sit from it.
    """
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    means = [sum(r["alc"] for r in v) / len(v) for v in by_episode.values()]
    centre = sum(means) / len(means)
    out = {"score": centre, "num_pairs": len(rows), "num_episodes": len(means)}
    for index, budget in enumerate(BUDGETS):
        out[f"brier_b{budget}"] = sum(sum(r["briers"][index] for r in v) / len(v)
                                      for v in by_episode.values()) / len(means)
    out["episode_sd"] = (sum((m - centre) ** 2 for m in means) / (len(means) - 1)) ** 0.5 if len(means) > 1 else float("nan")
    out["score_se"] = out["episode_sd"] / len(means) ** 0.5 if len(means) > 1 else float("nan")
    by_benchmark = {}
    for row in rows:
        by_benchmark.setdefault(row["source_benchmark"], []).append(row["alc"])
    out["by_benchmark"] = {name: sum(v) / len(v) for name, v in sorted(by_benchmark.items())}
    return out


def paired_difference(rows, other_rows):
    """Mean over episodes of (this run - other run), its standard error and the
    episodes where this run is better; both runs on the same local benchmark."""
    other = {(str(row["episode"]), row["subject_id"], row["benchmark_id"]): float(row["alc"]) for row in other_rows}
    by_episode = {}
    for row in rows:
        key = (str(row["episode"]), row["subject_id"], row["benchmark_id"])
        if key not in other:
            raise KeyError(f"pair {key} is not in the other run: not the same benchmark?")
        by_episode.setdefault(row["episode"], []).append(float(row["alc"]) - other[key])
    deltas = [sum(v) / len(v) for v in by_episode.values()]
    mean = sum(deltas) / len(deltas)
    sd = (sum((d - mean) ** 2 for d in deltas) / max(1, len(deltas) - 1)) ** 0.5
    return {"delta": mean, "delta_se": sd / len(deltas) ** 0.5,
            "wins": sum(d < 0 for d in deltas), "episodes": len(deltas)}

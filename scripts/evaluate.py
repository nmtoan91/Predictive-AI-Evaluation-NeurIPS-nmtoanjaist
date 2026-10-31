"""Score a model on a local benchmark: Brier ALC, lower is better.

    python scripts/evaluate.py --dataset paiec_sim                                # TagCover (submission/)
    python scripts/evaluate.py --dataset paiec_sim --model reference/BetaShrinkage
    python scripts/evaluate.py --dataset paiec_sim --set "features=[]"           # one setting changed
    python scripts/evaluate.py --dataset paiec_sim --compare output/BetaShrinkage_paiec_sim.csv

A model is a directory laid out like the submission archive: model.py with
predict(input, labeled), optionally labeling.py with acquisition_function, and
config.json. `--set key=value` changes one setting of config.json (the value is
read as JSON) in a copy under output/build/. Nothing is fitted on the public
benchmarks, so no run is scored on a benchmark its model was trained on.

The LLM ratings of the public items are read from features/ (shipped with the
repository; scripts/compute_ratings.py rebuilds them), and a missing rating
stops the run instead of loading the LLM. Writes
output/<name>_<dataset>[_<settings hash>].csv, one row per evaluated pair.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PAIEC_FEATURE_DIR", os.path.join(ROOT, "features"))
os.environ.setdefault("PAIEC_REQUIRE_CACHE", "1")

from paiec import data, evaluation  # noqa: E402

OUTPUT_DIR = os.path.join(ROOT, "output")
COLUMNS = (["episode", "subject_id", "benchmark_id", "source_benchmark", "source_subject",
            "n_eval", "n_rows", "base_rate", "alc"]
           + [f"brier_b{b}" for b in evaluation.BUDGETS] + [f"labels_b{b}" for b in evaluation.BUDGETS])


def model_name(model_dir):
    name = os.path.basename(os.path.normpath(model_dir))
    return "TagCover" if name == "submission" else name


def stage(model_dir, overrides):
    """(directory to load, settings hash): the model directory itself, or a temporary copy
    with `overrides` written into its config.json. `acquire=random` also drops labeling.py,
    which leaves label selection to the platform's uniform sampling."""
    if not overrides:
        return model_dir, ""
    path = os.path.join(model_dir, "config.json")
    config = {}
    if os.path.exists(path):
        with open(path) as handle:
            config = json.load(handle)
    for override in overrides:
        key, _, value = override.partition("=")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    tag = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    os.makedirs(os.path.join(OUTPUT_DIR, "build"), exist_ok=True)
    target = os.path.join(tempfile.mkdtemp(prefix=f"{model_name(model_dir)}_{tag}_", dir=os.path.join(OUTPUT_DIR, "build")),
                          "model")
    shutil.copytree(model_dir, target, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(target, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    if config.get("acquire") == "random" and os.path.exists(os.path.join(target, "labeling.py")):
        os.remove(os.path.join(target, "labeling.py"))
    return target, tag


def load(model_dir):
    """Import model.py (and labeling.py) as the platform does -> (predict, acquisition function or None)."""
    loaded, before = [], set(sys.modules)
    sys.path.insert(0, model_dir)
    try:
        for filename in ("model.py", "labeling.py"):
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                loaded.append(None)
                continue
            spec = importlib.util.spec_from_file_location(f"participant_{filename[:-3]}_{abs(hash(model_dir))}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    finally:
        sys.path.remove(model_dir)
        # helper modules imported by bare name (paiec_core) must not be reused by the next model
        for name in set(sys.modules) - before:
            origin = getattr(sys.modules[name], "__file__", None) or ""
            if os.path.abspath(origin).startswith(os.path.abspath(model_dir)):
                del sys.modules[name]
    model, labeling = loaded
    return model.predict, (labeling.acquisition_function if labeling is not None else None)


def evaluate(model_dir, dataset, overrides=(), max_episodes=0, compare="", verbose=True):
    started = time.time()
    staged, tag = stage(model_dir, list(overrides))
    predict, acquire = load(staged)
    if tag:
        shutil.rmtree(os.path.dirname(staged), ignore_errors=True)  # loaded: the copy is not read again
    hook = evaluation.AcquisitionHook(acquire)
    episodes = data.load_episodes(dataset)
    if max_episodes:
        episodes = episodes[:max_episodes]
    rows = []
    for episode in episodes:
        rows.extend(evaluation.run_episode(episode, predict, hook))
    summary = evaluation.summarize(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = os.path.join(OUTPUT_DIR, f"{model_name(model_dir)}_{dataset}{'_' + tag if tag else ''}.csv")
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([row["episode"], row["subject_id"], row["benchmark_id"], row["source_benchmark"],
                             row["source_subject"], row["n_eval"], row["n_rows"], f"{row['base_rate']:.4f}",
                             f"{row['alc']:.6f}"] + [f"{b:.6f}" for b in row["briers"]] + row["labels"])
    paired = None
    if compare:
        with open(compare, newline="") as handle:
            paired = evaluation.paired_difference(rows, list(csv.DictReader(handle)))

    if verbose:
        print(f"{model_name(model_dir)} on {dataset}: {summary['num_episodes']} episodes, {summary['num_pairs']} pairs"
              + (f", settings {tag}" if tag else ""))
        print(f"Brier ALC = {summary['score']:.6f} +- {summary['score_se']:.6f} "
              f"(one-episode sd {summary['episode_sd']:.4f})")
        print("  " + "  ".join(f"B{b}={summary[f'brier_b{b}']:.4f}" for b in evaluation.BUDGETS))
        print("  by benchmark: " + "  ".join(f"{k}={v:.4f}" for k, v in summary["by_benchmark"].items()))
        if paired:
            print(f"  vs {os.path.basename(compare)}: {paired['delta']:+.5f} +- {paired['delta_se']:.5f} "
                  f"(better in {paired['wins']}/{paired['episodes']} episodes)")
        print(f"  wrote {os.path.relpath(output, ROOT)} in {time.time() - started:.0f} s")
    return summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="paiec_sim", help="local benchmark under datasets/ (build_benchmark.py)")
    parser.add_argument("--model", default=os.path.join(ROOT, "submission"), help="model directory")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="change one setting of the model's config.json; repeatable")
    parser.add_argument("--max_episodes", type=int, default=0, help="score only the first N episodes")
    parser.add_argument("--compare", default="", help="per-pair CSV of another run on the same dataset")
    args = parser.parse_args()
    evaluate(os.path.abspath(args.model), args.dataset, args.set, args.max_episodes, args.compare)


if __name__ == "__main__":
    sys.exit(main())

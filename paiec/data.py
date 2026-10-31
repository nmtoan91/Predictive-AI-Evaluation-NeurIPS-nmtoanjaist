"""Public tables, local benchmarks, and the inputs a model sees.

Layout:
  datasets/measurement_db/raw/<benchmark>/{items,subjects,response}.parquet
      the public training pool (scripts/download_data.py)
  datasets/<id>/processed/{episodes.json,meta.json}
      a local benchmark: simulated formative evaluations (scripts/build_benchmark.py)

The sampler follows the starting kit's README:

  * eligible are subject-benchmark pairs with at least 80 distinct items;
  * the platform selects complete (benchmark, subject) groups under a draw limit
    of 1,000 distinct source items, with preference for benchmarks and subjects
    represented less often in the draw and random tie-breaking; groups that
    exceed the remaining budget are skipped;
  * a pair's items are split 50/50 into an acquisition pool and an evaluation
    pool (an odd count gives the extra item to acquisition); repeated response
    runs of an item stay in the same pool;
  * acquisition candidates arrive in a fixed random order, and acquiring one
    gives one recorded response; every response run of an evaluation item is
    scored.

An "episode" is one such draw. Two readings are ours: the 1,000 limit is counted
per pair and summed, and "subjects represented less often" is taken by model
name. The split and the stream order are redrawn in every episode, so a pair
that is drawn in many episodes does not repeat one label sequence.

Two pools of benchmarks to draw from:

  pool="whole"  the public benchmarks as they are (paiec_local). Draws hold about
                5 pairs, each alone on its benchmark.
  pool="sub"    a platform-like pool (paiec_sim): each public benchmark is cut into
                sub-benchmarks along its own structure (matharena by competition
                family, multi_swebench by language, mmdocrag and swe_rebench into
                item chunks), every episode sees a random 7 of them, and a pair is
                cut to at most 115 of its items (8.7 pairs per draw, about 40% with
                a co-subject, pair success rates from 1% to 97%).

`make_synthetic_raw` writes a tiny raw root in the public schema, so the smoke
test runs every code path without the (gated) real data.
"""

import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ROOT, "datasets")
RAW_ROOT = "datasets/measurement_db/raw"

# Field order of the inputs as documented by the organizers.
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "harness_version", "reasoning_effort", "subject_features_extra")

MIN_ITEMS = 80
DRAW_LIMIT = 1000

# matharena competitions grouped so that a subject has >= 80 items in a family.
MATH_FAMILIES = {"aime": "aime", "cmimc": "cmimc", "smt": "smt", "hmmt": "hmmt",
                 "kangaroo": "kangaroo", "apex": "apex", "brumo": "brumo", "imo": "imo"}
# Chunks of a benchmark without usable item structure: (number of chunks, items per chunk).
CHUNKS = {"mmdocrag": (3, 200), "swe_rebench": (2, 250)}


# --------------------------------------------------------------------------- tables and inputs

def processed_dir(dataset_id):
    return os.path.join(DATASETS_DIR, dataset_id, "processed")


def list_benchmarks(raw_root):
    """Benchmark directories under a raw root (those holding a response table)."""
    if not os.path.isdir(raw_root):
        return []
    return sorted(name for name in os.listdir(raw_root)
                  if os.path.exists(os.path.join(raw_root, name, "response.parquet")))


def _text(value):
    """Table cell -> the string the evaluator would send ('' for a missing value)."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def subject_dict(row):
    return {field: _text(row.get(field)) for field in SUBJECT_FIELDS}


def item_dict(row, anonymous_benchmark_id, interactors=""):
    return {"item_content": _text(row.get("content")), "item_features": _text(row.get("item_features")),
            "interactors": _text(interactors), "benchmark_id": anonymous_benchmark_id}


def load_tables(raw_root, benchmark, binarize=0.5):
    """One benchmark's tables, with responses as binary observations.

    Returns {"subjects", "items", "observations"}: `observations` has one row per
    recorded response (subject_id, item_id, label); a subject-item pair observed
    several times keeps every row, because every response run of an evaluation
    item is scored. Graded responses become 1 when response >= `binarize`.
    """
    import pandas as pd

    directory = os.path.join(raw_root, benchmark)
    subjects = pd.read_parquet(os.path.join(directory, "subjects.parquet"))
    items = pd.read_parquet(os.path.join(directory, "items.parquet"))
    responses = pd.read_parquet(os.path.join(directory, "response.parquet"),
                                columns=["subject_id", "item_id", "trial", "test_condition",
                                         "interactors", "response"])
    responses = responses[responses["response"].notna()]
    responses = responses.sort_values(["subject_id", "item_id", "trial"]).reset_index(drop=True)
    responses["label"] = (responses["response"] >= binarize).astype(int)
    observations = responses[["subject_id", "item_id", "interactors", "label"]]
    return {"subjects": subjects, "items": items, "observations": observations}


def load_meta(dataset_id):
    with open(os.path.join(processed_dir(dataset_id), "meta.json")) as handle:
        return json.load(handle)


def load_episodes(dataset_id):
    """Episodes of a local benchmark with every input dictionary materialized.

    episodes.json stores item ids, not item text; the text is joined back in
    here, from the tables the benchmark was built from.
    """
    meta = load_meta(dataset_id)
    with open(os.path.join(processed_dir(dataset_id), "episodes.json")) as handle:
        stored = json.load(handle)
    raw_root = os.path.join(ROOT, meta["raw_root"])

    needed = {}
    for episode in stored:
        for pair in episode["pairs"]:
            needed.setdefault(pair["source_benchmark"], set()).update(
                entry[0] for entry in pair["acquisition"] + pair["evaluation"])
    subjects, items = {}, {}
    for benchmark, item_ids in needed.items():
        tables = load_tables(raw_root, benchmark, meta["binarize"])
        for row in tables["subjects"].to_dict("records"):
            subjects[(benchmark, row["subject_id"])] = subject_dict(row)
        rows = tables["items"][tables["items"]["item_id"].isin(item_ids)]
        for row in rows.to_dict("records"):
            items[(benchmark, row["item_id"])] = row

    episodes = []
    for episode in stored:
        pairs = []
        for pair in episode["pairs"]:
            benchmark = pair["source_benchmark"]
            cache = {}

            def build(entries):
                # acquisition: (item, label); evaluation: (item, [label of every response run])
                out = []
                for item_id, label in entries:
                    if item_id not in cache:
                        cache[item_id] = item_dict(items[(benchmark, item_id)], pair["benchmark_id"])
                    out.append((cache[item_id], label))
                return out

            pairs.append({"subject_id": pair["subject_id"], "benchmark_id": pair["benchmark_id"],
                          "source_benchmark": benchmark, "source_subject": pair["source_subject"],
                          "sub_benchmark": pair.get("sub_benchmark", benchmark),
                          "subject": subjects[(benchmark, pair["source_subject"])],
                          "acquisition": build(pair["acquisition"]),
                          "evaluation": build(pair["evaluation"])})
        episodes.append({"episode": episode["episode"], "pairs": pairs})
    return episodes


# --------------------------------------------------------------------------- local benchmarks

def _digest(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def _anonymous(kind, key):
    """Stable alias, as on the platform: the same subject or benchmark keeps its id."""
    return f"{kind}_{int(_digest(kind, key)[:8], 16) % 900000 + 100000}"


def _features(text):
    out = {}
    for part in str(text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def sub_benchmark_of(benchmark, items):
    """item_id -> sub-benchmark name for pool="sub" (None: the item is left out)."""
    mapping = {}
    if benchmark == "matharena":
        for item_id, text in zip(items["item_id"], items["item_features"]):
            competition = _features(text).get("competition", "")
            family = next((name for prefix, name in MATH_FAMILIES.items()
                           if competition.startswith(prefix)), "misc")
            mapping[item_id] = f"matharena/{family}"
    elif benchmark == "multi_swebench":
        for item_id, text in zip(items["item_id"], items["item_features"]):
            mapping[item_id] = f"multi_swebench/{_features(text).get('lang', 'other')}"
    elif benchmark in CHUNKS:
        n_chunks, size = CHUNKS[benchmark]
        order = sorted(items["item_id"], key=lambda item_id: _digest("chunk", benchmark, item_id))
        for index, item_id in enumerate(order[:n_chunks * size]):
            mapping[item_id] = f"{benchmark}/chunk{index // size}"
    else:
        mapping = {item_id: benchmark for item_id in items["item_id"]}
    return mapping


def split_pair(item_ids, rng):
    """Random 50/50 split of one pair's items -> (acquisition, evaluation).

    The order inside the acquisition list is the stream order. Each pair is split
    on its own, so an item can sit in one subject's acquisition pool and in another
    subject's evaluation pool. Odd counts give acquisition the extra item.
    """
    ordered = sorted(item_ids)
    order = [ordered[i] for i in rng.permutation(len(ordered))]
    n_acquisition = (len(order) + 1) // 2
    return order[:n_acquisition], order[n_acquisition:]


def load_groups(raw_abs, names, binarize, draw_limit, pool="whole", max_items=0, verbose=True):
    """Drawable pairs: {(benchmark, subject_id): {"name", "parent", "labels": {item_id: [0/1, ...]}}}.

    `benchmark` is the name the sampler sees (a sub-benchmark for pool="sub"),
    `parent` the public benchmark it comes from.
    """
    groups, parents = {}, []
    for parent in names:
        tables = load_tables(raw_abs, parent, binarize)
        model = dict(zip(tables["subjects"]["subject_id"], tables["subjects"]["normalized_name"]))
        observations = tables["observations"]
        if pool == "sub":
            mapping = sub_benchmark_of(parent, tables["items"])
            observations = observations.assign(sub=observations["item_id"].map(mapping))
            observations = observations[observations["sub"].notna()]
        else:
            observations = observations.assign(sub=parent)
        n_eligible = n_drawable = 0
        for (sub, subject_id), group in observations.groupby(["sub", "subject_id"], sort=True):
            labels = group.groupby("item_id", sort=True)["label"].agg(list).to_dict()
            if len(labels) < MIN_ITEMS:
                continue
            n_eligible += 1
            if max_items and len(labels) > max_items:
                # a uniform random subset, persistent for the pair
                keep = sorted(labels, key=lambda item_id: _digest("cut", sub, subject_id, item_id))[:max_items]
                labels = {item_id: labels[item_id] for item_id in sorted(keep)}
            if len(labels) > draw_limit:
                continue
            n_drawable += 1
            groups[(sub, str(subject_id))] = {"name": model.get(subject_id) or str(subject_id),
                                              "parent": parent, "labels": labels}
        if n_drawable:
            parents.append(parent)
        if verbose:
            print(f"  {parent}: {n_eligible} pairs with >= {MIN_ITEMS} items, {n_drawable} drawable; "
                  f"base rate {tables['observations']['label'].mean():.3f}")
    return groups, parents


def draw_episode(groups, rng, draw_limit, benchmarks=None):
    """One formative draw: complete groups, least-represented benchmark and subject first."""
    remaining, chosen = draw_limit, []
    by_benchmark, by_subject = {}, {}
    keys = sorted(key for key in groups if benchmarks is None or key[0] in benchmarks)
    while True:
        fitting = [key for key in keys if key not in chosen and len(groups[key]["labels"]) <= remaining]
        if not fitting:
            return chosen
        ties = rng.random(len(fitting))
        best = min(range(len(fitting)), key=lambda i: (by_benchmark.get(fitting[i][0], 0),
                                                       by_subject.get(groups[fitting[i]]["name"], 0),
                                                       ties[i]))
        key = fitting[best]
        chosen.append(key)
        remaining -= len(groups[key]["labels"])
        by_benchmark[key[0]] = by_benchmark.get(key[0], 0) + 1
        by_subject[groups[key]["name"]] = by_subject.get(groups[key]["name"], 0) + 1


def build_benchmark(dataset_id, raw_root=RAW_ROOT, n_episodes=100, binarize=0.5, seed=51, benchmarks=None,
                    draw_limit=DRAW_LIMIT, pool="whole", pool_size=0, max_items=0, verbose=True):
    """Write datasets/<dataset_id>/processed/{episodes.json,meta.json}.

    pool_size > 0: each episode first picks that many benchmarks at random and
    draws only from them.
    """
    raw_abs = raw_root if os.path.isabs(raw_root) else os.path.join(ROOT, raw_root)
    names = benchmarks or list_benchmarks(raw_abs)
    if not names:
        raise FileNotFoundError(f"no benchmark tables under {raw_abs}; run scripts/download_data.py")
    rng = np.random.default_rng(seed)
    groups, parents = load_groups(raw_abs, names, binarize, draw_limit, pool, max_items, verbose)
    all_benchmarks = sorted({key[0] for key in groups})

    episodes, n_pairs, n_shared = [], 0, 0
    for index in range(n_episodes):
        allowed = None
        if pool_size and pool_size < len(all_benchmarks):
            allowed = {all_benchmarks[i] for i in rng.permutation(len(all_benchmarks))[:pool_size]}
        pairs = []
        for benchmark, subject_id in draw_episode(groups, rng, draw_limit, allowed):
            group = groups[(benchmark, subject_id)]
            labels = group["labels"]
            acquisition, evaluation = split_pair(labels, rng)
            pairs.append({
                "subject_id": _anonymous("subject", f"{benchmark}|{subject_id}"),
                "benchmark_id": _anonymous("benchmark", benchmark),
                "source_benchmark": group["parent"], "sub_benchmark": benchmark,
                "source_subject": subject_id,
                # acquiring a candidate gives ONE of its recorded responses
                "acquisition": [[item_id, int(labels[item_id][rng.integers(len(labels[item_id]))])]
                                for item_id in acquisition],
                "evaluation": [[item_id, [int(y) for y in labels[item_id]]] for item_id in evaluation]})
        counts = {}
        for pair in pairs:
            counts[pair["benchmark_id"]] = counts.get(pair["benchmark_id"], 0) + 1
        n_shared += sum(1 for pair in pairs if counts[pair["benchmark_id"]] > 1)
        n_pairs += len(pairs)
        episodes.append({"episode": index, "pairs": pairs})

    out = processed_dir(dataset_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.json"), "w") as handle:
        json.dump(episodes, handle)
    meta = {"dataset_id": dataset_id,
            "raw_root": os.path.relpath(raw_abs, ROOT).replace(os.sep, "/"),
            "benchmarks": parents, "n_episodes": n_episodes, "draw_limit": draw_limit,
            "pool": pool, "pool_size": pool_size, "max_items": max_items,
            "n_sub_benchmarks": len(all_benchmarks),
            "pairs_per_episode": round(n_pairs / n_episodes, 2),
            "share_with_co_subject": round(n_shared / max(1, n_pairs), 3),
            "binarize": binarize, "seed": seed}
    revision = os.path.join(raw_abs, "REVISION")
    if os.path.exists(revision):
        with open(revision) as handle:
            meta["revision"] = handle.read().strip()
    with open(os.path.join(out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    if verbose:
        print(f"  {dataset_id}: {n_episodes} episodes, {n_pairs} pairs ({n_pairs / n_episodes:.1f} per episode, "
              f"{100 * n_shared / max(1, n_pairs):.0f}% with a co-subject), {len(all_benchmarks)} benchmarks in the pool "
              f"-> {os.path.relpath(out, ROOT)}")
    return meta


def make_synthetic_raw(raw_root, n_benchmarks=3, n_subjects=8, n_items=120, seed=0):
    """A tiny raw root in the public schema, drawn from a Rasch model.

    Subjects are shared between benchmarks (same normalized_name), benchmarks
    differ in difficulty, and a fifth of the items have a second response run:
    enough structure for every code path, and nothing here resembles real results.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    ability = rng.normal(0.0, 1.0, n_subjects)
    for b in range(n_benchmarks):
        name = f"synthetic_{b}"
        directory = os.path.join(raw_root, name)
        os.makedirs(directory, exist_ok=True)
        difficulty = rng.normal(b - 1.0, 1.0, n_items)
        subjects = pd.DataFrame({
            "subject_id": [f"s{b}_{s}" for s in range(n_subjects)],
            "display_name": [f"model-{s}" for s in range(n_subjects)],
            "normalized_name": [f"Synthetic Model {s}" for s in range(n_subjects)],
            "provider": [f"Lab {s % 3}" for s in range(n_subjects)],
            "release_date": [f"2025-{s % 12 + 1:02d}-01" for s in range(n_subjects)],
            "access_date": None, "harness": None, "reasoning_effort": None,
            "harness_version": None, "subject_features_extra": None})
        items = pd.DataFrame({
            "item_id": [f"i{b}_{i}" for i in range(n_items)], "benchmark_id": name,
            "raw_item_id": [str(i) for i in range(n_items)],
            "content": [f"Synthetic task {i} of benchmark {b}: compute step {i % 7} of case {i % 11}." for i in range(n_items)],
            "item_features": [f"tier={i % 3}" for i in range(n_items)]})
        rows = []
        for s in range(n_subjects):
            p = 1.0 / (1.0 + np.exp(-(ability[s] - difficulty)))
            for i in np.flatnonzero(rng.random(n_items) < 0.95):
                for trial in (1, 2) if i % 5 == 0 else (1,):
                    rows.append({"subject_id": f"s{b}_{s}", "item_id": f"i{b}_{i}",
                                 "response_id": f"r{b}_{s}_{i}_{trial}", "benchmark_id": name,
                                 "trial": trial, "test_condition": None, "interactors": None,
                                 "response": float(rng.random() < p[i])})
        subjects.to_parquet(os.path.join(directory, "subjects.parquet"), index=False)
        items.to_parquet(os.path.join(directory, "items.parquet"), index=False)
        pd.DataFrame(rows).to_parquet(os.path.join(directory, "response.parquet"), index=False)

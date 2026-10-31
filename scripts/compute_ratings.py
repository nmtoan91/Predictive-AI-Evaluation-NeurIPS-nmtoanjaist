"""Rate every public item with the LLM rater of TagCover (a GPU is recommended).

    python scripts/compute_ratings.py                        # all items, one process
    python scripts/compute_ratings.py --shard 0 --of 4       # one of four parallel jobs
    python scripts/compute_ratings.py --merge --of 4         # join the four parts

Writes features/llm-<model>-c<chars>/{vectors.npy, keys.json, manifest.json}, the
cache the predictor reads during local scoring (key: SHA-1 of the item text
clipped to <chars> characters; values: log10 minutes and four 1-10 scales).
The repository ships this cache for Qwen/Qwen3-4B-Instruct-2507 at 2,000
characters (11,532 texts). It was made with 64 new tokens per text; the archive
generates 24, which cover the two scales it uses, and greedy decoding gives
those two the same values either way.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))

import paiec_core  # noqa: E402

from paiec import data  # noqa: E402


def public_texts(chars):
    """Distinct clipped item texts of every public benchmark, longest first."""
    import pandas as pd

    raw_root = os.path.join(ROOT, data.RAW_ROOT)
    texts = {}
    for benchmark in data.list_benchmarks(raw_root):
        items = pd.read_parquet(os.path.join(raw_root, benchmark, "items.parquet"), columns=["item_id", "content"])
        for content in items["content"]:
            content = content if isinstance(content, str) else ""
            texts.setdefault(paiec_core.text_key(paiec_core.clip_text(content, chars)), content)
    keys = sorted(texts, key=lambda key: -len(texts[key]))
    return keys, [texts[key] for key in keys]


def write_cache(directory, keys, vectors, args):
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "vectors.npy"), np.asarray(vectors, dtype=np.float32))
    with open(os.path.join(directory, "keys.json"), "w") as handle:
        json.dump(list(keys), handle)
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump({"model": args.model, "chars": args.chars, "tokens": args.tokens, "count": len(keys),
                   "dim": int(np.asarray(vectors).shape[1]),
                   "nan_rows": int(np.isnan(np.asarray(vectors, dtype=np.float32)).any(axis=1).sum())},
                  handle, indent=1, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--chars", type=float, default=2000)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="join the part files of --of shards")
    args = parser.parse_args()

    name = f"llm-{args.model.split('/')[-1]}-c{int(args.chars)}"
    directory = os.path.join(ROOT, "features", name)
    keys, texts = public_texts(args.chars)

    if args.merge:
        found = {}
        for shard in range(args.of):
            with np.load(os.path.join(directory, f"part-{shard:03d}-of-{args.of:03d}.npz")) as part:
                found.update(zip((str(key) for key in part["keys"]), part["vectors"]))
        write_cache(directory, keys, np.stack([found[key] for key in keys]), args)
        print(f"{name}: {len(keys)} ratings merged")
        return

    rater = paiec_core.Rater(args.model, "", args.chars, tokens=args.tokens)
    mine = [index for index in range(len(keys)) if index % args.of == args.shard]
    started, out = time.time(), []
    for start in range(0, len(mine), 64):
        out.append(rater.rate([texts[index] for index in mine[start:start + 64]], batch_size=args.batch))
        done = min(len(mine), start + 64)
        print(f"  {done}/{len(mine)}  {time.time() - started:.0f} s", flush=True)
    vectors = np.concatenate(out) if out else np.zeros((0, len(paiec_core.RATER_FIELDS)), dtype=np.float32)
    if args.of == 1:
        write_cache(directory, keys, vectors, args)
        print(f"{name}: {len(keys)} ratings written")
    else:
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, f"part-{args.shard:03d}-of-{args.of:03d}.npz"),
                 keys=np.array([keys[index] for index in mine]), vectors=vectors)
        print(f"{name}: shard {args.shard} of {args.of} written ({len(mine)} texts)")


if __name__ == "__main__":
    sys.exit(main())

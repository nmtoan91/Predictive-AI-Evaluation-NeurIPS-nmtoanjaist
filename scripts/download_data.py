"""Download the public training data (AI Measurement Data Bank) from Hugging Face.

    huggingface-cli login                 # the dataset is gated: request access once on its page
    python scripts/download_data.py       # curated tables, about 1.1 GB

Files land in datasets/measurement_db/raw/<benchmark>/... The default revision is
the one the report uses (bc8204d8); item and response ids change between
revisions, so another revision gives other local benchmarks.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "aims-foundations/measurement-db"
REVISION = "bc8204d811823da849c6686bf124d4ca9f82e4de"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--revision", default=REVISION, help="dataset commit (default: the report's)")
    parser.add_argument("--out", default=os.path.join(ROOT, "datasets", "measurement_db", "raw"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    os.makedirs(args.out, exist_ok=True)
    print(f"{REPO_ID} @ {args.revision} -> {args.out}")
    snapshot_download(REPO_ID, repo_type="dataset", revision=args.revision, local_dir=args.out,
                      ignore_patterns=["*/raw/*", "*/reproduction_checks/*"], max_workers=args.workers)
    with open(os.path.join(args.out, "REVISION"), "w") as handle:
        handle.write(args.revision + "\n")
    n_files = n_bytes = 0
    for directory, _, files in os.walk(args.out):
        if ".cache" in directory:
            continue
        for name in files:
            n_files += 1
            n_bytes += os.path.getsize(os.path.join(directory, name))
    print(f"done: {n_files} files, {n_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())

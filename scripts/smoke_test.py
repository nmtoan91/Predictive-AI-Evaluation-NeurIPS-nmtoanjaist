"""Run the whole pipeline on a tiny synthetic dataset: seconds, no download, no GPU.

    python scripts/smoke_test.py

Writes a synthetic raw root in the public schema (datasets/smoke/raw), builds a
local benchmark of three draws from it, scores TagCover (the LLM stage answers
with neutral ratings, as in the organizers' archive check) and the reference,
and zips the submission.
"""

import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"

import evaluate  # noqa: E402
import make_submission  # noqa: E402

from paiec import data  # noqa: E402


def main():
    data.make_synthetic_raw(os.path.join(ROOT, "datasets", "smoke", "raw"))
    data.build_benchmark("smoke_draws", raw_root="datasets/smoke/raw", n_episodes=3, seed=0)
    tagcover, _ = evaluate.evaluate(os.path.join(ROOT, "submission"), "smoke_draws")
    reference, _ = evaluate.evaluate(os.path.join(ROOT, "reference", "BetaShrinkage"), "smoke_draws")
    for name, summary in (("TagCover", tagcover), ("BetaShrinkage", reference)):
        if not 0.0 < summary["score"] < 1.0:
            raise SystemExit(f"{name}: Brier ALC {summary['score']} is not a valid score")
    zip_path = os.path.join(ROOT, "output", "smoke", "TagCover.zip")
    make_submission.make_zip(os.path.join(ROOT, "submission"), zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if "model.py" not in archive.namelist():
            raise SystemExit("model.py is not at the root of the archive")
    print("smoke test passed (the scores on synthetic data say nothing about the method)")


if __name__ == "__main__":
    sys.exit(main())

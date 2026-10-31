"""Build the local benchmarks of the report from the downloaded public tables.

    python scripts/build_benchmark.py                      # all four
    python scripts/build_benchmark.py --dataset paiec_sim

  paiec_sim          300 platform-like draws: sub-benchmarks, 7 per draw, pairs cut to 115 items
  paiec_local        100 draws over the public benchmarks as they are
  paiec_simfresh     600 draws like paiec_sim from a seed nothing was selected on
  paiec_localfresh   300 draws like paiec_local from that seed

paiec/data.py says what the sampler imitates. The report's estimate of a
Codabench score is blend = 0.529 sim + 0.471 local.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec.data import build_benchmark  # noqa: E402

DATASETS = {
    "paiec_sim": dict(n_episodes=300, seed=51, pool="sub", pool_size=7, max_items=115),
    "paiec_local": dict(n_episodes=100, seed=51),
    "paiec_simfresh": dict(n_episodes=600, seed=9001, pool="sub", pool_size=7, max_items=115),
    "paiec_localfresh": dict(n_episodes=300, seed=9001),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS), help="build only this one")
    args = parser.parse_args()
    for name, options in DATASETS.items():
        if args.dataset in (None, name):
            build_benchmark(name, **options)


if __name__ == "__main__":
    sys.exit(main())

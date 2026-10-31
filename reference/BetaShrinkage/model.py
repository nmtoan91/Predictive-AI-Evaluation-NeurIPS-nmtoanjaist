"""Two-level Beta shrinkage of a pair's success rate. Standard library only.

    benchmark rate  m_b = (S_others + k0 * m0) / (N_others + k0)
    prediction      p   = (S_own    + k1 * m_b) / (N_own    + k1)

`own` are the revealed labels of the target subject on the target benchmark,
`others` those of every other subject on that benchmark: `labeled` holds the
labels of all sampled pairs, so another model's results on the same benchmark
tell how hard it is before the target has a single label of its own. m0, k0,
k1 are `prior_mean`, `prior_strength`, `pair_strength` in config.json.

Every item of a pair gets the same probability; nothing is fitted offline and
nothing is kept between calls.
"""

import json
import os

_CONFIG = {"prior_mean": 0.5, "prior_strength": 8.0, "pair_strength": 3.0}
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(_PATH):
    with open(_PATH) as _handle:
        _CONFIG.update(json.load(_handle))


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    benchmark = item.get("benchmark_id")
    own_n = own_s = other_n = other_s = 0
    for (seen_subject, seen_item), label in labeled or ():
        if seen_item.get("benchmark_id") != benchmark:
            continue
        if seen_subject == subject:
            own_n += 1
            own_s += label
        else:
            other_n += 1
            other_s += label
    k0, k1 = _CONFIG["prior_strength"], _CONFIG["pair_strength"]
    benchmark_rate = (other_s + k0 * _CONFIG["prior_mean"]) / (other_n + k0)
    return float((own_s + k1 * benchmark_rate) / (own_n + k1))

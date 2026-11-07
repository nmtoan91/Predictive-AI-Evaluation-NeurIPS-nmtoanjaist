"""LLMDifficulty, our PAIEC 2026 submission: predict(input, labeled) -> P(correct).

The method is paiec_core.py (shipped beside this file); config.json holds its settings.
There is no labeling.py: every pair receives the labels of the first candidates of its
random stream, as the platform does by default.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paiec_core  # noqa: E402

predict = paiec_core.make_predict(__file__)

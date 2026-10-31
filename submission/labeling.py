"""Tag-balanced label acquisition of TagCover (config.json: `acquire`); see paiec_core.make_acquire."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paiec_core  # noqa: E402

acquisition_function = paiec_core.make_acquire(__file__)

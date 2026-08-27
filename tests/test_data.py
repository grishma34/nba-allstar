"""Tests for src/data.py — the parts that run without the raw CSVs.

The CSVs are release-attached, not committed, so these tests exercise only
the pure logic; the full pipeline is verified by `python -m src.data`.
"""

import re

from src.data import COMBINED_TEAM_PATTERN


def test_combined_team_pattern_matches_all_multi_team_codes():
    # The data contains 2TM through 5TM; hard-coding "2TM"/"3TM" (the two
    # values the spec happened to name) would silently drop 4TM and 5TM.
    for code in ["2TM", "3TM", "4TM", "5TM"]:
        assert re.match(COMBINED_TEAM_PATTERN, code)
    for code in ["ATL", "TOT", "2TMX", "TM", ""]:
        assert not re.match(COMBINED_TEAM_PATTERN, code)

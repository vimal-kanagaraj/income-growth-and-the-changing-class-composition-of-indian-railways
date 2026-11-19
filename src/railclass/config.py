"""Study constants: years, passenger classes, segments and the policy-regime coding.

Everything that encodes a modelling decision from the paper lives here, so the
analysis modules contain no magic strings or numbers.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Final

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- years
YEARS: Final = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)  # fmt: skip
YEAR_INDEX: Final = MappingProxyType({y: i for i, y in enumerate(YEARS)})

BASELINE_YEARS: Final = YEARS[:5]  # pre-pandemic estimation window
PRE_LOCKDOWN_YEARS: Final = YEARS[:4]  # robustness: baseline without 2019-20
PAX_BASELINE_YEARS: Final = YEARS[1:5]  # class-wise passengers not published for 2015-16
SUSPENSION_YEAR: Final = "2020-21"
RESTRICTED_YEAR: Final = "2021-22"
POST_YEARS: Final = YEARS[7:]  # unreserved coaches restored

REFERENCE_YEAR: Final = "2019-20"  # last pre-pandemic year; index base
LATEST_YEAR: Final = "2024-25"
LONG_DIFF_YEAR: Final = "2010-11"  # Year Book 2018-19 long-run check

# --------------------------------------------------------------------------- classes
AC_CLASSES: Final = (
    "AC First Class", "AC 2 Tier", "AC 3 Tier", "Executive Class", "AC Chair Car",
)  # fmt: skip
NON_AC_CLASSES: Final = (
    "First Class (M/E)", "First Class (Ordinary)",
    "Sleeper (M/E)", "Sleeper (Ordinary)",
    "Second Class (M/E)", "Second Class (Ordinary)",
)  # fmt: skip
SUBURBAN_CLASS: Final = "Suburban"
CLASSES: Final = (*AC_CLASSES, *NON_AC_CLASSES, SUBURBAN_CLASS)

FIRST_CLASSES: Final = ("First Class (M/E)", "First Class (Ordinary)")  # <0.05% of pkm
ORDINARY_CLASSES: Final = ("First Class (Ordinary)", "Sleeper (Ordinary)", "Second Class (Ordinary)")

# --------------------------------------------------------------------------- segments
AC: Final = "AC"
NON_AC: Final = "NonAC"
SUBURBAN: Final = "Suburban"
SEGMENTS: Final = (AC, NON_AC, SUBURBAN)

SEGMENT_OF: Final = MappingProxyType(
    {c: AC for c in AC_CLASSES} | {c: NON_AC for c in NON_AC_CLASSES} | {SUBURBAN_CLASS: SUBURBAN}
)

# --------------------------------------------------------------------------- regimes (Table 1)
# D_FARE: share of the year in which ordinary classes paid mail/express fares.
FARE_RELABEL_SCHEDULE: Final = MappingProxyType({"2021-22": 1.0, "2022-23": 1.0, "2023-24": 0.9})
# Robustness (Section 5.5): relabelling period ends in 2022-23 rather than February 2024.
FARE_RELABEL_SCHEDULE_ALT: Final = MappingProxyType({"2021-22": 1.0, "2022-23": 1.0})

# --------------------------------------------------------------------------- scenarios (Table 8)
PROJECTION_INCOME_GROWTH: Final = 0.05  # real per capita NNI, per year
PROJECTION_POP_GROWTH: Final = 0.01  # per year
PROJECTION_HORIZON_YEARS: Final = 5  # 2024-25 -> 2029-30
STRESS_TEST_AC_ELASTICITY: Final = 2.0  # upper end of Dargay & Gately (1999)

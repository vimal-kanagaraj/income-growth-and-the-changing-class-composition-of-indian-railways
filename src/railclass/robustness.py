"""Robustness checks promised in the Stage 2 plan (Section 5.5)."""

from __future__ import annotations

import pandas as pd

from railclass.benchmark import income_benchmark
from railclass.config import (
    AC,
    FARE_RELABEL_SCHEDULE_ALT,
    LATEST_YEAR,
    NON_AC,
    PAX_BASELINE_YEARS,
    PRE_LOCKDOWN_YEARS,
    SEGMENTS,
    SUBURBAN,
)
from railclass.elasticities import segment_aggregate_fits
from railclass.regime import nonac_class_panel_model


def baseline_excluding_2019_20(seg: pd.DataFrame) -> dict[str, float]:
    """Segment-aggregate fits on 2015-16 to 2018-19, and the latest-year gaps they imply."""
    fits = segment_aggregate_fits(seg, years=PRE_LOCKDOWN_YEARS)
    bm = income_benchmark(seg, fits)
    out = {f"beta_{s}": fits[s].beta for s in SEGMENTS}
    out |= {f"gap_pct_{s}_{LATEST_YEAR}": bm.loc[LATEST_YEAR, f"gap_pct_{s}"] for s in SEGMENTS}
    return out


def alternative_fare_coding(class_panel: pd.DataFrame) -> dict[str, float]:
    """Relabelling period ends in 2022-23 instead of February 2024."""
    return nonac_class_panel_model(class_panel, fare_schedule=FARE_RELABEL_SCHEDULE_ALT)


def passengers_as_outcome(seg: pd.DataFrame) -> dict[str, float]:
    """Originating passengers per person as the dependent variable (baseline 2016-17 to 2019-20)."""
    fits = segment_aggregate_fits(seg, years=PAX_BASELINE_YEARS, outcome="ln_pax")
    bm = income_benchmark(seg, fits, outcome_col="pax")
    return {
        "beta_AC": fits[AC].beta,
        "beta_NonAC": fits[NON_AC].beta,
        "beta_Suburban": fits[SUBURBAN].beta,
        **{f"gap_pct_{s}_{LATEST_YEAR}": bm.loc[LATEST_YEAR, f"gap_pct_{s}"] for s in SEGMENTS},
    }


def run_all(seg: pd.DataFrame, class_panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "baseline_excluding_2019_20": baseline_excluding_2019_20(seg),
        "alternative_fare_coding": alternative_fare_coding(class_panel),
        "passengers_as_outcome": passengers_as_outcome(seg),
    }

"""Construction of the class-year and segment-year analysis panels (Section 3.3-3.4)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from railclass.config import (
    FARE_RELABEL_SCHEDULE,
    ORDINARY_CLASSES,
    POST_YEARS,
    RESTRICTED_YEAR,
    SEGMENT_OF,
    SUSPENSION_YEAR,
    YEAR_INDEX,
    YEARS,
)


def population_mn(national_income: pd.DataFrame) -> pd.Series:
    """Population (million) = NNI at constant prices / per capita NNI at constant prices."""
    return national_income["nni_constant_cr"] * 1e7 / national_income["nni_pc_constant_rs"] / 1e6


def deflator(national_income: pd.DataFrame) -> pd.Series:
    """Implicit NNI deflator, 2011-12 = 100."""
    return 100 * national_income["nni_pc_current_rs"] / national_income["nni_pc_constant_rs"]


def add_regime_indicators(
    df: pd.DataFrame,
    fare_schedule: Mapping[str, float] = FARE_RELABEL_SCHEDULE,
    rail_class_col: str | None = "rail_class",
) -> pd.DataFrame:
    """Attach the Table 1 regime indicators. ``D_FARE`` applies to ordinary-class rows only."""
    out = df.copy()
    out["D_SUSP"] = (out["year"] == SUSPENSION_YEAR).astype(int)
    out["D_RESTR"] = (out["year"] == RESTRICTED_YEAR).astype(int)
    out["POST"] = out["year"].isin(POST_YEARS).astype(int)
    if rail_class_col is not None:
        fare = out["year"].map(fare_schedule).fillna(0.0)
        out["D_FARE"] = np.where(out[rail_class_col].isin(ORDINARY_CLASSES), fare, 0.0)
    return out


def build_class_panel(traffic: pd.DataFrame, national_income: pd.DataFrame) -> pd.DataFrame:
    """Class-by-year panel (12 classes x 10 years) with per-person travel, yields and logs."""
    df = traffic.copy()
    ni = national_income.loc[list(YEARS)]
    df["segment"] = df["rail_class"].map(SEGMENT_OF)
    df["t"] = df["year"].map(YEAR_INDEX)
    df["Y"] = df["year"].map(ni["nni_pc_constant_rs"])
    df["deflator"] = df["year"].map(deflator(ni))
    df["pop_mn"] = df["year"].map(population_mn(ni))
    df["pkm_pc"] = df["pkm_mn"] / df["pop_mn"]
    df["yield_p"] = np.where(df["pkm_mn"] > 0, df["earn_cr"] * 1000 / df["pkm_mn"], np.nan)
    df["real_yield"] = df["yield_p"] * 100 / df["deflator"]
    df = add_regime_indicators(df)
    df["ln_q"] = np.log(df["pkm_pc"].where(df["pkm_mn"] > 0))
    df["ln_Y"] = np.log(df["Y"])
    df["ln_P"] = np.log(df["real_yield"].where(df["real_yield"] > 0))
    return df


def build_segment_panel(class_panel: pd.DataFrame) -> pd.DataFrame:
    """Segment-by-year aggregates: volumes, shares, yields and per-person travel."""
    seg = (
        class_panel.groupby(["year", "segment"])
        .agg(
            pax=("pax_mn", lambda s: s.sum(min_count=1)),
            pkm=("pkm_mn", "sum"),
            earn=("earn_cr", "sum"),
            pop_mn=("pop_mn", "first"),
            Y=("Y", "first"),
            deflator=("deflator", "first"),
        )
        .reset_index()
    )
    for m in ("pax", "pkm", "earn"):
        seg[f"sh_{m}"] = seg.groupby("year")[m].transform(lambda s: 100 * s / s.sum(min_count=1))
    seg["yield_p"] = seg["earn"] * 1000 / seg["pkm"]
    seg["real_yield"] = seg["yield_p"] * 100 / seg["deflator"]
    seg["pkm_pc"] = seg["pkm"] / seg["pop_mn"]
    seg["pax_pc"] = seg["pax"] / seg["pop_mn"]
    seg["ln_q"] = np.log(seg["pkm_pc"])
    seg["ln_pax"] = np.log(seg["pax_pc"])
    seg["ln_Y"] = np.log(seg["Y"])
    return add_regime_indicators(seg, rail_class_col=None)


def segment_wide(seg: pd.DataFrame, value: str = "pkm") -> pd.DataFrame:
    """Year x segment table of one segment-panel column."""
    return seg.pivot(index="year", columns="segment", values=value).loc[list(YEARS)]

"""M3 - income-only benchmark paths (Objective O3; Figures 2-3, eq. 3).

Predicted PKM_st = POP_t * exp(a_s + b_s ln Y_t);  G_t = actual AC share_t - predicted AC share_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from railclass.config import AC, BASELINE_YEARS, LATEST_YEAR, NON_AC, REFERENCE_YEAR, SEGMENTS, SUBURBAN, YEARS
from railclass.elasticities import SegmentFit
from railclass.panel import segment_wide


def _share(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    cols = [prefix + s for s in SEGMENTS]
    return 100 * df[prefix + AC] / df[cols].sum(axis=1)


def income_benchmark(seg: pd.DataFrame, fits: dict[str, SegmentFit], outcome_col: str = "pkm") -> pd.DataFrame:
    """Actual vs income-predicted volumes and AC share by year, plus the baseline linear share trend.

    ``outcome_col`` is ``pkm`` for passenger-km or ``pax`` for passengers, matching the fits.
    """
    actual = segment_wide(seg, outcome_col)
    base = seg.groupby("year")[["ln_Y", "pop_mn"]].first().loc[list(YEARS)]
    bm = pd.concat([base, actual], axis=1)
    for s, f in fits.items():
        bm[f"pred_{s}"] = np.exp(f.predict_ln(bm["ln_Y"])) * bm["pop_mn"]
        bm[f"gap_pct_{s}"] = 100 * (bm[s] / bm[f"pred_{s}"] - 1)
    bm["sh_act"] = _share(bm)
    bm["sh_pred"] = _share(bm, "pred_")
    bm["gap_pp"] = bm["sh_act"] - bm["sh_pred"]
    n = len(BASELINE_YEARS)
    slope, intercept = np.polyfit(np.arange(n), bm["sh_act"].iloc[:n], 1)
    bm["sh_trend"] = intercept + slope * np.arange(len(bm))
    return bm


def baseline_share_band(bm: pd.DataFrame) -> float:
    """Largest absolute AC-share error of the fit over the baseline (pp)."""
    return float(bm.loc[list(BASELINE_YEARS), "gap_pp"].abs().max())


@dataclass(frozen=True)
class RupeeGap:
    """Value of the latest-year gap between actual and income-predicted pkm, at latest-year yields."""

    excess_ac_mn_pkm: float
    shortfall_nonac_mn_pkm: float
    shortfall_suburban_mn_pkm: float
    excess_ac_cr: float
    shortfall_nonac_cr: float
    shortfall_suburban_cr: float
    yield_ac: float
    yield_nonac: float
    yield_suburban: float
    total_earnings_cr: float

    @property
    def excess_ac_net_cr(self) -> float:
        """AC excess revenue net of what the same travellers would have paid in non-AC classes."""
        return self.excess_ac_mn_pkm * (self.yield_ac - self.yield_nonac) / 1000

    @property
    def excess_ac_share_of_earnings(self) -> float:
        return 100 * self.excess_ac_cr / self.total_earnings_cr


def rupee_gap(seg: pd.DataFrame, bm: pd.DataFrame, year: str = LATEST_YEAR) -> RupeeGap:
    latest = seg[seg["year"] == year].set_index("segment")
    gap = {s: bm.loc[year, s] - bm.loc[year, f"pred_{s}"] for s in SEGMENTS}
    yld = latest["yield_p"]
    return RupeeGap(
        excess_ac_mn_pkm=gap[AC],
        shortfall_nonac_mn_pkm=gap[NON_AC],
        shortfall_suburban_mn_pkm=gap[SUBURBAN],
        excess_ac_cr=gap[AC] * yld[AC] / 1000,
        shortfall_nonac_cr=gap[NON_AC] * yld[NON_AC] / 1000,
        shortfall_suburban_cr=gap[SUBURBAN] * yld[SUBURBAN] / 1000,
        yield_ac=yld[AC],
        yield_nonac=yld[NON_AC],
        yield_suburban=yld[SUBURBAN],
        total_earnings_cr=latest["earn"].sum(),
    )


def income_explained(bm: pd.DataFrame, start: str = REFERENCE_YEAR, end: str = LATEST_YEAR) -> dict[str, float]:
    """Fitted (income-only) vs actual change in AC share between two years (pp)."""
    pred = bm.loc[end, "sh_pred"] - bm.loc[start, "sh_pred"]
    act = bm.loc[end, "sh_act"] - bm.loc[start, "sh_act"]
    return {"predicted_rise_pp": pred, "actual_rise_pp": act, "share_explained_pct": 100 * pred / act}


def ac_growth_attribution(
    class_panel: pd.DataFrame, bm: pd.DataFrame, start: str = REFERENCE_YEAR, end: str = LATEST_YEAR
) -> dict[str, float]:
    """How much of the added AC pkm the income path accounts for, and AC 3-tier's share of it."""
    added = bm.loc[end, AC] - bm.loc[start, AC]
    income_added = bm.loc[end, f"pred_{AC}"] - bm.loc[start, f"pred_{AC}"]
    ac3 = class_panel[class_panel["rail_class"] == "AC 3 Tier"].set_index("year")["pkm_mn"]
    return {
        "ac_added_mn_pkm": added,
        "income_path_added_mn_pkm": income_added,
        "income_share_of_added_pct": 100 * income_added / added,
        "ac3_share_of_added_pct": 100 * (ac3[end] - ac3[start]) / added,
    }

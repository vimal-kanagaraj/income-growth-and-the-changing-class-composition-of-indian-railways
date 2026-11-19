"""Elasticity scenarios: anchored 2024-25 projections (Table 6) and 2029-30 planning range (Table 8)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from railclass.benchmark import RupeeGap
from railclass.config import (
    AC,
    LATEST_YEAR,
    NON_AC,
    PROJECTION_HORIZON_YEARS,
    PROJECTION_INCOME_GROWTH,
    PROJECTION_POP_GROWTH,
    REFERENCE_YEAR,
    STRESS_TEST_AC_ELASTICITY,
    SUBURBAN,
)
from railclass.elasticities import SegmentFit
from railclass.panel import segment_wide


@dataclass(frozen=True)
class ElasticitySet:
    ac: float
    nonac: float
    suburban: float

    def as_dict(self) -> dict[str, float]:
        return {AC: self.ac, NON_AC: self.nonac, SUBURBAN: self.suburban}

    def rounded(self, ndigits: int = 2) -> ElasticitySet:
        return ElasticitySet(round(self.ac, ndigits), round(self.nonac, ndigits), round(self.suburban, ndigits))

    def label(self) -> str:
        return f"({self.ac:.2f}, {self.nonac:.2f}, {self.suburban:.2f})"


def elasticity_sets(
    fits: Mapping[str, SegmentFit],
    fits_excl_2019_20: Mapping[str, SegmentFit],
    long_diff: pd.DataFrame,
) -> dict[str, ElasticitySet]:
    """The alternative elasticity sets of Table 6."""
    ld = long_diff["elasticity"]
    return {
        "Baseline segment fit": ElasticitySet(fits[AC].beta, fits[NON_AC].beta, fits[SUBURBAN].beta),
        "Leave-one-out bounds most favourable to income": ElasticitySet(
            fits[AC].loo_max, fits[NON_AC].loo_min, fits[SUBURBAN].loo_min
        ),
        "Long difference 2010-11 to 2019-20": ElasticitySet(
            ld["AC (upper class)"], ld["NonAC (second class)"], ld["Suburban"]
        ),
        "Baseline segment fit excluding 2019-20": ElasticitySet(
            fits_excl_2019_20[AC].beta, fits_excl_2019_20[NON_AC].beta, fits_excl_2019_20[SUBURBAN].beta
        ),
        "AC elasticity of 2.0 (stress test)": ElasticitySet(
            STRESS_TEST_AC_ELASTICITY, fits[NON_AC].beta, fits[SUBURBAN].beta
        ),
    }


def anchored_projection_table(
    seg: pd.DataFrame,
    sets: Mapping[str, ElasticitySet],
    start: str = REFERENCE_YEAR,
    end: str = LATEST_YEAR,
) -> pd.DataFrame:
    """Table 6: project each segment from its *actual* start-year level at actual income and population."""
    pkm = segment_wide(seg, "pkm")
    base = seg.groupby("year")[["Y", "pop_mn"]].first()
    income_ratio = base.loc[end, "Y"] / base.loc[start, "Y"]
    pop_ratio = base.loc[end, "pop_mn"] / base.loc[start, "pop_mn"]
    sh_start = 100 * pkm.loc[start, AC] / pkm.loc[start].sum()
    sh_end = 100 * pkm.loc[end, AC] / pkm.loc[end].sum()
    rows = []
    for name, es in sets.items():
        proj = {s: pkm.loc[start, s] * income_ratio**b * pop_ratio for s, b in es.as_dict().items()}
        share = 100 * proj[AC] / sum(proj.values())
        rows.append(
            {
                "elasticities": name,
                "beta_ac": es.ac,
                "beta_nonac": es.nonac,
                "beta_suburban": es.suburban,
                "pred_ac_share_pct": share,
                "income_rise_pp": share - sh_start,
                "actual_rise_pp": sh_end - sh_start,
                "share_explained_pct": 100 * (share - sh_start) / (sh_end - sh_start),
                "ac_above_income_path_pct": 100 * (pkm.loc[end, AC] / proj[AC] - 1),
            }
        )
    return pd.DataFrame(rows)


def projection_2030_table(
    seg: pd.DataFrame,
    bm: pd.DataFrame,
    sets: Mapping[str, ElasticitySet],
    year: str = LATEST_YEAR,
    income_growth: float = PROJECTION_INCOME_GROWTH,
    pop_growth: float = PROJECTION_POP_GROWTH,
    horizon: int = PROJECTION_HORIZON_YEARS,
) -> pd.DataFrame:
    """Table 8: income-only projection from the actual mix (excess persists) or the income path (excess unwinds)."""
    actual = segment_wide(seg, "pkm").loc[year] / 1000
    income_path = pd.Series({s: bm.loc[year, f"pred_{s}"] / 1000 for s in (AC, NON_AC, SUBURBAN)})
    starts = {"Actual (excess persists)": actual, "Income path (excess unwinds)": income_path}

    def growth(beta: float) -> float:
        return ((1 + income_growth) ** beta * (1 + pop_growth)) ** horizon

    rows = []
    for start_name, base in starts.items():
        for set_name, es in sets.items():
            proj = {s: base[s] * growth(b) for s, b in es.as_dict().items()}
            rows.append(
                {
                    "start": start_name,
                    "elasticities": f"{set_name} {es.label()}",
                    "ac_share_start_pct": 100 * base[AC] / base.sum(),
                    "ac_share_end_pct": 100 * proj[AC] / sum(proj.values()),
                    "ac_bn_pkm": proj[AC],
                    "nonac_bn_pkm": proj[NON_AC],
                    "suburban_bn_pkm": proj[SUBURBAN],
                }
            )
    return pd.DataFrame(rows)


def unwind_scenario(gap: RupeeGap, fraction: float = 0.5) -> dict[str, float]:
    """Revenue lost if ``fraction`` of the AC excess unwinds (Section 7)."""
    gross = fraction * gap.excess_ac_cr
    return {
        "fraction": fraction,
        "gross_loss_cr": gross,
        "gross_loss_pct_of_earnings": 100 * gross / gap.total_earnings_cr,
        "net_loss_cr": fraction * gap.excess_ac_net_cr,
    }

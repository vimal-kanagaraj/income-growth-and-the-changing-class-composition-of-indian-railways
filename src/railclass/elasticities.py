"""M2 - pre-pandemic income elasticities (Objective O2; Tables 5, B1).

Equation (1), segment aggregate:   ln(PKM/POP)_st = a_s + b_s ln Y_t + u_st
Equation (2), pooled with class FE: ln(PKM/POP)_ct = a_c + b_s ln Y_t + theta ln P_ct + e_ct

Income takes only five distinct values in the baseline, so standard errors are
descriptive; stability is judged by leave-one-year-out (LOO) ranges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from railclass.config import (
    AC,
    BASELINE_YEARS,
    CLASSES,
    FIRST_CLASSES,
    LONG_DIFF_YEAR,
    NON_AC,
    REFERENCE_YEAR,
    SEGMENTS,
    SUBURBAN,
)
from railclass.panel import population_mn, segment_wide

FIXED_PRICE_ELASTICITIES = (0.0, -0.5, -1.0)
COMBINED_GROUPS = {
    "Second combined": ("Second Class (M/E)", "Second Class (Ordinary)"),
    "Sleeper combined": ("Sleeper (M/E)", "Sleeper (Ordinary)"),
}


def ols(data: pd.DataFrame, formula: str) -> RegressionResultsWrapper:
    return smf.ols(formula, data).fit()


def leave_one_year_out(data: pd.DataFrame, formula: str, years: Sequence[str], param: str = "ln_Y") -> list[float]:
    """Coefficient on ``param`` re-estimated with each year in ``years`` dropped in turn."""
    return [ols(data[data["year"] != y], formula).params[param] for y in years]


# --------------------------------------------------------------------------- segment aggregate (eq. 1)
@dataclass(frozen=True)
class SegmentFit:
    segment: str
    intercept: float
    beta: float
    se: float
    r2: float
    loo_min: float
    loo_max: float
    residuals: np.ndarray = field(repr=False)

    def predict_ln(self, ln_y: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
        return self.intercept + self.beta * ln_y


def segment_aggregate_fits(
    seg: pd.DataFrame, years: Sequence[str] = BASELINE_YEARS, outcome: str = "ln_q"
) -> dict[str, SegmentFit]:
    """Fit eq. (1) for each segment on ``years``; ``outcome`` may be ``ln_q`` (pkm) or ``ln_pax``."""
    formula = f"{outcome} ~ ln_Y"
    fits = {}
    for s in SEGMENTS:
        d = seg[(seg["segment"] == s) & seg["year"].isin(years)]
        r = ols(d, formula)
        loo = leave_one_year_out(d, formula, years)
        fits[s] = SegmentFit(
            segment=s,
            intercept=r.params["Intercept"],
            beta=r.params["ln_Y"],
            se=r.bse["ln_Y"],
            r2=r.rsquared,
            loo_min=min(loo),
            loo_max=max(loo),
            residuals=r.resid.to_numpy(),
        )
    return fits


def segment_fits_frame(fits: dict[str, SegmentFit]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in f.__dict__.items() if k != "residuals"} for f in fits.values()]).set_index(
        "segment"
    )


# --------------------------------------------------------------------------- pooled with class FE (eq. 2)
def pooled_sample(class_panel: pd.DataFrame, segment: str, years: Sequence[str]) -> pd.DataFrame:
    d = class_panel[(class_panel["segment"] == segment) & class_panel["year"].isin(years) & class_panel["ln_q"].notna()]
    if segment == NON_AC:
        d = d[~d["rail_class"].isin(FIRST_CLASSES)]
    return d


def _fe(d: pd.DataFrame) -> str:
    return " + C(rail_class)" if d["rail_class"].nunique() > 1 else ""


def pooled_elasticities(class_panel: pd.DataFrame, years: Sequence[str] = BASELINE_YEARS) -> pd.DataFrame:
    """Pooled eq. (2) per segment: free and fixed price elasticities, LOO range, and without 2019-20."""
    rows = []
    for s in SEGMENTS:
        d = pooled_sample(class_panel, s, years)
        fe = _fe(d)
        r = ols(d, "ln_q ~ ln_Y" + fe)
        rp = ols(d, "ln_q ~ ln_Y + ln_P" + fe)
        loo = leave_one_year_out(d, "ln_q ~ ln_Y" + fe, years)
        row = {
            "segment": s,
            "n": int(r.nobs),
            "beta": r.params["ln_Y"],
            "se": r.bse["ln_Y"],
            "r2": r.rsquared,
            "beta_with_price": rp.params["ln_Y"],
            "theta": rp.params["ln_P"],
            "theta_se": rp.bse["ln_P"],
            "loo_min": min(loo),
            "loo_max": max(loo),
            "beta_excl_2019_20": ols(d[d["year"] != REFERENCE_YEAR], "ln_q ~ ln_Y" + fe).params["ln_Y"],
        }
        for theta in FIXED_PRICE_ELASTICITIES:
            fixed = ols(d.assign(q_net=d["ln_q"] - theta * d["ln_P"]), "q_net ~ ln_Y" + fe)
            row[f"beta_theta_{theta:+.1f}"] = fixed.params["ln_Y"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


# --------------------------------------------------------------------------- class level (Table B1)
def class_elasticities(class_panel: pd.DataFrame, years: Sequence[str] = BASELINE_YEARS) -> pd.DataFrame:
    """Exploratory class-by-class eq. (1) fits."""
    pkm_ref = class_panel[class_panel["year"] == REFERENCE_YEAR].set_index("rail_class")["pkm_mn"]
    rows = []
    for c in CLASSES:
        d = class_panel[
            (class_panel["rail_class"] == c) & class_panel["year"].isin(years) & class_panel["ln_q"].notna()
        ]
        r = ols(d, "ln_q ~ ln_Y")
        loo = leave_one_year_out(d, "ln_q ~ ln_Y", years)
        rows.append(
            {
                "rail_class": c,
                "share_pkm_2019_20": 100 * pkm_ref[c] / pkm_ref.sum(),
                "beta": r.params["ln_Y"],
                "se": r.bse["ln_Y"],
                "loo_min": min(loo),
                "loo_max": max(loo),
            }
        )
    return pd.DataFrame(rows).set_index("rail_class")


# --------------------------------------------------------------------------- combined ordinary + M/E (Section 5.3)
def combined_class_benchmarks(class_panel: pd.DataFrame, years: Sequence[str] = BASELINE_YEARS) -> pd.DataFrame:
    """Ordinary + Mail/Express series, which relabelling cannot distort, against their income path."""
    out = []
    for name, classes in COMBINED_GROUPS.items():
        d = (
            class_panel[class_panel["rail_class"].isin(classes)]
            .groupby("year", sort=False)
            .agg(pkm=("pkm_mn", "sum"), pop=("pop_mn", "first"), ln_Y=("ln_Y", "first"))
            .reset_index()
        )
        d["ln_q"] = np.log(d["pkm"] / d["pop"])
        r = ols(d[d["year"].isin(years)], "ln_q ~ ln_Y")
        d["pred"] = np.exp(r.params["Intercept"] + r.params["ln_Y"] * d["ln_Y"]) * d["pop"]
        d["gap_pct"] = 100 * (d["pkm"] / d["pred"] - 1)
        d.insert(0, "group", name)
        d["beta"] = r.params["ln_Y"]
        out.append(d[["group", "year", "beta", "pkm", "pred", "gap_pct"]])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- long difference (Table 5, last row)
def long_difference(
    seg: pd.DataFrame, class_panel: pd.DataFrame, national_income: pd.DataFrame, yearbook: pd.Series
) -> pd.DataFrame:
    """Elasticities from 2010-11 to 2019-20 at Year Book broad-class level.

    The Year Book "upper class" equals the AC classes plus First Class, so First
    Class is moved from the non-AC to the AC side for 2019-20.
    """
    pop = population_mn(national_income)
    y = national_income["nni_pc_constant_rs"]
    g = segment_wide(seg, "pkm")
    ref = class_panel[class_panel["year"] == REFERENCE_YEAR].set_index("rail_class")["pkm_mn"]
    fc = ref[list(FIRST_CLASSES)].sum()
    pairs = {
        "AC (upper class)": (yearbook["Upper class"], g.loc[REFERENCE_YEAR, AC] + fc),
        "NonAC (second class)": (
            yearbook["Second Class (M/E)"] + yearbook["Second Class (Ordinary)"],
            g.loc[REFERENCE_YEAR, NON_AC] - fc,
        ),
        "Suburban": (yearbook["Suburban"], g.loc[REFERENCE_YEAR, SUBURBAN]),
    }
    dln_y = np.log(y[REFERENCE_YEAR] / y[LONG_DIFF_YEAR])
    rows = []
    for k, (a0, a1) in pairs.items():
        q0, q1 = a0 / pop[LONG_DIFF_YEAR], a1 / pop[REFERENCE_YEAR]
        rows.append({"group": k, "pkm_pc_2010_11": q0, "pkm_pc_2019_20": q1, "elasticity": np.log(q1 / q0) / dln_y})
    return pd.DataFrame(rows).set_index("group")


def ac_upper_share_2010_11(yearbook: pd.Series) -> float:
    return 100 * yearbook["Upper class"] / yearbook["Total"]


__all__ = [
    "SegmentFit",
    "ac_upper_share_2010_11",
    "class_elasticities",
    "combined_class_benchmarks",
    "leave_one_year_out",
    "long_difference",
    "ols",
    "pooled_elasticities",
    "segment_aggregate_fits",
    "segment_fits_frame",
]

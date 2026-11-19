"""Table C1: tidy regression table for every reported model."""

from __future__ import annotations

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from railclass.config import AC, BASELINE_YEARS, NON_AC, SEGMENTS
from railclass.elasticities import ols, pooled_sample
from railclass.regime import CLASS_PANEL_FORMULA, REGIME_TERMS, SEGMENT_FORMULA, nonac_class_panel_sample


def _tidy(model: str, r: RegressionResultsWrapper, terms: tuple[str, ...]) -> list[dict]:
    return [
        {
            "model": model,
            "term": t,
            "coef": r.params[t],
            "se": r.bse[t],
            "t": r.tvalues[t],
            "n": int(r.nobs),
            "r2": r.rsquared,
        }
        for t in terms
    ]


def regression_table(class_panel: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for s in (AC, NON_AC):
        d = pooled_sample(class_panel, s, BASELINE_YEARS)
        rows += _tidy(f"M2 pooled {s}", ols(d, "ln_q ~ ln_Y + C(rail_class)"), ("ln_Y",))
        rows += _tidy(f"M2 pooled {s} with price", ols(d, "ln_q ~ ln_Y + ln_P + C(rail_class)"), ("ln_Y", "ln_P"))
    for s in SEGMENTS:
        base = seg[(seg["segment"] == s) & seg["year"].isin(BASELINE_YEARS)]
        rows += _tidy(f"M3 segment fit {s}", ols(base, "ln_q ~ ln_Y"), ("Intercept", "ln_Y"))
        rows += _tidy(f"M4 regime {s}", ols(seg[seg["segment"] == s], SEGMENT_FORMULA), REGIME_TERMS)
    rows += _tidy(
        "M4 class panel NonAC",
        ols(nonac_class_panel_sample(class_panel), CLASS_PANEL_FORMULA),
        (*REGIME_TERMS, "D_FARE"),
    )
    return pd.DataFrame(rows)

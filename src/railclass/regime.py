"""M4 - full-period regime model (Objective O3; Section 5.4, Table C1).

ln(PKM/POP)_st = a_s + b_s ln Y_t + g1 D_SUSP_t + g2 D_RESTR_t + g3 POST_t + e_st

With five parameters and ten observations, coefficients are conditional
differences, not causal effects.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from railclass.config import FARE_RELABEL_SCHEDULE, FIRST_CLASSES, NON_AC, SEGMENTS
from railclass.elasticities import ols
from railclass.panel import add_regime_indicators

REGIME_TERMS = ("ln_Y", "D_SUSP", "D_RESTR", "POST")
SEGMENT_FORMULA = "ln_q ~ " + " + ".join(REGIME_TERMS)
CLASS_PANEL_FORMULA = SEGMENT_FORMULA + " + D_FARE + C(rail_class)"


def segment_regime_models(seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SEGMENTS:
        r = ols(seg[seg["segment"] == s], SEGMENT_FORMULA)
        rows.append(
            {
                "segment": s,
                "beta": r.params["ln_Y"],
                "beta_se": r.bse["ln_Y"],
                "g_susp": r.params["D_SUSP"],
                "g_restr": r.params["D_RESTR"],
                "g_post": r.params["POST"],
                "g_post_se": r.bse["POST"],
                "post_pct": 100 * (np.exp(r.params["POST"]) - 1),
                "r2": r.rsquared,
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def nonac_class_panel_sample(class_panel: pd.DataFrame) -> pd.DataFrame:
    return class_panel[
        (class_panel["segment"] == NON_AC)
        & class_panel["ln_q"].notna()
        & ~class_panel["rail_class"].isin(FIRST_CLASSES)
    ]


def nonac_class_panel_model(
    class_panel: pd.DataFrame, fare_schedule: Mapping[str, float] = FARE_RELABEL_SCHEDULE
) -> dict[str, float]:
    """Non-AC class-level panel with D_FARE and class fixed effects (H3)."""
    d = add_regime_indicators(nonac_class_panel_sample(class_panel), fare_schedule=fare_schedule)
    r = ols(d, CLASS_PANEL_FORMULA)
    return {
        "beta": r.params["ln_Y"],
        "g_post": r.params["POST"],
        "d_fare": r.params["D_FARE"],
        "d_fare_se": r.bse["D_FARE"],
        "n": int(r.nobs),
    }

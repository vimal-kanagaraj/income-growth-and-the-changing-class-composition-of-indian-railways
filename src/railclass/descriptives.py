"""M1 - descriptive and decomposition analysis (Objective O1; Tables 3-4, B2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from railclass.config import AC, LATEST_YEAR, NON_AC, REFERENCE_YEAR, SUBURBAN
from railclass.panel import segment_wide

DECOMPOSITION_WINDOWS = (("2015-16", "2019-20"), ("2019-20", "2024-25"), ("2018-19", "2024-25"))


@dataclass(frozen=True)
class ShareDecomposition:
    """Shapley split of a change in the AC share of passenger-km (percentage points)."""

    window: str
    total: float
    from_ac_growth: float
    from_other_travel: float
    ac_pkm_change_pct: float
    other_pkm_change_pct: float


def _share(ac: float, other: float) -> float:
    return 100 * ac / (ac + other)


def shapley_share_decomposition(ac0: float, ac1: float, other0: float, other1: float) -> tuple[float, float, float]:
    """Return (total, AC component, other component); each component averages both orderings."""
    total = _share(ac1, other1) - _share(ac0, other0)
    from_ac = 0.5 * ((_share(ac1, other0) - _share(ac0, other0)) + (_share(ac1, other1) - _share(ac0, other1)))
    return total, from_ac, total - from_ac


def decompose_ac_share(seg: pd.DataFrame, start: str, end: str) -> ShareDecomposition:
    g = segment_wide(seg, "pkm")
    ac0, ac1 = g.loc[start, AC], g.loc[end, AC]
    o0, o1 = g.loc[start, [NON_AC, SUBURBAN]].sum(), g.loc[end, [NON_AC, SUBURBAN]].sum()
    total, from_ac, from_other = shapley_share_decomposition(ac0, ac1, o0, o1)
    return ShareDecomposition(
        window=f"{start}->{end}",
        total=total,
        from_ac_growth=from_ac,
        from_other_travel=from_other,
        ac_pkm_change_pct=100 * (ac1 / ac0 - 1),
        other_pkm_change_pct=100 * (o1 / o0 - 1),
    )


def decomposition_table(seg: pd.DataFrame) -> pd.DataFrame:
    """Table 4."""
    return pd.DataFrame([asdict(decompose_ac_share(seg, a, b)) for a, b in DECOMPOSITION_WINDOWS])


def segment_indicators(
    seg: pd.DataFrame, years: tuple[str, ...] = ("2015-16", REFERENCE_YEAR, LATEST_YEAR)
) -> pd.DataFrame:
    """Table 3: segment volumes, shares, per-person travel and real yields."""
    cols = {
        "pkm": "Passenger-km (mn)",
        "sh_pkm": "Share of passenger-km (%)",
        "sh_earn": "Share of earnings (%)",
        "pkm_pc": "Passenger-km per person",
        "real_yield": "Real yield (2011-12 paise/pkm)",
    }
    t = seg[seg["year"].isin(years)].set_index(["segment", "year"])[list(cols)].rename(columns=cols)
    return t.T


def per_person_index(seg: pd.DataFrame, base_year: str = REFERENCE_YEAR) -> pd.DataFrame:
    """Figure 1: passenger-km per person by segment, base year = 100."""
    w = segment_wide(seg, "pkm_pc")
    return 100 * w / w.loc[base_year]


def class_change_table(class_panel: pd.DataFrame, start: str = REFERENCE_YEAR, end: str = LATEST_YEAR) -> pd.DataFrame:
    """Table B2: passenger-km, shares and real-yield change by class between two years."""
    p = class_panel.pivot(index="rail_class", columns="year", values="pkm_mn")
    ry = class_panel.pivot(index="rail_class", columns="year", values="real_yield")
    y = class_panel.pivot(index="rail_class", columns="year", values="yield_p")
    return pd.DataFrame(
        {
            f"pkm_{start}_bn": p[start] / 1000,
            f"pkm_{end}_bn": p[end] / 1000,
            f"share_{start}_pct": 100 * p[start] / p[start].sum(),
            f"share_{end}_pct": 100 * p[end] / p[end].sum(),
            "pkm_change_pct": 100 * (p[end] / p[start] - 1),
            f"yield_{end}_paise": y[end],
            "real_yield_change_pct": 100 * (ry[end] / ry[start] - 1),
        }
    ).loc[class_panel["rail_class"].unique()]


def relative_yield(class_panel: pd.DataFrame, num: str = "AC 3 Tier", den: str = "Sleeper (M/E)") -> pd.Series:
    """Real yield of one class relative to another, by year (Section 5.4)."""
    p = class_panel.pivot(index="year", columns="rail_class", values="real_yield")
    return p[num] / p[den]

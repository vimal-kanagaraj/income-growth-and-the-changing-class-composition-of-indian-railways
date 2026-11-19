"""Integrity checks on the transcribed source data (Section 3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from railclass.config import CLASSES, YEARS
from railclass.io import RawData


def test_panel_shape(raw: RawData) -> None:
    assert len(raw.class_traffic) == len(YEARS) * len(CLASSES) == 120


def test_passengers_missing_only_for_2015_16(raw: RawData) -> None:
    missing = raw.class_traffic[raw.class_traffic["pax_mn"].isna()]
    assert set(missing["year"]) == {"2015-16"}
    assert len(missing) == len(CLASSES)


@pytest.mark.parametrize(("col", "tol"), [("pax_mn", 0.00025), ("pkm_mn", 0.00015)])
def test_class_rows_sum_to_printed_totals(raw: RawData, col: str, tol: float) -> None:
    """Class rows equal printed totals, except 2020-21 where they exceed them by 0.02% (pax) / 0.01% (pkm)."""
    sums = raw.class_traffic.groupby("year")[col].sum(min_count=1)
    printed = raw.printed_totals.set_index("year")[col]
    for year in YEARS:
        if np.isnan(printed[year]):
            continue
        rel = sums[year] / printed[year] - 1
        if year == "2020-21":
            assert 0 < rel <= tol
        else:
            assert abs(rel) < 1e-6, (year, rel)


def test_earnings_sum_to_printed_totals(raw: RawData) -> None:
    sums = raw.class_traffic.groupby("year")["earn_cr"].sum()
    printed = raw.printed_totals.set_index("year")["earn_cr"]
    np.testing.assert_allclose(sums.loc[list(YEARS)], printed.loc[list(YEARS)], rtol=1e-4)


def test_yearbook_total_consistent(raw: RawData) -> None:
    yb = raw.yearbook_2010_11
    assert yb.drop("Total").sum() == pytest.approx(yb["Total"], rel=0.001)

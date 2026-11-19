from __future__ import annotations

import pytest

from railclass.descriptives import shapley_share_decomposition


def test_shapley_components_sum_to_total() -> None:
    total, a, o = shapley_share_decomposition(100, 250, 900, 820)
    assert a + o == pytest.approx(total)


def test_shapley_no_change_in_other_attributes_all_to_ac() -> None:
    total, a, o = shapley_share_decomposition(100, 200, 900, 900)
    assert a == pytest.approx(total)
    assert o == pytest.approx(0)


def test_shapley_no_change_in_ac_attributes_all_to_other() -> None:
    total, a, o = shapley_share_decomposition(100, 100, 900, 400)
    assert a == pytest.approx(0)
    assert o == pytest.approx(total)

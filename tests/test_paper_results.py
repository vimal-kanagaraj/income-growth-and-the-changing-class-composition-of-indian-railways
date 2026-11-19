"""Published-result tests: pin every number the paper reports from this pipeline.

A failure means a data or code change has altered a reported result; if the change is
intended, update this test and the paper together. Tolerances follow the precision at
which each number is reported.
"""

from __future__ import annotations

import pytest

from railclass.benchmark import baseline_share_band
from railclass.config import AC, NON_AC, SUBURBAN
from railclass.pipeline import Results

approx = pytest.approx


# --------------------------------------------------------------- Section 4
def test_table3_shares(results: Results) -> None:
    seg = results.segment_panel.set_index(["year", "segment"])
    assert seg.loc[("2019-20", AC), "sh_pkm"] == approx(12.5, abs=0.05)
    assert seg.loc[("2024-25", AC), "sh_pkm"] == approx(23.1, abs=0.05)
    assert seg.loc[("2024-25", AC), "sh_earn"] == approx(54.2, abs=0.05)
    assert seg.loc[("2024-25", AC), "pkm_pc"] == approx(185.8, abs=0.05)
    assert seg.loc[("2015-16", NON_AC), "real_yield"] == approx(25.7, abs=0.05)


def test_table4_decomposition(results: Results) -> None:
    d = results.decomposition.set_index("window")
    expected = {
        "2015-16->2019-20": (3.33, 2.16, 1.17, 25.3, -11.4),
        "2019-20->2024-25": (10.60, 9.82, 0.78, 99.2, -5.3),
        "2018-19->2024-25": (12.19, 9.88, 2.31, 107.2, -15.5),
    }
    for window, (tot, ac, other, ac_pct, other_pct) in expected.items():
        row = d.loc[window]
        assert row.total == approx(tot, abs=0.005)
        assert row.from_ac_growth == approx(ac, abs=0.005)
        assert row.from_other_travel == approx(other, abs=0.005)
        assert row.ac_pkm_change_pct == approx(ac_pct, abs=0.05)
        assert row.other_pkm_change_pct == approx(other_pct, abs=0.05)


# --------------------------------------------------------------- Section 5.1 (Table 5)
def test_table5_segment_aggregate(results: Results) -> None:
    f = results.segment_fits
    assert (f[AC].beta, f[NON_AC].beta, f[SUBURBAN].beta) == approx((0.94, -0.68, -0.38), abs=0.005)
    assert (f[AC].loo_min, f[AC].loo_max) == approx((0.83, 1.20), abs=0.005)
    assert (f[NON_AC].loo_min, f[NON_AC].loo_max) == approx((-1.10, -0.20), abs=0.005)


def test_table5_pooled(results: Results) -> None:
    p = results.pooled
    assert p.loc[AC, "beta"] == approx(0.66, abs=0.005)
    assert p.loc[NON_AC, "beta"] == approx(-1.41, abs=0.005)
    assert p.loc[AC, "beta_theta_-0.5"] == approx(0.47, abs=0.005)
    assert p.loc[AC, "beta_theta_-1.0"] == approx(0.29, abs=0.005)
    assert p.loc[AC, "beta_with_price"] == approx(0.26, abs=0.005)
    assert p.loc[NON_AC, "theta"] == approx(-5.0, abs=0.005)


def test_table5_long_difference(results: Results) -> None:
    ld = results.long_difference["elasticity"]
    assert ld.to_numpy() == approx([1.50, -0.29, -0.29], abs=0.005)


def test_tableB1_ac3_luxury(results: Results) -> None:
    ac3 = results.class_elasticities.loc["AC 3 Tier"]
    assert (ac3.beta, ac3.loo_min, ac3.loo_max) == approx((1.35, 1.05, 1.90), abs=0.005)


# --------------------------------------------------------------- Section 5.2
def test_benchmark_shares_and_gaps(results: Results) -> None:
    bm = results.benchmark
    assert bm.loc["2019-20", "sh_pred"] == approx(11.6, abs=0.05)
    assert bm.loc["2024-25", "sh_pred"] == approx(15.1, abs=0.05)
    assert bm.loc["2024-25", "sh_trend"] == approx(16.0, abs=0.05)
    assert baseline_share_band(bm) == approx(0.9, abs=0.05)
    nonac = bm.loc[["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"], "gap_pct_NonAC"]
    assert nonac.round().tolist() == [-83, -53, -23, -11, -3]
    ac = bm.loc[["2022-23", "2023-24", "2024-25"], "gap_pct_AC"]
    assert ac.round().tolist() == [44, 54, 61]
    assert bm.loc["2024-25", "gap_pct_Suburban"] == approx(-13, abs=0.5)


def test_rupee_gap(results: Results) -> None:
    g = results.rupee_gap
    assert g.excess_ac_mn_pkm == approx(98_759, abs=1)
    assert g.excess_ac_cr == approx(15_422, abs=1)
    assert g.excess_ac_share_of_earnings == approx(20, abs=0.5)
    assert g.excess_ac_net_cr == approx(11_300, abs=50)
    assert -g.shortfall_nonac_cr == approx(1_014, abs=1)
    assert -g.shortfall_suburban_cr == approx(446, abs=1)


def test_table6_anchored_projections(results: Results) -> None:
    t = results.anchored_projections.set_index("elasticities")
    expected = {  # predicted share, % explained, AC above path
        "Baseline segment fit": (16.3, 35, 58),
        "Leave-one-out bounds most favourable to income": (18.0, 52, 50),
        "Long difference 2010-11 to 2019-20": (16.8, 41, 42),
        "Baseline segment fit excluding 2019-20": (14.8, 22, 61),
        "AC elasticity of 2.0 (stress test)": (19.3, 64, 29),
    }
    for name, (share, explained, above) in expected.items():
        row = t.loc[name]
        assert row.pred_ac_share_pct == approx(share, abs=0.05)
        assert row.share_explained_pct == approx(explained, abs=0.5)
        assert row.ac_above_income_path_pct == approx(above, abs=0.5)


def test_ac_growth_attribution(results: Results) -> None:
    assert results.ac_growth["ac3_share_of_added_pct"] == approx(77, abs=0.5)


# --------------------------------------------------------------- Sections 5.3-5.4
def test_regime_models(results: Results) -> None:
    r = results.regime_models
    assert r.loc[AC, "g_post"] == approx(0.378, abs=0.0005)
    assert r.loc[AC, "g_post_se"] == approx(0.049, abs=0.0005)
    assert r.loc[AC, "post_pct"] == approx(46, abs=0.5)
    assert r.loc[NON_AC, "post_pct"] == approx(-21, abs=0.5)
    assert r.loc[SUBURBAN, "post_pct"] == approx(-20, abs=0.5)


def test_fare_relabelling(results: Results) -> None:
    m = results.nonac_class_panel
    assert (m["d_fare"], m["d_fare_se"], m["n"]) == approx((-1.844, 0.451, 39), abs=0.0005)
    second = results.combined_classes.query("group == 'Second combined' and year == '2024-25'")
    assert second["gap_pct"].iloc[0] == approx(-2, abs=0.5)


def test_relative_yield(results: Results) -> None:
    ry = results.relative_yield
    assert 100 * (ry["2024-25"] / ry["2019-20"] - 1) == approx(-4.4, abs=0.05)


# --------------------------------------------------------------- Section 5.5
def test_robustness(results: Results) -> None:
    rb = results.robustness
    ex = rb["baseline_excluding_2019_20"]
    assert (ex["beta_AC"], ex["beta_NonAC"], ex["beta_Suburban"]) == approx((0.83, -0.20, -0.10), abs=0.005)
    assert ex["gap_pct_NonAC_2024-25"] == approx(-18, abs=0.5)
    alt = rb["alternative_fare_coding"]
    assert (alt["d_fare"], alt["d_fare_se"]) == approx((-1.54, 0.54), abs=0.005)
    pax = rb["passengers_as_outcome"]
    assert pax["beta_AC"] == approx(1.56, abs=0.005)
    assert pax["gap_pct_AC_2024-25"] == approx(40, abs=0.5)
    assert pax["gap_pct_NonAC_2024-25"] == approx(-15, abs=0.5)
    assert results.pooled.loc[AC, "beta_excl_2019_20"] == approx(0.86, abs=0.005)


# --------------------------------------------------------------- Section 7 (Table 8)
def test_table8_projection_2030(results: Results) -> None:
    shares = results.projection_2030["ac_share_end_pct"].round(1).tolist()
    assert shares == [30.6, 31.7, 20.7, 21.6]
    assert results.projection_2030["ac_bn_pkm"].round().tolist() == [346, 396, 215, 247]
    u = results.unwind
    assert u["gross_loss_cr"] == approx(7_700, abs=50)
    assert u["gross_loss_pct_of_earnings"] == approx(10, abs=0.5)
    assert u["net_loss_cr"] == approx(5_600, abs=50)

"""Figures 1-4 and B1 of the paper, drawn from the computed panel and benchmark series."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from railclass.config import (  # noqa: E402
    AC,
    REFERENCE_YEAR,
    RESTRICTED_YEAR,
    SEGMENTS,
    SUSPENSION_YEAR,
    YEARS,
)
from railclass.panel import segment_wide  # noqa: E402

SEGMENT_LABELS = {"AC": "AC", "NonAC": "Non-AC long-distance", "Suburban": "Suburban"}
SEGMENT_COLORS = {"AC": "#1f5fa8", "NonAC": "#c0612b", "Suburban": "#4d8f4a"}
POLICY_EVENTS = (  # (financial year, months after 1 April, label) - Figure 4
    ("2020-21", 2, "Fully reserved trains\n(1 Jun 2020)"),
    ("2021-22", 11, "Unreserved coaches\nrestored (28 Feb 2022)"),
    ("2023-24", 10, "Ordinary fares\nrestored* (Feb 2024)"),
)


def _shade_disruption(ax: Axes, years: list[str]) -> None:
    i0, i1 = years.index(SUSPENSION_YEAR), years.index(RESTRICTED_YEAR)
    ax.axvspan(i0 - 0.5, i1 + 0.5, color="0.9", zorder=0)


def _finish(ax: Axes, years: list[str], ylabel: str) -> None:
    ax.set_xticks(range(len(years)), years, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


def figure1_per_person_index(seg: pd.DataFrame) -> plt.Figure:
    w = segment_wide(seg, "pkm_pc")
    idx = 100 * w / w.loc[REFERENCE_YEAR]
    years = list(YEARS)
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade_disruption(ax, years)
    for s in SEGMENTS:
        ax.plot(range(len(years)), idx[s].to_numpy(), marker="o", color=SEGMENT_COLORS[s], label=SEGMENT_LABELS[s])
    ax.axhline(100, color="0.5", lw=0.8, ls=":")
    _finish(ax, years, f"Passenger-km per person ({REFERENCE_YEAR} = 100)")
    return fig


def figure2_ac_share(bm: pd.DataFrame) -> plt.Figure:
    years = list(bm.index)
    x = range(len(years))
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade_disruption(ax, years)
    ax.plot(x, bm["sh_act"], marker="o", color=SEGMENT_COLORS[AC], label="Actual")
    ax.plot(x, bm["sh_pred"], marker="s", ls="--", color="0.25", label="Income-predicted")
    ax.plot(x, bm["sh_trend"], ls=":", color="0.45", label="Pre-pandemic trend")
    _finish(ax, years, "AC share of passenger-km (%)")
    return fig


def figure3_gap_by_segment(bm: pd.DataFrame) -> plt.Figure:
    years = [y for y in bm.index if y > REFERENCE_YEAR]
    width = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, s in enumerate(SEGMENTS):
        xs = [j + (i - 1) * width for j in range(len(years))]
        ax.bar(xs, bm.loc[years, f"gap_pct_{s}"], width, color=SEGMENT_COLORS[s], label=SEGMENT_LABELS[s])
    ax.axhline(0, color="0.3", lw=0.8)
    _finish(ax, years, "Actual vs income-predicted passenger-km (%)")
    return fig


def figure4_second_class(class_panel: pd.DataFrame) -> plt.Figure:
    p = class_panel.pivot(index="year", columns="rail_class", values="pax_mn").dropna(how="all")
    years = list(p.index)
    x = range(len(years))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, p["Second Class (Ordinary)"], marker="o", color="#c0612b", label="Second Class (Ordinary)")
    ax.plot(x, p["Second Class (M/E)"], marker="s", color="#1f5fa8", label="Second Class (Mail/Express)")
    top = p[["Second Class (Ordinary)", "Second Class (M/E)"]].to_numpy().max()
    for k, (year, months, text) in enumerate(POLICY_EVENTS):
        if year in years:
            # each financial-year point sits mid-year; place the event at its month
            xi = years.index(year) - 0.5 + months / 12
            ax.axvline(xi, color="0.6", lw=0.8, ls="--")
            ax.text(xi + 0.05, top * (1.0 - 0.12 * k), text, fontsize=7, ha="left", va="top")
    ax.text(0, -0.2, "* press report", transform=ax.transAxes, fontsize=7)
    _finish(ax, years, "Originating passengers (million)")
    return fig


def figureB1_ac_shares(seg: pd.DataFrame) -> plt.Figure:
    years = list(YEARS)
    x = range(len(years))
    pkm = segment_wide(seg, "sh_pkm")[AC]
    earn = segment_wide(seg, "sh_earn")[AC]
    fig, ax = plt.subplots(figsize=(7, 4))
    _shade_disruption(ax, years)
    ax.plot(x, pkm, marker="o", color=SEGMENT_COLORS[AC], label="Share of passenger-km")
    ax.plot(x, earn, marker="s", color="#7a3e9d", label="Share of passenger earnings")
    _finish(ax, years, "AC classes' share (%)")
    return fig


def save_all(seg: pd.DataFrame, class_panel: pd.DataFrame, bm: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "figure1_per_person_index.png": figure1_per_person_index(seg),
        "figure2_ac_share.png": figure2_ac_share(bm),
        "figure3_gap_by_segment.png": figure3_gap_by_segment(bm),
        "figure4_second_class.png": figure4_second_class(class_panel),
        "figureB1_ac_shares.png": figureB1_ac_shares(seg),
    }
    paths = []
    for name, fig in figs.items():
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


__all__ = [
    "figure1_per_person_index",
    "figure2_ac_share",
    "figure3_gap_by_segment",
    "figure4_second_class",
    "figureB1_ac_shares",
    "save_all",
]

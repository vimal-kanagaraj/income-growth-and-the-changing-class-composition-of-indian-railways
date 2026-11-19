"""Loaders for the transcribed source data in ``data/raw``.

Each loader validates the shape of its file so that a transcription slip fails
loudly here rather than silently shifting an estimate downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from railclass.config import CLASSES, DEFAULT_DATA_DIR, LONG_DIFF_YEAR, YEARS

CLASS_TRAFFIC_FILE = "class_traffic.csv"
PRINTED_TOTALS_FILE = "printed_totals.csv"
NATIONAL_INCOME_FILE = "national_income.csv"
YEARBOOK_FILE = "yearbook_2010_11.csv"


@dataclass(frozen=True)
class RawData:
    """All source tables used by the study."""

    class_traffic: pd.DataFrame  # year, rail_class, pax_mn, pkm_mn, earn_cr
    printed_totals: pd.DataFrame  # year, pax_mn, pkm_mn, earn_cr (budget grand totals)
    national_income: pd.DataFrame  # year, nni_pc_constant_rs, nni_pc_current_rs, nni_constant_cr
    yearbook_2010_11: pd.Series  # broad-class pkm (mn) for 2010-11, indexed by group


def load_class_traffic(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / CLASS_TRAFFIC_FILE, dtype={"year": str})
    df = df.rename(columns={"class": "rail_class"})
    expected = pd.MultiIndex.from_product([YEARS, CLASSES], names=["year", "rail_class"])
    actual = pd.MultiIndex.from_frame(df[["year", "rail_class"]])
    if not actual.equals(expected):
        missing = expected.difference(actual).tolist()
        extra = actual.difference(expected).tolist()
        raise ValueError(f"{CLASS_TRAFFIC_FILE}: unexpected year/class rows; missing={missing} extra={extra}")
    return df


def load_printed_totals(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / PRINTED_TOTALS_FILE, dtype={"year": str})
    if tuple(df["year"]) != YEARS:
        raise ValueError(f"{PRINTED_TOTALS_FILE}: years must be {YEARS}")
    return df


def load_national_income(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / NATIONAL_INCOME_FILE, dtype={"year": str}).set_index("year")
    missing = {*YEARS, LONG_DIFF_YEAR} - set(df.index)
    if missing:
        raise ValueError(f"{NATIONAL_INCOME_FILE}: missing years {sorted(missing)}")
    return df


def load_yearbook(data_dir: Path = DEFAULT_DATA_DIR) -> pd.Series:
    return pd.read_csv(data_dir / YEARBOOK_FILE).set_index("group")["pkm_mn"]


def load_all(data_dir: Path = DEFAULT_DATA_DIR) -> RawData:
    return RawData(
        class_traffic=load_class_traffic(data_dir),
        printed_totals=load_printed_totals(data_dir),
        national_income=load_national_income(data_dir),
        yearbook_2010_11=load_yearbook(data_dir),
    )

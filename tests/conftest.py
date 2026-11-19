from __future__ import annotations

import pytest

from railclass.io import RawData, load_all
from railclass.pipeline import Results, run_analysis


@pytest.fixture(scope="session")
def raw() -> RawData:
    return load_all()


@pytest.fixture(scope="session")
def results(raw: RawData) -> Results:
    return run_analysis(raw)

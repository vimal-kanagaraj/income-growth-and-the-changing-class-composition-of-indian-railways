"""End-to-end analysis: load data -> build panels -> M1-M4 -> scenarios -> robustness -> outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from railclass import benchmark, descriptives, elasticities, regime, robustness, scenarios, tables
from railclass.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, PRE_LOCKDOWN_YEARS
from railclass.io import RawData, load_all
from railclass.panel import build_class_panel, build_segment_panel


@dataclass
class Results:
    class_panel: pd.DataFrame
    segment_panel: pd.DataFrame
    decomposition: pd.DataFrame  # Table 4
    segment_indicators: pd.DataFrame  # Table 3
    class_changes: pd.DataFrame  # Table B2
    segment_fits: dict[str, elasticities.SegmentFit]  # Table 5 row 1
    segment_fits_excl_2019_20: dict[str, elasticities.SegmentFit]
    pooled: pd.DataFrame  # Table 5 rows 2-4
    class_elasticities: pd.DataFrame  # Table B1
    combined_classes: pd.DataFrame  # Section 5.3
    long_difference: pd.DataFrame  # Table 5 row 5
    benchmark: pd.DataFrame  # Figures 2-3
    rupee_gap: benchmark.RupeeGap
    income_explained: dict[str, float]
    ac_growth: dict[str, float]
    regime_models: pd.DataFrame  # Section 5.4
    nonac_class_panel: dict[str, float]
    relative_yield: pd.Series
    anchored_projections: pd.DataFrame  # Table 6
    projection_2030: pd.DataFrame  # Table 8
    unwind: dict[str, float]
    robustness: dict[str, dict[str, float]]  # Section 5.5
    regression_table: pd.DataFrame  # Table C1

    def summary(self) -> dict:
        """JSON-serialisable headline numbers."""
        return {
            "decomposition": self.decomposition.to_dict(orient="records"),
            "segment_fits": elasticities.segment_fits_frame(self.segment_fits).to_dict(orient="index"),
            "segment_fits_excl_2019_20": elasticities.segment_fits_frame(self.segment_fits_excl_2019_20).to_dict(
                orient="index"
            ),
            "pooled": self.pooled.to_dict(orient="index"),
            "long_difference": self.long_difference.to_dict(orient="index"),
            "baseline_share_band_pp": benchmark.baseline_share_band(self.benchmark),
            "rupee_gap": asdict(self.rupee_gap)
            | {
                "excess_ac_net_cr": self.rupee_gap.excess_ac_net_cr,
                "excess_ac_share_of_earnings": self.rupee_gap.excess_ac_share_of_earnings,
            },
            "income_explained": self.income_explained,
            "ac_growth": self.ac_growth,
            "regime_models": self.regime_models.to_dict(orient="index"),
            "nonac_class_panel": self.nonac_class_panel,
            "anchored_projections": self.anchored_projections.to_dict(orient="records"),
            "projection_2030": self.projection_2030.to_dict(orient="records"),
            "unwind": self.unwind,
            "robustness": self.robustness,
        }


def run_analysis(raw: RawData) -> Results:
    panel = build_class_panel(raw.class_traffic, raw.national_income)
    seg = build_segment_panel(panel)

    fits = elasticities.segment_aggregate_fits(seg)
    fits_ex = elasticities.segment_aggregate_fits(seg, years=PRE_LOCKDOWN_YEARS)
    ld = elasticities.long_difference(seg, panel, raw.national_income, raw.yearbook_2010_11)

    bm = benchmark.income_benchmark(seg, fits)
    gap = benchmark.rupee_gap(seg, bm)

    sets = scenarios.elasticity_sets(fits, fits_ex, ld)
    planning_sets = {
        "Baseline": sets["Baseline segment fit"].rounded(),
        "Long difference": sets["Long difference 2010-11 to 2019-20"].rounded(),
    }

    return Results(
        class_panel=panel,
        segment_panel=seg,
        decomposition=descriptives.decomposition_table(seg),
        segment_indicators=descriptives.segment_indicators(seg),
        class_changes=descriptives.class_change_table(panel),
        segment_fits=fits,
        segment_fits_excl_2019_20=fits_ex,
        pooled=elasticities.pooled_elasticities(panel),
        class_elasticities=elasticities.class_elasticities(panel),
        combined_classes=elasticities.combined_class_benchmarks(panel),
        long_difference=ld,
        benchmark=bm,
        rupee_gap=gap,
        income_explained=benchmark.income_explained(bm),
        ac_growth=benchmark.ac_growth_attribution(panel, bm),
        regime_models=regime.segment_regime_models(seg),
        nonac_class_panel=regime.nonac_class_panel_model(panel),
        relative_yield=descriptives.relative_yield(panel),
        anchored_projections=scenarios.anchored_projection_table(seg, sets),
        projection_2030=scenarios.projection_2030_table(seg, bm, planning_sets),
        unwind=scenarios.unwind_scenario(gap),
        robustness=robustness.run_all(seg, panel),
        regression_table=tables.regression_table(panel, seg),
    )


def write_outputs(results: Results, out_dir: Path, figures: bool = True) -> list[Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "panel.csv": results.class_panel,
        "segment_year.csv": results.segment_panel,
        "table3_segment_indicators.csv": results.segment_indicators,
        "table4_decomposition.csv": results.decomposition,
        "table5_pooled_elasticities.csv": results.pooled,
        "table5_long_difference.csv": results.long_difference,
        "table6_anchored_projections.csv": results.anchored_projections,
        "table8_projection_2030.csv": results.projection_2030,
        "tableB1_class_elasticities.csv": results.class_elasticities,
        "tableB2_class_changes.csv": results.class_changes,
        "tableC1_regressions.csv": results.regression_table,
        "benchmark.csv": results.benchmark,
        "combined_classes.csv": results.combined_classes,
        "segment_fits.csv": elasticities.segment_fits_frame(results.segment_fits),
        "regime_models.csv": results.regime_models,
    }
    written = []
    for name, df in frames.items():
        path = tables_dir / name
        df.to_csv(path, index=not isinstance(df.index, pd.RangeIndex))
        written.append(path)
    summary = out_dir / "results.json"
    summary.write_text(json.dumps(results.summary(), indent=2, default=float))
    written.append(summary)
    if figures:
        from railclass.figures import save_all

        written += save_all(results.segment_panel, results.class_panel, results.benchmark, out_dir / "figures")
    return written


def run(data_dir: Path = DEFAULT_DATA_DIR, out_dir: Path = DEFAULT_OUTPUT_DIR, figures: bool = True) -> Results:
    results = run_analysis(load_all(data_dir))
    write_outputs(results, out_dir, figures=figures)
    return results

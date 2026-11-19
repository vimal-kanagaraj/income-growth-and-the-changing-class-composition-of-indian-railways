# railclass

Analysis code for the DA241O Managerial Economics term paper
**"Income Growth and the Changing Class Composition of Indian Railways, 2015-16 to 2024-25"**
(The Cartel, Group 1).

This repository is where the paper's analysis was developed. It builds the class-by-year
panel from the Union Budget railway statements, estimates the models, and generates the
tables and figures that appear in the paper.

Between 2019-20 and 2024-25 the AC classes' share of Indian Railways passenger-km
rose from 12.5% to 23.1%. The code estimates pre-pandemic income elasticities by
segment, projects the path income alone would have produced, and tests the
alternative explanation that unreserved services were suspended and slowly restored.

## Quick start

```bash
make install      # creates .venv and installs the package with dev tools
make run          # generates the paper's tables, figures and results.json in outputs/
make test         # checks data integrity and pins the published results
```

Without `make`:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHONPATH=src .venv/bin/python -m railclass --output-dir outputs
PYTHONPATH=src .venv/bin/pytest
```

From Python:

```python
from railclass import run_analysis
from railclass.io import load_all

r = run_analysis(load_all())
r.segment_fits["AC"].beta          # 0.94
r.anchored_projections             # Table 6
```

## From code to paper

`make run` writes every table and figure used in the paper to `outputs/` (git-ignored).
These files were placed into the paper; rerun `make run` after any change to the data
or code and update the paper from the new outputs.

| Paper element | Output file | Function |
| --- | --- | --- |
| Table 1 regime coding | (inputs to all models) | `config.FARE_RELABEL_SCHEDULE`, `panel.add_regime_indicators` |
| Table 3 segment indicators | `tables/table3_segment_indicators.csv` | `descriptives.segment_indicators` |
| Table 4 decomposition | `tables/table4_decomposition.csv` | `descriptives.decomposition_table` |
| Table 5 elasticities | `tables/table5_pooled_elasticities.csv`, `tables/segment_fits.csv`, `tables/table5_long_difference.csv` | `elasticities.segment_aggregate_fits`, `pooled_elasticities`, `long_difference` |
| Table 6 alternative elasticities | `tables/table6_anchored_projections.csv` | `scenarios.anchored_projection_table` |
| Table 7 hypotheses | assembled from Tables 5-6 and Sections 5.2-5.5 | — |
| Table 8 projection to 2029-30 | `tables/table8_projection_2030.csv` | `scenarios.projection_2030_table` |
| Table B1 class elasticities | `tables/tableB1_class_elasticities.csv` | `elasticities.class_elasticities` |
| Table B2 class changes | `tables/tableB2_class_changes.csv` | `descriptives.class_change_table` |
| Table C1 regressions | `tables/tableC1_regressions.csv` | `tables.regression_table` |
| Figure 1 per-person index | `figures/figure1_per_person_index.png` | `figures.figure1_per_person_index` |
| Figure 2 AC share vs income path | `figures/figure2_ac_share.png` | `figures.figure2_ac_share` |
| Figure 3 gap by segment | `figures/figure3_gap_by_segment.png` | `figures.figure3_gap_by_segment` |
| Figure 4 second-class passengers | `figures/figure4_second_class.png` | `figures.figure4_second_class` |
| Figure B1 AC shares | `figures/figureB1_ac_shares.png` | `figures.figureB1_ac_shares` |
| Section 5.2 rupee value of the gap | `results.json` → `rupee_gap` | `benchmark.rupee_gap` |
| Section 5.3 relabelling | `tables/combined_classes.csv`, `results.json` → `nonac_class_panel` | `elasticities.combined_class_benchmarks`, `regime.nonac_class_panel_model` |
| Section 5.4 regime model | `tables/regime_models.csv` | `regime.segment_regime_models` |
| Section 5.5 robustness | `results.json` → `robustness` | `robustness.run_all` |
| Section 7 revenue scenario | `results.json` → `unwind` | `scenarios.unwind_scenario` |

## Repository layout

```
railclass/
├── data/
│   ├── raw/                  # transcribed source data (read-only), see data/README.md
│   └── README.md             # sources and Union Budget file list
├── src/railclass/
│   ├── config.py             # years, classes, segments, regime coding, scenario assumptions
│   ├── io.py                 # validated CSV loaders
│   ├── panel.py              # class-year and segment-year panels, regime indicators
│   ├── descriptives.py       # M1: Tables 3, 4, B2 (Shapley decomposition)
│   ├── elasticities.py       # M2: Table 5, Table B1, long difference, combined classes
│   ├── benchmark.py          # M3: income-only paths, Figures 2-3, rupee value of the gap
│   ├── regime.py             # M4: full-period regime models, D_FARE class panel
│   ├── scenarios.py          # Table 6 (alternative elasticities) and Table 8 (2029-30)
│   ├── robustness.py         # Section 5.5 checks
│   ├── tables.py             # Table C1 regression table
│   ├── figures.py            # Figures 1-4 and B1
│   ├── pipeline.py           # orchestration and output writing
│   └── cli.py                # `railclass` command
├── tests/                    # data integrity, unit tests and published-result checks
├── reference/prototype/      # first-pass scripts (excerpted in the paper's Appendix C)
├── pyproject.toml
└── Makefile
```

## Tests

`tests/test_data.py` checks the transcribed data against the grand totals printed in the
budget statements. `tests/test_paper_results.py` pins the published results, so any change
to the data or code that would alter a number reported in the paper fails a test. If such a
change is intended, update the test and the paper together.

## Methodological caveats

These apply to every number the code produces. The baseline has five annual observations and
income is almost collinear with time, so standard errors are descriptive only and the
elasticities are indicative. Passenger-km are equilibrium quantities, so the AC excess
cannot be split into demand and supply. Real yield is an average-revenue proxy for price.

## Troubleshooting

On macOS with Python 3.13, an editable install inside a synced folder (e.g. `~/Documents`)
can fail with `ModuleNotFoundError: railclass` because the `.pth` file gets the `hidden` flag
and Python skips hidden `.pth` files. The Makefile sets `PYTHONPATH=src`, which avoids this;
or run `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`.

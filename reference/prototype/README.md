# Prototype scripts

These five scripts are the first working version of the analysis, written as flat
scripts before the code was organised into the `railclass` package. The paper's
Appendix C.2 prints an excerpt of them to document the method.

They are kept for reference only and are not part of the package. They hard-code
absolute paths, have to be run in a fixed order, and keep the data
inline in `data.py`. Use the package in `src/` for any new work. It gives the same
results, and `tests/test_paper_results.py` holds them fixed.

| Prototype script     | Now in the package                                                          |
| -------------------- | --------------------------------------------------------------------------- |
| `data.py`            | `data/raw/*.csv` + `railclass.io`                                           |
| `analysis.py`        | `panel`, `descriptives`, `elasticities`, `benchmark`, `regime`              |
| `analysis2.py`       | `elasticities.long_difference`, `scenarios.anchored_projection_table`       |
| `appendix_models.py` | `tables.regression_table`                                                   |
| `projection.py`      | `scenarios.projection_2030_table`, `scenarios.unwind_scenario`              |
| *(package only)*     | `robustness` (Section 5.5 checks), `figures` (Figures 1-4, B1)              |

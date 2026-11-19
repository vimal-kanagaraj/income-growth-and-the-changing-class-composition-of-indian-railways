# Data

All inputs come from public documents and were transcribed by hand into `raw/`.
Every table and figure in the paper is computed from these four files. Nothing in
`raw/` is generated; treat it as read-only and change it only to fix a
transcription against the source.

| File | Contents | Source |
| --- | --- | --- |
| `raw/class_traffic.csv` | Originating passengers (million), passenger-km (million) and passenger earnings (₹ crore) for 12 classes, 2015-16 to 2024-25 | Union Budget railway statements: "Revenue earning traffic performance targets" (railstat4, railstat6 from 2021-22) and "Railway receipts" (railstat3) |
| `raw/printed_totals.csv` | Grand totals printed in the same statements (used to validate the class rows) | as above |
| `raw/national_income.csv` | Real per capita NNI (₹, 2011-12 prices), nominal per capita NNI, and real NNI (₹ crore) | Economic Survey 2025-26, Statistical Appendix, Table 1.1 |
| `raw/yearbook_2010_11.csv` | 2010-11 passenger-km by broad class, for the long-difference check | Indian Railways Year Book 2018-19, Table II |

## Notes

- Actuals for year *t* appear in the Achievement (traffic) or Actuals (earnings)
  column of the budget presented in year *t + 2*. The file for each year is
  listed below (paths relative to `https://www.indiabudget.gov.in/`).
- Class-wise passenger counts were not published for 2015-16 (`pax_mn` blank).
- The class rows sum exactly to the printed totals in every year except
  2020-21, where they exceed them by 0.02% (passengers) and 0.01%
  (passenger-km). The class rows are kept; `tests/test_data.py` checks this.
- Sleeper (Ordinary) passenger-km for 2020-21 is printed as −14; it is kept as
  printed and dropped from log models.
- Population is derived as real NNI ÷ real per capita NNI, from a single vintage.

## Union Budget files (Appendix B, Table B3)

| Year | Budget | Traffic file | Earnings file |
| --- | --- | --- | --- |
| 2015-16 | 2017-18 | `budget2017-2018/ub2017-18/eb/railstat4.pdf` (Part B, pkm only) | same folder, `railstat3.pdf` |
| 2016-17 | 2018-19 | `budget2018-2019/ub2018-19/eb/railstat4.pdf` | same folder, `railstat3.pdf` |
| 2017-18 | 2019-20 | `budget2019-20/doc/eb/railstat4.pdf` | same folder, `railstat3.pdf` |
| 2018-19 | 2020-21 | `budget2020-21/doc/eb/railstat4.pdf` | same folder, `railstat3.pdf` |
| 2019-20 | 2021-22 | `budget2021-22/doc/eb/railstat6.pdf` | same folder, `railstat3.pdf` |
| 2020-21 | 2022-23 | `budget2022-23/doc/eb/railstat6.pdf` | same folder, `railstat3.pdf` |
| 2021-22 | 2023-24 | `budget2023-24/doc/eb/railstat6.pdf` | same folder, `railstat3.pdf` |
| 2022-23 | 2024-25 | `budget2024-25/doc/eb/railstat6.pdf` | same folder, `railstat3.pdf` |
| 2023-24 | 2025-26 | `budget2025-26/doc/eb/railstat6.pdf` | same folder, `railstat3.pdf` |
| 2024-25 | 2026-27 | `doc/eb/railstat6.pdf` (current budget) | same folder, `railstat3.pdf` |

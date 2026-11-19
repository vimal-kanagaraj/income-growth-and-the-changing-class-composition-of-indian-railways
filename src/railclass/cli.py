"""Command-line entry point: ``railclass`` (or ``python -m railclass``)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from railclass.config import AC, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, LATEST_YEAR, NON_AC, SUBURBAN
from railclass.pipeline import Results, run


def _print_headlines(r: Results) -> None:
    fits = r.segment_fits
    post = r.decomposition.iloc[1]
    print("Headline results")
    print(f"  AC share rise {post.window}: {post.total:+.2f} pp ({post.from_ac_growth:+.2f} from AC growth)")
    print(
        "  Income elasticities (segment aggregate): "
        f"AC {fits[AC].beta:.2f}, Non-AC {fits[NON_AC].beta:.2f}, Suburban {fits[SUBURBAN].beta:.2f}"
    )
    print(
        f"  {LATEST_YEAR} gap to income path: "
        + ", ".join(f"{s} {r.benchmark.loc[LATEST_YEAR, f'gap_pct_{s}']:+.0f}%" for s in (AC, NON_AC, SUBURBAN))
    )
    print(
        f"  AC excess revenue: Rs {r.rupee_gap.excess_ac_cr:,.0f} crore gross, {r.rupee_gap.excess_ac_net_cr:,.0f} net"
    )
    print(
        f"  POST coefficient (AC): {r.regime_models.loc[AC, 'g_post']:.3f}; D_FARE: {r.nonac_class_panel['d_fare']:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="railclass",
        description=(
            "Generate the tables and figures for 'Income Growth and the Changing Class Composition "
            "of Indian Railways, 2015-16 to 2024-25'."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory with the raw CSVs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where tables/figures are written")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args(argv)

    results = run(args.data_dir, args.output_dir, figures=not args.no_figures)
    _print_headlines(results)
    print(f"Outputs written to {args.output_dir}")
    return 0

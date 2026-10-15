"""Export the manuscript's tables from the complete verified protocol.

The conditioning table covers ratios up to 1e6. Selection depends only on
that range, not on whether a method succeeded. The raw protocol is preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from checks.render_contest_tables import ROOT, load_protocol, render_tables


def render_paper_tables(data):
    selected = dict(data, conditioning=[case for case in data["conditioning"]
                                       if case["sigma_ratio"] <= 1e6])
    if not selected["conditioning"]:
        raise ValueError("No conditioning cases in the manuscript's range")
    rendered = render_tables(selected)
    return rendered.replace(
        r"\caption{Разброс заданных гиперболических чисел:",
        r"\caption{Разброс заданных гиперболических чисел при $\kappa_\Sigma\le10^6$:",
        1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", type=Path, default=ROOT / "results/contest/tables.tex")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/contest/results.json")
    args = parser.parse_args()
    try:
        rendered = render_paper_tables(load_protocol(args.protocol))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    if args.check:
        if not args.export.exists() or args.export.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "Exported paper tables do not match the saved protocol\n")
        print("Paper tables match the source- and input-hash-verified protocol.")
    else:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(rendered, encoding="utf-8")
        print(f"Exported four paper tables to {args.export}.")


if __name__ == "__main__":
    main()

"""Run the experiments and export everything the webpage uses to assignment2/results/.

    uv run python run.py                # every section
    uv run python run.py --only conv    # just these sections (data.json keeps the others)
"""

import argparse

import matplotlib

matplotlib.use("Agg")                   # figures are only saved, never shown

from experiments import Experiments     # noqa: E402  (after the backend is set)
from report import Report               # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=Report.SECTIONS, metavar="SECTION",
                        help=f"sections to export: {', '.join(Report.SECTIONS)}")
    args = parser.parse_args()
    Report(Experiments()).build(args.only or Report.SECTIONS)


if __name__ == "__main__":
    main()

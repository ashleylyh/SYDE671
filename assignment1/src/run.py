"""Run the experiments and export everything the webpage uses to assignment1/results/.

    uv run python run.py                        # every section, reusing results/cache/*.json
    uv run python run.py --only pyramid b6      # just these sections (data.json keeps the others)
    uv run python run.py --fresh                # ignore the cache and rerun every search
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
    parser.add_argument("--fresh", action="store_true", help="ignore the JSON cache and rerun every search")
    args = parser.parse_args()
    Report(Experiments(reuse=not args.fresh)).build(args.only or Report.SECTIONS)


if __name__ == "__main__":
    main()

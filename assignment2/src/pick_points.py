"""Click the two reference points of each hybrid pair that needs aligning (the ginput part of align_images.m).

    uv run python pick_points.py                  # every pair with align=True
    uv run python pick_points.py derek_nutmeg     # just this one

For each image, click two points (e.g. the centres of the left and right eye), in the same order in
both images. The points are saved to results/align_points.json; run.py reads them from there.
"""

import json
import sys

import matplotlib.pyplot as plt

from algorithm.config import ALIGN_POINTS, HYBRIDS, STARTER_DIR, find_input
from algorithm.io import load_image


def click_two(path, title: str) -> list[list[float]]:
    plt.figure(figsize=(9, 9))
    plt.imshow(load_image(path))
    plt.title(title)
    plt.axis("off")
    points = plt.ginput(2, timeout=0)
    plt.close()
    return [[round(x, 1), round(y, 1)] for x, y in points]


def main() -> None:
    names = sys.argv[1:]
    saved = json.loads(ALIGN_POINTS.read_text()) if ALIGN_POINTS.exists() else {}
    for pair in HYBRIDS:
        if not pair.align or (names and pair.name not in names):
            continue
        saved[pair.name] = {
            role: click_two(find_input(stem, STARTER_DIR), f"{pair.title}: click 2 points (left eye, then right eye)")
            for role, stem in (("low", pair.low), ("high", pair.high))}
        print(pair.name, saved[pair.name])
    ALIGN_POINTS.write_text(json.dumps(saved, indent=1))
    print(f"saved {ALIGN_POINTS}")


if __name__ == "__main__":
    main()

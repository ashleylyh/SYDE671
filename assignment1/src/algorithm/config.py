"""Paths and the one fixed set of parameters used for every plate."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # assignment1/
DATA_DIR = ROOT / "data"
OWN_DIR = DATA_DIR / "own"                          # my own plates from the LoC collection
RESULTS_DIR = ROOT / "results"                      # everything the webpage shows:
IMAGES_DIR = RESULTS_DIR / "images"                 #   figures and result images
DATA_JSON = RESULTS_DIR / "data.json"               #   numerical results (offsets, scores, ...)
CACHE_DIR = RESULTS_DIR / "cache"                   # JSON cache of the slow searches
OWN_LOWRES_DIR = IMAGES_DIR / "own" / "lowres_plates"   # my own low-res versions of my plates
INDEX_HTML = ROOT / "index.html"                    # the webpage; report.py fills its generated blocks

METRICS = ("l2", "ncc")                             # PDF: results with L2 / NCC


@dataclass(frozen=True)
class Params:
    """PDF: "one fixed set of parameters for all images; don't tune per image".

    Frozen so nothing can be changed per plate; use dataclasses.replace() for a
    deliberate variant (e.g. the window demo or the pyramid check on low-res plates).
    """

    # Single scale. ±15 clipped R at dy = 15 on 01597v, 01598v, 01728v (true dy = 16-18), so ±20.
    window: int = 20
    # Interior crop. 10% (~34 px on low-res) > window, so np.roll's wrapped pixels are never scored.
    frac: float = 0.1
    # Pyramid. Stop halving once the smaller side is <= 400 px (full-res: 3238 -> ... -> 202).
    pyr_min_size: int = 400
    coarse_window: int = 20         # ±20 at the coarsest level = ±320 px at full res after 4 halvings
    refine_window: int = 2          # doubling only adds ±1 px of error; ±2 is a safe margin
    # B4 / B5
    wb_strength: float = 0.5
    color_strength: float = 5.0
    # B6: rotation (degrees) and scale candidates, blur on the edge maps
    angles: tuple[float, ...] = tuple(np.round(np.arange(-0.6, 0.61, 0.1), 1).tolist())
    scales: tuple[float, ...] = (0.99, 0.995, 1.0, 1.005, 1.01)
    b6_sigma: float = 2.0
    # Own plates: same height as the course's low-res *v.jpg plates
    lowres_height: int = 1024
    # Size of the results kept in memory (a full-res RGB result is ~145 MB)
    web_max_side: int = 1600

    def single_kwargs(self) -> dict:
        """Keyword arguments for align_single_scale."""
        return {"window": self.window, "frac": self.frac}

    def pyramid_kwargs(self) -> dict:
        """Keyword arguments for align_pyramid (and align_edges)."""
        return {"frac": self.frac, "min_size": self.pyr_min_size,
                "coarse_window": self.coarse_window, "refine_window": self.refine_window}

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

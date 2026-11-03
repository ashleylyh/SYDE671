"""Paths, input images and the fixed parameters."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # assignment3/
DATA_DIR = ROOT / "data"                            # my inputs (git-ignored)
MOSAIC_DIR = DATA_DIR / "mosaic"                    #   A.1 / A.4 / B.4: one folder per image set
RECTIFY_DIR = DATA_DIR / "rectify"                  #   A.3: photos with a known rectangle
RESULTS_DIR = ROOT / "results"                      # everything the webpage shows:
IMAGES_DIR = RESULTS_DIR / "images"                 #   figures and result images
DATA_JSON = RESULTS_DIR / "data.json"               #   numbers (homographies, runtimes, match counts, ...)
POINTS_JSON = RESULTS_DIR / "points.json"           #   clicked correspondences (written by pick_points.py)
INDEX_HTML = ROOT / "index.html"                    # the webpage; report.py fills its generated blocks


@dataclass(frozen=True)
class MosaicSet:
    """A.1: photos shot from a fixed centre of projection, rotating the camera (40-70% overlap).

    images: file stems in data/mosaic/<name>/, left to right; `reference` is left unwarped (A.4).
    """
    name: str
    title: str
    images: tuple[str, ...]
    reference: int = 0

    @property
    def folder(self) -> Path:
        return MOSAIC_DIR / self.name

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """A.2: neighbouring images (left, right), the pairs that share enough overlap to click correspondences in."""
        return list(zip(self.images, self.images[1:]))


# PDF A.4 / B.4: "Show 3 mosaics with source images" (A.1 needs at least 2 sets)
MOSAICS = (
    # Listed left to right in the scene, i.e. against the shooting order (every set was panned right to left); sets 4 / 5 are frames of a video (data/mosaic/*.MOV)
    MosaicSet("set1", "Atrium", ("IMG_1555", "IMG_1554", "IMG_1553"), reference=1),
    MosaicSet("set2", "Marina", ("IMG_1768", "IMG_1767", "IMG_1766"), reference=1),
    MosaicSet("set4", "Toronto skyline", ("IMG_0574_13s", "IMG_0574_11s", "IMG_0574_09s"), reference=1),
    MosaicSet("set5", "Niagara Falls", ("IMG_1051_09s", "IMG_1051_07s", "IMG_1051_05s", "IMG_1051_03s"), reference=1),
)


@dataclass(frozen=True)
class Rectification:
    """A.3: make a known rectangle (painting, poster, ...) rectangular. PDF: "Apply to 2+ images".

    The 4 corners are clicked (pick_points.py --rectify) in any order; experiments.order_corners sorts them.
    aspect: the rectangle's real width / height, if known; None estimates it from the clicked sides.
    """
    name: str
    title: str
    image: str                          # file stem in data/rectify/
    aspect: float | None = None


RECTIFICATIONS = (
    Rectification("renoir", "Renoir painting", "IMG_3548"),
    Rectification("almond", "Van Gogh, Almond Blossom", "IMG_4344", aspect=92.4 / 73.3),   # 73.3 x 92.4 cm canvas
    Rectification("posters", "Cocktail posters", "IMG_6530"),      # the Aperol Spritz poster; the wall is one plane
)


def mosaic_set(name: str) -> MosaicSet:
    return next(m for m in MOSAICS if m.name == name)


def find_input(stem: str, folder: Path = DATA_DIR) -> Path:
    """<folder>/<stem>.<any extension>."""
    matches = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"missing input {folder.name}/{stem}.*")
    return matches[0]


@dataclass(frozen=True)
class Params:
    """Fixed parameters, chosen once and used for every image."""

    max_side: int = 1200                # inputs resized so the longest side is <= this
    # A.3: the rectified output is cropped to the rectangle plus this fraction of its size on each side
    # (a plane seen at a grazing angle maps its far side towards infinity)
    rect_margin: float = 0.6
    # A.4
    blend_sigma: float = 6.0            # 2-band blending: Gaussian that splits each image into low / high frequencies
    max_canvas: int = 8000              # a mosaic wider / taller than this means a badly chained H ...
    clip_canvas: float = 1.5            # ... so it is clipped to the reference image +- this many times its size
    # B.1
    harris_edge_discard: int = 20       # harris.py asserts >= 20
    # MOPS paper, Section 2: only maxima above a minimum strength are interest points. harris.py keeps every local
    # maximum, most of them noise in flat regions (sky): their median strength is ~1e-3, real corners 0.1 - 3
    harris_threshold: float = 0.01
    anms_corners: int = 500             # MOPS paper, Section 3: n_ip = 500
    anms_c_robust: float = 0.9          # MOPS paper, Section 3: c_robust = 0.9
    # B.2: PDF "axis-aligned 8x8 patches ... sampled from the larger 40x40 window"
    window: int = 40
    patch: int = 8
    descriptor_sigma: float = 2.0       # blur before sampling every 5th pixel (anti-aliasing for the 5x subsampling)
    # B.3: PDF "Consult Figure 6b in the paper for picking the threshold"
    ratio_threshold: float = 0.6
    # B.4
    ransac_iters: int = 2000
    ransac_eps: float = 3.0             # inlier distance in pixels (2 px rejected good matches near the wide-angle edges)
    ransac_seed: int = 0                # fixed, so the page is reproducible

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

"""Paths, input images and the fixed parameters."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # assignment2/
DATA_DIR = ROOT / "data"                            # my inputs (git-ignored)
STARTER_DIR = ROOT / "hybrid"                       # course starter zip: DerekPicture.jpg, nutmeg.jpg
RESULTS_DIR = ROOT / "results"                      # everything the webpage shows:
IMAGES_DIR = RESULTS_DIR / "images"                 #   figures and result images
DATA_JSON = RESULTS_DIR / "data.json"               #   numbers (runtimes, thresholds, sigmas, ...)
INDEX_HTML = ROOT / "index.html"                    # the webpage; report.py fills its generated blocks

# Input images in data/, by file name without extension (any format Pillow reads: jpg, jpeg, png, avif, ...)
SELFIE = "selfie"                   # Part 1.1: "take a picture of yourself"
CAMERAMAN = "cameraman"             # Parts 1.2, 1.3, orientation
TAJ = "taj"                         # Part 2.1, the image from Learn
BLURRY = "cat_closeup"              # Part 2.1, my own blurry image
APPLE, ORANGE = "apple", "orange"   # Parts 2.3 / 2.4, the Burt & Adelson pair
SHARP = "skyline"                   # Part 2.1 evaluation: a sharp image to blur and re-sharpen


# Part 2.2: the two pairs of reference points per pair that needs aligning (written by pick_points.py)
ALIGN_POINTS = RESULTS_DIR / "align_points.json"


@dataclass(frozen=True)
class HybridPair:
    """One hybrid: the low frequencies of `low` + the high frequencies of `high`.

    PDF: "The cutoff-frequency of each filter should be chosen with some experimentation", so each pair
    has its own sigmas. `align`: the inputs come from the starter zip (hybrid/) and are aligned with
    clicked points, as in hybrid_image_starter.m; my own pairs were aligned when I made them.
    """
    name: str
    title: str
    low: str                            # file stem, seen from far away
    high: str                           # file stem, seen up close
    sigma_low: float
    sigma_high: float
    align: bool = False


HYBRIDS = (
    HybridPair("derek_nutmeg", "Derek + Nutmeg", "DerekPicture", "nutmeg", 7.0, 4.0, align=True),
    HybridPair("cat_jaguar", "Cat + jaguar", "cat_hybrid1", "jaguar_hybrid1", 7.0, 4.0),
    HybridPair("mainecoon_owl", "Maine coon + owl", "mainecoon_hybrid2", "owl_hybrid2", 7.0, 4.0),
)
FAVORITE = "derek_nutmeg"           # PDF: "For your favorite result ... illustrate the process through frequency analysis"


@dataclass(frozen=True)
class BlendPair:
    """Part 2.4: im_a where the mask is 1, im_b where it is 0.

    mask: "vertical" (step function, im_a on the left) or the file stem of a mask image (white = im_a).
    levels / sigma: the stacks (stacks.gaussian_stack). The deepest level's total blur should be about the
    size of the region being blended: a small mask needs fewer levels (see the third-eye comparison).
    """
    name: str
    title: str
    a: str
    b: str
    mask: str
    levels: int = 6
    sigma: float = 2.0
    max_side: int | None = None         # resize the inputs (and the mask) so the longest side is <= this
    zoom: tuple[int, int, int, int] | None = None      # (top, bottom, left, right) shown in the process figure


BLENDS = (
    BlendPair("oraple", "Oraple", "apple", "orange", "vertical"),
    BlendPair("earth_mars", "Earth + Mars", "earth_blend1", "mars_blend1", "vertical", max_side=700),
    BlendPair("third_eye", "Doctor Strange's third eye", "eye_aligned_blend2", "dr_strange_blend2", "eye_mask",
              levels=4, zoom=(110, 330, 470, 810)),
)
FAVORITE_BLEND = "third_eye"        # PDF: Laplacian stack "for your favorite result", like Figure 10 in the paper


def find_input(stem: str, folder: Path = DATA_DIR) -> Path:
    """<folder>/<stem>.<any extension>."""
    matches = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"missing input {folder.name}/{stem}.*")
    return matches[0]


@dataclass(frozen=True)
class Params:
    """Fixed parameters, chosen once and used for every image."""

    # Part 1.1
    box_size: int = 9                   # PDF: "write out a 9x9 box filter"
    selfie_max_side: int = 600          # selfie resized so its longest side is <= this (the photo is 5712 × 4284)
    # Parts 1.2 / 1.3: edge thresholds on the gradient magnitude (image in [0, 1]), chosen by eye
    fd_threshold: float = 0.25
    dog_sigma: float = 1.5              # Gaussian for 1.3; kernel size = ksize_for(sigma) (±3 sigma)
    dog_threshold: float = 0.10
    # Part 2.1: unsharp mask
    sharpen_sigma: float = 2.0
    sharpen_alpha: float = 1.5          # the amount used for the main results
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    eval_max_side: int = 800            # the sharp image is resized so its longest side is <= this
    # Part 2.2
    hybrid_max_side: int = 700          # inputs (after alignment) resized so the longest side is <= this
    sweep_low: tuple[float, ...] = (4.0, 7.0, 10.0)     # sigmas tried for the favourite pair
    sweep_high: tuple[float, ...] = (2.0, 4.0, 7.0)
    # Parts 2.3 / 2.4: stacks. Level i is blurred with sigma * 2^(i-1) (see stacks.gaussian_stack);
    # with 4 levels the oraple still had a visible seam, from 5 on it was smooth
    stack_levels: int = 6
    stack_sigma: float = 2.0
    fig342_levels: tuple[int, ...] = (0, 2, 4)  # the levels shown in Szeliski's Figure 3.42

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

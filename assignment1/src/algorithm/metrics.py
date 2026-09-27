"""Matching metrics between two channels."""

import numpy as np


def l2(im1: np.ndarray, im2: np.ndarray) -> float:
    """Euclidean distance. Lower is better.

    PDF: sqrt(sum(sum((image1 - image2).^2))), summed over the pixel values.
    """
    diff = im1 - im2
    return float(np.sqrt(np.sum(diff * diff)))


def ncc(im1: np.ndarray, im2: np.ndarray, zero_mean: bool = False) -> float:
    """Normalized cross-correlation. Higher is better (1 = identical up to scale).

    PDF: dot product between the normalized vectors image1/||image1|| and image2/||image2||.
    zero_mean=True subtracts each image's mean first, so a brightness offset between
    channels doesn't affect the score (range becomes [-1, 1]).
    """
    a = im1.ravel()
    b = im2.ravel()
    if zero_mean:
        a = a - a.mean()
        b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def interior(im: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Center crop with `frac` of the height/width removed from each side (a view, no copy).

    PDF: "Ignore borders when scoring; compute metrics on interior pixels."
    The crop also hides the pixels that np.roll wraps around from the opposite edge,
    as long as frac * size > the search window.
    """
    h, w = im.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    return im[dy:h - dy, dx:w - dx]


# All scores as "higher is better", so the search can always take the max
SCORES = {
    "l2": lambda a, b: -l2(a, b),
    "ncc": ncc,
    "ncc_zero_mean": lambda a, b: ncc(a, b, zero_mean=True),
}

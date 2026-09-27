"""Splitting a plate and aligning G and R to B."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import SCORES, interior
from .pyramid import build_pyramid

Offset = tuple[int, int]            # (dx, dy): x = columns, y = rows


def split_channels(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a plate into its (b, g, r) channels, ordered top to bottom.

    PDF: "divide the image into three equal parts" (organized BGR from top to bottom).
    Each part is exactly height // 3 rows, so all three have the same shape
    (1024 and 9715 are not divisible by 3; the 1-2 leftover rows at the bottom are dropped).
    Returns views into `plate`, so no memory is copied.
    """
    h = plate.shape[0] // 3
    return plate[:h], plate[h:2 * h], plate[2 * h:3 * h]


def shift(im: np.ndarray, offset: Offset) -> np.ndarray:
    """Shift an image by offset = (dx, dy) pixels with np.roll (x = columns, y = rows)."""
    dx, dy = offset
    return np.roll(im, (dy, dx), axis=(0, 1))


def stack_rgb(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Stack the channels as an RGB image (no alignment)."""
    return np.dstack([r, g, b])


def align_single_scale(moving: np.ndarray, ref: np.ndarray, window: int = 15,
                       metric: str = "ncc", frac: float = 0.1) -> Offset:
    """Exhaustive search for the (dx, dy) that best aligns `moving` to `ref`.

    PDF: "Implement a simple single-scale version first, using for loops, searching over
    a user-specified window of displacements" — here [-window, window] in x and y
    ((2 * window + 1)^2 = 961 candidates for window=15), shifted with np.roll and scored
    on interior pixels. The reference crop is computed once, outside the loops.
    """
    score = SCORES[metric]
    ref_in = interior(ref, frac)
    best_score, best_offset = -np.inf, (0, 0)
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            s = score(interior(shift(moving, (dx, dy)), frac), ref_in)
            if s > best_score:
                best_score, best_offset = s, (dx, dy)
    return best_offset


def align_pyramid(moving: np.ndarray, ref: np.ndarray, metric: str = "ncc", frac: float = 0.1,
                  min_size: int = 400, coarse_window: int = 20, refine_window: int = 2,
                  trace: list | None = None) -> Offset:
    """Coarse-to-fine alignment of `moving` to `ref`; returns (dx, dy) at full resolution.

    PDF: "processing is done sequentially starting from the coarsest scale (smallest image)
    and going down the pyramid, updating your estimate as you go ... by adding recursive
    calls to your original single-scale implementation."
    - Coarsest level: exhaustive search in ±coarse_window (the only expensive search, on a tiny image).
    - Every finer level: double the estimate, apply it with np.roll, refine in ±refine_window
      (25 candidates instead of 1681).
    `trace`, if given a list, receives (level shape, offset so far) per level, coarsest first.
    """
    moving_pyr = build_pyramid(moving, min_size)
    ref_pyr = build_pyramid(ref, min_size)
    # The interior crop must be at least the window, or np.roll's wrapped pixels get scored
    assert int(frac * min(ref_pyr[-1].shape)) >= coarse_window, "min_size too small for coarse_window"

    dx, dy = 0, 0
    for level in reversed(range(len(ref_pyr))):
        coarsest = level == len(ref_pyr) - 1
        if not coarsest:
            dx, dy = 2 * dx, 2 * dy
        ddx, ddy = align_single_scale(shift(moving_pyr[level], (dx, dy)), ref_pyr[level],
                                      window=coarse_window if coarsest else refine_window,
                                      metric=metric, frac=frac)
        dx, dy = dx + ddx, dy + ddy
        if trace is not None:
            trace.append((ref_pyr[level].shape, (dx, dy)))
    return dx, dy


@dataclass(frozen=True)
class Transform:
    """How one channel is moved onto B: rotate + scale about the centre (B6), then shift by (dx, dy).

    Translation-only aligners return plain (dx, dy) tuples; Transform.of() wraps them,
    so every aligner plugs into colorize() the same way.
    """

    dx: int = 0
    dy: int = 0
    angle: float = 0.0              # degrees
    scale: float = 1.0

    @classmethod
    def of(cls, t: "Transform | Offset") -> "Transform":
        return t if isinstance(t, cls) else cls(int(t[0]), int(t[1]))

    @property
    def offset(self) -> Offset:
        return self.dx, self.dy

    @property
    def is_translation(self) -> bool:
        return self.angle == 0 and self.scale == 1

    def apply(self, im: np.ndarray) -> np.ndarray:
        if not self.is_translation:
            from .bells_whistles import warp     # B6; imported here to avoid a circular import
            im = warp(im, self.angle, self.scale)
        return shift(im, self.offset)


Aligner = Callable[..., "Transform | Offset"]     # (moving, ref, **kwargs) -> Transform or (dx, dy)


def align_channels(plate: np.ndarray, align: Aligner = align_single_scale,
                   **align_kwargs) -> dict[str, Transform]:
    """Split a plate and align G and R to B: {"G": Transform, "R": Transform}."""
    b, g, r = split_channels(plate)
    return {ch: Transform.of(align(im, b, **align_kwargs)) for ch, im in (("G", g), ("R", r))}


def apply_transforms(b: np.ndarray, g: np.ndarray, r: np.ndarray,
                     transforms: dict[str, Transform]) -> np.ndarray:
    """Move G and R onto B and stack as RGB."""
    return stack_rgb(b, transforms["G"].apply(g), transforms["R"].apply(r))


def colorize(plate: np.ndarray, align: Aligner = align_single_scale,
             **align_kwargs) -> tuple[np.ndarray, dict[str, Transform]]:
    """Split a plate, align G and R to B, and stack as RGB.

    PDF: "divide the image into three equal parts and align the second and the third
    parts (G and R) to the first (B)." `align` is any function (moving, ref, **kw) -> offset,
    e.g. align_single_scale, align_pyramid, align_edges or align_similarity.
    Returns (rgb image, {"G": Transform, "R": Transform}).
    """
    transforms = align_channels(plate, align, **align_kwargs)
    return apply_transforms(*split_channels(plate), transforms), transforms

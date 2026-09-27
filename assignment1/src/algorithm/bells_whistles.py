"""Bells & whistles B1-B6 and the final pipeline.

Libraries are allowed here (PDF), as long as they are explained:
scipy.ndimage provides the Sobel filter, Gaussian blur and affine warp.

    B1  edge features          align on gradient magnitude instead of raw brightness
    B2  automatic cropping     remove plate borders and np.roll wrap-around
    B3  automatic contrast     linear percentile stretch
    B4  white balance          gray world / white patch
    B5  color mapping          3x3 matrices from plate filters to RGB
    B6  better transformations translation + rotation + scale
    restore()                  B1 -> B6 -> B2 -> B4 -> B3 on one plate
"""

from typing import Iterable

import numpy as np
import scipy.ndimage as ndi

from .align import Offset, Transform, align_channels, align_pyramid, align_single_scale, \
    apply_transforms, shift, split_channels
from .config import Params
from .metrics import SCORES, interior
from .pyramid import build_pyramid


# ---------------------------------------------------------------- B1. Edge features

def edge_features(im: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Gradient magnitude sqrt(Gx^2 + Gy^2) from 3x3 Sobel filters (scipy.ndimage.sobel).

    Why: an edge is in the same place in every channel even when the brightness is not
    (the red rugs in 01725u are bright in R and dark in B), so aligning edge maps
    compares structure instead of raw intensity. `sigma` optionally blurs the edge map (used in B6).
    """
    mag = np.hypot(ndi.sobel(im, axis=0), ndi.sobel(im, axis=1))
    if sigma:
        mag = ndi.gaussian_filter(mag, sigma)
    return mag.astype(np.float32)


def align_edges(moving: np.ndarray, ref: np.ndarray, **kwargs) -> Offset:
    """align_pyramid on edge features; same interface, so it plugs straight into colorize()."""
    return align_pyramid(edge_features(moving), edge_features(ref), **kwargs)


# ---------------------------------------------------------------- B2. Automatic cropping

def line_strength(ch: np.ndarray, axis: int) -> np.ndarray:
    """Mean |brightness jump| between neighbouring rows (axis=0) or columns (axis=1).

    A plate border is a straight line across the whole image, so its jump shows up in every
    column of that row and the mean is high. Edges inside the picture are scattered, so they average out.
    """
    return np.abs(np.diff(ch, axis=axis)).mean(axis=1 - axis)


def auto_crop_box(rgb: np.ndarray, transforms: Iterable[Transform | Offset], margin: float = 0.1,
                  k: float = 5.0, extra: int = 0, work_size: int = 800) -> tuple[int, int, int, int]:
    """Detect the (top, bottom, left, right) box inside the plate borders.

    1. Rows / columns that np.roll wrapped around from the opposite side are invalid:
       their size is known exactly from the alignment offsets.
    2. In the outer `margin` of each side, find the innermost border line in any channel:
       a row / column whose line_strength is > k x the median line strength of that channel.
    `extra` removes a few more pixels on every side (used for B6's rotated / scaled channels).
    Runs on a strided copy about `work_size` px tall (borders don't need full resolution).
    """
    H, W = rgb.shape[:2]
    step = max(1, H // work_size)
    small = rgb[::step, ::step]
    h, w = small.shape[:2]

    offsets = [Transform.of(t).offset for t in transforms]
    dxs = [dx // step for dx, _ in offsets] + [0]
    dys = [dy // step for _, dy in offsets] + [0]
    top, bottom = max(dys), h + min(dys)        # 1. wrapped-around rows / columns
    left, right = max(dxs), w + min(dxs)

    my, mx = int(h * margin), int(w * margin)   # 2. innermost border line near each side
    for c in range(3):
        rows, cols = line_strength(small[:, :, c], 0), line_strength(small[:, :, c], 1)
        r_thr, c_thr = k * np.median(rows), k * np.median(cols)
        top = max([top] + [i + 1 for i in range(0, my) if rows[i] > r_thr])
        bottom = min([bottom] + [i + 1 for i in range(h - 1 - my, h - 1) if rows[i] > r_thr])
        left = max([left] + [j + 1 for j in range(0, mx) if cols[j] > c_thr])
        right = min([right] + [j + 1 for j in range(w - 1 - mx, w - 1) if cols[j] > c_thr])

    e = -(-extra // step) if extra else 0
    return (top + e) * step, min((bottom - e) * step, H), (left + e) * step, min((right - e) * step, W)


def auto_crop(rgb: np.ndarray, transforms: Iterable[Transform | Offset], **kwargs) -> np.ndarray:
    t, b, l, r = auto_crop_box(rgb, transforms, **kwargs)
    return rgb[t:b, l:r]


# ---------------------------------------------------------------- B3. Automatic contrast

def auto_contrast(rgb: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Linear stretch so the darkest value (over all channels) -> 0 and the brightest -> 1.

    PDF: "rescale image intensities such that the darkest pixel is zero (on its darkest color channel)
    and the brightest pixel is 1 (on its brightest color channel)". Percentiles instead of min / max,
    so a few dust specks or scratches don't decide the stretch. One mapping for all three channels,
    so the colours don't shift. Percentiles are estimated on every 4th pixel (fast, same result).
    """
    lo, hi = np.percentile(rgb[::4, ::4], [low_pct, high_pct])
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


# ---------------------------------------------------------------- B4. White balance

def white_balance(rgb: np.ndarray, method: str = "gray_world", strength: float = 1.0,
                  pct: float = 99.0) -> np.ndarray:
    """(1) Estimate the illuminant, (2) scale each channel so it becomes neutral (von Kries / diagonal model).

    - "gray_world":  illuminant = average colour (assumes the scene averages to gray).
    - "white_patch": illuminant = the `pct` percentile of each channel (assumes the brightest colour is white).
    `strength` in [0, 1] moves the gains only part of the way from 1: full gray world over-corrects
    scenes dominated by one colour (a green landscape pushes the sky towards purple).
    """
    sample = rgb[::4, ::4].reshape(-1, 3)
    if method == "gray_world":
        illum = sample.mean(axis=0)
        gains = illum.mean() / illum
    elif method == "white_patch":
        illum = np.percentile(sample, pct, axis=0)
        gains = illum.max() / illum
    else:
        raise ValueError(method)
    gains = 1 + strength * (gains - 1)
    return np.clip(rgb * gains, 0, 1)


# ---------------------------------------------------------------- B5. Color mapping

M_IDENTITY = np.eye(3)

M_WARM = np.array([
    [1.10, -0.04, -0.02],
    [-0.01, 1.04, -0.01],
    [0.00, -0.03, 0.97],
])

M_COOL = np.array([
    [0.98, -0.02, 0.00],
    [-0.02, 1.03, -0.01],
    [-0.03, -0.04, 1.10],
])


def color_matrices(strength: float = 1.0) -> dict[str, np.ndarray]:
    """{"identity", "warm", "cool"} with each matrix's deviation from identity scaled: M' = I + s (M - I).

    The rows of warm / cool do not sum to 1 (warm at strength 5: 1.2, 1.1, 0.7), so besides
    boosting saturation they also tint the image: warm boosts red and cuts blue, cool the opposite.
    """
    return {k: M_IDENTITY + strength * (M - M_IDENTITY)
            for k, M in {"identity": M_IDENTITY, "warm": M_WARM, "cool": M_COOL}.items()}


def apply_color_matrix(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map every pixel through a 3x3 matrix: [r', g', b'] = M @ [r, g, b], clipped to [0, 1].

    The plate filters don't match sRGB's R, G, B primaries, so each output channel is a mix of the
    three plates. Positive diagonal + negative off-diagonal entries subtract some of the other
    channels, which undoes the "crosstalk" between overlapping filters and makes colors more saturated.
    """
    return np.clip(rgb @ M.T.astype(np.float32), 0, 1)


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colorfulness: spread + mean of the opponent channels r-g and (r+g)/2-b."""
    x = rgb[::4, ::4].reshape(-1, 3)
    rg, yb = x[:, 0] - x[:, 1], 0.5 * (x[:, 0] + x[:, 1]) - x[:, 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


# ---------------------------------------------------------------- B6. Rotation + scale

def warp(im: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate by `angle` degrees and scale by `scale` about the image centre.

    scipy.ndimage.affine_transform with bilinear interpolation (order=1); it maps every output
    pixel back to the input, so there are no holes. Edge pixels are repeated outside the image.
    """
    a = np.deg2rad(angle)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) / scale
    c = (np.array(im.shape) - 1) / 2
    return ndi.affine_transform(im, M, offset=c - M @ c, order=1, mode="nearest").astype(np.float32)


def align_similarity(moving: np.ndarray, ref: np.ndarray, metric: str = "ncc",
                     params: Params = Params(), info: dict | None = None) -> Transform:
    """Translation + rotation + scale, coarse to fine.

    1. Translation with the pyramid, on blurred edge maps (B1).
    2. At a coarse pyramid level (~400-800 px), try every (angle, scale) in angles x scales; for each,
       refine the translation by ±refine_window around the scaled estimate and keep the best score.
       Searching the 2 extra dimensions only on a small image keeps it fast (PDF hint).
    3. Warp the full-resolution edge map with the best (angle, scale) and redo the pyramid translation.
    Why blurred edges: warping resamples (slightly blurs) the image, and blurrier edge maps correlate
    better with anything, which would bias the search towards "some warp". Blurring both images first
    makes that extra blur negligible.
    `info`, if given a dict, receives the translation-only offset and the score before / after.
    """
    pyr_kw = params.pyramid_kwargs()
    em, er = edge_features(moving, params.b6_sigma), edge_features(ref, params.b6_sigma)
    t = align_pyramid(em, er, metric=metric, **pyr_kw)

    pm, pr = build_pyramid(em, params.pyr_min_size), build_pyramid(er, params.pyr_min_size)
    lv = max(len(pr) - 2, 0)
    f = 2 ** lv
    m, r = pm[lv], pr[lv]
    r_in = interior(r, params.frac)
    t_lv = (round(t[0] / f), round(t[1] / f))
    score = SCORES[metric]
    best = (-np.inf, 0.0, 1.0)
    for angle in params.angles:
        for scale in params.scales:
            wm = warp(m, angle, scale)
            d = align_single_scale(shift(wm, t_lv), r, window=params.refine_window,
                                   metric=metric, frac=params.frac)
            s = score(interior(shift(wm, (t_lv[0] + d[0], t_lv[1] + d[1])), params.frac), r_in)
            if s > best[0]:
                best = (s, float(angle), float(scale))
    _, angle, scale = best

    warped = warp(em, angle, scale)
    dx, dy = align_pyramid(warped, er, metric=metric, **pyr_kw)
    if info is not None:
        info["translation_only"] = t
        info["score_before"] = score(interior(shift(em, t), params.frac), interior(er, params.frac))
        info["score_after"] = score(interior(shift(warped, (dx, dy)), params.frac), interior(er, params.frac))
    return Transform(dx, dy, angle, scale)


def warp_margin(transforms: Iterable[Transform], shape: tuple[int, ...]) -> int:
    """Pixels near the border that a rotated / scaled channel fills with repeated edge pixels."""
    H, W = shape[:2]
    return max(int(np.ceil((abs(1 - t.scale) + abs(np.sin(np.deg2rad(t.angle)))) * max(H, W) / 2))
               for t in transforms)


# ---------------------------------------------------------------- Final pipeline

def restore(plate: np.ndarray, params: Params = Params(),
            transforms: dict[str, Transform] | None = None) -> np.ndarray:
    """Full pipeline for one plate: edges (B1) -> translation + rotation + scale (B6) -> auto-crop (B2)
    -> gray-world white balance at params.wb_strength (B4) -> auto contrast (B3).

    `transforms` (e.g. cached from align_similarity) skips the alignment search.
    """
    if transforms is None:
        transforms = align_channels(plate, align_similarity, params=params)
    rgb = apply_transforms(*split_channels(plate), transforms)
    rgb = auto_crop(rgb, transforms.values(), extra=warp_margin(transforms.values(), rgb.shape))
    rgb = white_balance(rgb, "gray_world", strength=params.wb_strength)
    return auto_contrast(rgb)

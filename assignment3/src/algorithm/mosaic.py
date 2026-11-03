"""A.4: Blend the images into a mosaic (weighted averaging instead of overwriting).

Every image is warped into the reference image's projection (the reference itself with H = I), all into one canvas
that holds them all (one-shot procedure). Each image carries a weight that is 1 at its centre and falls off linearly to
0 at its edges; the weight is warped with the image as an extra channel.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .warp import Bounds, output_bounds, warpImageBilinear


def chain_to_reference(pair_Hs: list[np.ndarray], reference: int) -> list[np.ndarray]:
    """Homographies image i -> reference image, from the neighbour homographies pair_Hs[i]: image i -> image i + 1.

    Left of the reference, multiply forward: H_{i->r} = H_{r-1} ... H_{i+1} H_i.
    Right of it, invert the chain back: H_{i->r} = (H_{i-1} ... H_r)^-1.
    """
    n = len(pair_Hs) + 1
    Hs = []
    for i in range(n):
        H = np.eye(3)
        if i < reference:
            for k in range(i, reference):
                H = pair_Hs[k] @ H
        else:
            for k in range(reference, i):
                H = pair_Hs[k] @ H
            H = np.linalg.inv(H)
        Hs.append(H / H[2, 2])
    return Hs


def canvas_bounds(shapes: list[tuple[int, ...]], Hs: list[np.ndarray]) -> Bounds:
    """The union of every image's output_bounds: the size of the final mosaic."""
    b = np.array([output_bounds(s, H) for s, H in zip(shapes, Hs)])
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def tent_weight(shape: tuple[int, ...]) -> np.ndarray:
    """PDF: alpha "1 at the center of each (unwarped) image and ... fall off linearly until it hits 0 at the edges"."""
    h, w = shape[:2]
    wx = 1 - np.abs(np.linspace(-1, 1, w))
    wy = 1 - np.abs(np.linspace(-1, 1, h))
    return (np.outer(wy, wx) + 1e-4).astype(np.float32)      # never exactly 0 inside the image (no black seams)


def warp_with_weight(im: np.ndarray, H: np.ndarray, bounds: Bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(warped RGB, warped weight, alpha) of one image on the canvas."""
    out = warpImageBilinear(np.dstack([im, tent_weight(im.shape)]), H, bounds)
    return out[..., :3], out[..., 3], out[..., 4]


def overwrite(warped: list[np.ndarray], alphas: list[np.ndarray]) -> np.ndarray:
    """Baseline without blending: each image is pasted on top of the previous ones."""
    out = np.zeros_like(warped[0])
    for im, a in zip(warped, alphas):
        out[a > 0] = im[a > 0]
    return out


def feather(warped: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    """Weighted average at every pixel: sum_i w_i I_i / sum_i w_i."""
    num = sum(w[..., None] * im for im, w in zip(warped, weights))
    den = sum(weights)[..., None]
    return np.where(den > 0, num / np.maximum(den, 1e-8), 0).astype(np.float32)


def two_band(warped: list[np.ndarray], weights: list[np.ndarray], alphas: list[np.ndarray], sigma: float) -> np.ndarray:
    """2-level Laplacian pyramid blend (PDF: "If your only problem is ghosting of high-frequency terms, then a 2-level
    pyramid should be enough").

    Low band: each image blurred with a Gaussian (normalised by its blurred alpha, so the black outside the image does
    not leak in), then feathered. High band (image - low): taken whole from the image with the largest weight at each
    pixel, so edges are never averaged with a misaligned copy.
    """
    lows, highs = [], []
    for im, a in zip(warped, alphas):
        blur_a = gaussian_filter(a, sigma)
        low = np.stack([gaussian_filter(im[..., c] * a, sigma) for c in range(3)], axis=-1)
        low = np.where(blur_a[..., None] > 1e-3, low / np.maximum(blur_a, 1e-3)[..., None], 0)
        lows.append(low)
        highs.append((im - low) * a[..., None])
    best = np.argmax(np.stack(weights), axis=0)
    high = np.zeros_like(warped[0])
    for i, h in enumerate(highs):
        high[best == i] = h[best == i]
    covered = (sum(alphas) > 0)[..., None]
    return np.clip(np.where(covered, feather(lows, weights) + high, 0), 0, 1).astype(np.float32)

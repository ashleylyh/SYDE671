"""A.3: Warp the images, with inverse warping and two interpolations written from scratch.

PDF: no cv2.warpPerspective / skimage.transform.warp / scipy.interpolate.griddata.
Convention: (0, 0) is the centre of the top-left pixel, so pixel (row i, column j) sits at (x, y) = (j, i).

Both warps return an RGBA-style image: the warped channels plus an alpha channel that is 1 where the output pixel came
from inside the source image and 0 elsewhere (PDF: "Consider using an alpha mask").
"""

import numpy as np

Bounds = tuple[int, int, int, int]      # (x_min, y_min, x_max, y_max), inclusive, in output coordinates


def corners(shape: tuple[int, ...]) -> np.ndarray:
    """The four corner pixel centres of an image, as 4-by-2 (x, y): TL, TR, BR, BL."""
    h, w = shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def output_bounds(shape: tuple[int, ...], H: np.ndarray) -> Bounds:
    """Bounding box of the image's corners mapped through H (PDF: "predict the bounding box by piping the four
    corners of the image through your H")."""
    p = np.column_stack([corners(shape), np.ones(4)]) @ H.T
    p = p[:, :2] / p[:, 2:3]
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def source_coords(H: np.ndarray, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    """Inverse warping: for every output pixel in bounds, where it comes from in the source (H^-1 applied)."""
    x0, y0, x1, y1 = bounds
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
    Hi = np.linalg.inv(H)
    w = Hi[2, 0] * xs + Hi[2, 1] * ys + Hi[2, 2]
    return (Hi[0, 0] * xs + Hi[0, 1] * ys + Hi[0, 2]) / w, (Hi[1, 0] * xs + Hi[1, 1] * ys + Hi[1, 2]) / w


def _channels(im: np.ndarray) -> np.ndarray:
    return im[..., None] if im.ndim == 2 else im


def warpImageNearestNeighbor(im: np.ndarray, H: np.ndarray, bounds: Bounds | None = None) -> np.ndarray:
    """Inverse warp; each output pixel takes the source pixel whose centre is nearest (rounded coordinates).

    Returns (h, w, C + 1): the C warped channels and alpha. bounds defaults to output_bounds(im.shape, H).
    """
    im = _channels(im)
    bounds = bounds or output_bounds(im.shape, H)
    sx, sy = source_coords(H, bounds)
    h, w = im.shape[:2]
    xi, yi = np.floor(sx + 0.5).astype(np.int64), np.floor(sy + 0.5).astype(np.int64)   # round half up
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(sx.shape + (im.shape[2] + 1,), dtype=np.float32)
    out[valid, :-1] = im[yi[valid], xi[valid]]
    out[valid, -1] = 1
    return out


def warpImageBilinear(im: np.ndarray, H: np.ndarray, bounds: Bounds | None = None) -> np.ndarray:
    """Inverse warp; each output pixel is the weighted average of the 4 source pixels around its source position,
    weighted by (1 - dx)(1 - dy), dx(1 - dy), (1 - dx)dy, dx dy.

    Returns (h, w, C + 1): the C warped channels and alpha. bounds defaults to output_bounds(im.shape, H).
    """
    im = _channels(im)
    bounds = bounds or output_bounds(im.shape, H)
    sx, sy = source_coords(H, bounds)
    h, w = im.shape[:2]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x, y = sx[valid], sy[valid]
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2)    # top-left neighbour; on the last row / column dx = 1
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    out = np.zeros(sx.shape + (im.shape[2] + 1,), dtype=np.float32)
    out[valid, :-1] = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x0 + 1] +
                       (1 - dx) * dy * im[y0 + 1, x0] + dx * dy * im[y0 + 1, x0 + 1])
    out[valid, -1] = 1
    return out

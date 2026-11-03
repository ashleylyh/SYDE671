"""B.1-B.3, following "Multi-Image Matching using Multi-Scale Oriented Patches" (Brown et al.), single scale.

Corner coordinates are 2 x n arrays (ys, xs), as returned by harris.get_harris_corners.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .harris import dist2


def anms(h: np.ndarray, coords: np.ndarray, n: int, c_robust: float, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """B.1, Section 3: Adaptive Non-Maximal Suppression.

    Each corner i gets the suppression radius r_i = min_j |x_i - x_j| over the corners j that are clearly stronger,
    f(x_i) < c_robust f(x_j); the strongest corner has r = infinity. The n corners with the largest r are kept: strong
    corners that are also local maxima over a large neighbourhood, so the selection is spread over the whole image
    instead of clustering where the texture is strongest.

    Returns (indices into coords of the kept corners, largest r first; r of every corner).
    """
    f = h[coords[0], coords[1]]
    order = np.argsort(-f)                              # strongest first: the stronger corners of i are a prefix
    pts = coords.T[order].astype(np.float64)            # (k, 2) (y, x)
    fs = f[order]
    r2 = np.full(len(pts), np.inf)
    for start in range(1, len(pts), chunk):             # chunks of rows: the full k x k distance matrix is too big
        stop = min(start + chunk, len(pts))
        d = dist2(pts[start:stop], pts[:stop])          # squared distances to every corner up to this chunk
        stronger = fs[start:stop, None] < c_robust * fs[None, :stop]
        r2[start:stop] = np.where(stronger, d, np.inf).min(axis=1)
    radii = np.empty(len(pts))
    radii[order] = np.sqrt(r2)
    keep = np.argsort(-radii, kind="stable")[:n]
    return keep, radii


def descriptors(im: np.ndarray, coords: np.ndarray, window: int, patch: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """B.2, Section 4: axis-aligned patch x patch descriptors sampled from a window x window region around each corner.

    The grayscale image is blurred first (the paper samples from a coarser pyramid level), then every
    (window / patch)-th pixel of the window is taken: a 40 x 40 window gives an 8 x 8 descriptor with spacing 5, which
    is robust to small errors in the corner position. Each descriptor is bias / gain normalised (zero mean, unit
    standard deviation), so it does not change with brightness and contrast.

    Returns (descriptors (k, patch * patch), the raw window x window patches (k, window, window) for figures).
    The corners must be at least window / 2 from the border (harris.py discards 20 px).
    """
    step = window // patch
    blurred = gaussian_filter(im, sigma)
    half = window // 2
    rows = []
    windows = []
    for y, x in coords.T:
        win = blurred[y - half:y + half, x - half:x + half]
        windows.append(im[y - half:y + half, x - half:x + half])
        d = win[step // 2::step, step // 2::step].ravel()   # sample centres of the 5 x 5 cells
        rows.append((d - d.mean()) / (d.std() + 1e-8))
    return np.array(rows), np.array(windows)


def nearest_two(desc1: np.ndarray, desc2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every descriptor of image 1: its nearest neighbour in image 2, and the ratio e_1-NN / e_2-NN of the
    (Euclidean) distances to its nearest and second-nearest neighbours."""
    d = np.sqrt(np.maximum(dist2(desc1, desc2), 0))
    two = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(len(d))
    return two[:, 0], d[rows, two[:, 0]] / np.maximum(d[rows, two[:, 1]], 1e-12)


def match(desc1: np.ndarray, desc2: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """B.3, Section 5: Lowe's ratio test. A feature is matched to its nearest neighbour only if that neighbour is
    clearly better than the second nearest (e_1-NN / e_2-NN < ratio): a correct match has one good partner, a wrong one
    is about as close to several features (Figure 6b of the paper).

    Returns ((m, 2) index pairs (i in image 1, j in image 2), the ratio of every feature of image 1).
    """
    nn, ratios = nearest_two(desc1, desc2)
    i = np.nonzero(ratios < ratio)[0]
    return np.column_stack([i, nn[i]]), ratios

"""Every experiment on the webpage: one function per section, computing numbers and images.

Nothing here writes files (report.py does). Experiments computes each result on first use.
"""

import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from algorithm.config import MOSAICS, POINTS_JSON, RECTIFICATIONS, RECTIFY_DIR, MosaicSet, Params, Rectification, find_input
from algorithm.features import anms, descriptors, match, nearest_two
from algorithm.harris import get_harris_corners
from algorithm.homography import apply_H, computeH, homography_system, reprojection_error
from algorithm.io import load_image, pair_key, read_points, scaled_points
from algorithm.ransac import ransac_H
from algorithm.mosaic import canvas_bounds, chain_to_reference, feather, overwrite, two_band, warp_with_weight
from algorithm.warp import Bounds, output_bounds, warpImageBilinear, warpImageNearestNeighbor


def _timed(fn, *args, repeats: int = 1):
    """Result and run time in seconds (the fastest of `repeats` runs, which filters out OS noise)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - start)
    return out, best


# ---------------------------------------------------------------- A.1

@dataclass
class Shots:
    """One mosaic set: its photos, left to right."""
    mset: MosaicSet
    images: list[np.ndarray]


def load_shots(mset: MosaicSet, params: Params) -> Shots:
    return Shots(mset, [load_image(find_input(stem, mset.folder), max_side=params.max_side) for stem in mset.images])


# ---------------------------------------------------------------- A.2

@dataclass
class PairH:
    """A.2 for one pair of neighbouring images: H maps im1 (left) points onto im2 (right)."""
    mset: MosaicSet
    names: tuple[str, str]
    im1: np.ndarray
    im2: np.ndarray
    pts1: np.ndarray                    # n-by-2 (x, y), clicked
    pts2: np.ndarray
    A: np.ndarray                       # (2n, 8)
    b: np.ndarray                       # (2n,)
    H: np.ndarray                       # (3, 3), H[2, 2] = 1

    @property
    def projected(self) -> np.ndarray:
        """pts1 mapped through H, drawn on im2 next to pts2."""
        return apply_H(self.H, self.pts1)

    @property
    def errors(self) -> np.ndarray:
        return reprojection_error(self.H, self.pts1, self.pts2)


def make_pair(shots: Shots, i: int, points: dict) -> PairH:
    """Correspondences of images i and i + 1 of the set (from points.json), the system Ah = b and H."""
    mset = shots.mset
    names = (mset.images[i], mset.images[i + 1])
    entry = points.get(mset.name, {}).get(pair_key(*names))
    if entry is None:
        raise FileNotFoundError(f"no points for {mset.name} {names[0]} -> {names[1]} (run pick_points.py {mset.name})")
    im1, im2 = shots.images[i], shots.images[i + 1]
    pts1, pts2 = scaled_points(entry, "im1", im1.shape), scaled_points(entry, "im2", im2.shape)
    A, b = homography_system(pts1, pts2)
    return PairH(mset, names, im1, im2, pts1, pts2, A, b, computeH(pts1, pts2))


@dataclass
class HCheck:
    """A.2 sanity check on a known homography: recover it from exact points (n = 4) and from noisy ones (n > 4)."""
    H_true: np.ndarray
    rows: list[dict]                    # n, noise (px), max |H - H_true| (relative), mean reprojection error (px)


def make_h_check(seed: int = 0, size: tuple[int, int] = (1200, 900)) -> HCheck:
    """Points spread over a size[0] x size[1] image, mapped through a random but realistic H (small rotation,
    shift and perspective), then fitted again with computeH."""
    rng = np.random.default_rng(seed)
    H_true = np.array([[0.95, -0.08, 310.0], [0.06, 1.02, -25.0], [1.5e-4, -4e-5, 1.0]])
    rows = []
    for n, noise in ((4, 0.0), (8, 0.0), (8, 1.0), (20, 1.0), (50, 1.0)):
        pts1 = rng.uniform((0, 0), size, (n, 2))
        pts2 = apply_H(H_true, pts1)
        noisy = pts2 + rng.normal(0, noise, pts2.shape) if noise else pts2
        H = computeH(pts1, noisy)
        grid = rng.uniform((0, 0), size, (500, 2))           # error over the whole image, not just at the clicked points
        rows.append({"n": n, "noise": noise, "H_diff": float(np.abs(H - H_true).max() / np.abs(H_true).max()),
                     "error": float(np.linalg.norm(apply_H(H, grid) - apply_H(H_true, grid), axis=1).mean())})
    return HCheck(H_true, rows)


# ---------------------------------------------------------------- A.3

@dataclass
class Rectified:
    """A.3: one photo warped so that the clicked quadrilateral becomes an axis-aligned rectangle."""
    rect: Rectification
    image: np.ndarray
    quad: np.ndarray                    # 4-by-2 clicked corners (TL, TR, BR, BL)
    target: np.ndarray                  # 4-by-2 rectangle they are mapped to
    H: np.ndarray
    bounds: Bounds
    nn: np.ndarray                      # RGBA
    bilinear: np.ndarray
    t_nn: float
    t_bilinear: float

    @property
    def size(self) -> tuple[int, int]:
        """Rectangle (width, height) in output pixels."""
        return round(self.target[1, 0] - self.target[0, 0]), round(self.target[3, 1] - self.target[0, 1])


def order_corners(quad: np.ndarray) -> np.ndarray:
    """The 4 clicked corners in the order TL, TR, BR, BL, whatever order they were clicked in: sorted by angle around
    their centroid (clockwise on screen, y down), starting from the one with the smallest x + y."""
    c = quad.mean(axis=0)
    ordered = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def rectangle_for(quad: np.ndarray, aspect: float | None) -> np.ndarray:
    """The target rectangle: anchored at the clicked top-left corner, as tall as the mean of the clicked left / right
    sides, and as wide as aspect * height (or the mean of the top / bottom sides when the aspect is unknown)."""
    tl, tr, br, bl = quad
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    width = aspect * height if aspect else (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    x, y = tl
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


def crop_bounds(full: Bounds, target: np.ndarray, margin: float) -> Bounds:
    """full, limited to the rectangle plus margin * its size on each side."""
    (x0, y0), (x1, y1) = target.min(axis=0), target.max(axis=0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (max(full[0], int(x0 - mx)), max(full[1], int(y0 - my)), min(full[2], int(x1 + mx)), min(full[3], int(y1 + my)))


def make_rectified(rect: Rectification, points: dict, params: Params) -> Rectified:
    entry = points.get("rectify", {}).get(rect.name)
    if entry is None:
        raise FileNotFoundError(f"no corners for {rect.name} (run pick_points.py --rectify)")
    image = load_image(find_input(rect.image, RECTIFY_DIR), max_side=params.max_side)
    quad = order_corners(scaled_points(entry, "im1", image.shape))
    target = rectangle_for(quad, rect.aspect)
    H = computeH(quad, target)                          # exactly 4 points: the exact solution
    bounds = crop_bounds(output_bounds(image.shape, H), target, params.rect_margin)
    nn, t_nn = _timed(warpImageNearestNeighbor, image, H, bounds, repeats=3)
    bil, t_bil = _timed(warpImageBilinear, image, H, bounds, repeats=3)
    return Rectified(rect, image, quad, target, H, bounds, nn, bil, t_nn, t_bil)


# ---------------------------------------------------------------- A.4

@dataclass
class Mosaic:
    """A.4: every image of a set warped into the reference image's projection, and three ways to combine them."""
    shots: Shots
    Hs: list[np.ndarray]                # image i -> reference
    bounds: Bounds
    warped: list[np.ndarray]            # RGB on the canvas
    weights: list[np.ndarray]           # warped tent weights
    alphas: list[np.ndarray]
    overwrite: np.ndarray
    feather: np.ndarray
    two_band: np.ndarray
    seconds: float                      # warping + blending
    clipped: bool = False               # canvas cut to a window around the reference (see make_mosaic)


def make_mosaic(shots: Shots, pair_Hs: list[np.ndarray], params: Params) -> Mosaic:
    """pair_Hs[i]: image i -> image i + 1 (from clicked points in A.4, from RANSAC in B.4)."""
    start = time.perf_counter()
    Hs = chain_to_reference(pair_Hs, shots.mset.reference)
    bounds = canvas_bounds([im.shape for im in shots.images], Hs)
    w, h = bounds[2] - bounds[0] + 1, bounds[3] - bounds[1] + 1
    clipped = max(w, h) > params.max_canvas
    if clipped:
        # A chained H sends an image towards infinity: keep a window around the reference image instead
        rh, rw = shots.images[shots.mset.reference].shape[:2]
        print(f"[experiments] {shots.mset.name}: mosaic would be {w} x {h} px (check the clicked points); "
              f"clipped to the reference image +- {params.clip_canvas} x its size")
        k = params.clip_canvas
        bounds = (max(bounds[0], int(-k * rw)), max(bounds[1], int(-k * rh)),
                  min(bounds[2], int((k + 1) * rw)), min(bounds[3], int((k + 1) * rh)))
    parts = [warp_with_weight(im, H, bounds) for im, H in zip(shots.images, Hs)]
    warped, weights, alphas = (list(x) for x in zip(*parts))
    return Mosaic(shots, Hs, bounds, warped, weights, alphas, overwrite(warped, alphas), feather(warped, weights),
                  two_band(warped, weights, alphas, params.blend_sigma), time.perf_counter() - start, clipped)


# ---------------------------------------------------------------- B.1 / B.2

@dataclass
class Features:
    """B.1 / B.2 for one image: Harris corners, the ANMS selection and its descriptors."""
    name: str
    gray: np.ndarray
    h: np.ndarray                       # Harris response
    coords: np.ndarray                  # 2 x k (ys, xs): the Harris corners above the strength threshold
    n_raw: int                          # Harris corners before the threshold
    keep: np.ndarray                    # indices of the ANMS corners, largest radius first
    radii: np.ndarray                   # ANMS suppression radius of every corner
    descs: np.ndarray                   # (n, 64) normalised descriptors of the ANMS corners
    windows: np.ndarray                 # (n, 40, 40) the raw windows they come from
    t_harris: float
    t_anms: float

    @property
    def anms_coords(self) -> np.ndarray:
        return self.coords[:, self.keep]

    @property
    def strongest(self) -> np.ndarray:
        """The same number of corners as ANMS keeps, chosen by Harris strength alone (for comparison)."""
        f = self.h[self.coords[0], self.coords[1]]
        return self.coords[:, np.argsort(-f)[:len(self.keep)]]


def make_features(name: str, image: np.ndarray, params: Params) -> Features:
    gray = image.mean(axis=2)
    (h, raw), t_harris = _timed(get_harris_corners, gray, params.harris_edge_discard)
    coords = raw[:, h[raw[0], raw[1]] > params.harris_threshold]
    (keep, radii), t_anms = _timed(anms, h, coords, params.anms_corners, params.anms_c_robust)
    descs, windows = descriptors(gray, coords[:, keep], params.window, params.patch, params.descriptor_sigma)
    return Features(name, gray, h, coords, raw.shape[1], keep, radii, descs, windows, t_harris, t_anms)


# ---------------------------------------------------------------- B.3 / B.4

@dataclass
class AutoPair:
    """B.3 / B.4 for one pair of neighbouring images: descriptor matches, RANSAC inliers and H (left -> right)."""
    mset: MosaicSet
    f1: Features
    f2: Features
    matches: np.ndarray                 # (m, 2) indices into the ANMS features of image 1 / image 2
    ratios: np.ndarray                  # e_1-NN / e_2-NN of every ANMS feature of image 1
    nn: np.ndarray                      # its nearest neighbour in image 2
    H: np.ndarray
    inliers: np.ndarray                 # (m,) bool
    seconds: float                      # matching + RANSAC

    @property
    def names(self) -> tuple[str, str]:
        return self.f1.name, self.f2.name

    @property
    def pts1(self) -> np.ndarray:
        """Matched points of image 1 as n-by-2 (x, y)."""
        return self.f1.anms_coords[::-1, self.matches[:, 0]].T.astype(np.float64)

    @property
    def pts2(self) -> np.ndarray:
        return self.f2.anms_coords[::-1, self.matches[:, 1]].T.astype(np.float64)


def make_auto_pair(mset: MosaicSet, f1: Features, f2: Features, params: Params) -> AutoPair:
    start = time.perf_counter()
    matches, ratios = match(f1.descs, f2.descs, params.ratio_threshold)
    nn, _ = nearest_two(f1.descs, f2.descs)
    pair = AutoPair(mset, f1, f2, matches, ratios, nn, np.eye(3), np.zeros(len(matches), dtype=bool), 0.0)
    pair.H, pair.inliers = ransac_H(pair.pts1, pair.pts2, params.ransac_iters, params.ransac_eps,
                                    np.random.default_rng(params.ransac_seed))
    pair.seconds = time.perf_counter() - start
    return pair


def h_distance(H1: np.ndarray, H2: np.ndarray, shape: tuple[int, ...], step: int = 20) -> float:
    """Mean distance (px) between where two homographies send a grid of points covering the image."""
    h, w = shape[:2]
    xs, ys = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    return float(np.linalg.norm(apply_H(H1, grid) - apply_H(H2, grid), axis=1).mean())


class Experiments:
    """Every result above, computed on first access and kept in memory."""

    def __init__(self, params: Params = Params()):
        self.params = params

    @cached_property
    def shots(self) -> list[Shots]:
        return [load_shots(mset, self.params) for mset in MOSAICS]

    @cached_property
    def points(self) -> dict:
        return read_points(POINTS_JSON)

    @cached_property
    def h_check(self) -> HCheck:
        return make_h_check()

    @cached_property
    def pairs(self) -> list[PairH]:
        """Every neighbouring pair that has clicked points; pairs without points are listed and skipped."""
        pairs = []
        for s in self.shots:
            for i in range(len(s.images) - 1):
                try:
                    pairs.append(make_pair(s, i, self.points))
                except FileNotFoundError as e:
                    print(f"[experiments] {e}")
        if not pairs:
            raise FileNotFoundError(f"no clicked points in {POINTS_JSON.name} yet (run pick_points.py)")
        return pairs

    @cached_property
    def _features(self) -> dict[tuple[str, int], Features]:
        return {}

    def features(self, shots: Shots, i: int) -> Features:
        """B.1 / B.2 of image i of a set, computed once."""
        key = (shots.mset.name, i)
        if key not in self._features:
            self._features[key] = make_features(shots.mset.images[i], shots.images[i], self.params)
        return self._features[key]

    @cached_property
    def auto_pairs(self) -> list[AutoPair]:
        """B.3 / B.4 for every neighbouring pair of every set."""
        return [make_auto_pair(s.mset, self.features(s, i), self.features(s, i + 1), self.params)
                for s in self.shots for i in range(len(s.images) - 1)]

    @cached_property
    def auto_mosaics(self) -> list[Mosaic]:
        """B.4: the A.4 mosaics again, with the RANSAC homographies instead of the clicked ones."""
        return [make_mosaic(s, [p.H for p in self.auto_pairs if p.mset.name == s.mset.name], self.params)
                for s in self.shots]

    @cached_property
    def rectified(self) -> list[Rectified]:
        """Every rectification whose corners are clicked; the others are listed and skipped."""
        out = []
        for rect in RECTIFICATIONS:
            try:
                out.append(make_rectified(rect, self.points, self.params))
            except FileNotFoundError as e:
                print(f"[experiments] {e}")
        if not out:
            raise FileNotFoundError("no rectification corners clicked yet (run pick_points.py --rectify)")
        return out

    @cached_property
    def mosaics(self) -> list[Mosaic]:
        """Every set whose neighbouring pairs all have clicked points."""
        out = []
        for shots in self.shots:
            pairs = [p for p in self.pairs if p.mset.name == shots.mset.name]
            if len(pairs) == len(shots.images) - 1:
                out.append(make_mosaic(shots, [p.H for p in pairs], self.params))
            else:
                print(f"[experiments] {shots.mset.name}: {len(pairs)} of {len(shots.images) - 1} pairs clicked, no mosaic")
        if not out:
            raise FileNotFoundError("no set has all its pairs clicked")
        return out

"""B.4: 4-point RANSAC for a robust homography (from scratch)."""

import numpy as np

from .homography import apply_H, computeH


def ransac_H(pts1: np.ndarray, pts2: np.ndarray, iters: int, eps: float,
             rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(H, inlier mask) mapping pts1 onto pts2 (n-by-2 (x, y)), robust to wrong matches.

    Each iteration samples 4 matches, computes the exact H through them, and counts the inliers: the matches that H
    maps to within eps pixels. The largest inlier set wins, and H is refitted with least squares (computeH) on all of
    its inliers, then the inliers are recounted with the refitted H.
    """
    rng = rng or np.random.default_rng(0)
    n = len(pts1)
    if n < 4:
        raise ValueError(f"RANSAC needs >= 4 matches, got {n}")
    best = np.zeros(n, dtype=bool)
    for _ in range(iters):
        sample = rng.choice(n, 4, replace=False)
        try:
            H = computeH(pts1[sample], pts2[sample])
        except np.linalg.LinAlgError:                  # degenerate sample (e.g. collinear points)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            inliers = np.linalg.norm(apply_H(H, pts1) - pts2, axis=1) < eps
        if inliers.sum() > best.sum():
            best = inliers
    H = computeH(pts1[best], pts2[best])
    inliers = np.linalg.norm(apply_H(H, pts1) - pts2, axis=1) < eps
    return H, inliers

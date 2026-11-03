"""A.2: Recover homographies, p' = Hp.

Points are n-by-2 arrays of (x, y) (column, row), as in the PDF.
"""

import numpy as np


def homography_system(im1_pts: np.ndarray, im2_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The linear system A h = b for the 8 unknowns h = (h11, h12, h13, h21, h22, h23, h31, h32), h33 = 1.

    Each correspondence (x, y) -> (x', y') gives two rows, from x' = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
    and the same for y'. Returns A of shape (2n, 8) and b of shape (2n,).
    """
    A, b = [], []
    for (x, y), (xp, yp) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        b.extend([xp, yp])
    return np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)


def computeH(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """3x3 homography H with H @ [x1, y1, 1] ~ [x2, y2, 1], h33 = 1.

    PDF: with n = 4 the system has an exact solution; with n > 4 it is overdetermined and solved with least squares.
    """
    assert im1_pts.shape == im2_pts.shape and len(im1_pts) >= 4, "need >= 4 correspondences (n-by-2)"
    A, b = homography_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map n-by-2 (x, y) points through H (homogeneous coordinates, then divide by w)."""
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def reprojection_error(H: np.ndarray, im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """Distance in pixels between each H-mapped im1 point and its clicked im2 point."""
    return np.linalg.norm(apply_H(H, im1_pts) - im2_pts, axis=1)

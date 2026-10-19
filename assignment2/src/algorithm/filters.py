"""Parts 1.2, 1.3, the orientation bells & whistles, and 2.1: derivatives, Gaussians, DoG, unsharp masking.

PDF (1.2): "You can use scipy.signal.convolve2d" from here on. Every convolution uses
mode="same" (output size = input size) and boundary="symm" (mirror the image at its border),
so the border does not look like an edge the way zero fill does (see Part 1.1).
"""

import math

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

from .convolution import DX, DY


def conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution, same size, mirrored border."""
    return convolve2d(image, kernel, mode="same", boundary="symm")


def conv_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """conv on each color channel (a grayscale image is passed straight through)."""
    if image.ndim == 2:
        return conv(image, kernel)
    return np.stack([conv(image[..., c], kernel) for c in range(image.shape[2])], axis=-1)


def ksize_for(sigma: float) -> int:
    """Odd kernel size covering ±3 sigma, where the Gaussian has fallen to ~1% of its peak."""
    return 2 * math.ceil(3 * sigma) + 1


def gaussian_2d(sigma: float, ksize: int | None = None) -> np.ndarray:
    """PDF: "use cv2.getGaussianKernel() to create a 1D gaussian and then take an outer product with its transpose"."""
    g = cv2.getGaussianKernel(ksize or ksize_for(sigma), sigma)     # (k, 1), sums to 1
    return g @ g.T


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Convolution with gaussian_2d(sigma), done as two 1-D passes (G = g gᵀ is separable, so the result is the same).

    For the large sigmas of Part 2 (a 49×49 kernel at sigma = 8) this is ~25× fewer multiplies than the 2-D kernel.
    """
    g = cv2.getGaussianKernel(ksize_for(sigma), sigma)
    return conv_rgb(conv_rgb(image, g), g.T)


def gradients(image: np.ndarray, dx: np.ndarray = DX, dy: np.ndarray = DY) -> tuple[np.ndarray, np.ndarray]:
    """Partial derivatives (df/dx, df/dy). With Dx = [1, 0, -1], df/dx[i, j] = f[i, j+1] - f[i, j-1]."""
    return conv(image, dx), conv(image, dy)


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx ** 2 + gy ** 2)


def dog_filters(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of Gaussian filters G * Dx and G * Dy, as one kernel each.

    mode="full" keeps the whole result (one column / row larger than G), so by associativity
    (f * G) * Dx = f * (G * Dx) holds exactly away from the border.
    """
    return convolve2d(g, DX, mode="full"), convolve2d(g, DY, mode="full")


def arctan2_own(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Angle of the vector (x, y) in (-pi, pi], with no built-in angle function.

    1. Fold every vector into the first octant: a = min(|x|, |y|) / max(|x|, |y|) is in [0, 1].
    2. atan(a) on [0, 1] from the polynomial of Abramowitz & Stegun 4.4.49 (error <= 1e-5 rad).
    3. Unfold: swap to pi/2 - angle if |y| > |x|, mirror to pi - angle if x < 0, negate if y < 0.
    """
    ax, ay = np.abs(x), np.abs(y)
    big, small = np.maximum(ax, ay), np.minimum(ax, ay)
    a = np.divide(small, big, out=np.zeros_like(big, dtype=np.float64), where=big > 0)
    s = a * a
    r = a * (0.9998660 + s * (-0.3302995 + s * (0.1801410 + s * (-0.0851330 + s * 0.0208351))))
    r = np.where(ay > ax, np.pi / 2 - r, r)
    r = np.where(x < 0, np.pi - r, r)
    return np.where(y < 0, -r, r)


def orientation_rgb(gx: np.ndarray, gy: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Gradient direction as HSV hue, saturation 1, brightness = `value` (in [0, 1]).

    Image rows grow downwards, so the angle is measured counter-clockwise from +x with y pointing up:
    theta = atan2(-gy, gx). Hue = theta / 2pi wraps around, so 0 and 2pi get the same color.
    """
    theta = arctan2_own(-gy, gx)
    hue = (theta / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.clip(value, 0, 1)], axis=-1))


def unsharp_kernel(g: np.ndarray, alpha: float) -> np.ndarray:
    """Unsharp mask as one filter: f + alpha (f - f * G) = f * ((1 + alpha) e - alpha G), e = unit impulse."""
    impulse = np.zeros_like(g)
    impulse[g.shape[0] // 2, g.shape[1] // 2] = 1.0
    return (1 + alpha) * impulse - alpha * g

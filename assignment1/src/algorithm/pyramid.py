"""Downsampling and Gaussian pyramids (own implementation; reused in Assignment 2).

PDF: library Gaussian / Laplacian pyramids are not allowed, so the kernel and the
blur-then-subsample step are written here; only the 1-D convolution is from scipy.
"""

from typing import Callable

import numpy as np
import scipy.ndimage as ndi


def gaussian_kernel(sigma: float = 1.0, radius: int = 2) -> np.ndarray:
    """Normalized 1-D Gaussian kernel of length 2 * radius + 1."""
    k = np.exp(-0.5 * np.arange(-radius, radius + 1) ** 2 / sigma ** 2)
    return k / k.sum()


def downsample_naive(im: np.ndarray) -> np.ndarray:
    """Keep every 2nd row and column. No blur, so fine textures alias."""
    return im[::2, ::2]


def downsample_box(im: np.ndarray) -> np.ndarray:
    """2x2 block average (odd sizes are trimmed by one row / column)."""
    h, w = im.shape[:2]
    im = im[:h - h % 2, :w - w % 2]
    return im.reshape(h // 2, 2, w // 2, 2, *im.shape[2:]).mean(axis=(1, 3))


def downsample(im: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur, then keep every 2nd row and column.

    PDF: "blur before downsampling to avoid aliasing". The 5-tap kernel is separable,
    so it is applied along rows and then columns.
    """
    k = gaussian_kernel(sigma)
    blurred = ndi.convolve1d(im, k, axis=0, mode="reflect")
    blurred = ndi.convolve1d(blurred, k, axis=1, mode="reflect")
    return blurred[::2, ::2]


def build_pyramid(im: np.ndarray, min_size: int = 64,
                  down: Callable[[np.ndarray], np.ndarray] = downsample) -> list[np.ndarray]:
    """Gaussian pyramid [full, 1/2, 1/4, ...] down to `min_size` in the smallest dimension.

    PDF: "Implement a multi-scale version using an image pyramid."
    The last level is the first one with min(height, width) <= min_size.
    """
    pyramid = [im]
    while min(pyramid[-1].shape[:2]) > min_size:
        pyramid.append(down(pyramid[-1]))
    return pyramid


def make_lowres(plate: np.ndarray, target_height: int = 1024) -> np.ndarray:
    """Own low-res version of a plate (PDF: "You will have to create your own low-res versions").

    Halves the plate with downsample() while that brings its height closer to target_height
    (e.g. 9700 -> 4850 -> 2425 -> 1212 px).
    """
    while abs(plate.shape[0] / 2 - target_height) < abs(plate.shape[0] - target_height):
        plate = downsample(plate)
    return plate

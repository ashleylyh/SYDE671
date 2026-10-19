"""Part 1.1: 2-D convolution from scratch, numpy only.

PDF: "Implement it with four for loops, then two for loops. Implement padding, with zero fill
values." Both versions return the same size as the input, so they match
scipy.signal.convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0).
"""

import numpy as np

# PDF: finite difference operators
DX = np.array([[1, 0, -1]], dtype=np.float32)
DY = np.array([[1], [0], [-1]], dtype=np.float32)


def box_filter(size: int = 9) -> np.ndarray:
    """size x size averaging kernel (sums to 1)."""
    return np.full((size, size), 1.0 / size ** 2, dtype=np.float32)


def zero_pad(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Surround the image with zeros: a new array with the image copied into its middle."""
    h, w = image.shape
    out = np.zeros((h + top + bottom, w + left + right), dtype=image.dtype)
    out[top:top + h, left:left + w] = image
    return out


def _prepare(image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip the kernel (convolution, not correlation) and zero-pad the image for a "same"-size output.

    Same centring as convolve2d(mode="same"): kh // 2 rows on top, (kh - 1) // 2 below
    (equal for odd kernels), and likewise for the columns.
    """
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = zero_pad(image, kh // 2, (kh - 1) // 2, kw // 2, (kw - 1) // 2)
    return padded, flipped


def convolve_4loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """out[i, j] = sum over (m, n) of flipped[m, n] * padded[i + m, j + n], one multiply at a time."""
    padded, flipped = _prepare(image, kernel)
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            total = 0.0
            for m in range(kh):
                for n in range(kw):
                    total += flipped[m, n] * padded[i + m, j + n]
            out[i, j] = total
    return out


def convolve_2loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same as convolve_4loops, but the two kernel loops become one array multiply-and-sum per pixel."""
    padded, flipped = _prepare(image, kernel)
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(flipped * padded[i:i + kh, j:j + kw])
    return out

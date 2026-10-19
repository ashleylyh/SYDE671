"""Part 2.2: hybrid images (Oliva, Torralba & Schyns, SIGGRAPH 2006).

align_images is a port of the course's align_images.m. The two pairs of reference points
(clicked by pick_points.py) come in as arguments, so a rerun doesn't ask for clicks again.
"""

import math

import cv2
import numpy as np
import scipy.ndimage as ndi

from .filters import gaussian_blur


# ---------------------------------------------------------------- alignment (port of align_images.m)

Points = tuple[tuple[float, float], tuple[float, float]]       # ((x1, y1), (x2, y2)), pixel coordinates


def _pad_to_centre(im: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Pad with zeros so (cx, cy) becomes the image centre.

    align_images.m: tx = round((w/2 - cx) * 2), padded before ('pre') if tx > 0, else after ('post').
    """
    h, w = im.shape[:2]
    tx = round(((w - 1) / 2 - cx) * 2)          # 0-based pixel centres: the centre is (w - 1) / 2
    ty = round(((h - 1) / 2 - cy) * 2)
    pad = [(max(ty, 0), max(-ty, 0)), (max(tx, 0), max(-tx, 0))] + [(0, 0)] * (im.ndim - 2)
    return np.pad(im, pad)


def _resize(im: np.ndarray, scale: float) -> np.ndarray:
    """imresize(im, scale, 'bilinear'); INTER_AREA when shrinking, which averages like imresize's antialiasing."""
    h, w = im.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def shrink_to(im: np.ndarray, max_side: int) -> np.ndarray:
    """Resize so the longest side is <= max_side (never enlarges)."""
    scale = max_side / max(im.shape[:2])
    return _resize(im, scale) if scale < 1 else im


def _angle(p: Points) -> float:
    """theta = atan2(-(y2 - y1), x2 - x1): y points down in an image, so -dy makes the angle counter-clockwise."""
    (x1, y1), (x2, y2) = p
    return math.atan2(-(y2 - y1), x2 - x1)


def _crop_to(im: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop equally from both sides (ceil on the top / left, floor on the bottom / right, as in align_images.m)."""
    top, left = math.ceil((im.shape[0] - h) / 2), math.ceil((im.shape[1] - w) / 2)
    return im[top:top + h, left:left + w]


def align_images(im1: np.ndarray, im2: np.ndarray, p1: Points, p2: Points) -> tuple[np.ndarray, np.ndarray]:
    """Translate, scale and rotate so that p1 in im1 lands on p2 in im2; both are returned at the same size.

    1. Translate: pad each image so the midpoint of its two points is the image centre.
    2. Scale: shrink the image whose two points are farther apart, so both distances are equal.
    3. Rotate im1 about its centre by the difference of the two angles (scipy's positive angle
       is counter-clockwise on screen, like imrotate). reshape=True keeps the whole rotated image.
    4. Crop both, on both sides, to the smaller height and width.
    """
    im1 = _pad_to_centre(im1, *np.mean(p1, axis=0))
    im2 = _pad_to_centre(im2, *np.mean(p2, axis=0))

    len1, len2 = math.dist(*p1), math.dist(*p2)
    if len2 / len1 < 1:
        im1 = _resize(im1, len2 / len1)
    else:
        im2 = _resize(im2, len1 / len2)

    dtheta = _angle(p2) - _angle(p1)
    im1 = ndi.rotate(im1, math.degrees(dtheta), axes=(1, 0), reshape=True, order=1)

    h, w = min(im1.shape[0], im2.shape[0]), min(im1.shape[1], im2.shape[1])
    return _crop_to(im1, h, w), _crop_to(im2, h, w)


def fill_holes(im: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Give every invalid pixel the value of the nearest valid one, so a hole's border isn't a sharp edge."""
    rows, cols = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return im[rows, cols]


def align_and_crop(im1: np.ndarray, im2: np.ndarray, p1: Points, p2: Points
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """align_images, then crop both to the frame of im2 (the reference), so im2 is kept whole.

    The starter crops the black padding by hand. Here a channel of ones is aligned along with each image;
    after padding, scaling and rotating it is 1 exactly where real pixels are. Where im1 doesn't cover
    im2's frame (e.g. the corners left by rotating it), its holes are filled with the nearest real pixel,
    and the returned mask (True = real pixel of im1) lets the caller fade im1's contribution out there.
    """
    ones = lambda im: np.dstack([im, np.ones(im.shape[:2], dtype=im.dtype)])
    a1, a2 = align_images(ones(im1), ones(im2), p1, p2)
    real1, real2 = a1[..., -1] > 0.99, a2[..., -1] > 0.99
    rows, cols = np.flatnonzero(real2.any(axis=1)), np.flatnonzero(real2.any(axis=0))
    box = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    real1 = real1[box]
    out1 = fill_holes(a1[box][..., :-1], real1)
    return np.clip(out1, 0, 1), np.clip(a2[box][..., :-1], 0, 1), real1


# ---------------------------------------------------------------- hybrid

def low_pass(im: np.ndarray, sigma: float) -> np.ndarray:
    """PDF: "For a low-pass filter, Oliva et al. suggest using a standard 2D Gaussian filter"."""
    return gaussian_blur(im, sigma)


def high_pass(im: np.ndarray, sigma: float) -> np.ndarray:
    """PDF: "the impulse filter minus the Gaussian filter (... subtracting the Gaussian-filtered image from the original)"."""
    return im - gaussian_blur(im, sigma)


def hybrid_image(im_low: np.ndarray, im_high: np.ndarray, sigma_low: float, sigma_high: float) -> np.ndarray:
    """Low frequencies of im_low + high frequencies of im_high (added; the high-pass part is zero-mean)."""
    return np.clip(low_pass(im_low, sigma_low) + high_pass(im_high, sigma_high), 0, 1)


def cutoff_frequency(sigma: float) -> float:
    """Frequency (cycles / pixel) where a Gaussian of std sigma passes half the amplitude.

    The Gaussian's Fourier transform is exp(-2 pi^2 sigma^2 f^2); it equals 1/2 at
    f = sqrt(ln 2 / 2) / (pi sigma) ~= 0.187 / sigma. The high-pass filter (impulse - Gaussian) passes
    1/2 at the same frequency.
    """
    return math.sqrt(math.log(2) / 2) / (math.pi * sigma)


def gaussian_response(f: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-2 * math.pi ** 2 * sigma ** 2 * f ** 2)


def log_spectrum(gray: np.ndarray) -> np.ndarray:
    """PDF: np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))); a tiny epsilon avoids log(0)."""
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def to_gray(im: np.ndarray) -> np.ndarray:
    """Luminance (ITU-R 601 weights, as rgb2gray / Pillow "L")."""
    return im if im.ndim == 2 else im @ np.array([0.299, 0.587, 0.114], dtype=im.dtype)

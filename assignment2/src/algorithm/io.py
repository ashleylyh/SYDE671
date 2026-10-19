"""Reading and writing images."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: Path, gray: bool = False, max_side: int | None = None) -> np.ndarray:
    """Load an image as float32 in [0, 1]: (H, W) if gray, else (H, W, 3).

    EXIF orientation is applied (phone photos), and the image is shrunk so its longest side
    is <= max_side (PDF: "use medium-size images").
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L" if gray else "RGB")
        if max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(im) -> np.ndarray:
    """Float image in [0, 1] (or a uint8 array / PIL image) -> uint8 array."""
    arr = np.asarray(im)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_pil(im) -> Image.Image:
    return im if isinstance(im, Image.Image) else Image.fromarray(to_uint8(im))


def save_jpg(im, path: Path, max_side: int | None = None, quality: int = 90) -> Path:
    """Save a [0, 1] float array, uint8 array or PIL image as JPG."""
    out = to_pil(im).copy()
    if max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=quality)
    return path

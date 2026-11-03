"""Reading and writing images and clicked points."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()                  # lets Pillow open the iPhone .HEIC photos


def load_image(path: Path, gray: bool = False, max_side: int | None = None) -> np.ndarray:
    """Load an image as float32 in [0, 1]: (H, W) if gray, else (H, W, 3).

    EXIF orientation is applied (phone photos), and the image is shrunk so its longest side is <= max_side.
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


# ---------------------------------------------------------------- clicked points (pick_points.py)
#
# points.json: {set name: {"<left>|<right>": {"im1": [[x, y], ...], "im2": [[x, y], ...], "size1": [w, h], "size2": [w, h]}}}
# (x, y) are pixel coordinates with (0, 0) at the centre of the top-left pixel (PDF A.3 tip), in an image of size1 / size2.

def pair_key(left: str, right: str) -> str:
    return f"{left}|{right}"


def read_points(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_points(path: Path, points: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(points, indent=1))


def scaled_points(entry: dict, which: str, shape: tuple[int, ...]) -> np.ndarray:
    """entry[which] ("im1" / "im2") as an n-by-2 (x, y) array, rescaled to an image of this shape.

    The points are clicked at one max_side; rescaling keeps them valid if Params.max_side changes later.
    """
    pts = np.asarray(entry[which], dtype=np.float64)
    w, h = entry["size1" if which == "im1" else "size2"]
    sx, sy = shape[1] / w, shape[0] / h
    # pixel centres: x_new + 0.5 = (x_old + 0.5) * scale
    return np.column_stack([(pts[:, 0] + 0.5) * sx - 0.5, (pts[:, 1] + 0.5) * sy - 0.5])

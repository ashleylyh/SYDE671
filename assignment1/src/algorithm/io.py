"""Reading plates and writing images."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Full-size plates are ~36 MP; disable Pillow's decompression-bomb guard
Image.MAX_IMAGE_PIXELS = None


def load_plate(path: Path) -> np.ndarray:
    """Load a glass-plate scan as a 2-D float32 array in [0, 1].

    PDF: "Convert images to floats and the same scale. JPGs are uint8; TIFFs may be uint16."
    8-bit data (JPG) is divided by 255 and anything wider by 65535 (16-bit TIFF, whether Pillow
    reads it as uint16 "I;16" or as int32 "I"), so every plate lands in [0, 1].
    float32 keeps a 9715 x 3741 plate at ~145 MB.
    """
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:                       # saved as RGB: all channels are equal
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        scale = 65535.0
    else:                                   # already float
        scale = 1.0
    out = arr.astype(np.float32) / scale
    assert 0.0 <= out.min() and out.max() <= 1.0, f"{path}: values outside [0, 1] after scaling"
    return out


def build_catalog(data_dir: Path, suffixes: tuple[str, ...] = (".jpg",)) -> pd.DataFrame:
    """One row per plate, read from the file headers only (no pixels are decoded).

    `type` is the file-name suffix (v / a / u). The index is the file stem
    (e.g. "00125v"), so a plate is loaded with load_plate(df.loc[name, "path"]).
    A missing directory gives an empty catalog.
    """
    records = []
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in suffixes) if data_dir.is_dir() else []
    for path in paths:
        with Image.open(path) as img:
            width, height = img.size
        records.append({"name": path.stem, "path": path, "type": path.stem[-1], "shape": (height, width)})
    columns = ["name", "path", "type", "shape"]
    return pd.DataFrame(records, columns=columns).set_index("name").sort_values(["type", "name"])


def to_uint8(im) -> np.ndarray:
    """Float image in [0, 1] (or a uint8 array / PIL image) -> uint8 array."""
    arr = np.asarray(im)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_pil(im) -> Image.Image:
    return im if isinstance(im, Image.Image) else Image.fromarray(to_uint8(im))


def to_web(im, max_side: int) -> Image.Image:
    """Small uint8 copy of an image, longest side <= max_side (resizing is allowed by the PDF)."""
    out = to_pil(im).copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def save_jpg(im, path: Path, max_side: int | None = None, quality: int = 90) -> Path:
    """Save a [0, 1] float array, uint8 array or PIL image as JPG (PDF: "Save outputs as JPG")."""
    out = to_web(im, max_side) if max_side else to_pil(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=quality)
    return path

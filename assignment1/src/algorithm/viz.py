"""Image grids and crops for the notebook and the webpage figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .io import to_pil


def image_grid(images, titles, size: float = 4.0, cols: int | None = None, max_side: int = 1024,
               title_size: float | None = None) -> plt.Figure:
    """Images (float [0, 1], uint8 or PIL) in a grid, one row by default.

    Large images are strided down to about `max_side` for display only.
    """
    cols = cols or len(images)
    rows = -(-len(images) // cols)                          # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, title in zip(axes.flat, images, titles):
        im = np.asarray(im)
        step = max(1, -(-max(im.shape[:2]) // max_side))
        ax.imshow(im[::step, ::step], cmap="gray", vmin=0, vmax=255 if im.dtype == np.uint8 else 1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=title_size)
    fig.tight_layout()
    return fig


def show_images(images, titles, **kwargs) -> None:
    """image_grid, shown inline (notebook)."""
    image_grid(images, titles, **kwargs)
    plt.show()


def save_panel(images, titles, path: Path, size: float = 3.4, cols: int | None = None,
               dpi: int = 110, quality: int = 85) -> Path:
    """image_grid, saved as one labelled JPG figure (webpage)."""
    fig = image_grid(images, titles, size=size, cols=cols, max_side=1200, title_size=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", pil_kwargs={"quality": quality})
    plt.close(fig)
    return path


def center_crop(im: np.ndarray, size: int) -> np.ndarray:
    """size x size crop around the image centre."""
    h, w = im.shape[:2]
    return im[h // 2 - size // 2:h // 2 + size // 2, w // 2 - size // 2:w // 2 + size // 2]


def corner_crop(im: np.ndarray, size: int = 600, at: float = 0.88) -> np.ndarray:
    """size x size crop ending at (at * H, at * W): the bottom-right, inside the plate border.

    Rotation and scale errors grow with distance from the centre, so B6 is judged here.
    """
    h, w = im.shape[:2]
    y, x = int(at * h), int(at * w)
    return im[y - size:y, x - size:x]


def draw_box(im, box: tuple[int, int, int, int], color=(0, 255, 0), width: int | None = None) -> Image.Image:
    """Copy of the image with a (top, bottom, left, right) box drawn on it."""
    out = to_pil(im).convert("RGB")
    top, bottom, left, right = box
    width = width or max(3, out.height // 150)
    ImageDraw.Draw(out).rectangle([left, top, right - 1, bottom - 1], outline=color, width=width)
    return out

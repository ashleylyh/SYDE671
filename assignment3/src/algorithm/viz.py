"""Image grids and the correspondence / corner / match figures for the webpage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .io import to_pil

# Categorical colours for numbered points (cycled); distinct on both bright sky and dark buildings
POINT_COLORS = ((255, 214, 0), (0, 200, 255), (255, 80, 160), (120, 255, 90), (255, 140, 0), (190, 120, 255))


def image_grid(images, titles, size: float = 4.0, cols: int | None = None,
               title_size: float | None = None) -> plt.Figure:
    """Images (float [0, 1] or uint8; gray or RGB) in a grid, one row by default."""
    cols = cols or len(images)
    rows = -(-len(images) // cols)                          # ceil division
    h, w = np.asarray(images[0]).shape[:2]                  # cells follow the first image's shape (+ room for a title)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, (size * h / w + 0.35) * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, title in zip(axes.flat, images, titles):
        im = np.asarray(im)
        ax.imshow(im, cmap="gray", vmin=0, vmax=255 if im.dtype == np.uint8 else 1, interpolation="nearest")
        ax.set_title(title, fontsize=title_size)
    fig.tight_layout()
    return fig


def show_images(images, titles, **kwargs) -> None:
    image_grid(images, titles, **kwargs)
    plt.show()


def save_panel(images, titles, path: Path, size: float = 3.4, cols: int | None = None,
               dpi: int = 110, quality: int = 85) -> Path:
    """image_grid, saved as one labelled JPG figure (webpage)."""
    fig = image_grid(images, titles, size=size, cols=cols, title_size=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": quality})
    plt.close(fig)
    return path


# ---------------------------------------------------------------- A.2 figures

def mark_points(im, points, projected=None) -> Image.Image:
    """Copy of the image with numbered rings at `points`; `projected` (optional) drawn as crosses in the same colours."""
    out = to_pil(im).convert("RGB")
    draw = ImageDraw.Draw(out)
    r = max(6, out.width // 90)
    lw = max(2, r // 3)
    for i, (x, y) in enumerate(points):
        c = POINT_COLORS[i % len(POINT_COLORS)]
        draw.ellipse([x - r, y - r, x + r, y + r], outline=c, width=lw)
        draw.text((x + r + 3, y - 2 * r), str(i + 1), fill=c, font_size=2 * r, stroke_width=2, stroke_fill=(0, 0, 0))
    for i, (x, y) in enumerate(projected if projected is not None else []):
        c = POINT_COLORS[i % len(POINT_COLORS)]
        draw.line([x - r, y - r, x + r, y + r], fill=c, width=lw)
        draw.line([x - r, y + r, x + r, y - r], fill=c, width=lw)
    return out


def correspondences(im1, im2, pts1, pts2, projected=None, gap: int = 12) -> Image.Image:
    """The two images side by side (same height) with their numbered correspondences."""
    a, b = mark_points(im1, pts1), mark_points(im2, pts2, projected)
    out = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    return out


# ---------------------------------------------------------------- A.3 / A.4 figures

def on_black(rgba: np.ndarray) -> np.ndarray:
    """A warped RGBA image as RGB, black where alpha = 0."""
    return rgba[..., :-1] * rgba[..., -1:]


def draw_quad(im, quad, color=(255, 214, 0)) -> Image.Image:
    """Copy of the image with the clicked quadrilateral outlined and its corners numbered."""
    out = to_pil(im).convert("RGB")
    draw = ImageDraw.Draw(out)
    lw = max(2, out.width // 300)
    pts = [tuple(p) for p in quad]
    draw.line(pts + [pts[0]], fill=color, width=lw)
    r = 3 * lw
    for i, (x, y) in enumerate(pts, 1):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=lw)
        draw.text((x + r + 2, y - 3 * r), str(i), fill=color, font_size=5 * r, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def zoom(im, box: tuple[int, int, int, int], scale: int = 3) -> Image.Image:
    """Crop (left, top, right, bottom) of an image, enlarged with nearest neighbour so single pixels stay visible."""
    crop = to_pil(im).crop(box)
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


# ---------------------------------------------------------------- B.1 / B.2 figures

def save_corners(im, coord_sets, titles, path: Path, size: float = 5.0, dpi: int = 110) -> Path:
    """The image once per coordinate set (2 x n, (ys, xs)), with those corners as small dots."""
    h, w = np.asarray(im).shape[:2]
    fig, axes = plt.subplots(1, len(coord_sets), figsize=(size * len(coord_sets), size * h / w + 0.4), squeeze=False)
    for ax, coords, title in zip(axes[0], coord_sets, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.scatter(coords[1], coords[0], s=4 if coords.shape[1] > 2000 else 9, c="#ffd600", edgecolors="none")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": 85})
    plt.close(fig)
    return path


def save_descriptors(im, coords, windows, descs, path: Path, window: int, dpi: int = 110) -> Path:
    """Left: the image with each feature's window outlined and numbered. Right: per feature, the window and its
    normalised 8 x 8 descriptor (gray scale -3..3 standard deviations)."""
    k = coords.shape[1]
    side = int(round(np.sqrt(descs.shape[1])))
    fig = plt.figure(figsize=(16, 3.9))
    grid = fig.add_gridspec(2, k + 3)
    ax = fig.add_subplot(grid[:, :3])
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    for i, (y, x) in enumerate(coords.T, 1):
        ax.add_patch(plt.Rectangle((x - window / 2, y - window / 2), window, window, fill=False, ec="#ffd600", lw=1.5))
        ax.text(x + window / 2 + 4, y, str(i), color="#ffd600", fontsize=10, fontweight="bold", va="center")
    ax.axis("off")
    for i in range(k):
        a = fig.add_subplot(grid[0, 3 + i])
        a.imshow(windows[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"{i + 1}: {window}×{window}", fontsize=10)
        a.axis("off")
        b = fig.add_subplot(grid[1, 3 + i])
        b.imshow(descs[i].reshape(side, side), cmap="gray", vmin=-3, vmax=3, interpolation="nearest")
        b.set_title(f"{side}×{side} normalised", fontsize=10)
        b.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": 85})
    plt.close(fig)
    return path


# ---------------------------------------------------------------- B.3 / B.4 figures

INK, MUTED, GRID = "#1a2233", "#5a6475", "#dde3ec"      # page text tokens (--fg, --muted, --border)
SERIES = ("#2a78d6", "#eb6834")                         # categorical slots 1-2 (blue, orange), in order
INLIER, OUTLIER = (40, 220, 90), (255, 60, 60)


def draw_matches(im1, im2, pts1, pts2, colors=None, gap: int = 12) -> Image.Image:
    """The two images side by side with a line (and end dots) per match. colors: one RGB per match, default cycled."""
    a, b = to_pil(im1).convert("RGB"), to_pil(im2).convert("RGB")
    out = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    draw = ImageDraw.Draw(out)
    r = max(3, out.width // 500)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        c = colors[i] if colors is not None else POINT_COLORS[i % len(POINT_COLORS)]
        x2 += a.width + gap
        draw.line([x1, y1, x2, y2], fill=c, width=max(1, r // 2))
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=c, width=max(1, r // 2))
    return out


def save_ratio_histogram(correct: np.ndarray, incorrect: np.ndarray, threshold: float, path: Path, dpi: int = 110) -> Path:
    """Our version of Figure 6b of the MOPS paper: the distribution of e_1-NN / e_2-NN for correct and incorrect
    nearest neighbours, with the chosen threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(0, 1, 41)
    for values, color, label in zip((correct, incorrect), SERIES, ("correct nearest neighbour", "incorrect nearest neighbour")):
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=color)
        ax.hist(values, bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{label} (n = {len(values):,})")
    ax.axvline(threshold, color=INK, lw=1.5, ls="--")
    ax.annotate(f"threshold {threshold:g}", (threshold, ax.get_ylim()[1] * 0.55), xytext=(6, 0), textcoords="offset points",
                ha="left", color=INK, fontsize=10)
    ax.set_xlabel("e₁-NN / e₂-NN", color=INK)
    ax.set_ylabel("probability density", color=INK)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", pil_kwargs={"quality": 90})
    plt.close(fig)
    return path

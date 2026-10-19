"""Image grids for the notebook and the webpage figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .io import to_pil


def signed_to_unit(im: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Map a signed response (e.g. a derivative) to [0, 1] with 0 at mid-gray, symmetric about 0.

    The `pct` percentile of |im| maps to black / white (larger values clip), so a few strong
    edges don't make everything else look flat. pct=100 uses the true peak (e.g. for kernels).
    """
    peak = float(np.percentile(np.abs(im), pct)) or 1.0
    return np.clip(0.5 + 0.5 * im / peak, 0, 1)


def bands_to_unit(images, pct: float = 99.5) -> list[np.ndarray]:
    """Signed band-pass images (Laplacian levels) on one shared scale: 0 -> mid-gray, so a row can be compared."""
    peak = float(np.percentile(np.abs(np.stack(images)), pct)) or 1.0
    return [np.clip(0.5 + 0.5 * im / peak, 0, 1) for im in images]


def unit_scale(im: np.ndarray, pct: float = 100) -> np.ndarray:
    """Non-negative image (magnitude, kernel) scaled so its `pct` percentile is white."""
    return np.clip(im / (float(np.percentile(im, pct)) or 1.0), 0, 1)


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
    """image_grid, shown inline (notebook)."""
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


# ---------------------------------------------------------------- Part 2.2 figures

INK, MUTED, GRID = "#1a2233", "#5a6475", "#dde3ec"      # page text tokens (--fg, --muted, --border)
SERIES = ("#2a78d6", "#eb6834")                         # categorical slots 1-2 (blue, orange), in order


def save_spectra(spectra, titles, path: Path, size: float = 3.2, dpi: int = 110) -> Path:
    """Log-magnitude Fourier spectra in one row, on a shared color scale so they can be compared."""
    lo = min(np.percentile(s, 1) for s in spectra)
    hi = max(s.max() for s in spectra)
    fig, axes = plt.subplots(1, len(spectra), figsize=(size * len(spectra), size + 0.5), squeeze=False)
    for ax, s, title in zip(axes.flat, spectra, titles):
        im = ax.imshow(s, cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.colorbar(im, ax=list(axes.flat), fraction=0.012, pad=0.01, label="log |F|")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08, pil_kwargs={"quality": 85})
    plt.close(fig)
    return path


def save_filter_responses(freqs, low, high, cutoffs: tuple[float, float], labels: tuple[str, str], path: Path) -> Path:
    """The two filters' gains against frequency, with each half-gain cutoff marked."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # cutoff labels on opposite sides of their points, so close cutoffs don't collide
    for y, color, label, fc, (dx, ha) in zip((low, high), SERIES, labels, cutoffs, ((-7, "right"), (7, "left"))):
        ax.plot(freqs, y, color=color, lw=2, label=label)
        ax.axvline(fc, color=color, lw=1, ls="--")
        ax.annotate(f"{fc:.3f}", (fc, 0.5), xytext=(dx, 4), textcoords="offset points", ha=ha, color=INK, fontsize=9)
        ax.plot([fc], [0.5], "o", color=color, ms=6, mec="white", mew=1.5)
    ax.set_xlim(0, freqs.max())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frequency (cycles / pixel)", color=MUTED)
    ax.set_ylabel("gain", color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(frameon=False, labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)
    return path


def distance_strip(im, scales=(1.0, 0.5, 0.25, 0.125), gap: int = 12) -> Image.Image:
    """The same image at shrinking sizes, bottom-aligned: a stand-in for looking at it from farther away."""
    base = to_pil(im).convert("RGB")
    copies = [base.resize((max(1, round(base.width * s)), max(1, round(base.height * s))), Image.Resampling.LANCZOS)
              for s in scales]
    out = Image.new("RGB", (sum(c.width for c in copies) + gap * (len(copies) - 1), base.height), "white")
    x = 0
    for c in copies:
        out.paste(c, (x, base.height - c.height))
        x += c.width + gap
    return out


def mark_points(im, points, color=(255, 230, 0)) -> Image.Image:
    """Copy of the image with the clicked reference points drawn as numbered rings."""
    out = to_pil(im).convert("RGB")
    draw = ImageDraw.Draw(out)
    r = max(8, out.width // 60)
    for i, (x, y) in enumerate(points, 1):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=max(3, r // 3))
        draw.text((x + r + 4, y - r), str(i), fill=color, font_size=2 * r)
    return out

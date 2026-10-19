"""Every experiment on the webpage: one function per section, computing numbers and images.

Nothing here writes files (report.py does). Experiments computes each result on first use.
"""

import json
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.signal import convolve2d

from algorithm.config import (ALIGN_POINTS, APPLE, BLENDS, BLURRY, CAMERAMAN, DATA_DIR, FAVORITE, FAVORITE_BLEND, HYBRIDS, ORANGE, SELFIE, SHARP, STARTER_DIR, TAJ,
                              BlendPair, HybridPair, Params, find_input)
from algorithm.convolution import DX, DY, box_filter, convolve_2loops, convolve_4loops
from algorithm.filters import (arctan2_own, conv, conv_rgb, dog_filters, gaussian_2d, gaussian_blur, gradients, magnitude,
                               orientation_rgb, unsharp_kernel)
from algorithm.hybrid import align_and_crop, high_pass, low_pass, shrink_to, to_gray
from algorithm.io import load_image
from algorithm.stacks import blend_stacks, collapse, gaussian_stack, laplacian_stack, step_mask

FD_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)        # thresholds shown for 1.2
DOG_SWEEP = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15)       # and for 1.3


def _timed(fn, *args) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def scipy_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The built-in reference: same output size, zero fill outside the image (like my versions)."""
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1] (higher = closer)."""
    return float(10 * np.log10(1.0 / np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)))


# ---------------------------------------------------------------- Part 1.1

@dataclass
class ConvDemo:
    selfie: np.ndarray                      # grayscale, [0, 1]
    kernels: dict[str, np.ndarray]          # box / Dx / Dy
    results: dict[str, np.ndarray]          # my 2-loop result on the full selfie, per kernel
    rows: list[dict]                        # runtime and max difference per kernel and implementation


def make_conv_demo(params: Params) -> ConvDemo:
    """Convolve my selfie with the 9x9 box, Dx and Dy; compare my two versions with convolve2d.

    The 4-loop version is far too slow for the whole selfie, so all three are timed on a
    centre crop of timing_side x timing_side; the 2-loop version and scipy also run on the full selfie.
    """
    selfie = load_image(find_input(SELFIE), gray=True, max_side=params.selfie_max_side)
    h, w = selfie.shape
    s = params.timing_side
    crop = selfie[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    kernels = {f"{params.box_size}x{params.box_size} box": box_filter(params.box_size), "Dx": DX, "Dy": DY}

    results, rows = {}, []
    for name, k in kernels.items():
        ref_crop, t_scipy_crop = _timed(scipy_same, crop, k)
        out4, t4 = _timed(convolve_4loops, crop, k)
        out2_crop, t2_crop = _timed(convolve_2loops, crop, k)
        ref, t_scipy = _timed(scipy_same, selfie, k)
        out2, t2 = _timed(convolve_2loops, selfie, k)
        results[name] = out2
        rows.append({"kernel": name, "crop": [s, s], "full": [h, w],
                     "t4_crop": t4, "t2_crop": t2_crop, "scipy_crop": t_scipy_crop,
                     "t2_full": t2, "scipy_full": t_scipy,
                     "diff4": float(np.abs(out4 - ref_crop).max()),
                     "diff2": float(np.abs(out2 - ref).max())})
    return ConvDemo(selfie, kernels, results, rows)


# ---------------------------------------------------------------- Part 1.2

@dataclass
class Edges:
    """Gradients of one image, their magnitude and the binarized edge map."""
    gx: np.ndarray
    gy: np.ndarray
    mag: np.ndarray
    threshold: float

    @property
    def edges(self) -> np.ndarray:
        return self.mag > self.threshold


def edges_of(gx: np.ndarray, gy: np.ndarray, threshold: float) -> Edges:
    return Edges(gx, gy, magnitude(gx, gy), threshold)


def make_finite_diff(image: np.ndarray, params: Params) -> Edges:
    """Dx and Dy directly on the cameraman image."""
    return edges_of(*gradients(image), params.fd_threshold)


# ---------------------------------------------------------------- Part 1.3

@dataclass
class DogDemo:
    gaussian: np.ndarray                    # 2-D kernel
    blurred: np.ndarray
    two_step: Edges                         # (f * G) * D
    dog_x: np.ndarray                       # G * Dx
    dog_y: np.ndarray
    one_step: Edges                         # f * (G * D)
    diff_full: float                        # max |two_step - one_step| over the whole image (gx and gy)
    diff_interior: float                    # the same, ignoring a kernel-size band along the border


def make_dog(image: np.ndarray, params: Params) -> DogDemo:
    g = gaussian_2d(params.dog_sigma)
    blurred = conv(image, g)
    two_step = edges_of(*gradients(blurred), params.dog_threshold)
    dog_x, dog_y = dog_filters(g)
    one_step = edges_of(conv(image, dog_x), conv(image, dog_y), params.dog_threshold)
    diffs = [np.abs(a - b) for a, b in ((two_step.gx, one_step.gx), (two_step.gy, one_step.gy))]
    m = g.shape[0]
    return DogDemo(g, blurred, two_step, dog_x, dog_y, one_step,
                   float(max(d.max() for d in diffs)), float(max(d[m:-m, m:-m].max() for d in diffs)))


# ---------------------------------------------------------------- Bells & whistles: orientation

@dataclass
class OrientationDemo:
    shaded: np.ndarray                      # hue = direction, brightness = gradient magnitude
    on_edges: np.ndarray                    # hue = direction, only on edge pixels
    wheel: np.ndarray                       # legend: the color of every gradient direction
    atan_error: float                       # max |arctan2_own - np.arctan2| on the image (check only)


def color_wheel(size: int = 241) -> np.ndarray:
    """Legend: each pixel colored as a gradient pointing from the centre to that pixel."""
    r = np.linspace(-1, 1, size)
    gx, gy = np.meshgrid(r, r)                         # gy grows downwards, like image rows
    rad = magnitude(gx, gy)
    wheel = orientation_rgb(gx, gy, (rad <= 1) & (rad > 0.25))
    wheel[rad > 1] = 1.0                               # white outside the disk
    return wheel


def make_orientation(dog: DogDemo) -> OrientationDemo:
    """Gradient direction of the DoG gradients (less noisy than raw finite differences)."""
    e = dog.one_step
    value = e.mag / np.percentile(e.mag, 99)           # 99th percentile -> full brightness
    error = np.abs(arctan2_own(-e.gy, e.gx) - np.arctan2(-e.gy, e.gx))
    error = np.minimum(error, 2 * np.pi - error)       # -pi and pi are the same direction
    return OrientationDemo(orientation_rgb(e.gx, e.gy, value), orientation_rgb(e.gx, e.gy, e.edges),
                           color_wheel(), float(error.max()))


# ---------------------------------------------------------------- Part 2.1

@dataclass
class SharpenDemo:
    name: str
    original: np.ndarray
    blurred: np.ndarray                     # f * G (low frequencies)
    high: np.ndarray                        # f - f * G (high frequencies, signed)
    sharpened: np.ndarray                   # f * unsharp kernel, alpha = params.sharpen_alpha
    sweep: list[np.ndarray]                 # sharpened with each of params.alphas
    diff: float                             # max |single filter - (f + alpha * high)|: the two are the same


def sharpen(image: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(conv_rgb(image, unsharp_kernel(g, alpha)), 0, 1)


def make_sharpen(name: str, image: np.ndarray, params: Params) -> SharpenDemo:
    g = gaussian_2d(params.sharpen_sigma)
    blurred = conv_rgb(image, g)
    high = image - blurred
    single = conv_rgb(image, unsharp_kernel(g, params.sharpen_alpha))
    diff = float(np.abs(single - (image + params.sharpen_alpha * high)).max())
    return SharpenDemo(name, image, np.clip(blurred, 0, 1), high, np.clip(single, 0, 1),
                       [sharpen(image, g, a) for a in params.alphas], diff)


@dataclass
class SharpenEval:
    original: np.ndarray                    # sharp skyline photo (color)
    blurred: np.ndarray
    resharpened: list[np.ndarray]           # the blurred image sharpened with each of params.alphas
    psnr_blurred: float
    psnr: list[float]                       # PSNR of each resharpened image vs. the original


def make_sharpen_eval(params: Params) -> SharpenEval:
    """PDF: "pick a sharp image, blur it and then try to sharpen it again"."""
    original = load_image(find_input(SHARP), max_side=params.eval_max_side)
    g = gaussian_2d(params.sharpen_sigma)
    blurred = np.clip(conv_rgb(original, g), 0, 1)
    again = [sharpen(blurred, g, a) for a in params.alphas]
    return SharpenEval(original, blurred, again, psnr(blurred, original), [psnr(a, original) for a in again])


# ---------------------------------------------------------------- Part 2.2

@dataclass
class Hybrid:
    pair: HybridPair
    low_orig: np.ndarray                    # inputs as loaded (color)
    high_orig: np.ndarray
    points: dict | None                     # {"low": [[x, y], [x, y]], "high": ...} if the pair was aligned
    low_in: np.ndarray                      # inputs after alignment and resizing (color, same size)
    high_in: np.ndarray
    low: np.ndarray                         # low-pass of the gray low input
    high: np.ndarray                        # high-pass of the gray high input (signed)
    hybrid: np.ndarray                      # gray hybrid
    high_real: np.ndarray                   # True where high_in has real pixels (False: filled holes after alignment)
    high_weight: np.ndarray                 # high_real blurred: the high-pass part fades out over the holes


def make_hybrid(pair: HybridPair, params: Params) -> Hybrid:
    """Load (and align) a pair, then low-pass one, high-pass the other and add them, in grayscale (like the starter)."""
    folder = STARTER_DIR if pair.align else DATA_DIR
    low_orig, high_orig = load_image(find_input(pair.low, folder)), load_image(find_input(pair.high, folder))
    points = None
    if pair.align:
        points = json.loads(ALIGN_POINTS.read_text())[pair.name]
        # hybrid_image_starter.m: [im2, im1] = align_images(im2, im1): the high-pass image is moved onto the
        # low-pass one, and the low-pass (far) image keeps its whole frame
        high_in, low_in, high_real = align_and_crop(high_orig, low_orig, points["high"], points["low"])
    else:
        assert low_orig.shape == high_orig.shape, f"{pair.name}: my own pairs must already be aligned"
        low_in, high_in, high_real = low_orig, high_orig, np.ones(low_orig.shape[:2], dtype=bool)
    low_in, high_in = shrink_to(low_in, params.hybrid_max_side), shrink_to(high_in, params.hybrid_max_side)
    high_real = shrink_to(high_real.astype(np.float32), params.hybrid_max_side) > 0.5
    # Over filled holes the high-pass part fades to 0 (the far image shows alone), with no sharp edge
    high_weight = gaussian_blur(high_real.astype(np.float32), pair.sigma_high) * high_real
    low = low_pass(to_gray(low_in), pair.sigma_low)
    high = high_pass(to_gray(high_in), pair.sigma_high) * high_weight
    return Hybrid(pair, low_orig, high_orig, points, low_in, high_in, low, high, np.clip(low + high, 0, 1),
                  high_real, high_weight)


def color_variants(h: Hybrid) -> dict[str, np.ndarray]:
    """Bells & whistles: color for the low-frequency image, the high-frequency image, both, or neither."""
    p = h.pair
    low_rgb, high_rgb = low_pass(h.low_in, p.sigma_low), high_pass(h.high_in, p.sigma_high) * h.high_weight[..., None]
    gray3 = lambda im: np.repeat(im[..., None], 3, axis=-1)
    return {"gray low + gray high": gray3(h.hybrid),
            "color low + gray high": np.clip(low_rgb + h.high[..., None], 0, 1),
            "gray low + color high": np.clip(h.low[..., None] + high_rgb, 0, 1),
            "color low + color high": np.clip(low_rgb + high_rgb, 0, 1)}


def sigma_sweep(h: Hybrid, params: Params) -> list[list[np.ndarray]]:
    """Hybrids for every (sigma_low, sigma_high) in the sweep: rows = sigma_low, columns = sigma_high."""
    lo, hi = to_gray(h.low_in), to_gray(h.high_in)
    weight = lambda sh: gaussian_blur(h.high_real.astype(np.float32), sh) * h.high_real
    return [[np.clip(low_pass(lo, sl) + high_pass(hi, sh) * weight(sh), 0, 1) for sh in params.sweep_high]
            for sl in params.sweep_low]


# ---------------------------------------------------------------- Part 2.3

@dataclass
class StackDemo:
    apple: np.ndarray
    orange: np.ndarray
    mask: np.ndarray                        # step: 1 on the left half (apple)
    gauss_apple: list[np.ndarray]
    gauss_orange: list[np.ndarray]
    lap_apple: list[np.ndarray]
    lap_orange: list[np.ndarray]
    blend: dict                             # everything blend_stacks returns
    reconstruction_error: float             # max |sum(Laplacian stack) - image|


def make_stacks(params: Params) -> StackDemo:
    """Gaussian and Laplacian stacks of the apple and the orange, and the oraple (Figure 3.42)."""
    apple, orange = load_image(find_input(APPLE)), load_image(find_input(ORANGE))
    n, s = params.stack_levels, params.stack_sigma
    ga, go = gaussian_stack(apple, n, s), gaussian_stack(orange, n, s)
    la, lo = laplacian_stack(ga), laplacian_stack(go)
    mask = step_mask(apple.shape[:2])
    error = max(np.abs(collapse(la) - apple).max(), np.abs(collapse(lo) - orange).max())
    return StackDemo(apple, orange, mask, ga, go, la, lo, blend_stacks(apple, orange, mask, n, s), float(error))


# ---------------------------------------------------------------- Part 2.4

FEATHER_SIGMAS = (4.0, 30.0)        # single-level alpha blending, for the oraple comparison
LEVEL_CHOICES = (2, 4, 6)           # numbers of levels compared on the third eye


@dataclass
class Blend:
    pair: BlendPair
    a: np.ndarray                           # color inputs, same size
    b: np.ndarray
    mask: np.ndarray                        # (h, w) in [0, 1], 1 = take a
    color: dict                             # blend_stacks on the color images
    gray: np.ndarray                        # the same blend done on grayscale versions


def blend_inputs(pair: BlendPair) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = load_image(find_input(pair.a), max_side=pair.max_side)
    b = load_image(find_input(pair.b), max_side=pair.max_side)
    assert a.shape == b.shape, f"{pair.name}: the two images must have the same size"
    if pair.mask == "vertical":
        mask = step_mask(a.shape[:2])
    else:                                   # a drawn mask; its anti-aliased JPG edge is kept as soft values
        mask = load_image(find_input(pair.mask), gray=True, max_side=pair.max_side).astype(np.float64)
    return a, b, mask


def make_blend(pair: BlendPair) -> Blend:
    a, b, mask = blend_inputs(pair)
    color = blend_stacks(a, b, mask, pair.levels, pair.sigma)
    gray = blend_stacks(to_gray(a), to_gray(b), mask, pair.levels, pair.sigma)["result"]
    return Blend(pair, a, b, mask, color, gray)


def compare_blend_methods(oraple: Blend) -> dict[str, np.ndarray]:
    """Why multiresolution: the oraple with simpler methods, from a hard cut to the blurred-mask stacks."""
    a, b, m, p = oraple.a, oraple.b, oraple.mask[..., None], oraple.pair
    feather = lambda s: np.clip(gaussian_blur(oraple.mask, s)[..., None] * a
                                + (1 - gaussian_blur(oraple.mask, s)[..., None]) * b, 0, 1)
    out = {"direct paste": m * a + (1 - m) * b}
    out.update({f"one blurred mask, σ = {s:g}": feather(s) for s in FEATHER_SIGMAS})
    out["stacks, sharp mask (p. 226)"] = blend_stacks(a, b, oraple.mask, p.levels, p.sigma, blur_mask=False)["result"]
    out["stacks, blurred mask (p. 230)"] = oraple.color["result"]
    return out


def compare_levels(blend: Blend) -> dict[int, np.ndarray]:
    """The same blend with different numbers of levels."""
    p = blend.pair
    return {n: blend.color["result"] if n == p.levels else blend_stacks(blend.a, blend.b, blend.mask, n, p.sigma)["result"]
            for n in LEVEL_CHOICES}


# ---------------------------------------------------------------- All experiments, computed lazily

class Experiments:
    """Every result above, computed on first access and kept in memory."""

    def __init__(self, params: Params = Params()):
        self.params = params

    @cached_property
    def conv(self) -> ConvDemo:
        return make_conv_demo(self.params)

    @cached_property
    def cameraman(self) -> np.ndarray:
        return load_image(find_input(CAMERAMAN), gray=True)

    @cached_property
    def finite_diff(self) -> Edges:
        return make_finite_diff(self.cameraman, self.params)

    @cached_property
    def dog(self) -> DogDemo:
        return make_dog(self.cameraman, self.params)

    @cached_property
    def orientation(self) -> OrientationDemo:
        return make_orientation(self.dog)

    @cached_property
    def sharpen(self) -> list[SharpenDemo]:
        return [make_sharpen(name, load_image(find_input(name)), self.params) for name in (TAJ, BLURRY)]

    @cached_property
    def sharpen_eval(self) -> SharpenEval:
        return make_sharpen_eval(self.params)

    @cached_property
    def hybrids(self) -> list[Hybrid]:
        return [make_hybrid(pair, self.params) for pair in HYBRIDS]

    @cached_property
    def favorite(self) -> Hybrid:
        return next(h for h in self.hybrids if h.pair.name == FAVORITE)

    @cached_property
    def sweep(self) -> list[list[np.ndarray]]:
        return sigma_sweep(self.favorite, self.params)

    @cached_property
    def stacks(self) -> StackDemo:
        return make_stacks(self.params)

    @cached_property
    def blends(self) -> list[Blend]:
        return [make_blend(pair) for pair in BLENDS]

    def blend(self, name: str) -> Blend:
        return next(b for b in self.blends if b.pair.name == name)

    @cached_property
    def blend_methods(self) -> dict[str, np.ndarray]:
        return compare_blend_methods(self.blend("oraple"))

    @cached_property
    def blend_levels(self) -> dict[int, np.ndarray]:
        return compare_levels(self.blend(FAVORITE_BLEND))

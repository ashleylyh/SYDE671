"""Parts 2.3 / 2.4: Gaussian and Laplacian stacks, and multiresolution blending (Burt & Adelson, 1983).

PDF: no built-in pyramid functions (cv2.pyrDown, skimage.transform.pyramid_gaussian); the stacks are
built here from gaussian_blur (cv2.getGaussianKernel + convolve2d) and array arithmetic.
"""

import numpy as np

from .filters import gaussian_blur


def gaussian_stack(im: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    """[G_0, ..., G_{levels-1}], all the size of the input. G_0 = im, G_i = G_{i-1} * Gaussian(sigma * 2^(i-1)).

    PDF: "apply the Gaussian filter at each level, but do not subsample", so that the stack "behaves
    similarly to a pyramid that was down sampled to half its size at each level". A pyramid applies the
    same small filter to an image half the size each time, which, measured in pixels of the original, is a
    filter twice as wide each level. So the sigma doubles from one level to the next here.
    """
    stack = [im]
    for i in range(1, levels):
        stack.append(gaussian_blur(stack[-1], sigma * 2 ** (i - 1)))
    return stack


def laplacian_stack(gaussian: list[np.ndarray]) -> list[np.ndarray]:
    """L_i = G_i - G_{i+1} (one frequency band each), and the last level keeps the rest: L_{n-1} = G_{n-1}.

    The levels telescope, so sum(L) = G_0: the stack reconstructs the image exactly.
    """
    return [g - g_next for g, g_next in zip(gaussian, gaussian[1:])] + [gaussian[-1]]


def collapse(stack: list[np.ndarray]) -> np.ndarray:
    """Sum of a Laplacian stack (no upsampling needed: every level has the full size)."""
    return np.sum(stack, axis=0)


def _match(mask: np.ndarray, im: np.ndarray) -> np.ndarray:
    """A 2-D mask broadcast over the color channels of im."""
    return mask[..., None] if im.ndim == 3 and mask.ndim == 2 else mask


def blend_stacks(im_a: np.ndarray, im_b: np.ndarray, mask: np.ndarray, levels: int, sigma: float,
                 blur_mask: bool = True) -> dict:
    """Multiresolution blending with a mask (Burt & Adelson, the algorithm on p. 230).

    LS_i = GR_i * LA_i + (1 - GR_i) * LB_i, blended = sum_i LS_i, where LA / LB are the Laplacian stacks of
    the two images and GR is the Gaussian stack of the mask (1 = take im_a). The mask is blurred more at
    each level, so low frequencies are mixed over a wide band around the seam and fine details over a
    narrow one. Returns every stack, so the process can be shown.

    blur_mask=False uses the same sharp mask at every level: the algorithm on p. 226 as-is, which the PDF
    warns leaves a clear seam with stacks (with a pyramid, downsampling and upsampling blur the seam instead).
    """
    la = laplacian_stack(gaussian_stack(im_a, levels, sigma))
    lb = laplacian_stack(gaussian_stack(im_b, levels, sigma))
    mask = mask.astype(np.float64)
    gr = [_match(m, im_a) for m in (gaussian_stack(mask, levels, sigma) if blur_mask else [mask] * levels)]
    part_a = [m * l for m, l in zip(gr, la)]                     # im_a weighted by the mask, per level
    part_b = [(1 - m) * l for m, l in zip(gr, lb)]
    blended = [pa + pb for pa, pb in zip(part_a, part_b)]
    return {"la": la, "lb": lb, "mask": gr, "part_a": part_a, "part_b": part_b, "levels": blended,
            "result": np.clip(collapse(blended), 0, 1)}


def step_mask(shape: tuple[int, int], vertical: bool = True) -> np.ndarray:
    """PDF: "For the vertical or horizontal seam, your mask will simply be a step function". 1 on the left / top half."""
    h, w = shape
    mask = np.zeros((h, w))
    if vertical:
        mask[:, : w // 2] = 1
    else:
        mask[: h // 2] = 1
    return mask

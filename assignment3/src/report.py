"""Writes every figure and number the webpage uses: results/images/**/*.jpg and results/data.json.

One method per webpage section. Each pulls its results from an Experiments object,
saves web-sized JPGs and fills its keys of data.json. Blocks of index.html that show
code or numbers are rewritten between <!-- generated:<key> start --> and <!-- generated:<key> end -->.
"""

import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from algorithm.config import DATA_JSON, IMAGES_DIR, INDEX_HTML
from algorithm.io import save_jpg
from algorithm.homography import apply_H
from algorithm.viz import (INLIER, OUTLIER, correspondences, draw_matches, draw_quad, on_black, save_corners,
                           save_descriptors, save_panel, save_ratio_histogram, zoom)
from experiments import AutoPair, Experiments, Mosaic, PairH, h_distance

THUMBS = "thumbs"                       # inline copies shown on the page (and in a printed PDF)
THUMB_SIDE, THUMB_QUALITY = 1200, 75       # wide multi-image panels need the width
SYSTEM_PAIR = 0                         # A.2: the pair whose full system Ah = b is written out on the page


def _figure(rel: str, title: str, caption: str = "", cls: str = "") -> str:
    return (f'<figure class="{cls}"><a href="results/images/{rel}" target="_blank" rel="noopener">'
            f'<img src="results/images/{rel}" alt="{html.escape(title)}"></a>'
            f"<figcaption><b>{title}</b> {caption}</figcaption></figure>")


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tex_num(v: float, small: float = 1e-2) -> str:
    """A number for LaTeX: 4 significant digits; very small ones as m × 10^e (so h31, h32 do not show as 0.00)."""
    if v == 0:
        return "0"
    if abs(v) < small:
        m, e = f"{v:.2e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    if v == 1:
        return "1"
    return f"{v:.2f}" if abs(v) >= 10 else f"{v:.4f}"       # translations to 0.01 px, the rest to 4 decimals


def _tex_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A and b as strings, formatted per column of A: coordinates (x, y) to 1 decimal, the 0 / 1 entries as they are,
    the products -x x', -y x' (up to ~10^6) rounded to integers; b (x', y') to 1 decimal."""
    cells = np.empty(A.shape, dtype=object)
    for c in range(A.shape[1]):
        for r in range(A.shape[0]):
            v = A[r, c]
            cells[r, c] = f"{v:.0f}" if c >= 6 else (f"{v:.1f}" if c in (0, 1, 3, 4) and v != 0 else f"{v:.0f}")
    return cells, np.array([[f"{v:.1f}"] for v in b], dtype=object)


def _bmatrix(M: np.ndarray, fmt=_tex_num) -> str:
    rows = np.atleast_2d(M)
    return "\\begin{bmatrix}" + " \\\\ ".join(" & ".join(fmt(v) for v in row) for row in rows) + "\\end{bmatrix}"


def _display(tex: str) -> str:
    """Display math (KaTeX auto-render), scrolling sideways on narrow screens."""
    return f'<div class="math">$${tex}$$</div>'


def _secs(t: float) -> str:
    return f"{t * 1000:.0f} ms" if t < 1 else f"{t:.2f} s"


def _seam_box(m: Mosaic, size: tuple[int, int] = (360, 240)) -> tuple[int, int, int, int]:
    """Canvas crop centred on the overlap of the first two images, where blending artifacts show."""
    ys, xs = np.nonzero((m.alphas[0] > 0) & (m.alphas[1] > 0))
    cx, cy = int(np.median(xs)), int(np.median(ys))
    h, w = m.alphas[0].shape
    left, top = min(max(cx - size[0] // 2, 0), w - size[0]), min(max(cy - size[1] // 2, 0), h - size[1])
    return left, top, left + size[0], top + size[1]


# A.4: what each set's seam figure shows (from looking at the figures)
SEAM_NOTES = {
    "set1": "Overwriting leaves a hard edge and feathering a faint ghost of slightly misaligned lines; 2-band blending "
            "removes both.",
    "set2": "All three fail: overwriting breaks the building at the seam, feathering doubles the mast, and 2-band "
            "blending still shows it twice. The misalignment (parallax) is too large for blending to hide.",
    "set4": "All three look clean; only overwriting leaves a faint brightness step in the sky. The views align to "
            "within 1 px, so there is nothing to ghost.",
    "set5": "All three look clean: the views align to within about 1 px, so there is nothing to ghost.",
}

CORRECT_PX = 8                          # B.3: a nearest neighbour counts as correct within this distance of the clicked H


def _pair_figures(figs: list[tuple[str, str]], first_of_set: list[bool], summary: str) -> str:
    """The first pair of each set shown, the others in one collapsed panel."""
    shown = [f for f, first in zip(figs, first_of_set) if first]
    hidden = [f for f, first in zip(figs, first_of_set) if not first]
    return "\n".join(shown) + f"\n<details><summary>{summary}</summary><div>\n" + "\n".join(hidden) + "\n</div></details>"


def _pair_name(p) -> str:
    return f"{p.mset.title}: {p.names[0]} → {p.names[1]}"


class Report:
    """Exports results for the webpage. build() runs the sections, then writes data.json."""

    SECTIONS = ("params", "shoot", "homography_check", "homography", "rectify", "mosaic", "corners", "descriptors", "matching", "ransac", "hero")

    def __init__(self, ex: Experiments, images_dir: Path = IMAGES_DIR, data_json: Path = DATA_JSON,
                 index_html: Path = INDEX_HTML, max_side: int = 1000):
        self.ex = ex
        self.images_dir = images_dir
        self.data_json = data_json
        self.index_html = index_html
        self.max_side = max_side
        # Start from the previous export: keeps sections that aren't rebuilt this time
        self.data = json.loads(data_json.read_text()) if data_json.exists() else {}
        self.html: dict[str, str] = {}      # generated index.html blocks, by key

    # ------------------------------------------------------------ helpers

    def save_web(self, im, rel: str, max_side: int | None = None, quality: int = 85) -> str:
        """Save an image as a web-sized JPG under results/images/."""
        save_jpg(im, self.images_dir / rel, max_side=max_side or self.max_side, quality=quality)
        return rel

    def save_panel(self, images, titles, rel: str, **kwargs) -> str:
        """Save a labelled row / grid of images as one JPG figure under results/images/."""
        save_panel(images, titles, self.images_dir / rel, **kwargs)
        return rel

    def build(self, sections=SECTIONS) -> None:
        for section in sections:
            print(f"[report] {section}")
            try:
                getattr(self, section)()
            except FileNotFoundError as e:          # an input image / clicked points aren't there yet
                print(f"[report] {section} skipped: {e}")
            except NotImplementedError as e:        # a TODO in algorithm/ isn't written yet
                print(f"[report] {section} skipped, not implemented: {e}")
        self.write()

    def write(self) -> None:
        self.data_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_json, "w") as f:
            json.dump(self.data, f, indent=1)
        self.thumbnails()
        files = [p for p in self.images_dir.rglob("*") if p.is_file()]
        print(f"{self.images_dir.name}/: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB; "
              f"numbers in {self.data_json.name}")
        self.write_html()

    def thumbnails(self) -> None:
        """Small copy of every result JPG in results/images/thumbs/ (same relative path).

        The page shows these inline and links to the full image. A thumbnail is only rewritten when its source is newer.
        """
        for src in self.images_dir.rglob("*.jpg"):
            rel = src.relative_to(self.images_dir)
            if rel.parts[0] == THUMBS:
                continue
            out = self.images_dir / THUMBS / rel
            if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
                save_jpg(Image.open(src).convert("RGB"), out, max_side=THUMB_SIDE, quality=THUMB_QUALITY)

    def write_html(self) -> None:
        """Replace each generated block of index.html, then point every inline <img> at its thumbnail."""
        text = self.index_html.read_text()
        for key, block in self.html.items():
            start, end = f"<!-- generated:{key} start (report.py) -->", f"<!-- generated:{key} end -->"
            i, j = text.find(start), text.find(end)
            if i < 0 or j < i:
                print(f"[report] {self.index_html.name}: no '{key}' markers, block not updated")
                continue
            text = text[:i + len(start)] + "\n" + block + "\n" + text[j:]

        def to_thumb(m: re.Match) -> str:
            rel = m.group(2)
            return m.group(0) if not (self.images_dir / THUMBS / rel).exists() else \
                f'{m.group(1)}src="results/images/{THUMBS}/{rel}"'
        text = re.sub(r'(<img\b[^>]*?\s)src="results/images/(?!' + THUMBS + r'/)([^"]+\.jpg)"', to_thumb, text)
        self.index_html.write_text(text)
        print(f"[report] {self.index_html.name}: updated {', '.join(self.html) or 'image links'}")

    # ------------------------------------------------------------ sections

    def params(self) -> None:
        self.data["params"] = self.ex.params.to_dict()

    def shoot(self) -> None:
        """A.1: the source images of every set, left to right."""
        figures, data = [], {}
        for s in self.ex.shots:
            m = s.mset
            rel = self.save_panel(s.images, list(m.images), f"shoot/{m.name}.jpg", size=4)
            h, w = s.images[0].shape[:2]
            data[m.name] = {"title": m.title, "images": list(m.images), "reference": m.images[m.reference], "size": [w, h]}
            figures.append(_figure(rel, m.title, f"{len(s.images)} images, {w} × {h} after resizing; "
                                                 f"reference (unwarped in A.4): {m.images[m.reference]}.", cls="plain wide"))
        self.data["shoot"] = data
        self.html["shoot_sets"] = "\n".join(figures)

    def homography_check(self) -> None:
        """A.2: a check that computeH recovers a known H (exactly from 4 points, robustly from more)."""
        c = self.ex.h_check
        self.data["homography_check"] = {"H_true": c.H_true.tolist(), "rows": c.rows}
        self.html["homography_check"] = (
            _display(f"H_{{\\text{{true}}}} = {_bmatrix(c.H_true)}") + "\n" +
            _table(["n points", "noise σ (px)", "max |H − H_true| / max |H_true|", "mean error over the image (px)"],
                   [[str(r["n"]), f'{r["noise"]:g}', f'{r["H_diff"]:.1e}', f'{r["error"]:.3f}'] for r in c.rows]))

    def homography(self) -> None:
        """A.2: the clicked correspondences, one full system Ah = b, and H with its fitting error per pair."""
        pairs = self.ex.pairs
        figures, rows, data = [], [], {}
        for p in pairs:
            key = f"{p.names[0]}_{p.names[1]}"
            rel = self.save_web(correspondences(p.im1, p.im2, p.pts1, p.pts2, p.projected),
                                f"homography/{p.mset.name}/{key}.jpg", max_side=1600)
            err = p.errors
            figures.append(_figure(rel, _pair_name(p), f"{len(p.pts1)} correspondences; mean error {err.mean():.2f} px.",
                                   cls="plain wide"))
            rows.append([_pair_name(p), str(len(p.pts1)), f"{err.mean():.2f}", f"{err.max():.2f}",
                         f"\\({_bmatrix(p.H)}\\)"])
            data.setdefault(p.mset.name, {})[key] = {"n": len(p.pts1), "H": p.H.tolist(),
                                                     "mean_error": float(err.mean()), "max_error": float(err.max())}
        self.data["homography"] = data
        self.html["homography_points"] = "\n".join(figures)

        h_vec = _bmatrix(np.array([["h_{11}"], ["h_{12}"], ["h_{13}"], ["h_{21}"], ["h_{22}"], ["h_{23}"],
                                   ["h_{31}"], ["h_{32}"]]), fmt=str)

        def system(p: PairH) -> str:
            """The numeric system A h = b of one pair (rows in the order of the numbered points), then its H."""
            A, b = _tex_system(p.A, p.b)
            return (_display(f"\\underbrace{{{_bmatrix(A, str)}}}_{{A\\;({len(p.b)}\\times 8)}}"
                             f"{h_vec} = \\underbrace{{{_bmatrix(b, str)}}}_{{\\mathbf{{b}}}}") +
                    _display(f"H = {_bmatrix(p.H)}"))

        def points_table(p: PairH) -> str:
            err = p.errors
            return _table(["#", "x (left)", "y (left)", "x′ (right)", "y′ (right)", "rows of A", "error (px)"],
                          [[str(i + 1), f"{a[0]:.2f}", f"{a[1]:.2f}", f"{b[0]:.2f}", f"{b[1]:.2f}",
                            f"{2 * i + 1}, {2 * i + 2}", f"{e:.2f}"] for i, (a, b, e) in enumerate(zip(p.pts1, p.pts2, err))])

        first, rest = pairs[SYSTEM_PAIR], [p for i, p in enumerate(pairs) if i != SYSTEM_PAIR]
        self.html["homography_system"] = (
            f"<p><b>{_pair_name(first)}</b>: {len(first.pts1)} correspondences give {len(first.b)} equations in 8 "
            f"unknowns. Point k gives rows 2k − 1 (x′) and 2k (y′).</p>\n" + points_table(first) + "\n"
            f'<div class="keep">{system(first)}</div>\n'
            f"<details><summary>The system A h = b and H of the other {len(rest)} pairs</summary><div>\n" +
            "\n".join(f'<div class="keep"><h4>{_pair_name(p)}</h4>\n<p>{len(p.pts1)} correspondences, {len(p.b)} '
                      f"equations.</p>\n{system(p)}</div>" for p in rest) + "\n</div></details>")
        self.html["homography_table"] = _table(["Pair", "n", "mean error (px)", "max error (px)", "H"], rows)

    def rectify(self) -> None:
        """A.3: each rectification with both interpolations, a pixel-level zoom and the run times."""
        figures, rows, data = [], [], {}
        for r in self.ex.rectified:
            name = r.rect.name
            nn, bil = on_black(r.nn), on_black(r.bilinear)
            self.save_panel([draw_quad(r.image, r.quad), nn, bil],
                            ["input, clicked corners", "nearest neighbour", "bilinear"], f"rectify/{name}.jpg", size=4.5)
            # zoom: a quarter of the rectangle around its centre, 3x enlarged
            (x0, y0), (x1, y1) = r.target.min(axis=0) - r.bounds[:2], r.target.max(axis=0) - r.bounds[:2]
            cx, cy, dx, dy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 8, (y1 - y0) / 8
            box = (int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy))
            self.save_panel([zoom(nn, box), zoom(bil, box)], ["nearest neighbour (3x zoom)", "bilinear (3x zoom)"],
                            f"rectify/{name}_zoom.jpg", size=4.5)
            w, h = r.size
            out_h, out_w = r.nn.shape[:2]
            aspect = f"{r.rect.aspect:.3f} (known)" if r.rect.aspect else f"{w / h:.3f} (from the clicked sides)"
            figures.append(_figure(f"rectify/{name}.jpg", r.rect.title,
                                   f"The clicked quadrilateral is mapped to a {w} × {h} px rectangle, width / height {aspect}.",
                                   cls="plain wide"))
            figures.append(_figure(f"rectify/{name}_zoom.jpg", f"{r.rect.title}, zoomed",
                                   "Nearest neighbour repeats or skips source pixels (jagged edges); bilinear averages them "
                                   "(smooth, slightly softer).", cls="plain wide"))
            rows.append([r.rect.title, f"{out_w} × {out_h}", _secs(r.t_nn), _secs(r.t_bilinear),
                         f"{r.t_bilinear / r.t_nn:.1f}×"])
            data[name] = {"H": r.H.tolist(), "rectangle": [w, h], "output": [out_w, out_h],
                          "t_nn": r.t_nn, "t_bilinear": r.t_bilinear}
        self.data["rectify"] = data
        self.html["rectify_results"] = "\n".join(figures)
        self.html["rectify_table"] = _table(["Image", "Output (w × h)", "Nearest neighbour", "Bilinear", "Bilinear / NN"],
                                            rows)

    def mosaic(self) -> None:
        """A.4 per set: sources, the mosaic, the 3 blends at a seam, and the warped images with their weights."""
        blocks, rows, data = [], [], {}
        for m in self.ex.mosaics:
            ms = m.shots.mset
            w, h = m.bounds[2] - m.bounds[0] + 1, m.bounds[3] - m.bounds[1] + 1
            self.save_web(m.two_band, f"mosaic/{ms.name}.jpg", max_side=2400, quality=88)
            self.save_panel(m.shots.images, list(ms.images), f"mosaic/{ms.name}_sources.jpg", size=3)
            self.save_panel(m.warped + [w_ / max(float(w_.max()), 1e-8) for w_ in m.weights],
                            [f"{n} warped" for n in ms.images] + [f"{n} weight" for n in ms.images],
                            f"mosaic/{ms.name}_layers.jpg", cols=len(m.warped), size=3.4)
            box = _seam_box(m)
            self.save_panel([zoom(m.overwrite, box, 2), zoom(m.feather, box, 2), zoom(m.two_band, box, 2)],
                            ["overwrite", "feathering (weighted average)", "2-band blending"],
                            f"mosaic/{ms.name}_seam.jpg", size=4)
            clip = (" The full projective canvas would be enormous (the outermost frame reaches towards the horizon), so it is cut "
                    "to a window around the reference image." if m.clipped else "")
            blocks.append(f"""<h4>{ms.title}</h4>
{_figure(f"mosaic/{ms.name}_sources.jpg", "Sources", f"left to right; reference: {ms.images[ms.reference]}.", cls="plain wide")}
{_figure(f"mosaic/{ms.name}.jpg", f"{ms.title} mosaic", f"{len(ms.images)} images, {w} × {h} px, 2-band blending.{clip}", cls="plain wide")}
{_figure(f"mosaic/{ms.name}_seam.jpg", "Seam between the first two images", f"(2× zoom) {SEAM_NOTES.get(ms.name, '')}", cls="plain wide")}
<details><summary>Each image warped onto the canvas, with its weight</summary><div>
{_figure(f"mosaic/{ms.name}_layers.jpg", "Warped onto the canvas", "Top: each image in the reference projection. Bottom: its warped weight (1 at the image centre, 0 at its edges).", cls="plain wide")}
</div></details>""")
            rows.append([ms.title, str(len(ms.images)), ms.images[ms.reference], f"{w} × {h}", _secs(m.seconds)])
            data[ms.name] = {"canvas": [w, h], "Hs": [H.tolist() for H in m.Hs], "seconds": m.seconds, "clipped": m.clipped}
        self.data["mosaic"] = data
        self.html["mosaic_results"] = "\n".join(blocks)
        self.html["mosaic_table"] = _table(["Mosaic", "Images", "Reference", "Canvas (w × h)", "Warp + blend"], rows)

    def corners(self) -> None:
        """B.1: for the reference image of each set, all Harris corners, the strongest n, and the n chosen by ANMS."""
        n = self.ex.params.anms_corners
        figures, rows, data = [], [], {}
        for shots in self.ex.shots:
            ms = shots.mset
            f = self.ex.features(shots, ms.reference)
            rel = f"corners/{ms.name}.jpg"
            save_corners(f.gray, [f.coords, f.strongest, f.anms_coords],
                         [f"Harris: {f.coords.shape[1]:,} corners", f"strongest {n} (no ANMS)", f"ANMS: {n} corners"],
                         self.images_dir / rel)
            r_min = float(f.radii[f.keep].min())
            figures.append(_figure(rel, f"{ms.title} ({f.name})",
                                   f"After ANMS, no kept corner has a corner more than 1 / c<sub>robust</sub> ≈ 1.1× "
                                   f"stronger within {r_min:.0f} px of it.", cls="plain wide"))
            rows.append([f"{ms.title} ({f.name})", f"{f.gray.shape[1]} × {f.gray.shape[0]}", f"{f.n_raw:,}", f"{f.coords.shape[1]:,}",
                         str(len(f.keep)), f"{r_min:.1f}", _secs(f.t_harris), _secs(f.t_anms)])
            data[ms.name] = {"image": f.name, "harris_raw": f.n_raw, "harris": int(f.coords.shape[1]), "anms": len(f.keep),
                             "min_radius": r_min, "t_harris": f.t_harris, "t_anms": f.t_anms}
        self.data["corners"] = data
        self.html["corners_results"] = "\n".join(figures)
        self.html["corners_table"] = _table(["Image", "Size", "Local maxima", f"Harris corners (f > {self.ex.params.harris_threshold:g})",
                                             "After ANMS", "Smallest kept radius (px)",
                                             "Harris", "ANMS"], rows)

    def descriptors(self) -> None:
        """B.2: a few ANMS features of each reference image: where they are, their 40 x 40 windows and 8 x 8 descriptors."""
        p = self.ex.params
        figures = []
        for shots in self.ex.shots:
            ms = shots.mset
            f = self.ex.features(shots, ms.reference)
            pick = np.linspace(0, len(f.keep) - 1, 8).astype(int)     # spread over the ANMS ranking
            rel = f"descriptors/{ms.name}.jpg"
            save_descriptors(f.gray, f.anms_coords[:, pick], f.windows[pick], f.descs[pick], self.images_dir / rel, p.window)
            figures.append(_figure(rel, f"{ms.title} ({f.name})",
                                   f"8 of the {len(f.keep)} features. Top: the {p.window} × {p.window} window around the corner. "
                                   f"Bottom: its {p.patch} × {p.patch} descriptor (mid-gray = the patch mean, white / black = "
                                   "±3 standard deviations).", cls="plain wide"))
        self.data["descriptors"] = {"window": p.window, "patch": p.patch, "sigma": p.descriptor_sigma}
        self.html["descriptors_results"] = "\n".join(figures)

    def _manual(self) -> dict:
        return {(p.mset.name, p.names): p for p in self.ex.pairs}

    def matching(self) -> None:
        """B.3: the matches of every pair after the ratio test, and our Figure 6b justifying the threshold."""
        pairs, manual, t = self.ex.auto_pairs, self._manual(), self.ex.params.ratio_threshold
        figures, firsts, rows, data = [], [], [], {}
        correct_r, wrong_r = [], []
        for a in pairs:
            key = f"{a.names[0]}_{a.names[1]}"
            rel = self.save_web(draw_matches(a.f1.gray, a.f2.gray, a.pts1, a.pts2), f"matching/{a.mset.name}/{key}.jpg",
                                max_side=1600)
            figures.append(_figure(rel, _pair_name(a), f"{len(a.matches)} matches with e₁-NN / e₂-NN &lt; {t:g}, out of "
                                                      f"{len(a.ratios)} ANMS features.", cls="plain wide"))
            firsts.append(a.names[0] == a.mset.images[0])
            # label every nearest neighbour with the clicked homography of Part A (independent of the automatic pipeline)
            m = manual[(a.mset.name, a.names)]
            p1 = a.f1.anms_coords[::-1].T.astype(np.float64)
            p2 = a.f2.anms_coords[::-1, a.nn].T.astype(np.float64)
            ok = np.linalg.norm(apply_H(m.H, p1) - p2, axis=1) < CORRECT_PX
            correct_r.append(a.ratios[ok])
            wrong_r.append(a.ratios[~ok])
            kept = a.ratios < t
            rows.append([_pair_name(a), str(len(a.ratios)), str(int(ok.sum())), str(int(kept.sum())),
                         f"{ok[kept].mean():.0%}", f"{ok[~kept].mean():.0%}"])
            data[key] = {"features": len(a.ratios), "matches": int(kept.sum()), "correct_nn": int(ok.sum()),
                         "precision": float(ok[kept].mean())}
        correct, wrong = np.concatenate(correct_r), np.concatenate(wrong_r)
        save_ratio_histogram(correct, wrong, t, self.images_dir / "matching/ratio_hist.jpg")
        self.data["matching"] = {"threshold": t, "correct_px": CORRECT_PX, "pairs": data,
                                 "correct_below": float((correct < t).mean()), "wrong_below": float((wrong < t).mean())}
        self.html["matching_hist"] = _figure(
            "matching/ratio_hist.jpg", "e₁-NN / e₂-NN on my images (cf. Figure 6b of the paper).",
            f"The nearest neighbour of every ANMS feature, in all 9 pairs, labelled correct when it lies within {CORRECT_PX} px of "
            f"where the clicked homography of Part A puts the feature. Below {t:g}: {(correct < t).mean():.0%} of the correct "
            f"and {(wrong < t).mean():.0%} of the incorrect ones.", cls="plain result")
        self.html["matching_results"] = _pair_figures(figures, firsts, "Matches of the other pairs")
        self.html["matching_table"] = _table(["Pair", "ANMS features", "Correct nearest neighbours",
                                              f"Matches (ratio &lt; {t:g})", "Correct among matches",
                                              "Correct among rejected"], rows)

    def ransac(self) -> None:
        """B.4: inliers / outliers per pair, the automatic H against the clicked one, and the mosaics side by side."""
        pairs, manual, p = self.ex.auto_pairs, self._manual(), self.ex.params
        figures, firsts, rows, data = [], [], [], {}
        for a in pairs:
            key = f"{a.names[0]}_{a.names[1]}"
            colors = [INLIER if i else OUTLIER for i in a.inliers]
            rel = self.save_web(draw_matches(a.f1.gray, a.f2.gray, a.pts1, a.pts2, colors), f"ransac/{a.mset.name}/{key}.jpg",
                                max_side=1600)
            figures.append(_figure(rel, _pair_name(a), f"RANSAC: {int(a.inliers.sum())} inliers (green) and "
                                                      f"{int((~a.inliers).sum())} outliers (red) of {len(a.matches)} matches.",
                                   cls="plain wide"))
            firsts.append(a.names[0] == a.mset.images[0])
            m = manual[(a.mset.name, a.names)]
            res = np.linalg.norm(apply_H(a.H, a.pts1[a.inliers]) - a.pts2[a.inliers], axis=1)
            diff = h_distance(a.H, m.H, a.f1.gray.shape)
            rows.append([_pair_name(a), str(len(a.matches)), str(int(a.inliers.sum())), f"{a.inliers.mean():.0%}",
                         f"{np.median(res):.2f}"])
            data[key] = {"matches": len(a.matches), "inliers": int(a.inliers.sum()), "H": a.H.tolist(),
                         "residual": float(res.mean()), "median_residual": float(np.median(res)), "vs_manual": diff,
                         "seconds": a.seconds}
        self.data["ransac"] = data
        self.html["ransac_results"] = _pair_figures(figures, firsts, "Inliers and outliers of the other pairs")
        self.html["ransac_table"] = _table(["Pair", "Matches", "Final inliers", "Inlier fraction",
                                            "Median inlier error (px)"], rows)

        blocks = []
        for man, auto in zip(self.ex.mosaics, self.ex.auto_mosaics):
            ms = auto.shots.mset
            self.save_web(auto.two_band, f"auto/{ms.name}.jpg", max_side=2400, quality=88)
            self.save_panel([man.two_band, auto.two_band], ["manual (clicked points, A.4)", "automatic (RANSAC, B.4)"],
                            f"auto/{ms.name}_compare.jpg", size=7)
            w, h = auto.bounds[2] - auto.bounds[0] + 1, auto.bounds[3] - auto.bounds[1] + 1
            clip = " Cut to a window around the reference image, as in A.4." if auto.clipped else ""
            blocks.append(f"<h4>{ms.title}</h4>\n" +
                          _figure(f"auto/{ms.name}_compare.jpg", "Manual vs automatic", "", cls="plain wide") + "\n" +
                          _figure(f"auto/{ms.name}.jpg", f"{ms.title}, automatic mosaic",
                                  f"{len(ms.images)} images, {w} × {h} px, no clicked points.{clip}", cls="plain wide"))
        self.html["auto_mosaics"] = "\n".join(blocks)

    def hero(self) -> None:
        """Four square crops for the page header (same layout as Assignments 1 and 2)."""
        auto = {m.shots.mset.name: m.two_band for m in self.ex.auto_mosaics}
        manual = {m.shots.mset.name: m.two_band for m in self.ex.mosaics}
        almond = next(r for r in self.ex.rectified if r.rect.name == "almond")
        (x0, y0), (x1, y1) = almond.target.min(axis=0) - almond.bounds[:2], almond.target.max(axis=0) - almond.bounds[:2]
        crops = {   # name: (image, centre x, centre y as fractions, side as a fraction of the image height)
            "atrium": (manual["set1"], 0.5, 0.56, 0.62),
            "skyline": (auto["set4"], 0.24, 0.5, 0.85),
            "almond": (on_black(almond.bilinear), (x0 + x1) / 2 / almond.bilinear.shape[1],
                       (y0 + y1) / 2 / almond.bilinear.shape[0], (y1 - y0) * 1.15 / almond.bilinear.shape[0]),
            "niagara": (auto["set5"], 0.63, 0.5, 0.2),
        }
        for name, (im, cx, cy, frac) in crops.items():
            h, w = im.shape[:2]
            side = int(frac * h)
            left = int(np.clip(cx * w - side / 2, 0, w - side))
            top = int(np.clip(cy * h - side / 2, 0, h - side))
            self.save_web(im[top:top + side, left:left + side], f"hero/{name}.jpg", max_side=600)

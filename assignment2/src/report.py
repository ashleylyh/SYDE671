"""Writes every figure and number the webpage uses: results/images/**/*.jpg and results/data.json.

One method per webpage section. Each pulls its results from an Experiments object,
saves web-sized JPGs and fills its keys of data.json. Blocks of index.html that show
code or numbers are rewritten between <!-- generated:<key> start --> and <!-- generated:<key> end -->.
"""

import html
import inspect
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from algorithm import convolution, filters, hybrid, stacks
from algorithm.config import BLURRY, DATA_JSON, FAVORITE_BLEND, IMAGES_DIR, INDEX_HTML, TAJ
from algorithm.io import save_jpg
from algorithm.hybrid import cutoff_frequency, gaussian_response, log_spectrum, to_gray
from algorithm.stacks import collapse
from algorithm.viz import (bands_to_unit, distance_strip, mark_points, save_filter_responses, save_panel, save_spectra,
                           signed_to_unit, unit_scale)
from experiments import DOG_SWEEP, FD_SWEEP, Edges, Experiments, color_variants

THUMBS = "thumbs"                       # inline copies shown on the page (and in a printed PDF)
THUMB_SIDE, THUMB_QUALITY = 1200, 75       # wide multi-image panels need the width
LABELS = {TAJ: "Taj Mahal", BLURRY: "My blurry cat"}
# Part 2.1 alpha sweep: zoom into the cat's eyes so the difference is visible; the Taj Mahal is shown whole
ZOOM = (slice(80, 330), slice(200, 450))       # cameraman: head, camera and tripod top, for the 1.2 vs 1.3 zoom
SWEEP_CROPS = {BLURRY: (slice(170, 400), slice(230, 620))}


def _constants(module, *names: str) -> str:
    """Source lines that assign module-level constants (inspect.getsource can't read arrays)."""
    lines = inspect.getsource(module).splitlines()
    return "\n".join(line for line in lines if line.split(" =")[0] in names)


def _code(*objects) -> str:
    """Source of functions (or ready-made source strings) as an escaped <pre> block, so the page shows the code that ran."""
    src = "\n\n".join(o if isinstance(o, str) else inspect.getsource(o).rstrip() for o in objects)
    return f'<pre class="code"><code>{html.escape(src)}</code></pre>'


def _figure(rel: str, title: str, caption: str = "", cls: str = "") -> str:
    return (f'<figure class="{cls}"><a href="results/images/{rel}" target="_blank" rel="noopener">'
            f'<img src="results/images/{rel}" alt="{html.escape(title)}"></a>'
            f"<figcaption><b>{title}</b> {caption}</figcaption></figure>")


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _slider(before: str, after: str, name: str, left: str = "before", right: str = "after") -> str:
    """Before / after comparison with a drag slider (same markup as Assignment 1)."""
    return (f'<figure class="compare" style="--pos: 50%">\n  <div class="compare-frame">\n'
            f'    <img src="results/images/{before}" alt="{name} {left}">\n'
            f'    <img class="after" src="results/images/{after}" alt="{name} {right}">\n'
            f'    <span class="tag tag-l">{left}</span><span class="tag tag-r">{right}</span>\n'
            f'    <input type="range" min="0" max="100" value="50" aria-label="Slider for {name}">\n'
            f'  </div>\n  <figcaption><b>{name}</b></figcaption>\n</figure>')


def _edge_panel(e: Edges) -> tuple[list, list[str]]:
    """The four images of a gradient computation: d/dx, d/dy, magnitude, edges."""
    return ([signed_to_unit(e.gx), signed_to_unit(e.gy), unit_scale(e.mag, 99.5), e.edges],
            ["∂f/∂x", "∂f/∂y", "gradient magnitude", f"edges (|∇f| > {e.threshold:g})"])


def _secs(t: float) -> str:
    return f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"


class Report:
    """Exports results for the webpage. build() runs the sections, then writes data.json."""

    SECTIONS = ("params", "conv", "finite_diff", "dog", "orientation", "sharpen", "hybrid", "hybrid_color", "stacks", "blend", "blend_color", "hero")

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
            except FileNotFoundError as e:          # an input image isn't in data/ yet
                print(f"[report] {section} skipped: {e}")
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

    def conv(self) -> None:
        """Part 1.1: code, runtime / correctness table, and the selfie convolved with box, Dx, Dy."""
        self.html["conv_code"] = _code(_constants(convolution, "DX", "DY"), convolution.box_filter, convolution.zero_pad,
                                       convolution._prepare, convolution.convolve_4loops,
                                       convolution.convolve_2loops)
        demo = self.ex.conv
        names = list(demo.results)
        images = [demo.selfie] + [demo.results[n] if n.endswith("box") else signed_to_unit(demo.results[n])
                                  for n in names]
        self.save_panel(images, ["selfie (grayscale)"] + [f"selfie * {n}" for n in names], "conv/selfie.jpg", size=4)
        self.data["conv"] = demo.rows

        rows = demo.rows
        s = rows[0]["crop"][0]
        h, w = rows[0]["full"]
        self.html["conv_table"] = _table(
            ["Kernel", f"4 loops ({s}×{s})", f"2 loops ({s}×{s})", f"convolve2d ({s}×{s})",
             f"2 loops ({h}×{w})", f"convolve2d ({h}×{w})", "max |mine − scipy|"],
            [[r["kernel"], _secs(r["t4_crop"]), _secs(r["t2_crop"]), _secs(r["scipy_crop"]),
              _secs(r["t2_full"]), _secs(r["scipy_full"]), f'{max(r["diff4"], r["diff2"]):.1e}'] for r in rows])
        self.html["conv_results"] = _figure(
            "conv/selfie.jpg", "My selfie",
            f"({h}×{w}, grayscale) convolved with the {names[0]}, Dx and Dy, using my 2-loop version. "
            "Dx / Dy are shown with mid-gray = 0, white = positive, black = negative.", cls="plain")

    def finite_diff(self) -> None:
        """Part 1.2: derivatives, magnitude, edges, and the threshold sweep that justifies the choice."""
        e = self.ex.finite_diff
        images, titles = _edge_panel(e)
        self.save_panel([self.ex.cameraman] + images, ["cameraman"] + titles, "finite_diff/steps.jpg", size=3.2)
        self.save_panel([e.mag > t for t in FD_SWEEP],
                        [f"|∇f| > {t:g}" + ("  (chosen)" if t == e.threshold else "") for t in FD_SWEEP],
                        "finite_diff/sweep.jpg", cols=3, size=3.6)
        self.data["finite_diff"] = {"threshold": e.threshold, "max": float(e.mag.max()),
                                    "percentiles": {p: float(np.percentile(e.mag, p)) for p in (50, 90, 99)}}

    def dog(self) -> None:
        """Part 1.3: blur then differentiate, the DoG filters, one convolution, and the comparison with 1.2."""
        d, fd, p = self.ex.dog, self.ex.finite_diff, self.ex.params
        k = d.gaussian.shape[0]
        images, titles = _edge_panel(d.two_step)
        self.save_panel([d.blurred] + images, [f"f * G (σ = {p.dog_sigma:g}, {k}×{k})"] + titles,
                        "dog/two_step.jpg", size=3.2)
        self.save_panel([unit_scale(d.gaussian), signed_to_unit(d.dog_x, 100), signed_to_unit(d.dog_y, 100)],
                        [f"G ({k}×{k})", f"G * Dx ({d.dog_x.shape[0]}×{d.dog_x.shape[1]})",
                         f"G * Dy ({d.dog_y.shape[0]}×{d.dog_y.shape[1]})"], "dog/filters.jpg", size=3.4)
        images, titles = _edge_panel(d.one_step)
        self.save_panel(images, [t.replace("∂f/∂", "f * DoG_") for t in titles[:2]] + titles[2:],
                        "dog/one_step.jpg", size=3.2)
        self.save_panel([fd.edges, d.one_step.edges, fd.edges[ZOOM], d.one_step.edges[ZOOM]],
                        ["finite difference", "derivative of Gaussian", "finite difference (zoom)", "DoG (zoom)"], "dog/compare.jpg", size=3.4)
        self.save_panel([d.one_step.mag > t for t in DOG_SWEEP],
                        [f"|∇(f*G)| > {t:g}" + ("  (chosen)" if t == d.one_step.threshold else "") for t in DOG_SWEEP],
                        "dog/sweep.jpg", cols=3, size=3.6)
        self.data["dog"] = {"sigma": p.dog_sigma, "ksize": k, "threshold": d.one_step.threshold,
                            "diff_full": d.diff_full, "diff_interior": d.diff_interior}
        self.html["dog_check"] = (
            f'<div class="callout ok"><p><b>Same result.</b> The largest difference between the two-step and the one-step '
            f"gradients is <code>{d.diff_full:.1e}</code>, which is floating-point rounding. Convolution is associative, so "
            "(f * G) * D = f * (G * D). This holds even at the border here: mirroring the image and then blurring with "
            "the symmetric G gives an image that is itself mirror-symmetric, so mirroring before or after the blur "
            "makes no difference.</p></div>")
        self.html["dog_code"] = _code(filters.gaussian_2d, filters.dog_filters)

    def orientation(self) -> None:
        """Bells & whistles: gradient direction as hue."""
        o = self.ex.orientation
        self.save_web(o.shaded, "orientation/shaded.jpg")
        self.save_web(o.on_edges, "orientation/edges.jpg")
        self.save_web(o.wheel, "orientation/wheel.jpg")
        self.data["orientation"] = {"atan_error": o.atan_error}
        self.html["orientation_results"] = (
            '<div class="grid-3">'
            + _figure("orientation/shaded.jpg", "Direction × strength", "hue = direction, brightness = |∇f|")
            + _figure("orientation/edges.jpg", "Direction on edges", "only pixels above the DoG edge threshold")
            + _figure("orientation/wheel.jpg", "Legend", "the color of a gradient pointing from the centre "
                      "towards each point (towards brighter)", cls="plain")
            + "</div>"
            f'<p>Check: on this image my <code>arctan2_own</code> is within <code>{o.atan_error:.1e}</code> rad of '
            "<code>np.arctan2</code> (used only for this check, never for the result).</p>")
        self.html["orientation_code"] = _code(filters.arctan2_own, filters.orientation_rgb)

    def sharpen(self) -> None:
        """Part 2.1: the unsharp mask steps, the amount sweep, and the blur -> re-sharpen evaluation."""
        p = self.ex.params
        steps, sliders, sweeps = [], [], []
        for demo in self.ex.sharpen:
            label = LABELS[demo.name]
            rel = self.save_panel([demo.original, demo.blurred, signed_to_unit(demo.high), demo.sharpened],
                                  ["original f", "blurred f * G", "high frequencies f − f * G",
                                   f"sharpened (α = {p.sharpen_alpha:g})"], f"sharpen/{demo.name}_steps.jpg", size=3.4)
            steps.append(_figure(rel, label, f"single-filter result matches f + α(f − f*G) to "
                                             f"<code>{demo.diff:.1e}</code>.", cls="plain"))
            before = self.save_web(demo.original, f"sharpen/{demo.name}_original.jpg")
            after = self.save_web(demo.sharpened, f"sharpen/{demo.name}_sharpened.jpg")
            sliders.append(_slider(before, after, f"{label}, α = {p.sharpen_alpha:g}", "original", "sharpened"))
            crop = SWEEP_CROPS.get(demo.name, (slice(None), slice(None)))
            rel = self.save_panel([im[crop] for im in demo.sweep], [f"α = {a:g}" for a in p.alphas],
                                  f"sharpen/{demo.name}_alphas.jpg", size=3.2)
            sweeps.append(_figure(rel, label, "(zoomed in)" if demo.name in SWEEP_CROPS else "", cls="plain"))
        self.html["sharpen_steps"] = '<div class="wide">' + "".join(steps) + "</div>"
        self.html["sharpen_sliders"] = '<div class="compare-grid">' + "".join(sliders) + "</div>"
        self.html["sharpen_alphas"] = '<div class="wide">' + "".join(sweeps) + "</div>"
        self.html["sharpen_code"] = _code(filters.unsharp_kernel)

        ev = self.ex.sharpen_eval
        figs = [_figure(self.save_web(ev.original, "sharpen/eval/original.jpg"), "Original (sharp)"),
                _figure(self.save_web(ev.blurred, "sharpen/eval/blurred.jpg"), "Blurred",
                        f"PSNR {ev.psnr_blurred:.1f} dB")]
        for a, im, score in zip(p.alphas, ev.resharpened, ev.psnr):
            if a > 0:
                figs.append(_figure(self.save_web(im, f"sharpen/eval/alpha_{a:g}.jpg"),
                                    f"Re-sharpened, α = {a:g}", f"PSNR {score:.1f} dB"))
        self.html["sharpen_eval"] = '<div class="grid-2">' + "".join(figs) + "</div>"
        self.data["sharpen"] = {"sigma": p.sharpen_sigma, "alpha": p.sharpen_alpha, "alphas": list(p.alphas),
                                "eval_psnr_blurred": ev.psnr_blurred, "eval_psnr": ev.psnr}

    def hybrid(self) -> None:
        """Part 2.2: the whole process for the favourite pair, then input + result for every pair."""
        p = self.ex.params
        fav = self.ex.favorite
        f, d = fav.pair, f"hybrid/{fav.pair.name}"
        fc_low, fc_high = cutoff_frequency(f.sigma_low), cutoff_frequency(f.sigma_high)
        parts = []

        # 1. originals (with the clicked points) and the aligned, cropped inputs
        if fav.points:
            row = [_figure(self.save_web(mark_points(fav.low_orig, fav.points["low"]), f"{d}/low_points.jpg"),
                           "Low-pass input", "with my two clicked points"),
                   _figure(self.save_web(mark_points(fav.high_orig, fav.points["high"]), f"{d}/high_points.jpg"),
                           "High-pass input", "with my two clicked points")]
            parts.append('<h4>1. Original images and alignment</h4><div class="grid-2">' + "".join(row) + "</div>")
        row = [_figure(self.save_web(fav.low_in, f"{d}/low_aligned.jpg"), "Aligned low-pass input"),
               _figure(self.save_web(fav.high_in * fav.high_real[..., None], f"{d}/high_aligned.jpg"),
                       "Aligned high-pass input", "scaled, rotated and cropped to the frame of the first; "
                       "black = no cat pixels there")]
        parts.append('<div class="grid-2">' + "".join(row) + "</div>")

        # 2. filtered images
        row = [_figure(self.save_web(fav.low, f"{d}/low_pass.jpg"), "Low-pass", f"G<sub>σ={f.sigma_low:g}</sub> * gray(low input)"),
               _figure(self.save_web(np.clip(fav.high + 0.5, 0, 1), f"{d}/high_pass.jpg"), "High-pass",
                       f"gray(high input) − G<sub>σ={f.sigma_high:g}</sub> * gray(high input), shown + 0.5")]
        parts.append('<h4>2. Filtered images</h4><div class="grid-2">' + "".join(row) + "</div>")

        # 3. Fourier analysis
        spectra = [log_spectrum(to_gray(fav.low_in)), log_spectrum(to_gray(fav.high_in)), log_spectrum(fav.low),
                   log_spectrum(fav.high), log_spectrum(fav.hybrid)]
        save_spectra(spectra, ["low-pass input", "high-pass input", "low-passed", "high-passed", "hybrid"],
                     self.images_dir / d / "spectra.jpg")
        parts.append('<h4>3. Frequency analysis</h4><div class="wide">'
                     + _figure(f"{d}/spectra.jpg", "Log magnitude of the 2-D Fourier transform",
                               "(centre = frequency 0; same color scale in every panel)", cls="plain") + "</div>")

        # 4. cutoff frequencies
        freqs = np.linspace(0, 0.5, 501)
        save_filter_responses(freqs, gaussian_response(freqs, f.sigma_low), 1 - gaussian_response(freqs, f.sigma_high),
                              (fc_low, fc_high), (f"low-pass, σ = {f.sigma_low:g}", f"high-pass, σ = {f.sigma_high:g}"),
                              self.images_dir / d / "cutoffs.jpg")
        sweep = [im for row in self.ex.sweep for im in row]
        titles = [f"σ_low {sl:g}, σ_high {sh:g}" + ("  (chosen)" if (sl, sh) == (f.sigma_low, f.sigma_high) else "")
                  for sl in p.sweep_low for sh in p.sweep_high]
        self.save_panel(sweep, titles, f"{d}/sweep.jpg", cols=len(p.sweep_high), size=3.6)
        parts.append(
            '<h4>4. Choosing the cutoff frequencies</h4><div class="wide">'
            + _figure(f"{d}/cutoffs.jpg", "Filter gains",
                      f"cutoff = where the gain is ½: {fc_low:.3f} cycles/pixel for the low-pass (σ = {f.sigma_low:g}) "
                      f"and {fc_high:.3f} for the high-pass (σ = {f.sigma_high:g})", cls="plain")
            + "</div>" + '<div class="wide">'
            + _figure(f"{d}/sweep.jpg", "Sigma sweep",
                      "rows: σ of the low-pass, columns: σ of the high-pass", cls="plain")
            + "</div>")

        # 5. result
        self.save_web(fav.hybrid, f"{d}/hybrid.jpg")
        self.save_web(distance_strip(fav.hybrid), f"{d}/hybrid_far.jpg", max_side=1400)
        parts.append('<h4>5. Hybrid image</h4><div class="wide">'
                     + _figure(f"{d}/hybrid_far.jpg", f.title,
                               "the same hybrid at 100%, 50%, 25% and 12.5%: shrinking it removes the high "
                               "frequencies, like stepping back", cls="plain") + "</div>")
        self.html["hybrid_process"] = "\n".join(parts)

        # every pair: inputs and result
        blocks, rows = [], []
        for h in self.ex.hybrids:
            q, d = h.pair, f"hybrid/{h.pair.name}"
            self.save_web(h.hybrid, f"{d}/hybrid.jpg")
            self.save_web(distance_strip(h.hybrid), f"{d}/hybrid_far.jpg", max_side=1400)
            figs = [_figure(self.save_web(h.low_orig, f"{d}/low_input.jpg", max_side=700), "Far: low-pass input"),
                    _figure(self.save_web(h.high_orig, f"{d}/high_input.jpg", max_side=700), "Close: high-pass input"),
                    _figure(f"{d}/hybrid.jpg", "Hybrid", f"σ_low = {q.sigma_low:g}, σ_high = {q.sigma_high:g}")]
            blocks.append(f'<h4>{q.title}</h4><div class="grid-3">' + "".join(figs) + "</div>"
                          + '<div class="wide">' + _figure(f"{d}/hybrid_far.jpg", "Viewed from farther away",
                                                           "100%, 50%, 25%, 12.5%", cls="plain") + "</div>")
            rows.append({"name": q.name, "sigma_low": q.sigma_low, "sigma_high": q.sigma_high,
                         "cutoff_low": cutoff_frequency(q.sigma_low), "cutoff_high": cutoff_frequency(q.sigma_high),
                         "size": [int(v) for v in h.hybrid.shape], "points": h.points})
        self.html["hybrid_results"] = "\n".join(blocks)
        self.html["hybrid_code"] = _code(hybrid.align_images, hybrid.low_pass, hybrid.high_pass, hybrid.hybrid_image,
                                         hybrid.cutoff_frequency)
        self.data["hybrid"] = rows

    def hybrid_color(self) -> None:
        """Bells & whistles: color in the low frequencies, the high frequencies, both or neither, for every pair."""
        figs = []
        for h in self.ex.hybrids:
            variants = color_variants(h)
            rel = self.save_panel(list(variants.values()), list(variants), f"hybrid_color/{h.pair.name}.jpg", size=3.6)
            figs.append(_figure(rel, h.pair.title, cls="plain"))
        self.html["hybrid_color"] = '<div class="wide">' + "".join(figs) + "</div>"

    def stacks(self) -> None:
        """Part 2.3: the Gaussian and Laplacian stacks of the apple and the orange, and Figure 3.42 (a)-(l)."""
        st, p = self.ex.stacks, self.ex.params
        n = p.stack_levels
        # total blur of level i: the level sigmas add in quadrature
        total = [float(np.sqrt(sum((p.stack_sigma * 2 ** (k - 1)) ** 2 for k in range(1, i + 1)))) for i in range(n)]
        cols = [f"level {i}" + (f" (σ ≈ {t:.0f})" if i else " (original)") for i, t in enumerate(total)]

        # (h, w, 1) -> (h, w), with a thin gray frame so the white half doesn't vanish into the page
        masks = [np.pad(m[..., 0], 2, constant_values=0.5) for m in st.blend["mask"]]
        self.save_panel(st.gauss_apple + st.gauss_orange + masks,
                        [f"apple, {c}" for c in cols] + [f"orange, {c}" for c in cols] + [f"mask, {c}" for c in cols],
                        "stacks/gaussian.jpg", cols=n, size=2.6)
        lap_titles = [f"L{i} = G{i} − G{i + 1}" for i in range(n - 1)] + [f"L{n - 1} = G{n - 1}"]
        show = lambda lap: [bands_to_unit([l])[0] for l in lap[:-1]] + [lap[-1]]
        self.save_panel(show(st.lap_apple) + show(st.lap_orange),
                        [f"apple, {t}" for t in lap_titles] + [f"orange, {t}" for t in lap_titles],
                        "stacks/laplacian.jpg", cols=n, size=2.6)

        # Figure 3.42: rows = Laplacian levels 0, 2, 4, then all levels summed; columns = apple part, orange part, sum
        b, images, titles, letters = st.blend, [], [], iter("abcdefghijkl")
        for i in p.fig342_levels:
            images += bands_to_unit([b["part_a"][i], b["part_b"][i], b["levels"][i]])
            titles += [f"({next(letters)}) apple × mask, L{i}", f"({next(letters)}) orange × (1 − mask), L{i}",
                       f"({next(letters)}) sum, L{i}"]
        images += [np.clip(collapse(b["part_a"]), 0, 1), np.clip(collapse(b["part_b"]), 0, 1), b["result"]]
        titles += ["(j) apple × mask, all levels", "(k) orange × (1 − mask), all levels", "(l) oraple"]
        self.save_panel(images, titles, "stacks/fig342.jpg", cols=3, size=3.2)
        self.save_web(b["result"], "stacks/oraple.jpg")

        self.data["stacks"] = {"levels": n, "sigma": p.stack_sigma, "total_sigma": total,
                               "reconstruction_error": st.reconstruction_error}
        self.html["stacks_check"] = (f'<div class="callout ok"><p><b>Check:</b> summing each Laplacian stack gives back '
                                     f"the original image to within <code>{st.reconstruction_error:.1e}</code>.</p></div>")
        self.html["stacks_code"] = _code(stacks.gaussian_stack, stacks.laplacian_stack, stacks.blend_stacks)

    def blend(self) -> None:
        """Part 2.4: every blend with its inputs and mask, why multiresolution, the levels choice, and the
        Laplacian-stack illustration of the favourite (like Figure 10 in Burt & Adelson)."""
        results = []
        for bl in self.ex.blends:
            q, d = bl.pair, f"blend/{bl.pair.name}"
            framed = np.pad(bl.mask[2:-2, 2:-2], 2, constant_values=0.5)     # gray frame: white stays visible
            self.save_panel([bl.a, bl.b, framed], ["image A (mask = 1)", "image B (mask = 0)", "mask"],
                            f"{d}/inputs.jpg", size=3.2)
            self.save_web(bl.color["result"], f"{d}/result.jpg")
            seam = "vertical step" if q.mask == "vertical" else "irregular, hand-drawn"
            results.append(f"<h4>{q.title}</h4>"
                           + '<div class="wide">' + _figure(f"{d}/inputs.jpg", "Inputs", f"mask: {seam}", cls="plain")
                           + "</div>" + '<div class="wide">'
                           # small results (oraple 300 px) are shown at a moderate width instead of stretched
                           + _figure(f"{d}/result.jpg", q.title, f"{q.levels} levels, σ = {q.sigma:g}",
                                     cls="plain" if bl.a.shape[1] >= 900 else "plain result") + "</div>")
        self.html["blend_results"] = "\n".join(results)

        methods = self.ex.blend_methods
        self.save_panel(list(methods.values()), list(methods), "blend/methods.jpg", size=3.2)
        self.save_panel([im[60:240, 90:210] for im in methods.values()], list(methods), "blend/methods_zoom.jpg", size=3.2)

        fav = self.ex.blend(FAVORITE_BLEND)
        t, b_, l, r = fav.pair.zoom
        zoom = (slice(t, b_), slice(l, r))
        levels = self.ex.blend_levels
        self.save_panel([fav.b[zoom]] + [im[zoom] for im in levels.values()],
                        ["image B (no blend)"] + [f"{n} levels" + ("  (chosen)" if n == fav.pair.levels else "")
                                                 for n in levels], "blend/levels.jpg", size=3.2)

        # Figure 10-style: every Laplacian level of the favourite, zoomed: A part, B part, sum
        c, n = fav.color, fav.pair.levels
        images, titles = [], []
        for i in range(n):
            row = [c["part_a"][i][zoom], c["part_b"][i][zoom], c["levels"][i][zoom]]
            images += bands_to_unit(row) if i < n - 1 else [np.clip(im, 0, 1) for im in row]
            kind = "band" if i < n - 1 else "low-pass rest"
            titles += [f"L{i} of A × mask ({kind})", f"L{i} of B × (1 − mask)", f"L{i} blended"]
        images += [np.clip(collapse(c["part_a"])[zoom], 0, 1), np.clip(collapse(c["part_b"])[zoom], 0, 1),
                   c["result"][zoom]]
        titles += ["A × mask, all levels", "B × (1 − mask), all levels", "result"]
        self.save_panel(images, titles, "blend/process.jpg", cols=3, size=3.2)
        self.data["blend"] = [{"name": bl.pair.name, "levels": bl.pair.levels, "sigma": bl.pair.sigma,
                               "size": [int(v) for v in bl.a.shape[:2]]} for bl in self.ex.blends]

    def blend_color(self) -> None:
        """Bells & whistles: each blend in grayscale and in color."""
        figs = []
        for bl in self.ex.blends:
            rel = self.save_panel([bl.gray, bl.color["result"]], ["grayscale", "color"],
                                  f"blend_color/{bl.pair.name}.jpg", size=4)
            figs.append(_figure(rel, bl.pair.title, cls="plain"))
        self.html["blend_color"] = '<div class="grid-3">' + "".join(figs) + "</div>"

    def hero(self) -> None:
        """Four square crops for the strip under the page title: one result from each kind of experiment."""
        ex = self.ex
        hyb = next(h for h in ex.hybrids if h.pair.name == "derek_nutmeg").hybrid
        eye = ex.blend("third_eye").color["result"]
        crops = {"orientation": ex.orientation.shaded,
                 "hybrid": hyb[: hyb.shape[1]],                          # top square: both heads
                 "earth_mars": ex.blend("earth_mars").color["result"],
                 "third_eye": eye[24:528, 348:852]}                      # the face (full image is 1200 × 900)
        for name, im in crops.items():
            self.save_web(im, f"hero/{name}.jpg", max_side=500)

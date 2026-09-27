"""Writes every figure and number the webpage uses: results/images/**/*.jpg and results/data.json.

One method per webpage section. Each pulls its results from an Experiments object
(computing or loading them from the cache on first use), saves web-sized JPGs and
fills its keys of data.json. All bells & whistles images go to one folder,
results/images/bells_whistles/, prefixed with their number (b1_..., b2_..., ...).

Sections whose content depends on the input plates (e.g. my own plates) also rewrite
their block of index.html, between <!-- generated:<key> start --> and <!-- generated:<key> end -->.
"""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from algorithm.align import stack_rgb
from algorithm.bells_whistles import color_matrices
from algorithm.config import DATA_JSON, IMAGES_DIR, INDEX_HTML, METRICS
from algorithm.io import save_jpg, to_uint8
from algorithm.viz import center_crop, save_panel
from experiments import FAIL, WINDOW_DEMO, Experiments

B3_NAMES = ("01007a", "01657u")         # subsets of the notebook demos shown on the page
B5_NAMES = ("01047u", "00125v")
BW = "bells_whistles"                   # one folder for all of B1-B6
THUMBS = "thumbs"                       # inline copies shown on the page (and in a printed PDF)
THUMB_SIDE, THUMB_QUALITY = 700, 75
# Judged by eye after looking at the results; these rows are marked "misaligned" on the page
OWN_FAILURES = {"00170a", "00153a"}


def _tup(x) -> list[int]:
    return [int(v) for v in x]


def _short(name: str) -> str:
    """LoC file stem -> plate id, e.g. "master-pnp-prok-00100-00166a" -> "00166a"."""
    return name.split("-")[-1]


def _off(offset) -> str:
    return f"({offset[0]}, {offset[1]})"


def _count(n: int) -> str:
    words = "zero one two three four five six seven eight nine ten".split()
    return words[n] if n < len(words) else str(n)


def _slider(before: str, after: str, name: str) -> str:
    """Before / after comparison with a drag slider (same markup as the course plates' final results)."""
    return (f'<figure class="compare" style="--pos: 50%">\n  <div class="compare-frame">\n'
            f'    <img src="{before}" alt="{name} before" loading="lazy">\n'
            f'    <img class="after" src="{after}" alt="{name} after" loading="lazy">\n'
            '    <span class="tag tag-l">before</span><span class="tag tag-r">after</span>\n'
            f'    <input type="range" min="0" max="100" value="50" aria-label="Before / after slider for {name}">\n'
            f'  </div>\n  <figcaption><b>{name}</b></figcaption>\n</figure>')


def _cards(names, variants, states: dict, label=lambda n: n, full: bool = False) -> str:
    """One card per plate showing several results side by side, each with its G / R offsets underneath.

    variants: [(tag, image path under results/images/ for a name, transforms for a name)].
    states: name -> "" | "diff" | "fail" (caption highlight). full: one card per row (for 3+ variants).
    """
    cards = []
    for name in names:
        links = "".join(
            f'<a href="results/images/{src(name)}" target="_blank" rel="noopener">'
            f'<img src="results/images/{src(name)}" alt="{label(name)}, {tag}" loading="lazy">'
            f'<span class="tag tag-l">{tag}</span></a>' for tag, src, _ in variants)
        offsets = "".join(f'<span class="mono">G {_off(tf(name)["G"].offset)} · R {_off(tf(name)["R"].offset)}</span>'
                          for _, _, tf in variants)
        state = f" {states[name]}" if states.get(name) else ""
        cards.append(f'<figure class="metric-card{state}" style="--n: {len(variants)}">'
                     f'<div class="metric-pair">{links}</div>'
                     f"<figcaption><b>{label(name)}</b>{offsets}</figcaption></figure>")
    return f'<div class="metric-grid{" full" if full else ""}">{"".join(cards)}</div>'


class Report:
    """Exports results for the webpage. build() runs the sections, then writes data.json."""

    SECTIONS = ("params", "overview", "sanity", "single", "pyramid", "failure", "own",
                "b1", "b2", "b3", "b4", "b5", "b6", "final")

    def __init__(self, ex: Experiments, images_dir: Path = IMAGES_DIR, data_json: Path = DATA_JSON,
                 index_html: Path = INDEX_HTML, max_side: int = 1000):
        self.ex = ex
        self.images_dir = images_dir
        self.data_json = data_json
        self.index_html = index_html
        self.max_side = max_side
        # Start from the previous export: keeps sections that aren't rebuilt this time,
        # e.g. "part1" (its photos only exist web-sized in results/images/part1).
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
            getattr(self, section)()
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

        The page shows these inline and links to the full image, which keeps the page and a printed PDF small.
        A thumbnail is only rewritten when its source is newer.
        """
        for src in self.images_dir.rglob("*.jpg"):
            rel = src.relative_to(self.images_dir)
            if rel.parts[0] == THUMBS:
                continue
            out = self.images_dir / THUMBS / rel
            if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
                save_jpg(Image.open(src).convert("RGB"), out, max_side=THUMB_SIDE, quality=THUMB_QUALITY)

    def write_html(self) -> None:
        """Replace each generated block of index.html, then point every inline <img> at its thumbnail.

        Links (<a href>) keep pointing at the full-size image. Images load eagerly, so a printed PDF
        contains every image, not only the ones that were scrolled into view.
        """
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
        text = text.replace(' loading="lazy"', "")
        self.index_html.write_text(text)
        print(f"[report] {self.index_html.name}: updated {', '.join(self.html) or 'image links'}")

    # ------------------------------------------------------------ sections

    def params(self) -> None:
        self.data["params"] = self.ex.params.to_dict()

    def overview(self) -> None:
        """Plate -> channels -> unaligned stack -> aligned (single scale, NCC)."""
        ov = self.ex.overview
        b, g, r = ov.channels
        self.save_web(ov.plate, "overview/plate.jpg", max_side=700)
        self.save_panel([b, g, r], ["B (top third)", "G (middle third)", "R (bottom third)"], "overview/channels.jpg")
        self.save_web(stack_rgb(b, g, r), "overview/unaligned.jpg", max_side=700)
        self.save_web(ov.aligned, "overview/aligned.jpg", max_side=700)
        self.data["overview"] = {"name": ov.name, "shape": _tup(ov.plate.shape), "channel": _tup(b.shape),
                                 "G": _tup(ov.transforms["G"].offset), "R": _tup(ov.transforms["R"].offset)}

    def sanity(self) -> None:
        table = self.ex.sanity
        self.data["sanity"] = {"true_offset": _tup(table.attrs["true_offset"]),
                               "rows": [{"case": case, **{m: float(v) for m, v in row.items()}}
                                        for case, row in table.iterrows()]}

    def single(self) -> None:
        """Sections 4-5: single-scale results, and why the window is ±20."""
        run = self.ex.single
        for (name, metric), im in run.images.items():
            self.save_web(im, f"single/{metric}/{name}.jpg", max_side=600)
        self.data["single"] = [{"name": row["name"], "metric": row["metric"], "G": _tup(row["G"]),
                                "R": _tup(row["R"]), "time": float(row["time_s"])}
                               for _, row in run.table.iterrows()]

        titles = [f"window ±{win}:  R {tf['R'].offset}" for win, _, tf in self.ex.window_demo]
        self.save_panel([center_crop(rgb, 200) for _, rgb, _ in self.ex.window_demo], titles,
                        "single/window_demo.jpg", size=4.5)
        self.data["window_demo"] = {"name": WINDOW_DEMO, "titles": titles}
        self.html["single_gallery"] = self._metric_cards(run, "single")

    def _metric_cards(self, run, folder: str, failures=()) -> str:
        """One card per plate with the L2 and NCC results side by side.

        Cards where the metrics differ are highlighted; plates in `failures` (judged by eye) are marked as failed.
        """
        names = list(dict.fromkeys(run.table["name"]))
        variants = [(m.upper(), lambda n, m=m: f"{folder}/{m}/{n}.jpg", lambda n, m=m: run.transforms(n, m))
                    for m in METRICS]
        states = {}
        for n in names:
            same = all(run.transforms(n, "l2")[ch].offset == run.transforms(n, "ncc")[ch].offset for ch in "GR")
            states[n] = "fail" if n in failures else "" if same else "diff"
        return _cards(names, variants, states)

    def pyramid(self) -> None:
        """Sections 6-8: levels, downsampling methods, per-level trace, check vs. single scale, results."""
        demo = self.ex.pyramid_demo
        self.save_panel(demo.levels, [f"level {i}: {im.shape[0]}×{im.shape[1]}" for i, im in enumerate(demo.levels)],
                        "pyramid/levels.jpg", size=2.6)
        patches = [np.asarray(Image.fromarray(to_uint8(p)).resize((360, 360), Image.Resampling.NEAREST))
                   for p in demo.downsampling.values()]
        self.save_panel(patches, list(demo.downsampling), "pyramid/downsampling.jpg", size=3.6)
        self.data["trace"] = {"name": demo.name, **{ch: [{"shape": _tup(s), "offset": _tup(o)} for s, o in trace]
                                                    for ch, trace in demo.traces.items()}}

        self.data["verify"] = [{"name": n, "G_single": _tup(row["G single"]), "G_pyr": _tup(row["G pyramid"]),
                                "R_single": _tup(row["R single"]), "R_pyr": _tup(row["R pyramid"]),
                                "diff": int(row["max diff (px)"])} for n, row in self.ex.verify.iterrows()]

        run = self.ex.pyramid
        for (name, metric), im in run.images.items():
            self.save_web(im, f"pyramid/{metric}/{name}.jpg")
        self.data["pyramid"] = [{"name": row["name"], "type": row["type"], "metric": row["metric"],
                                 "G": _tup(row["G"]), "R": _tup(row["R"]), "time": float(row["time_s"])}
                                for _, row in run.table.iterrows()]
        self.html["pyramid_gallery"] = self._metric_cards(run, "pyramid", failures={FAIL})

    def failure(self) -> None:
        """Section 9: channels of the failed plate and L2 vs. NCC crops."""
        f = self.ex.failure
        self.save_panel(list(f.channels), ["B", "G", "R"], "failure/channels.jpg")
        self.save_panel([f.crops[m] for m in METRICS], [f"{m.upper()}:  R {f.traces[m][-1]}" for m in METRICS],
                        "failure/crops.jpg", size=5)
        self.data["failure"] = {"name": f.name,
                                "stats": {ch: {k: float(v) for k, v in col.items()} for ch, col in f.stats.items()},
                                **{f"trace_{m}": [_tup(o) for o in f.traces[m]] for m in METRICS}}

    def own(self) -> None:
        """Section 10: my own plates, low-res (single scale) and full (pyramid)."""
        run = self.ex.own
        for (name, version, metric), im in run.images.items():
            self.save_web(im, f"own/{version}/{metric}/{_short(name)}.jpg")
        self.data["own"] = [{"name": _short(row["name"]), "version": row["version"], "metric": row["metric"],
                             "channel_height": int(row["channel_shape"]), "G": _tup(row["G"]),
                             "R": _tup(row["R"]), "time": float(row["time_s"])}
                            for _, row in run.table.iterrows()]
        self.html["own"] = self._own_html()

    def _own_html(self) -> str:
        """index.html block for my own plates: intro, NCC galleries (full + low-res), offsets table."""
        run, catalog = self.ex.own, self.ex.own_catalog
        names = list(dict.fromkeys(run.table["name"]))
        sizes = [p.stat().st_size / 1e6 for p in catalog.loc[names, "path"]]
        heights = run.table[run.table["version"] == "full"].groupby("name")["channel_shape"].first()
        depths = {}                                         # bit depth -> number of plates
        for path in catalog.loc[names, "path"]:
            with Image.open(path) as im:
                bits = 8 if im.mode in ("L", "RGB") else 16
            depths[bits] = depths.get(bits, 0) + 1
        if len(depths) == 1:
            kind = f"{next(iter(depths))}-bit TIFFs"
        else:
            kind = "TIFFs: " + ", ".join(f"{_count(n)} {b}-bit" for b, n in sorted(depths.items(), reverse=True))

        # Every plate: low-res (single scale) and full size (pyramid), each with L2 and NCC
        variants = [(f"{v} · {m.upper()}", lambda n, v=v, m=m: f"own/{v}/{m}/{_short(n)}.jpg",
                     lambda n, v=v, m=m: run.transforms(n, v, m)) for v in ("low-res", "full") for m in METRICS]
        states = {n: "fail" if _short(n) in OWN_FAILURES else "" for n in names}
        cards = _cards(names, variants, states, label=_short, full=True)

        head = "".join(f"<th>{v} {ch} ({m.upper()})</th>" for v in ("low-res", "full") for m in METRICS for ch in "GR")
        rows = []
        for name in names:
            cells = [_off(run.transforms(name, v, m)[ch].offset) for v in ("low-res", "full") for m in METRICS for ch in "GR"]
            cls = "fail" if _short(name) in OWN_FAILURES else ""
            rows.append(f'<tr class="{cls}"><td>{_short(name)}</td><td>{heights[name]}</td>'
                        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

        return "\n".join([
            f"<p>I downloaded {_count(len(names))} full-size plates ({kind}, "
            f"{min(sizes):.0f}–{max(sizes):.0f} MB each) from the",
            '<a href="https://www.loc.gov/collections/prokudin-gorskii/" target="_blank" rel="noopener">'
            "Library of Congress collection</a>,",
            "made my own low-res versions with my Gaussian <code>downsample</code> (halving 3 times, "
            "to about 1,200 px tall), and ran the",
            "same fixed-parameter pipeline: single-scale on the low-res version, the pyramid on the full plate.</p>",
            "<h4>Results: low-res (single scale) and full size (pyramid), L2 vs. NCC</h4>",
            cards,
            '<div class="table-wrap"><table class=""><thead><tr><th>Plate</th><th>channel height</th>'
            f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
            '<div class="legend"><span class="l-fail">misaligned</span></div>',
        ])

    def b1(self) -> None:
        """Edge maps, L2 raw vs. L2 edges on the failed plate, edge offsets for all plates."""
        maps = self.ex.edge_maps
        self.save_panel(list(maps.values()), list(maps), f"{BW}/b1_edges.jpg", size=3.2)
        before_after = self.ex.edges_before_after
        self.save_panel([crop for _, crop in before_after.values()],
                        [f"L2 on {label}:  R {tf['R'].offset}" for label, (tf, _) in before_after.items()],
                        f"{BW}/b1_before_after.jpg", size=5)
        table = self.ex.edge_table
        cols = [f"{ch} {kind} {m}" for kind in ("raw", "edges") for m in METRICS for ch in "GR"]
        self.data["edges"] = [{"name": n, **{c.replace(" ", "_"): _tup(row[c]) for c in cols}}
                              for n, row in table.iterrows()]

        # All 18 plates: raw L2, raw NCC, edges L2, edges NCC. The plate L2 fails on raw pixels is "fixed by edges".
        head = "".join(f"<th>{c.split()[0]} {c.split()[1]} {c.split()[2].upper()}</th>" for c in cols)
        rows = "".join(f'<tr class="{"fixed" if n == FAIL else ""}"><td>{n}</td>'
                       + "".join(f"<td>{_off(row[c])}</td>" for c in cols) + "</tr>" for n, row in table.iterrows())
        self.html["b1_table"] = (
            f'<details><summary>Show edge-based offsets for all {len(table)} plates</summary><div>\n'
            f'<div class="table-wrap"><table class=""><thead><tr><th>Plate</th>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table></div>\n"
            '<div class="legend"><span class="l-fixed">fixed by edges</span></div></div></details>')
        self._b1_own()

    def _b1_own(self) -> None:
        """B1 on my own plate OWN_EDGES_DEMO: full-size pyramid on raw pixels vs. on edges, L2 and NCC."""
        demo = self.ex.own_edges_demo
        name = next(iter(self.ex.own_edges.table["name"]))
        plate = _short(name)
        for (kind, m), (_, im) in demo.items():
            self.save_web(im, f"{BW}/b1_{plate}_{kind}_{m}.jpg")
        variants = [(f"{kind} · {m.upper()}", lambda n, k=kind, m=m: f"{BW}/b1_{plate}_{k}_{m}.jpg",
                     lambda n, k=kind, m=m: demo[k, m][0]) for kind, m in demo]
        card = _cards([name], variants, {}, label=_short, full=True)

        raw_l2, raw_ncc = (self.ex.own.transforms(name, "full", m) for m in METRICS)
        edges = {m: self.ex.own_edges.transforms(name, m) for m in METRICS}
        low = self.ex.own.transforms(name, "low-res", "ncc")["R"].offset
        factor = round(self.ex.own.table.query("name == @name")["channel_shape"].max()
                       / self.ex.own.table.query("name == @name")["channel_shape"].min())
        raw = {"l2": raw_l2, "ncc": raw_ncc}
        self.data["edges_own"] = {"name": plate,
                                  **{f"{ch}_raw_{m}": _tup(raw[m][ch].offset) for m in METRICS for ch in "GR"},
                                  **{f"{ch}_{m}": _tup(edges[m][ch].offset) for m in METRICS for ch in "GR"}}

        # Columns: raw L2, raw NCC, edges L2, edges NCC (G and R each)
        cells = [src[m][ch].offset for src in (raw, edges) for m in METRICS for ch in "GR"]
        head = "".join(f"<th>{ch} {kind} {m.upper()}</th>" for kind in ("raw", "edges") for m in METRICS for ch in "GR")
        if edges["l2"]["R"].offset == edges["ncc"]["R"].offset:
            on_edges = f"L2 and NCC both give R {_off(edges['ncc']['R'].offset)}"
        else:
            on_edges = f"L2 gives R {_off(edges['l2']['R'].offset)} and NCC R {_off(edges['ncc']['R'].offset)}"
        self.html["b1_own"] = "\n".join([
            f"<h4>My own plate {plate}</h4>",
            f"<p>On the full-size plate, the pyramid on raw pixels misaligns R with both metrics "
            f"(NCC {_off(raw_ncc['R'].offset)}, L2 {_off(raw_l2['R'].offset)}). The channels disagree about what is "
            "bright: in R the robe is much darker than the wall behind it, while in B the two are about equally bright, "
            "so the raw brightness patterns don't match. "
            f"On edges, {on_edges}, close to the low-res single-scale offset scaled up, "
            f"R {_off(low)} × {factor} = {_off((low[0] * factor, low[1] * factor))}.</p>",
            card,
            f'<div class="table-wrap"><table class=""><thead><tr><th>Plate</th>{head}</tr></thead>'
            f'<tbody><tr class="fixed"><td>{plate}</td>{"".join(f"<td>{_off(c)}</td>" for c in cells)}</tr>'
            "</tbody></table></div>",
            '<div class="legend"><span class="l-fixed">fixed by edges</span></div>',
        ])

    def b2(self) -> None:
        for name, ims in self.ex.crop_demo.items():
            self.save_web(ims["box"], f"{BW}/b2_{name}_box.jpg", max_side=700)
            self.save_web(ims["cropped"], f"{BW}/b2_{name}_cropped.jpg", max_side=700)
        self.data["crop"] = [{"name": n, "box": _tup(row["box (top, bottom, left, right)"]), "kept": row["kept"]}
                             for n, row in self.ex.crop.iterrows()]

    def b3(self) -> None:
        for name in B3_NAMES:
            for k, im in self.ex.contrast_demo[name].items():
                self.save_web(im, f"{BW}/b3_{name}_{k}.jpg", max_side=700)

    def b4(self) -> None:
        for name, ims in self.ex.white_balance_demo.items():
            for k, im in ims.items():
                self.save_web(im, f"{BW}/b4_{name}_{k}.jpg", max_side=600)

    def b5(self) -> None:
        images, colorfulness = self.ex.color_demo
        for name in B5_NAMES:
            for k, im in images[name].items():
                self.save_web(im, f"{BW}/b5_{name}_{k}.jpg", max_side=600)
        matrices = color_matrices(self.ex.params.color_strength)
        self.data["color"] = {"matrices": {k: np.round(M, 3).tolist() for k, M in matrices.items()},
                              "colorfulness": [{"name": n, **{k: float(v) for k, v in row.items()}}
                                               for n, row in colorfulness.iterrows()]}

    def b6(self) -> None:
        for k, im in self.ex.similarity_corner.items():
            self.save_web(im, f"{BW}/b6_corner_{k}.jpg", max_side=600)
        self.data["similarity"] = [{"name": n, "channel": ch, "angle": float(row["angle"]),
                                    "scale": float(row["scale"]), "t_only": _tup(row["translation only"]),
                                    "t": _tup(row["(dx, dy)"]), "before": float(row["NCC before"]),
                                    "after": float(row["NCC after"])}
                                   for (n, ch), row in self.ex.similarity.table.iterrows()]

    def final(self) -> None:
        for name, im in self.ex.final.items():
            self.save_web(im, f"final/{name}.jpg")
        # My own plates: before = basic full-size NCC pyramid, after = the same final pipeline
        for name, im in self.ex.own_final.items():
            self.save_web(im, f"final/own/{_short(name)}.jpg")
        sliders = "\n".join(_slider(f"results/images/own/full/ncc/{_short(n)}.jpg",
                                    f"results/images/final/own/{_short(n)}.jpg", _short(n))
                            for n in self.ex.own_final)
        self.html["final_own"] = (f"<h4>My own plates ({len(self.ex.own_final)})</h4>\n"
                                  f'<div class="compare-grid">\n{sliders}\n</div>')

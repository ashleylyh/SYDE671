"""Every experiment from the notebook, as functions that return numbers and images.

Nothing here writes to results/images/ (report.py does that). The slow searches cache their
offsets / transforms as JSON in results/cache/; images are always rebuilt from those.

    ex = Experiments()          # lazy: each result is computed on first access, then kept
    ex.pyramid.table            # offsets for all 18 plates x (L2, NCC)
    ex.pyramid.compare_metrics()
"""

import json
import time
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from algorithm.align import (Aligner, Transform, align_channels, align_pyramid, align_single_scale,
                             apply_transforms, colorize, shift, split_channels)
from algorithm.bells_whistles import (align_edges, align_similarity, apply_color_matrix, auto_contrast,
                                      auto_crop_box, color_matrices, colorfulness, edge_features, restore,
                                      white_balance)
from algorithm.config import CACHE_DIR, DATA_DIR, METRICS, OWN_DIR, OWN_LOWRES_DIR, Params
from algorithm.io import build_catalog, load_plate, save_jpg, to_web
from algorithm.metrics import SCORES, interior, l2, ncc
from algorithm.pyramid import build_pyramid, downsample, downsample_box, downsample_naive, make_lowres
from algorithm.viz import center_crop, corner_crop, draw_box

# Plates used for the demos (fixed, so the figures are reproducible)
SANITY_OFFSET = (7, -4)
OVERVIEW = "01164v"
WINDOW_DEMO = "01728v"                  # largest R shift of the low-res plates
PYRAMID_DEMO = "01007a"
FAIL = "01725u"                         # fails with L2 in the pyramid; NCC aligns it
CROP_DEMO = ("00458u", "01725u", "00125v")
CONTRAST_DEMO = ("01007a", "01657u", "31421v")
WB_DEMO = ("01657u", "01047u", "00125v")
COLOR_DEMO = ("01657u", "01047u", "00125v", "01164v")
SIMILARITY_DEMO = "00458u"
OWN_EDGES_DEMO = "00153a"               # own plate (portrait in a blue robe): raw pyramid fails at full size

Channels = tuple[np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------- JSON cache

class JsonCache:
    """Results of slow searches (offsets / transforms only) as <cache_dir>/<key>.json.

    Rows are cached per plate (every row has a "name"):
    - plates already cached are reused, only new plates are computed (e.g. a plate added to data/own/);
    - plates no longer in the input are dropped;
    - a file saved with different Params is ignored, so changing a parameter reruns the
      search instead of silently mixing old and new results.
    """

    def __init__(self, params: Params, cache_dir: Path = CACHE_DIR, reuse: bool = True):
        self.params = params
        self.cache_dir = cache_dir
        self.reuse = reuse

    def get(self, key: str, names, compute: Callable[[list[str]], list[dict]]) -> list[dict]:
        """Rows for every plate in `names`, in that order; compute(missing names) -> their rows."""
        names = list(names)
        path = self.cache_dir / f"{key}.json"
        cached = []
        if self.reuse and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("params") == self.params.to_dict():
                cached = saved["results"]
            else:
                print(f"[cache] {path.name}: saved with other params, recomputing")

        done = {row["name"] for row in cached}
        missing = [n for n in names if n not in done]
        if missing:
            print(f"[cache] {path.name}: computing {len(missing)} plate(s): {', '.join(missing)}")
        order = {n: i for i, n in enumerate(names)}
        rows = [row for row in cached + (compute(missing) if missing else []) if row["name"] in order]
        rows.sort(key=lambda row: order[row["name"]])      # stable: keeps metric / version order per plate

        if rows != cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"params": self.params.to_dict(), "results": rows}, indent=1))
        return rows


# ---------------------------------------------------------------- Alignment runs

@dataclass
class AlignmentRun:
    """One row per alignment (plate x metric [x version]) with its G and R offsets, plus the aligned images.

    `images` is keyed like the rows, e.g. images["00125v", "ncc"].
    """

    table: pd.DataFrame
    images: dict = field(default_factory=dict)
    key: tuple[str, ...] = ("name", "metric")

    def transforms(self, *key) -> dict[str, Transform]:
        row = self.table.set_index(list(self.key)).loc[key]
        return {ch: Transform.of(row[ch]) for ch in ("G", "R")}

    def compare_metrics(self) -> pd.DataFrame:
        """G / R offsets side by side per metric; `same` marks rows where L2 and NCC agree."""
        index = [k for k in self.key if k != "metric"]
        t = self.table.pivot(index=index, columns="metric", values=["G", "R", "time_s"])
        t["same"] = [row["G", "l2"] == row["G", "ncc"] and row["R", "l2"] == row["R", "ncc"]
                     for _, row in t.iterrows()]
        return t


def _row(name: str, metric: str, transforms: dict[str, Transform], seconds: float) -> dict:
    return {"name": name, "metric": metric, "G": list(transforms["G"].offset),
            "R": list(transforms["R"].offset), "time_s": round(seconds, 2)}


def _table(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Cached rows -> DataFrame with (dx, dy) tuples (JSON stores them as lists)."""
    table = pd.DataFrame(rows, columns=columns)
    for ch in ("G", "R"):
        table[ch] = table[ch].map(tuple)
    return table


def _on_window_edge(table: pd.DataFrame, window: int) -> list[bool]:
    """An offset on the window edge means the true shift may be larger than the search range."""
    return [any(abs(v) == window for off in (g, r) for v in off) for g, r in zip(table["G"], table["R"])]


def _align_plates(catalog: pd.DataFrame, names, align: Aligner, kwargs: dict, label: str) -> list[dict]:
    """Align G and R to B on every plate with every metric."""
    rows = []
    for name in names:
        plate = load_plate(catalog.loc[name, "path"])
        for metric in METRICS:
            start = time.perf_counter()
            tf = align_channels(plate, align, metric=metric, **kwargs)
            rows.append(_row(name, metric, tf, time.perf_counter() - start))
            print(f"[{label}] {name} {metric:>3}: G {tf['G'].offset}, R {tf['R'].offset}  "
                  f"[{rows[-1]['time_s']:.1f} s]")
    return rows


def _rebuild_images(table: pd.DataFrame, key: tuple[str, ...], channels: Callable[..., Channels],
                    keep: Callable[[np.ndarray], object]) -> dict:
    """Re-apply the offsets of every row. channels(*key[:-1]) loads the plate, once per consecutive group."""
    images, loaded, bgr = {}, None, None
    for _, row in table.iterrows():
        k = tuple(row[c] for c in key)
        if k[:-1] != loaded:
            loaded, bgr = k[:-1], channels(*k[:-1])
        images[k] = keep(apply_transforms(*bgr, {ch: Transform.of(row[ch]) for ch in ("G", "R")}))
    return images


def plate_channels(catalog: pd.DataFrame, name: str) -> Channels:
    return split_channels(load_plate(catalog.loc[name, "path"]))


def aligned_rgb(catalog: pd.DataFrame, name: str, transforms: dict[str, Transform]) -> np.ndarray:
    """Full-resolution RGB for a plate with the given transforms."""
    return apply_transforms(*plate_channels(catalog, name), transforms)


# ---------------------------------------------------------------- Sections 1-5: metrics + single scale

def check_metrics(catalog: pd.DataFrame, true_offset=SANITY_OFFSET) -> pd.DataFrame:
    """Section 3 sanity check: move B by a known amount; only the correct undo gives L2 = 0 and NCC = 1."""
    name = catalog.index[catalog["type"] == "v"][0]
    b = plate_channels(catalog, name)[0]
    moved = shift(b, true_offset)
    undo = (-true_offset[0], -true_offset[1])
    candidates = {"undo (correct)": undo, "no shift": (0, 0),
                  "off by 1 in x": (undo[0] + 1, undo[1]), "off by 1 in y": (undo[0], undo[1] + 1)}
    table = pd.DataFrame({case: {m: SCORES[m](interior(shift(moved, off)), interior(b)) for m in SCORES}
                          for case, off in candidates.items()}).T
    table.attrs.update(name=name, true_offset=true_offset)
    return table


@dataclass
class Overview:
    name: str
    plate: np.ndarray
    channels: Channels
    aligned: np.ndarray
    transforms: dict[str, Transform]


def make_overview(catalog: pd.DataFrame, params: Params, name: str = OVERVIEW) -> Overview:
    """Section 2 / 4: plate -> B, G, R -> single-scale NCC alignment."""
    plate = load_plate(catalog.loc[name, "path"])
    aligned, tf = colorize(plate, align_single_scale, metric="ncc", **params.single_kwargs())
    return Overview(name, plate, split_channels(plate), aligned, tf)


def run_single_scale(catalog: pd.DataFrame, params: Params, cache: JsonCache) -> AlignmentRun:
    """Section 5: exhaustive ±window search on every low-res plate, L2 and NCC."""
    names = list(catalog.index[catalog["type"] == "v"])
    rows = cache.get("single_scale", names, lambda todo: _align_plates(
        catalog, todo, align_single_scale, params.single_kwargs(), "single"))
    table = _table(rows, ["name", "metric", "G", "R", "time_s"])
    table["at_window_edge"] = _on_window_edge(table, params.window)
    images = _rebuild_images(table, ("name", "metric"), lambda n: plate_channels(catalog, n), keep=lambda rgb: rgb)
    return AlignmentRun(table, images)


def make_window_demo(catalog: pd.DataFrame, params: Params, name: str = WINDOW_DEMO,
                     windows=(15, 20)) -> list[tuple[int, np.ndarray, dict[str, Transform]]]:
    """Why ±20 instead of ±15: [(window, rgb, transforms)] on the plate with the largest R shift."""
    plate = load_plate(catalog.loc[name, "path"])
    return [(win, *colorize(plate, align_single_scale, window=win, metric="ncc", frac=params.frac))
            for win in windows]


# ---------------------------------------------------------------- Sections 6-8: pyramid

@dataclass
class PyramidDemo:
    name: str
    levels: list[np.ndarray]
    downsampling: dict[str, np.ndarray]                 # method -> center patch after 3 halvings
    traces: dict[str, list[tuple[tuple, tuple]]]        # channel -> [(level shape, offset so far)]


def make_pyramid_demo(catalog: pd.DataFrame, params: Params, name: str = PYRAMID_DEMO,
                      halvings: int = 3, patch: int = 120) -> PyramidDemo:
    """Section 6-7: pyramid levels, naive vs. box vs. Gaussian downsampling, per-level offsets."""
    b, g, r = plate_channels(catalog, name)
    patches = {}
    for label, down in (("naive [::2, ::2]", downsample_naive), ("2×2 block average", downsample_box),
                        ("Gaussian blur + subsample", downsample)):
        x = b
        for _ in range(halvings):
            x = down(x)
        patches[label] = center_crop(x, patch)
    traces = {}
    for ch, im in (("G", g), ("R", r)):
        traces[ch] = []
        align_pyramid(im, b, metric="ncc", trace=traces[ch], **params.pyramid_kwargs())
    return PyramidDemo(name, build_pyramid(b, params.pyr_min_size), patches, traces)


def verify_pyramid(catalog: pd.DataFrame, params: Params, single: AlignmentRun) -> pd.DataFrame:
    """Section 7 check: the pyramid gives the single-scale NCC offsets on the low-res plates.

    Low-res plates are only 341 px tall, so with pyr_min_size they'd have a single level.
    Force 2 levels (341 -> 170 px) so the doubling + refinement is actually tested;
    ±15 at half size covers ±30 px at full size.
    """
    kwargs = {**params.pyramid_kwargs(), "min_size": 200, "coarse_window": 15}
    rows = []
    for name in single.table["name"].unique():
        b, g, r = plate_channels(catalog, name)
        pyr = {ch: align_pyramid(im, b, metric="ncc", **kwargs) for ch, im in (("G", g), ("R", r))}
        ref = {ch: t.offset for ch, t in single.transforms(name, "ncc").items()}
        rows.append({
            "name": name,
            "G single": ref["G"], "G pyramid": pyr["G"],
            "R single": ref["R"], "R pyramid": pyr["R"],
            "max diff (px)": max(abs(p - s) for ch in "GR" for p, s in zip(pyr[ch], ref[ch])),
        })
    return pd.DataFrame(rows).set_index("name")


def run_pyramid(catalog: pd.DataFrame, params: Params, cache: JsonCache) -> AlignmentRun:
    """Section 8: coarse-to-fine alignment on all plates, L2 and NCC. Images are kept web-sized."""
    rows = cache.get("pyramid", catalog.index, lambda todo: _align_plates(
        catalog, todo, align_pyramid, params.pyramid_kwargs(), "pyramid"))
    table = _table(rows, ["name", "metric", "G", "R", "time_s"])
    table.insert(1, "type", catalog.loc[table["name"], "type"].to_numpy())
    images = _rebuild_images(table, ("name", "metric"), lambda n: plate_channels(catalog, n),
                             keep=lambda rgb: to_web(rgb, params.web_max_side))
    return AlignmentRun(table, images)


# ---------------------------------------------------------------- Section 9: failure analysis

@dataclass
class FailureAnalysis:
    name: str
    channels: Channels
    stats: pd.DataFrame                 # mean / std of each channel's interior
    traces: dict[str, list[tuple]]      # metric -> R offset at each pyramid level, coarsest first
    coarse_shape: tuple
    coarse_scores: pd.DataFrame         # both metrics' coarse picks, scored with both metrics
    full_scores: pd.DataFrame           # both metrics' final offsets, scored at full resolution
    crops: dict[str, np.ndarray]        # metric -> center crop of the result


def _score_picks(moving: np.ndarray, ref: np.ndarray, picks: dict[str, tuple], frac: float) -> pd.DataFrame:
    ref_in = interior(ref, frac)
    return pd.DataFrame([{"pick": label, "offset": off,
                          "l2": l2(interior(shift(moving, off), frac), ref_in),
                          "ncc": ncc(interior(shift(moving, off), frac), ref_in)}
                         for label, off in picks.items()]).set_index("pick")


def analyze_failure(catalog: pd.DataFrame, params: Params, pyramid: AlignmentRun,
                    name: str = FAIL) -> FailureAnalysis:
    """Why L2 misaligns R on `name`: brightness mismatch -> wrong coarsest pick -> doubled up the pyramid."""
    b, g, r = plate_channels(catalog, name)
    stats = pd.DataFrame({ch: {"mean": float(interior(im, params.frac).mean()),
                               "std": float(interior(im, params.frac).std())}
                          for ch, im in zip("BGR", (b, g, r))})
    traces = {}
    for metric in METRICS:
        trace = []
        align_pyramid(r, b, metric=metric, trace=trace, **params.pyramid_kwargs())
        traces[metric] = [off for _, off in trace]

    coarse_b = build_pyramid(b, params.pyr_min_size)[-1]
    coarse_r = build_pyramid(r, params.pyr_min_size)[-1]
    coarse = _score_picks(coarse_r, coarse_b, {f"{m.upper()} pick": traces[m][0] for m in METRICS}, params.frac)
    full = _score_picks(r, b, {f"{m.upper()} result": traces[m][-1] for m in METRICS}, params.frac)
    crops = {m: center_crop(np.asarray(pyramid.images[name, m]), 500) for m in METRICS}
    return FailureAnalysis(name, (b, g, r), stats, traces, coarse_b.shape, coarse, full, crops)


# ---------------------------------------------------------------- Section 10: own plates

def load_own_plate(path: Path, version: str, params: Params) -> np.ndarray:
    """"full" plate, or my own "low-res" version of it (also saved to results/images/own/lowres_plates/)."""
    full = load_plate(path)                  # 8-bit TIFFs are scaled by 255, 16-bit by 65535
    if version == "full":
        return full
    low = make_lowres(full, params.lowres_height)
    out = OWN_LOWRES_DIR / f"{path.stem.split('-')[-1]}.jpg"     # e.g. 00166a.jpg, like the results
    if not out.exists():                     # PDF: "You will have to create your own low-res versions"
        save_jpg(low, out)
    return low


def run_own(own_catalog: pd.DataFrame, params: Params, cache: JsonCache) -> AlignmentRun:
    """Section 10: same fixed parameters on my own plates.

    Single scale on my low-res version, pyramid on the full-size plate, both metrics.
    """
    settings = (("low-res", align_single_scale, params.single_kwargs()),
                ("full", align_pyramid, params.pyramid_kwargs()))

    def compute(names):
        rows = []
        for name, path in own_catalog.loc[names, "path"].items():
            for version, align, kwargs in settings:
                plate = load_own_plate(path, version, params)
                for metric in METRICS:
                    start = time.perf_counter()
                    tf = align_channels(plate, align, metric=metric, **kwargs)
                    rows.append({"name": name, "version": version, "channel_shape": plate.shape[0] // 3,
                                 **_row(name, metric, tf, time.perf_counter() - start)})
                    print(f"[own] {name} {version:>7} {metric:>3}: G {tf['G'].offset}, R {tf['R'].offset}  "
                          f"[{rows[-1]['time_s']:.1f} s]")
        return rows

    rows = cache.get("own", own_catalog.index, compute)
    table = _table(rows, ["name", "version", "channel_shape", "metric", "G", "R", "time_s"])
    # single-scale only: an offset on the window edge means the true shift may be larger
    table["at_window_edge"] = [v == "low-res" and e for v, e in
                               zip(table["version"], _on_window_edge(table, params.window))]
    key = ("name", "version", "metric")
    images = _rebuild_images(
        table, key, lambda n, v: split_channels(load_own_plate(own_catalog.loc[n, "path"], v, params)),
        keep=lambda rgb: to_web(rgb, params.web_max_side))
    return AlignmentRun(table, images, key)


# ---------------------------------------------------------------- B1: edges

def run_edges(catalog: pd.DataFrame, params: Params, cache: JsonCache) -> AlignmentRun:
    """B1: pyramid alignment on edge maps for all plates, L2 and NCC (offsets only, no images)."""
    rows = cache.get("edges", catalog.index, lambda todo: _align_plates(
        catalog, todo, align_edges, params.pyramid_kwargs(), "edges"))
    return AlignmentRun(_table(rows, ["name", "metric", "G", "R", "time_s"]))


def compare_edges(edges: AlignmentRun, pyramid: AlignmentRun) -> pd.DataFrame:
    """Raw-pixel offsets from Section 8 next to the edge-based ones, both metrics.

    Columns: G / R raw l2, G / R raw ncc, G / R edges l2, G / R edges ncc.
    """
    rows = []
    for name in edges.table["name"].unique():
        row = {"name": name}
        for kind, run in (("raw", pyramid), ("edges", edges)):
            for metric in METRICS:
                tf = run.transforms(name, metric)
                row[f"G {kind} {metric}"], row[f"R {kind} {metric}"] = tf["G"].offset, tf["R"].offset
        rows.append(row)
    table = pd.DataFrame(rows).set_index("name")
    table["L2 == NCC"] = ((table["G edges l2"] == table["G edges ncc"])
                          & (table["R edges l2"] == table["R edges ncc"]))
    return table


def make_edge_maps(catalog: pd.DataFrame, name: str = FAIL) -> dict[str, np.ndarray]:
    """B and R raw vs. their edge maps (same scale for both edge maps)."""
    b, g, r = plate_channels(catalog, name)
    eb, er = edge_features(b), edge_features(r)
    scale = max(np.percentile(eb, 99.5), np.percentile(er, 99.5))
    return {"B raw": b, "R raw": r, "B edges": eb / scale, "R edges": er / scale}


def make_edges_before_after(catalog: pd.DataFrame, pyramid: AlignmentRun, edges: AlignmentRun,
                            name: str = FAIL, size: int = 800) -> dict[str, tuple[dict, np.ndarray]]:
    """L2 on raw pixels vs. L2 on edges: {label: (transforms, center crop)}."""
    out = {}
    for label, run in (("raw pixels", pyramid), ("edges", edges)):
        tf = run.transforms(name, "l2")
        out[label] = (tf, center_crop(aligned_rgb(catalog, name, tf), size).copy())
    return out


def own_name(own_catalog: pd.DataFrame, plate_id: str) -> str:
    """Plate id -> catalog name, e.g. "00153a" -> "master-pnp-prok-00100-00153a"."""
    matches = [n for n in own_catalog.index if n.endswith(plate_id)]
    if not matches:
        raise KeyError(f"no plate ending in {plate_id!r} in {OWN_DIR}")
    return matches[0]


def run_own_edges(own_catalog: pd.DataFrame, params: Params, cache: JsonCache, names) -> AlignmentRun:
    """B1 on my own full-size plates: pyramid on edge maps, L2 and NCC (offsets only)."""
    rows = cache.get("own_edges", names, lambda todo: _align_plates(
        own_catalog, todo, align_edges, params.pyramid_kwargs(), "own edges"))
    return AlignmentRun(_table(rows, ["name", "metric", "G", "R", "time_s"]))


def make_own_edges_demo(own_catalog: pd.DataFrame, params: Params, own: AlignmentRun, own_edges: AlignmentRun,
                        name: str) -> dict[str, tuple[dict, object]]:
    """Full-size pyramid on raw pixels vs. on edges, L2 and NCC.

    {(kind, metric): (transforms, web-sized image)} for kind in ("raw", "edges").
    """
    b, g, r = split_channels(load_plate(own_catalog.loc[name, "path"]))
    out = {}
    for kind in ("raw", "edges"):
        for m in METRICS:
            tf = own.transforms(name, "full", m) if kind == "raw" else own_edges.transforms(name, m)
            out[kind, m] = (tf, to_web(apply_transforms(b, g, r, tf), params.web_max_side))
    return out


# ---------------------------------------------------------------- B2-B5: crop, contrast, white balance, color

def run_crop(catalog: pd.DataFrame, edges: AlignmentRun) -> pd.DataFrame:
    """B2: crop box and kept area for every plate (edge-based NCC offsets)."""
    rows = []
    for name in catalog.index:
        tf = edges.transforms(name, "ncc")
        rgb = aligned_rgb(catalog, name, tf)
        t, b, l, r = auto_crop_box(rgb, tf.values())
        rows.append({"name": name, "box (top, bottom, left, right)": (t, b, l, r),
                     "kept": f"{(b - t) * (r - l) / (rgb.shape[0] * rgb.shape[1]):.0%}"})
    return pd.DataFrame(rows).set_index("name")


def cropped_rgb(catalog: pd.DataFrame, edges: AlignmentRun, name: str) -> np.ndarray:
    """Aligned (edge-based NCC) and auto-cropped (B2) plate: the input to B3-B5."""
    tf = edges.transforms(name, "ncc")
    rgb = aligned_rgb(catalog, name, tf)
    t, b, l, r = auto_crop_box(rgb, tf.values())
    return rgb[t:b, l:r]


def make_crop_demo(catalog: pd.DataFrame, params: Params, edges: AlignmentRun,
                   names=CROP_DEMO) -> dict[str, dict]:
    """B2 before / after: {name: {"box": plate with the detected box, "cropped": result}} (web-sized)."""
    out = {}
    for name in names:
        tf = edges.transforms(name, "ncc")
        rgb = aligned_rgb(catalog, name, tf)
        t, b, l, r = auto_crop_box(rgb, tf.values())
        out[name] = {"box": to_web(draw_box(rgb, (t, b, l, r)), params.web_max_side),
                     "cropped": to_web(rgb[t:b, l:r], params.web_max_side)}
    return out


def make_contrast_demo(catalog: pd.DataFrame, params: Params, edges: AlignmentRun,
                       names=CONTRAST_DEMO) -> dict[str, dict]:
    """B3 before / after (web-sized)."""
    out = {}
    for name in names:
        rgb = cropped_rgb(catalog, edges, name)
        out[name] = {"before": to_web(rgb, params.web_max_side),
                     "after": to_web(auto_contrast(rgb), params.web_max_side)}
    return out


def make_white_balance_demo(catalog: pd.DataFrame, params: Params, edges: AlignmentRun,
                            names=WB_DEMO) -> dict[str, dict]:
    """B4: contrast only vs. gray world (full / params.wb_strength) vs. white patch (web-sized)."""
    out = {}
    for name in names:
        rgb = auto_contrast(cropped_rgb(catalog, edges, name))
        versions = {
            "none": rgb,
            "grayworld": auto_contrast(white_balance(rgb, "gray_world")),
            "grayworld_half": auto_contrast(white_balance(rgb, "gray_world", strength=params.wb_strength)),
            "whitepatch": auto_contrast(white_balance(rgb, "white_patch")),
        }
        out[name] = {k: to_web(v, params.web_max_side) for k, v in versions.items()}
    return out


def make_color_demo(catalog: pd.DataFrame, params: Params, edges: AlignmentRun,
                    names=COLOR_DEMO) -> tuple[dict[str, dict], pd.DataFrame]:
    """B5: every color matrix, then white balance + contrast. Returns (web-sized images, colorfulness).

    Pipeline order: align -> crop -> color matrix -> white balance -> contrast.
    The matrix goes first because it converts plate colors to RGB; B4 and B3 then work on proper RGB.
    """
    images, rows = {}, []
    for name in names:
        rgb = cropped_rgb(catalog, edges, name)
        versions = {k: auto_contrast(white_balance(apply_color_matrix(rgb, M), "gray_world",
                                                   strength=params.wb_strength))
                    for k, M in color_matrices(params.color_strength).items()}
        images[name] = {k: to_web(v, params.web_max_side) for k, v in versions.items()}
        rows.append({"name": name, **{k: round(colorfulness(v), 3) for k, v in versions.items()}})
    return images, pd.DataFrame(rows).set_index("name")


# ---------------------------------------------------------------- B6: rotation + scale

@dataclass
class SimilarityRun:
    table: pd.DataFrame                              # index (name, channel)
    transforms: dict[str, dict[str, Transform]]      # name -> {"G": ..., "R": ...}


def run_similarity(catalog: pd.DataFrame, params: Params, cache: JsonCache,
                   key: str = "similarity") -> SimilarityRun:
    """B6: best (angle, scale) per channel and the NCC gain over translation only (cached under `key`)."""

    def compute(names):
        rows = []
        for name in names:
            b, g, r = plate_channels(catalog, name)
            start = time.perf_counter()
            for ch, im in (("G", g), ("R", r)):
                info = {}
                tf = align_similarity(im, b, params=params, info=info)
                rows.append({"name": name, "channel": ch, "transform": asdict(tf),
                             "translation_only": list(info["translation_only"]),
                             "score_before": info["score_before"], "score_after": info["score_after"]})
            print(f"[similarity] {name}: {time.perf_counter() - start:.1f} s")
        return rows

    rows = cache.get(key, catalog.index, compute)
    transforms, table_rows = {}, []
    for row in rows:
        tf = Transform(**row["transform"])
        transforms.setdefault(row["name"], {})[row["channel"]] = tf
        table_rows.append({"name": row["name"], "channel": row["channel"], "angle": tf.angle, "scale": tf.scale,
                           "translation only": tuple(row["translation_only"]), "(dx, dy)": tf.offset,
                           "NCC before": round(row["score_before"], 4), "NCC after": round(row["score_after"], 4)})
    return SimilarityRun(pd.DataFrame(table_rows).set_index(["name", "channel"]), transforms)


def check_similarity_control(catalog: pd.DataFrame, params: Params, name: str = SIMILARITY_DEMO,
                             offset=(17, -9)) -> Transform:
    """Control: B against a shifted copy of itself must give angle 0, scale 1 and the opposite shift."""
    b = plate_channels(catalog, name)[0]
    return align_similarity(shift(b, offset), b, params=params)


def make_similarity_corner(catalog: pd.DataFrame, edges: AlignmentRun, similarity: SimilarityRun,
                           name: str = SIMILARITY_DEMO) -> dict[str, np.ndarray]:
    """B6 before / after on the bottom-right corner, where rotation and scale errors are largest."""
    return {"before": corner_crop(aligned_rgb(catalog, name, edges.transforms(name, "ncc"))).copy(),
            "after": corner_crop(aligned_rgb(catalog, name, similarity.transforms[name])).copy()}


# ---------------------------------------------------------------- Final pipeline

def run_final(catalog: pd.DataFrame, params: Params, similarity: SimilarityRun) -> dict:
    """All bells & whistles on every plate, reusing the B6 transforms. {name: web-sized image}."""
    out = {}
    for name, path in catalog["path"].items():
        out[name] = to_web(restore(load_plate(path), params, similarity.transforms[name]), params.web_max_side)
        print(f"[final] {name}")
    return out


# ---------------------------------------------------------------- All experiments, computed lazily

class Experiments:
    """Every result above, computed on first access and kept in memory.

    Dependencies resolve themselves: `ex.failure` runs (or loads) the pyramid first.
    `reuse=False` ignores the JSON cache and reruns every search.
    """

    def __init__(self, params: Params = Params(), reuse: bool = True):
        self.params = params
        self.cache = JsonCache(params, reuse=reuse)

    @cached_property
    def catalog(self) -> pd.DataFrame:
        return build_catalog(DATA_DIR)

    @cached_property
    def own_catalog(self) -> pd.DataFrame:
        return build_catalog(OWN_DIR, suffixes=(".tif", ".tiff", ".jpg", ".jpeg"))

    @cached_property
    def sanity(self) -> pd.DataFrame:
        return check_metrics(self.catalog)

    @cached_property
    def overview(self) -> Overview:
        return make_overview(self.catalog, self.params)

    @cached_property
    def single(self) -> AlignmentRun:
        return run_single_scale(self.catalog, self.params, self.cache)

    @cached_property
    def window_demo(self) -> list:
        return make_window_demo(self.catalog, self.params)

    @cached_property
    def pyramid_demo(self) -> PyramidDemo:
        return make_pyramid_demo(self.catalog, self.params)

    @cached_property
    def verify(self) -> pd.DataFrame:
        return verify_pyramid(self.catalog, self.params, self.single)

    @cached_property
    def pyramid(self) -> AlignmentRun:
        return run_pyramid(self.catalog, self.params, self.cache)

    @cached_property
    def failure(self) -> FailureAnalysis:
        return analyze_failure(self.catalog, self.params, self.pyramid)

    @cached_property
    def own(self) -> AlignmentRun:
        return run_own(self.own_catalog, self.params, self.cache)

    @cached_property
    def edges(self) -> AlignmentRun:
        return run_edges(self.catalog, self.params, self.cache)

    @cached_property
    def edge_table(self) -> pd.DataFrame:
        return compare_edges(self.edges, self.pyramid)

    @cached_property
    def edge_maps(self) -> dict:
        return make_edge_maps(self.catalog)

    @cached_property
    def edges_before_after(self) -> dict:
        return make_edges_before_after(self.catalog, self.pyramid, self.edges)

    @cached_property
    def own_edges(self) -> AlignmentRun:
        return run_own_edges(self.own_catalog, self.params, self.cache,
                             [own_name(self.own_catalog, OWN_EDGES_DEMO)])

    @cached_property
    def own_edges_demo(self) -> dict:
        return make_own_edges_demo(self.own_catalog, self.params, self.own, self.own_edges,
                                   own_name(self.own_catalog, OWN_EDGES_DEMO))

    @cached_property
    def crop(self) -> pd.DataFrame:
        return run_crop(self.catalog, self.edges)

    @cached_property
    def crop_demo(self) -> dict:
        return make_crop_demo(self.catalog, self.params, self.edges)

    @cached_property
    def contrast_demo(self) -> dict:
        return make_contrast_demo(self.catalog, self.params, self.edges)

    @cached_property
    def white_balance_demo(self) -> dict:
        return make_white_balance_demo(self.catalog, self.params, self.edges)

    @cached_property
    def color_demo(self) -> tuple[dict, pd.DataFrame]:
        return make_color_demo(self.catalog, self.params, self.edges)

    @cached_property
    def similarity(self) -> SimilarityRun:
        return run_similarity(self.catalog, self.params, self.cache)

    @cached_property
    def similarity_corner(self) -> dict:
        return make_similarity_corner(self.catalog, self.edges, self.similarity)

    @cached_property
    def final(self) -> dict:
        return run_final(self.catalog, self.params, self.similarity)

    @cached_property
    def own_similarity(self) -> SimilarityRun:
        return run_similarity(self.own_catalog, self.params, self.cache, key="own_similarity")

    @cached_property
    def own_final(self) -> dict:
        """The final pipeline on my own full-size plates."""
        return run_final(self.own_catalog, self.params, self.own_similarity)

"""Click correspondences between neighbouring images of each mosaic set (A.2); saved to results/points.json.

    uv run python pick_points.py                  # every pair that has no points yet
    uv run python pick_points.py set1 set4        # just these sets
    uv run python pick_points.py set1 --redo      # click set1 again from scratch, replacing its saved points
    uv run python pick_points.py set1 --edit      # open set1's saved points to refine them
    uv run python pick_points.py --rectify        # A.3: the 4 corners of each rectangle (--redo / --edit work too)

Pairs: click a point in the LEFT image, then the same point in the RIGHT image, and repeat (PDF A.2: more than 4 pairs,
spread over the overlap; corners of windows, signs, rooftops... not water or clouds, which move).
Rectify: click the rectangle's 4 corners, in any order.

The last clicked point is selected (white cross); the zoom panel(s) on the right show it magnified, pixel by pixel.
    arrow keys            move the selected point by 1 px (Shift + arrow: 0.25 px)
    click in a zoom       put the selected point exactly there
    click near a point    select it (to fix an earlier point);  [ / ]  select the previous / next point
    + / -                 zoom in / out
    Backspace, right-click  delete the last point
    Enter                 save          Esc: quit without saving
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from algorithm.config import MOSAICS, POINTS_JSON, RECTIFICATIONS, RECTIFY_DIR, Params, find_input
from algorithm.io import load_image, pair_key, read_points, write_points
from algorithm.viz import POINT_COLORS

MIN_POINTS = 6
STEP, FINE_STEP = 1.0, 0.25             # arrow-key moves, in image pixels
PICK_RADIUS = 12                        # a click this close (image px) to a point selects it instead of adding one
MOVES = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

# Matplotlib binds the arrow keys and Backspace to the toolbar's back / forward; we need them
plt.rcParams["keymap.back"] = []
plt.rcParams["keymap.forward"] = []


class Picker:
    """Points on 1 image (rectify) or corresponding points on 2 images (clicked alternately), with keyboard nudging.

    Points are (x, y) in image pixels with (0, 0) at the centre of the top-left pixel (imshow's convention).
    """

    def __init__(self, images, title: str, n_points: int | None = None, initial=None, zoom: float = 20.0):
        self.images, self.title, self.n_points, self.zoom = images, title, n_points, zoom
        self.pts = [[list(p) for p in pts] for pts in initial] if initial else [[] for _ in images]
        self.selected = (len(images) - 1, len(self.pts[-1]) - 1) if self.pts[-1] else None
        self.saved = False
        k = len(images)
        self.fig = plt.figure(figsize=(8 * k + 4, 8))
        grid = self.fig.add_gridspec(2, k + 1, width_ratios=[3] * k + [1.2])
        self.axes = [self.fig.add_subplot(grid[:, i]) for i in range(k)]
        self.zooms = [self.fig.add_subplot(grid[i, k]) for i in range(k)]
        for ax, im in zip(self.axes + self.zooms, images + images):
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        self.artists: list = []
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    # ------------------------------------------------------------ state

    @property
    def turn(self) -> int:
        """The image the next new point goes in: pairs alternate left, right."""
        return 0 if len(self.pts) == 1 or len(self.pts[0]) == len(self.pts[1]) else 1

    @property
    def complete(self) -> int:
        return min(len(p) for p in self.pts)

    def order(self) -> list[tuple[int, int]]:
        """Every point as (image, index), in click order."""
        out = []
        for i in range(max(len(p) for p in self.pts)):
            out += [(k, i) for k in range(len(self.pts)) if i < len(self.pts[k])]
        return out

    def focus(self, k: int):
        """The point shown in image k's zoom: the selected one if it is in image k, else that image's last."""
        if self.selected and self.selected[0] == k:
            return self.pts[k][self.selected[1]]
        return self.pts[k][-1] if self.pts[k] else None

    # ------------------------------------------------------------ drawing

    def redraw(self) -> None:
        for a in self.artists:
            a.remove()
        self.artists = []
        for k, (ax, zax) in enumerate(zip(self.axes, self.zooms)):
            for i, (x, y) in enumerate(self.pts[k]):
                c = [v / 255 for v in POINT_COLORS[i % len(POINT_COLORS)]]
                sel = self.selected == (k, i)
                self.artists += ax.plot(x, y, "o", mfc="none", mec="white" if sel else c, ms=14 if sel else 10, mew=2)
                self.artists.append(ax.text(x + 10, y - 10, str(i + 1), color=c, fontsize=11, fontweight="bold"))
                self.artists += zax.plot(x, y, "+", color="white" if sel else c, ms=18, mew=1.5)
            p = self.focus(k)
            if p is not None:
                zax.set_xlim(p[0] - self.zoom, p[0] + self.zoom)
                zax.set_ylim(p[1] + self.zoom, p[1] - self.zoom)
                zax.set_title(f"zoom: ({p[0]:.2f}, {p[1]:.2f})", fontsize=10)
            else:
                zax.set_title("zoom", fontsize=10)
        if self.n_points:
            todo = f"{len(self.pts[0])} / {self.n_points} points"
        else:
            todo = f"{self.complete} pairs - next: {'LEFT' if self.turn == 0 else 'RIGHT'} image"
        self.fig.suptitle(f"{self.title}   [{todo}]\nclick to add  ·  arrows move the selected point (Shift: 0.25 px)  ·  "
                          "click in a zoom to place it  ·  [ ] select  ·  + - zoom  ·  Backspace delete  ·  Enter save",
                          fontsize=11)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ events

    def on_click(self, event) -> None:
        toolbar = self.fig.canvas.toolbar
        if event.inaxes is None or (toolbar is not None and toolbar.mode != ""):     # not while panning / zooming
            return
        if event.button == 3:
            self.delete_last()
        elif event.button == 1 and event.inaxes in self.zooms:
            k = self.zooms.index(event.inaxes)
            p = self.focus(k)
            if p is not None:
                p[:] = [event.xdata, event.ydata]
                self.selected = (k, self.pts[k].index(p))
        elif event.button == 1 and event.inaxes in self.axes:
            k = self.axes.index(event.inaxes)
            near = [i for i, (x, y) in enumerate(self.pts[k]) if np.hypot(x - event.xdata, y - event.ydata) < PICK_RADIUS]
            if near:
                self.selected = (k, near[-1])
            elif k == self.turn and not (self.n_points and len(self.pts[0]) >= self.n_points):
                self.pts[k].append([event.xdata, event.ydata])
                self.selected = (k, len(self.pts[k]) - 1)
        self.redraw()

    def on_key(self, event) -> None:
        key = event.key or ""
        base = key.removeprefix("shift+")
        if base in MOVES and self.selected:
            step = FINE_STEP if key.startswith("shift+") else STEP
            k, i = self.selected
            self.pts[k][i][0] += MOVES[base][0] * step
            self.pts[k][i][1] += MOVES[base][1] * step
        elif key in ("[", "]") and self.selected:
            order = self.order()
            self.selected = order[(order.index(self.selected) + (1 if key == "]" else -1)) % len(order)]
        elif key in ("+", "="):
            self.zoom = max(3.0, self.zoom / 1.5)
        elif key == "-":
            self.zoom = min(200.0, self.zoom * 1.5)
        elif key == "backspace":
            self.delete_last()
        elif key == "enter":
            need = self.n_points or MIN_POINTS
            if self.complete < need or len({len(p) for p in self.pts}) > 1 or (self.n_points and len(self.pts[0]) != need):
                print(f"  need {'exactly' if self.n_points else '>='} {need} complete {'points' if self.n_points else 'pairs'}")
                return
            self.saved = True
            plt.close(self.fig)
            return
        elif key == "escape":
            plt.close(self.fig)
            return
        self.redraw()

    def delete_last(self) -> None:
        """Remove the most recently added point."""
        order = self.order()
        if order:
            k, i = order[-1]
            self.pts[k].pop(i)
            rest = self.order()
            self.selected = rest[-1] if rest else None

    def run(self) -> list[list[list[float]]]:
        plt.show()
        if not self.saved:
            raise SystemExit("window closed without saving")
        return [[[round(x, 2), round(y, 2)] for x, y in pts] for pts in self.pts]


def pick_rectify(saved: dict, params: Params, redo: bool, edit: bool) -> None:
    for rect in RECTIFICATIONS:
        entry = saved.get("rectify", {}).get(rect.name)
        if entry and not (redo or edit):
            print(f"rectify {rect.name}: already saved (use --redo or --edit)")
            continue
        im = load_image(find_input(rect.image, RECTIFY_DIR), max_side=params.max_side)
        initial = [entry["im1"]] if entry and edit else None
        (pts,) = Picker([im], f"{rect.title}: the 4 corners of the rectangle", n_points=4, initial=initial).run()
        saved.setdefault("rectify", {})[rect.name] = {"im1": pts, "size1": [im.shape[1], im.shape[0]]}
        write_points(POINTS_JSON, saved)
        print(f"rectify {rect.name}: saved to {POINTS_JSON.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="*", help="set names (default: all)")
    parser.add_argument("--redo", action="store_true", help="click saved pairs again from scratch")
    parser.add_argument("--edit", action="store_true", help="open saved pairs with their points, to refine them")
    parser.add_argument("--rectify", action="store_true", help="A.3: click the corners of each rectangle instead")
    args = parser.parse_args()

    params = Params()
    saved = read_points(POINTS_JSON)
    if args.rectify:
        pick_rectify(saved, params, args.redo, args.edit)
        return
    for mset in MOSAICS:
        if args.sets and mset.name not in args.sets:
            continue
        for left, right in mset.pairs:
            key = pair_key(left, right)
            entry = saved.get(mset.name, {}).get(key)
            if entry and not (args.redo or args.edit):
                print(f"{mset.name} {key}: already saved (use --redo or --edit)")
                continue
            im1, im2 = (load_image(find_input(s, mset.folder), max_side=params.max_side) for s in (left, right))
            # saved points are in the image size they were clicked at; --edit opens them at the current size
            initial = None
            if entry and args.edit:
                if entry["size1"] != [im1.shape[1], im1.shape[0]] or entry["size2"] != [im2.shape[1], im2.shape[0]]:
                    raise SystemExit(f"{mset.name} {key}: saved at another image size; use --redo")
                initial = [entry["im1"], entry["im2"]]
            pts1, pts2 = Picker([im1, im2], f"{mset.title}: {left} -> {right}", initial=initial).run()
            saved.setdefault(mset.name, {})[key] = {
                "im1": pts1, "im2": pts2,
                "size1": [im1.shape[1], im1.shape[0]], "size2": [im2.shape[1], im2.shape[0]]}
            write_points(POINTS_JSON, saved)                # after every pair, so nothing is lost
            print(f"{mset.name} {key}: {len(pts1)} pairs saved to {POINTS_JSON.name}")


if __name__ == "__main__":
    main()

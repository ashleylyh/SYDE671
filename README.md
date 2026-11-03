# SYDE 671 Assignments

Published with GitHub Pages at <https://ashleylyh.github.io/SYDE671/>.

| Assignment | Page |
|---|---|
| Assignment 1: Becoming friends with the camera & Colorizing the Prokudin-Gorskii collection | <https://ashleylyh.github.io/SYDE671/assignment1/> |
| Assignment 2: Fun with Filters and Frequencies | <https://ashleylyh.github.io/SYDE671/assignment2/> |
| Assignment 3: [Auto]Stitching Photo Mosaics | <https://ashleylyh.github.io/SYDE671/assignment3/> |

## Layout

```
SYDE671/
├── index.html              # landing page that links to each assignment
├── assets/style.css        # shared stylesheet
├── .nojekyll               # serve files as-is (no Jekyll processing)
├── pyproject.toml, uv.lock # Python environment shared by all assignments
└── assignment1/
    ├── index.html          # the write-up page → /SYDE671/assignment1/
    ├── src/                # code (see "Assignment 1: running the code")
    ├── results/
    │   ├── images/         # every figure and result image on the page (thumbs/ = the small inline copies)
    │   ├── data.json       # every number on the page (offsets, scores, crop boxes, ...)
    │   └── cache/          # JSON cache of the slow alignment searches
    └── data/               # input plates (git-ignored, not published)
└── assignment2/            # same structure; results/align_points.json holds the clicked alignment points
└── assignment3/            # same structure; results/points.json holds the clicked correspondences and corners
```

PDFs, zips, raw TIFF scans and `data/` folders are git-ignored, so course materials and large inputs stay local.

## Environment

Managed with [uv](https://docs.astral.sh/uv/) (Python 3.12), shared by all assignments.

```
uv sync                         # create .venv and install dependencies
uv add <package>                # add a dependency
uv run python script.py         # run a script inside the environment
```

For notebooks in VS Code, select the `.venv` interpreter as the kernel.

## Assignment 1: running the code

### 1. Put the input plates in place

```
assignment1/data/
├── 00056v.jpg ... 31421v.jpg   # the 18 course plates from Learn (v = low-res, a / u = full-size)
└── own/
    └── *.tif                   # my own full-size plates from the Library of Congress collection
```

The plate type comes from the last letter of the file name (`v`, `a` or `u`). Any `.tif`, `.tiff`, `.jpg` or `.jpeg`
added to `data/own/` is processed automatically on the next run.

### 2. Run everything

From the repository root:

```
uv sync
cd assignment1/src
uv run python run.py                     # every section; reuses results/cache/*.json
uv run python run.py --only pyramid b1   # only these sections (data.json keeps the others)
uv run python run.py --fresh             # ignore the cache and rerun every alignment search
```

Sections: `params overview sanity single pyramid failure own b1 b2 b3 b4 b5 b6 final`.

A run writes:

- `results/images/**` — every figure and result image, plus small copies in `results/images/thumbs/`;
- `results/data.json` — every number shown on the page;
- `results/cache/*.json` — offsets and transforms of the slow searches;
- the generated blocks of `index.html` (between `<!-- generated:... -->` markers): galleries, offset tables and
  the own-plate sections, so the page always matches the results.

With the cache a full run takes about 20 s. Without it (`--fresh`, or the first run) it takes a few minutes, mostly
the B6 rotation / scale search (about 3 s per full-size plate). The cache is only reused when it was made with the
same parameters, and only the plates that are new are computed.

### 3. Code

```
assignment1/src/
├── algorithm/            # reusable functions: arrays in, arrays / offsets out, no file writing
│   ├── config.py         # paths + Params, the one fixed parameter set used for every plate
│   ├── io.py             # load_plate (→ float32 in [0, 1]), build_catalog, save_jpg
│   ├── metrics.py        # L2, NCC, interior (border crop)
│   ├── pyramid.py        # own Gaussian downsampling and pyramid (reused in Assignment 2)
│   ├── align.py          # split_channels, shift (np.roll), single-scale and pyramid search, Transform, colorize
│   ├── bells_whistles.py # B1 edges, B2 crop, B3 contrast, B4 white balance, B5 color matrix, B6 rotation / scale, restore()
│   └── viz.py            # image grids for the notebook and the page figures
├── experiments.py        # one function per page section; computes numbers and images (only saves my low-res plate versions)
├── report.py             # writes results/images, results/data.json and the generated blocks of index.html
├── run.py                # command-line entry point
└── assignment1-2.ipynb   # exploration notebook (earlier version of the code; run.py is the source of truth)
```

### 4. Save the page as PDF

Open `assignment1/index.html` in Chrome → Print → **Save as PDF**, with Margins set to **Default**. The page has print
styles: collapsed sections open, before / after sliders become side-by-side pairs, and wide tables show every column.

## Assignment 2: running the code

### 1. Put the input images in place

```
assignment2/data/              # any format Pillow reads (jpg, jpeg, png, avif); found by file name without extension
├── selfie.*                   # 1.1
├── cameraman.*                # 1.2, 1.3, orientation
├── taj.*, cat_closeup.*       # 2.1 (the Taj Mahal from Learn, my own blurry image)
├── skyline.*                  # 2.1 evaluation (a sharp image to blur and re-sharpen)
├── cat_hybrid1.*, jaguar_hybrid1.*, mainecoon_hybrid2.*, owl_hybrid2.*     # 2.2, already aligned
├── apple.*, orange.*          # 2.3, 2.4
└── earth_blend1.*, mars_blend1.*, dr_strange_blend2.*, eye_aligned_blend2.*, eye_mask.*   # 2.4
assignment2/hybrid/            # the starter zip from Learn, unzipped: DerekPicture.jpg, nutmeg.jpg (git-ignored)
```

### 2. Run everything

From `assignment2/src`:

```
uv run python run.py                          # every section, about 30 s
uv run python run.py --only hybrid blend      # just these sections (data.json keeps the others)
uv run python pick_points.py                  # re-click the Derek / Nutmeg alignment points (opens a window)
```

Sections: `params conv finite_diff dog orientation sharpen hybrid hybrid_color stacks blend blend_color hero`.
A run writes `results/images/**` (plus `thumbs/`), `results/data.json` and the generated blocks of `index.html`.
Hybrid pairs, blend pairs and every parameter are in `src/algorithm/config.py`.

### 3. Code

```
assignment2/src/
├── algorithm/
│   ├── config.py         # paths, inputs, Params, the hybrid and blend pairs
│   ├── io.py             # load_image (→ float32 in [0, 1]), save_jpg
│   ├── convolution.py    # 1.1: zero padding, 4-loop and 2-loop convolution (numpy only)
│   ├── filters.py        # 1.2-1.3, orientation, 2.1: gradients, Gaussian / DoG, own arctan2, unsharp mask
│   ├── hybrid.py         # 2.2: port of align_images.m, low / high pass, cutoff frequency, spectra
│   ├── stacks.py         # 2.3-2.4: Gaussian / Laplacian stacks, multiresolution blending
│   └── viz.py            # image panels, spectra, filter-gain chart
├── experiments.py        # one function per page section; computes numbers and images
├── report.py             # writes results/images, results/data.json and the generated blocks of index.html
├── pick_points.py        # click the alignment points
└── run.py                # command-line entry point
```

## Assignment 3: running the code

### 1. Put the input images in place

```
assignment3/data/               # git-ignored; any format Pillow reads, including iPhone .HEIC (pillow-heif)
├── mosaic/
│   ├── set1/IMG_1553-1555.*    # Atrium (photos)
│   ├── set2/IMG_1766-1768.*    # Marina (photos)
│   ├── set4/IMG_0574_*s.jpg    # Toronto skyline: frames at 9, 11, 13 s of IMG_0574.MOV
│   └── set5/IMG_1051_*s.jpg    # Niagara Falls: frames at 3, 5, 7, 9 s of IMG_1051.MOV
└── rectify/
    └── IMG_3548.*, IMG_4344.*, IMG_6530.*    # Renoir, Almond Blossom, cocktail posters
```

The sets (left-to-right order, reference image) and the rectification targets are in `src/algorithm/config.py`.

### 2. Click the points (Part 1 only)

From `assignment3/src`:

```
uv run python pick_points.py                  # every neighbouring pair without saved points
uv run python pick_points.py set2 --edit      # refine the saved points of a set
uv run python pick_points.py set2 --redo      # click a set again from scratch
uv run python pick_points.py --rectify        # the 4 corners of each rectangle (any order)
```

Click a point in the left image, then the same point in the right image (at least 6 pairs). The selected point can be
moved with the arrow keys (1 px, Shift: 0.25 px) while a zoom panel shows it; `[` / `]` select earlier points, Backspace
deletes the last one, Enter saves. Points go to `results/points.json`. Part 2 needs no clicking.

### 3. Run everything

```
uv run python run.py                          # every section, about 35 s
uv run python run.py --only homography mosaic # just these sections (data.json keeps the others)
```

Sections: `params shoot homography_check homography rectify mosaic corners descriptors matching ransac hero`.
A run writes `results/images/**` (plus `thumbs/`), `results/data.json` and the generated blocks of `index.html`.
Sections whose points are not clicked yet are skipped with a message.

### 4. Code

```
assignment3/src/
├── algorithm/
│   ├── config.py         # paths, image sets, rectifications, Params
│   ├── io.py             # load_image (HEIC too), save_jpg, clicked-point storage
│   ├── homography.py     # A.2: computeH (least squares), apply_H
│   ├── warp.py           # A.3: inverse warping, warpImageNearestNeighbor, warpImageBilinear (numpy only)
│   ├── mosaic.py         # A.4: chaining to the reference, canvas, weights, overwrite / feathering / 2-band blending
│   ├── harris.py         # B.1: the provided Harris code (unchanged)
│   ├── features.py       # B.1-B.3: ANMS, 8 x 8 descriptors, ratio-test matching
│   ├── ransac.py         # B.4: 4-point RANSAC
│   └── viz.py            # image panels, correspondence / corner / match figures, ratio histogram
├── experiments.py        # one function per page section; computes numbers and images
├── report.py             # writes results/images, results/data.json and the generated blocks of index.html
├── pick_points.py        # click correspondences and rectangle corners
└── run.py                # command-line entry point
```

### 5. Save the page as PDF

As for Assignment 1, with **Background graphics** on. The equations are rendered with KaTeX from a CDN, so the page needs
an internet connection; wait until they appear before printing. Print styles keep every equation, system of equations
and numbered step on one page.

## Adding a new assignment

1. Copy `assignment3/` to `assignmentN/` and edit its `index.html`.
2. Add a link to it in the root `index.html`.

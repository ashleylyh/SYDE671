# SYDE 671 Assignments

Published with GitHub Pages at <https://ashleylyh.github.io/SYDE671/>.

| Assignment | Page |
|---|---|
| Assignment 1: Becoming friends with the camera & Colorizing the Prokudin-Gorskii collection | <https://ashleylyh.github.io/SYDE671/assignment1/> |
| Assignment 2 | <https://ashleylyh.github.io/SYDE671/assignment2/> |

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

## Adding a new assignment

1. Copy `assignment2/` to `assignmentN/` and edit its `index.html`.
2. Add a link to it in the root `index.html`.

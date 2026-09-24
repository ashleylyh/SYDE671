# SYDE 671 Assignments

Published with GitHub Pages at `https://<username>.github.io/SYDE671/`.

| Assignment | Page |
|---|---|
| Assignment 1 | `https://<username>.github.io/SYDE671/assignment1/` |
| Assignment 2 | `https://<username>.github.io/SYDE671/assignment2/` |

## Layout

```
SYDE671/
├── index.html          # landing page that links to each assignment
├── assets/style.css    # shared stylesheet
├── .nojekyll           # serve files as-is (no Jekyll processing)
└── assignmentN/
    ├── index.html      # the write-up page → /SYDE671/assignmentN/
    ├── code/           # source code
    ├── results/        # images shown on the page
    └── data/           # raw inputs (git-ignored)
```

PDFs, zips and `data/` folders are git-ignored, so course materials stay local.

## Adding a new assignment

1. Copy `assignment2/` to `assignmentN/` and edit its `index.html`.
2. Add a link to it in the root `index.html`.

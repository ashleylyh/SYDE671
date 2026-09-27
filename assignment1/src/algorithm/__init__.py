"""Colorizing the Prokudin-Gorskii collection (SYDE 671, Assignment 1, Part Two).

Pure algorithms: every function takes arrays and returns arrays / offsets.
Nothing here writes files; experiments.py runs these and report.py saves the outputs.

    config          paths + Params (the one fixed parameter set)
    io              load plates, catalog, save JPGs
    metrics         L2, NCC, interior crop
    pyramid         downsampling + Gaussian pyramid (reused in Assignment 2)
    align           split, shift, single-scale + pyramid search, Transform, colorize
    bells_whistles  B1-B6 + the final restore() pipeline
    viz             image grids for the notebook and the webpage
"""

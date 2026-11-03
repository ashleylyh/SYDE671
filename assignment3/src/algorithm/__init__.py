"""[Auto]Stitching Photo Mosaics (SYDE 671, Assignment 3).

Pure algorithms: every function takes arrays and returns arrays / numbers.
Nothing here writes files; experiments.py runs these and report.py saves the outputs.

    config          paths, the image sets + Params (the fixed parameters)
    io              load images (float32 in [0, 1], HEIC too), save JPGs, read/write clicked points
    homography      A.2: homography_system (Ah = b), computeH (least squares, h33 = 1), apply_H
    warp            A.3: inverse warping, nearest-neighbour and bilinear interpolation (from scratch)
    mosaic          A.4: bounding box, alpha masks, weighted-average / 2-level Laplacian blending
    harris          B.1: course sample code (get_harris_corners, dist2)
    features        B.1-B.3: ANMS, 8x8 descriptors from 40x40 windows, Lowe ratio matching
    ransac          B.4: 4-point RANSAC
    viz             correspondence / corner / match figures for the webpage
"""

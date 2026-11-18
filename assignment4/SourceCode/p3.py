import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # Use only the inhomogeneous (x, y) image coordinates.
    x1 = points_im1[:, :2]
    x2 = points_im2[:, :2]

    # 1. Center the points in each image on their centroid; this removes
    #    the translation component of the affine cameras.
    x1_centered = x1 - x1.mean(axis=0)
    x2_centered = x2 - x2.mean(axis=0)

    # 2. Build the 2m x n measurement matrix D (m = 2 cameras, n points):
    #    rows are x and y of image 1, then x and y of image 2.
    D = np.vstack((x1_centered.T, x2_centered.T))

    # 3. SVD of D. In the ideal (noise-free affine) case D has rank 3.
    U, S, Vt = np.linalg.svd(D)
    print("Singular values of D:", S)

    # 4. Keep the rank-3 part and split the singular values evenly:
    #    D ~= M S, with M = U3 sqrt(W3) (motion, 2m x 3) and
    #    S = sqrt(W3) V3^T (structure, 3 x n).
    U3 = U[:, :3]
    W3 = np.diag(S[:3])
    V3t = Vt[:3, :]
    motion = U3.dot(np.sqrt(W3))
    structure = np.sqrt(W3).dot(V3t)
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()

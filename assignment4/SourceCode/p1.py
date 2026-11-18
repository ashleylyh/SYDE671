import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # Each correspondence p' = (u', v', 1), p = (u, v, 1) gives one linear
    # equation p'^T F p = 0 in the 9 entries of F (row-major):
    #   [u'u, u'v, u', v'u, v'v, v', u, v, 1] . f = 0
    u1, v1 = points1[:, 0], points1[:, 1]
    u2, v2 = points2[:, 0], points2[:, 1]
    ones = np.ones(points1.shape[0])
    W = np.column_stack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, ones])

    # Least-squares solution of W f = 0 with ||f|| = 1: the right singular
    # vector associated with the smallest singular value.
    _, _, Vt = np.linalg.svd(W)
    F_hat = Vt[-1].reshape(3, 3)

    # Enforce the rank-two constraint (PDF 1(a)): zero out the smallest
    # singular value of F_hat.
    U, S, Vt = np.linalg.svd(F_hat)
    S[2] = 0
    F = U.dot(np.diag(S)).dot(Vt)
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # Normalizing transform: translate the centroid to the origin and scale
    # so that the mean squared distance from the origin is 2.
    def normalization_matrix(points):
        centroid = points[:, :2].mean(axis=0)
        centered = points[:, :2] - centroid
        scale = np.sqrt(2.0 / np.mean(np.sum(centered**2, axis=1)))
        return np.array([[scale, 0, -scale*centroid[0]],
                         [0, scale, -scale*centroid[1]],
                         [0, 0, 1]])

    T1 = normalization_matrix(points1)
    T2 = normalization_matrix(points2)
    q1 = T1.dot(points1.T).T
    q2 = T2.dot(points2.T).T

    # Run the eight point algorithm (including the rank-two constraint via
    # SVD) on the normalized points.
    Fq = lls_eight_point_alg(q1, q2)

    # De-normalize: q2^T Fq q1 = p2^T (T2^T Fq T1) p1
    F = T2.T.dot(Fq).dot(T1)
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # Epipolar lines in image 1 induced by points2: l = F^T p2 = (a, b, c)
    lines = F.T.dot(points2.T)          # 3 x N
    a, b = lines[0], lines[1]

    # Point-to-line distance |a x + b y + c| / sqrt(a^2 + b^2), with p1
    # in homogeneous coordinates (x, y, 1).
    numer = np.abs(np.sum(lines * points1.T, axis=0))
    distances = numer / np.sqrt(a**2 + b**2)
    average_distance = np.mean(distances)
    return average_distance

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()

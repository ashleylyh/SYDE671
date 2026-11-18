import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # PDF 2(a): the epipole satisfies F^T e = 0, i.e. e spans the null space
    # of F^T. Take the right singular vector of F^T for the smallest singular
    # value and scale it to homogeneous form [x y 1].
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    epipole = e / e[2]
    return epipole

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    # PDF 2(b) ("find_H"): H = T^-1 G R T
    height, width = im.shape[0], im.shape[1]

    # 1. Translate the image center to the origin.
    T = np.array([[1, 0, -width / 2.0],
                  [0, 1, -height / 2.0],
                  [0, 0, 1]])

    # 2. Rotate the translated epipole Te = (e1, e2, 1) onto the
    #    horizontal axis at (f, 0, 1).
    Te = T.dot(e)
    Te = Te / Te[2]
    e1, e2 = Te[0], Te[1]
    alpha = 1.0 if e1 >= 0 else -1.0
    norm = np.sqrt(e1**2 + e2**2)
    R = np.array([[ alpha * e1 / norm, alpha * e2 / norm, 0],
                  [-alpha * e2 / norm, alpha * e1 / norm, 0],
                  [0, 0, 1]])

    # 3. Send (f, 0, 1) to the point at infinity (f, 0, 0).
    f = R.dot(Te)[0]
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1.0 / f, 0, 1]])

    # 4. Undo the translation.
    H = np.linalg.inv(T).dot(G).dot(R).dot(T)
    return H

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # PDF 2(c): first H2 maps the epipole e2 to infinity.
    H2 = compute_H(e2, im2)

    # M = [e]_x F + e v^T, with v^T = [1 1 1]
    e_cross = np.array([[0, -e2[2], e2[1]],
                        [e2[2], 0, -e2[0]],
                        [-e2[1], e2[0], 0]])
    v = np.ones(3)
    M = e_cross.dot(F) + np.outer(e2, v)

    # Transformed points: p_hat = H2 M p (image 1), p_hat' = H2 p' (image 2)
    p_hat = H2.dot(M).dot(points1.T)
    p_hat = p_hat / p_hat[2]
    p_hat_prime = H2.dot(points2.T)
    p_hat_prime = p_hat_prime / p_hat_prime[2]

    # Solve W a = b in the least-squares sense, where each row of W is
    # [x_hat, y_hat, 1] and b holds the matching x_hat'.
    W = np.column_stack([p_hat[0], p_hat[1], np.ones(p_hat.shape[1])])
    b = p_hat_prime[0]
    a = np.linalg.lstsq(W, b, rcond=None)[0]

    H_A = np.array([[a[0], a[1], a[2]],
                    [0, 1, 0],
                    [0, 0, 1]])
    H1 = H_A.dot(H2).dot(M)
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()

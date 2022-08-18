# The Pilot Project

# from distutils.command.build import build
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

# # Print Versions:
# print("python version:", sys.version)   # python version: 3.10.6
# print("numpy version:", np.__version__) # numpy version: 1.23.0


# Define the image matrix (2D)
im_pix_sz = 1.0                     # physical size of a pixel. Arbitrary units
im_mat_sz = (10, 10)                # matrix size: N, M
N, M = im_mat_sz                    # for convenient use
im_mat = np.zeros(im_mat_sz)   # image matrix
im_mat[2, 3] = 1.0                  # single pixel image


# # quick image plot
# fig, ax = plt.subplots()

# ax.imshow(im_mat, 
#             cmap=cm.get_cmap("plasma"))   # cm.gray

# plt.show()


# Define the detector array (1D)
det_elm_sz = 1.0
det_sz = 10

# Pixel Coordinates
# 
# First, we need to get the (x,y) coordinates for each image pixel.
# for X, Y axes passing through the image center, the center locations (x, y) of
# a pixel in the (u, v) row/column is given by:

def pixel_coords(u, v, matrix_size, im_pix_size=1.0):
    # u, v are te row, column indices, respectively
    # N, M are the height and width of the image matrix (in pixels)
    # returns (x, y) coordinates of (u, v) in physical units
    N, M = matrix_size

    x = - (M - 1) / 2.0 + v
    y =   (N - 1) / 2.0 - u

    return (im_pix_size * x, im_pix_size * y)

# print("x, y physical coords for the (0,0) and (9,9) elements in a 10x10 matrix with unit pixel length:")
# print(pixel_coords(0, 0, (10,10), 1.0), pixel_coords(9, 9, (10,10), 1.0)) # (-4.5, 4.5) (4.5, -4.5)


# Rotation of the image matrix
# First, build separate x and y coordinate matrices
# 0 indicates before rotation applied

# # initializing of original (0) x and y coordinate matrices:
# mat_x_0 = np.zeros(im_mat_sz)
# mat_y_0 = np.zeros(im_mat_sz)

# N, M = im_mat_sz
# for u in range(N):
#     for v in range(M):
#         mat_x_0[u, v] = -(M - 1) / 2.0 + v
#         mat_y_0[u, v] = (N - 1) / 2.0 - u

# mat_x_0 *= im_pix_sz    # 1.0 here. for completeness
# mat_y_0 *= im_pix_sz

# Let's make a function to take care of generating mat_x and mat_y

def gen_mat_x_mat_y(im_mat_sz, im_pix_sz=1.0):
    mat_x = np.zeros(im_mat_sz)
    mat_y = np.zeros(im_mat_sz)

    N, M = im_mat_sz
    for u in range(N):
        for v in range(M):
            mat_x[u, v] = -(M - 1) / 2.0 + v
            mat_y[u, v] = (N - 1) / 2.0 - u
    # normalize physical coordinates
    mat_x *= im_pix_sz
    mat_y *= im_pix_sz

    return mat_x, mat_y

mat_x_0, mat_y_0 = gen_mat_x_mat_y(im_mat_sz, im_pix_sz)

# print("x, y physical coords for the (0,0) and (9,9) elements in a 10x10 matrix with unit pixel length:")
# print((mat_x_0[0, 0], mat_y_0[0, 0]), (mat_x_0[9, 9], mat_y_0[9, 9]))   # (-4.5, 4.5) (4.5, -4.5)


# Applying the rotation (the right hand rule):
# inputs are the coordinate matrices. 
# outputs are new x and y rotated matrices
# 
# Note: Rotating Detector by θ is equivalent to rotating the image matrix by -θ
# for the 2D case, a rotation by θ in the (x,y) plane is defined by:
# [ [ cos(θ) , -sin(θ)] , [sin(θ) , cos(θ)] ]
# using degrees for theta
def rotate_xy(mat_x, mat_y, theta):

    if mat_x.shape != mat_y.shape:
        raise Exception("Input x and y matrices should be the same size.") 
    
    N, M = mat_x.shape

    cos_th = np.cos(theta * np.pi / 180)
    sin_th = np.sin(theta * np.pi / 180)

    mat_x_th = mat_x * cos_th - mat_y * sin_th
    mat_y_th = mat_x * sin_th + mat_y * cos_th

    return mat_x_th, mat_y_th

mat_x_45, mat_y_45 = rotate_xy(mat_x_0, mat_y_0, 45)

# print("After the 45deg rotation: x, y coords for the (0,0) and (9,9) elements :")
# print((mat_x_45[0, 0], mat_y_45[0, 0]), (mat_x_45[9, 9], mat_y_45[9, 9]))   # (-6.3639610306789285, 0.0) (6.3639610306789285, 0.0)
# print( 4.5 * np.sqrt(2))    # 6.3639610306789285


# Project the matrix to the detector for each angle θ.
# 
# First we define a simple "nearest-element-center" projection rule:
# (1) Each pixel representative (center) is projected vertically on a detector plane below
# (2) The detector element assigned to the pixel is the one where the pixel center lands
# (3) Later, we can try other forms of dividing pixel information amonf nearby elements
# (4) Since we are projecting along the y-axis, we only need the x coords.

# First, a function to assign detector element index to each matrix element:
# General formula for assigning detector element index for an matrix elm with xcoord:

# k_x = floor(x/d) + det_sz/2                           : for det_sz even
# k_x = floor(x/d - 0.5) + (det_sz + 1)/2               : for det_sz odd

# inputs:
# mat_x: physical x coords of each mtrix elm,
# det_sz: number of detector elements
# det_elm_len: physical length of the detector element (arb units)
# crop_outside: indices outside the detector range, should we turn to None, or leave them as is
# output:
# mat_det_idx: matrix with the index of detector elm where each pixel is projected on
def mat_det_idx_y(mat_x, det_sz, det_elm_sz=1.0, crop_outside=True):
    mat_det_idx = np.zeros(mat_x.shape)
    
    if det_sz % 2 == 0:
        mat_det_idx = np.floor(mat_x / det_elm_sz) + det_sz / 2
    else:
        mat_det_idx = np.floor(mat_x / det_elm_sz - 0.5) + (det_sz + 1) / 2

    trans1 = lambda z: np.round(z).astype("int")
    if crop_outside:
        trans2 = lambda z: z if (0 <= z <= det_sz - 1) else None
    else:
        trans2 = lambda z: z   # do nothing

    return np.array([trans2(trans1((z))) for z in mat_det_idx.ravel()]).reshape(mat_x.shape)
    # return np.round(mat_det_idx).astype("int")

# project the original (and rotated matrices on a detector of size 10, elm length 1.0:
mat_det_idx_0 = mat_det_idx_y(mat_x_0, 10, 1.0)
mat_det_idx_45 = mat_det_idx_y(mat_x_45, 10, 1.0, crop_outside=True)


# print("Projection indices of the original and the 45deg rotated matrices (first and last rows)")
# print(mat_det_idx_0[0, :], mat_det_idx_0[-1, :])
# # [0 1 2 3 4 5 6 7 8 9] [0 1 2 3 4 5 6 7 8 9]
# print(mat_det_idx_45[0, :], mat_det_idx_45[-1, :])
# # [-2 -1  0  0  1  2  2  3  4  5] [ 5  5  6  7  7  8  9  9 10 11]   (crop_outside False)
# # [None None 0 0 1 2 2 3 4 5] [5 5 6 7 7 8 9 9 None None]           (with crop_outside True)

# Note: after the 45deg rotation, some of the pixels fall outside the array

# Next, after finding the detector element indices, we calculate the actual projection 
# values by summing values for detector indices in the index matrix
# For sparse matrices, it's much faster to sum over the nonzero elements.
def mat_det_proj(img_mat, mat_det_idx, det_sz, is_sparse=False, sparse_idx=[]):
    det_out = np.zeros(det_sz)
    N, M = img_mat.shape

    if not is_sparse:                   # for dense mats, sum over all elms
        for u in range(N):
            for v in range(M):
                d_idx = mat_det_idx[u, v]
                if (d_idx is not None) and (0 <= d_idx <= det_sz - 1):
                    det_out[d_idx] += img_mat[u, v]
    else:                               # for sparse mats, sum over nonzero indices:
        for s_idx in sparse_idx:
            d_idx = mat_det_idx[s_idx]  # detector elm index
            det_out[d_idx] += img_mat[s_idx]

    return det_out


# Projection for all angles
#
# Define the angles of projection
proj_angs = np.linspace(-90, +90, 10, endpoint=False)
proj_sz = len(proj_angs)
# print(proj_angs)    # [-90. -72. -54. -36. -18.   0.  18.  36.  54.  72.]
# print(proj_sz)      # 10

# Define the projection matrix (2D)
proj_mat = np.zeros((proj_sz, det_sz))

# run projection for all angles:
for t_idx, theta in enumerate(proj_angs):
    # -theta since matrix is rotated opposite w.r.t the detector
    mat_x_th, mat_y_th = rotate_xy(mat_x_0, mat_y_0, -1 * theta)
    mat_det_idx = mat_det_idx_y(mat_x_th, det_sz, det_elm_sz, crop_outside=True)
    proj_mat[t_idx] = mat_det_proj(im_mat, mat_det_idx, det_sz)


# print("projection matrix for (2, 3) one-hot matrix")
# print(proj_mat)
# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]

# # quick plot of projections
# fig, ax = plt.subplots()

# ax.imshow(proj_mat, 
#             cmap=cm.get_cmap("plasma"))   # cm.gray

# plt.show()


# Now, let's build the one-hot-pixel images
# Here we will be using just a dense representation. At some point, we will be 
# switching to sparse representations

# retuens a 4D matrix, where the i,j component is a matrix with index i,j eq to one, rest zero.
# A shortcut way of building it is to reshape the identity matrix of size NM by NM
def build_one_hot_mats(im_mat_sz):
    N, M = im_mat_sz
    # one_hot_out_mats = np.zeros((N, M, N, M))
    one_hot_2D = np.eye(N * M)
    one_hot_out_mats = one_hot_2D.reshape((N, M, N, M))

    return one_hot_out_mats


one_hot_mats = build_one_hot_mats(im_mat_sz)

# print("one-hot 10x10 matrix elemets for index (3, 8) ")
# print(one_hot_mats[3, 8])
# # [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]      <=== one hot at [3, 8]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

# fig, ax = plt.subplots()
# ax.imshow(one_hot_mats[3, 8], 
#             cmap=cm.get_cmap("plasma"))   # cm.gray
# plt.show()


# Now let's generate projections for all the one-hot images
# putting everything together here
# here we will use sparseness of the one-hot matrices to generate projections
def gen_one_hot_projs(one_hot_mats, proj_angs, det_sz, im_pix_sz=1.0, det_elm_sz=1.0):
    # validate and process inputs
    N, M, NN, MM = one_hot_mats.shape
    if (N != NN or M != MM):
        raise Exception(f"Input `one_hot_mats` should be of shape (A B A B). You gave me: {one_hot_mats.shape}")
    angles_sz = len(proj_angs)

    # initialize the 4D projection matrix
    proj_mat_one_hot = np.zeros((N, M, angles_sz, det_sz))

    # get physical coordinates mat_x_0 mat_y_0 before any rotation
    mat_x_0, mat_y_0 = gen_mat_x_mat_y((N, M), im_pix_sz=im_pix_sz)

    # for each angle theta, get the mat_x_th for -theta rotation
    # then calculate projections for all one hot matrices for that theta

    for (th_idx, theta) in enumerate(proj_angs):
        mat_x_th, _ = rotate_xy(mat_x_0, mat_y_0, -1 * theta)    # rotate by -theta
        mat_det_idx = mat_det_idx_y(mat_x_th, det_sz, det_elm_sz, crop_outside=True)    # generate index mat
        # do the projection for all one-hot matrices
        for i in range(N):
            for j in range(M):
                proj_mat_one_hot[i, j, th_idx] = mat_det_proj(one_hot_mats[i, j],
                                                                mat_det_idx, det_sz, 
                                                                is_sparse=True,         # <===
                                                                sparse_idx=[(i, j)])    # <===

    return proj_mat_one_hot


proj_mats_one_hot = gen_one_hot_projs(one_hot_mats,
                                        proj_angs,
                                        det_sz,
                                        im_pix_sz=im_pix_sz,
                                        det_elm_sz=det_elm_sz)

# print("projections of the (2, 3) one-hot image:")
# print(np.squeeze(proj_mats_one_hot[2, 3]))
# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]

# # quick plot of projections for the 2-3 one-hot image
# fig, ax = plt.subplots()
# ax.imshow(proj_mats_one_hot[2, 3], 
#             cmap=cm.get_cmap("plasma"))   # cm.gray
# plt.show()


# Linear Modeling Using Scikit-Learn
# for now, not using sparse erpresentations
sk_lr = LinearRegression(fit_intercept=False, n_jobs=-1)

# prepare X and y:
# basically reshaping the input one_hot projection matrix to (n_samples, n_features) = (N*M, NumAngs * DetLen)  <=== fixed
# and reshaping the target image matrix to (n_samples, n_targets) = (N*M, N*M)
def prepare_one_hot_Xy(proj_mats_one_hot, one_hot_mats, num_angs, det_sz):
    # validate
    N, M, NN, MM = one_hot_mats.shape
    if (N != NN or M != MM):
        raise Exception(f"Input `one_hot_mats` should be of shape (A B A B). You gave me: {one_hot_mats.shape}")
    
    proj_mat_size = proj_mats_one_hot.size
    exp_proj_size = N * M * num_angs * det_sz
    if (proj_mat_size != exp_proj_size):
        raise Exception(f"Input `proj_mats_one_hot` should have the length {exp_proj_size}. Not with {(N, M, num_angs, det_sz)} => {proj_mat_size}.")

    # transpose since we are switching the order to (n_samples, n_features)
    # X = proj_mats_one_hot.reshape(N * M, num_angs * det_sz).transpose()     # <===
    X = proj_mats_one_hot.reshape(N * M, num_angs * det_sz)
    y = one_hot_mats.reshape(N * M, N * M)
    return X, y


X, y = prepare_one_hot_Xy(proj_mats_one_hot, one_hot_mats, len(proj_angs), det_sz)

# print(f"X shape: {X.shape}, y shape: {y.shape}" )
# # X shape: (100, 100), y shape: (100, 100)

# Let's look at the training data rank:
X_rank = np.linalg.matrix_rank(X)
# print(f"Rank of Training Matrix X: ==> {X_rank} <==\n\
#     required number of rows for full recon: {N * M}\n\
#     for:\
#     image matrix size {im_mat_sz}\n\
#     projection angles: {proj_angs}\n\
#     total number of projs: {proj_angs.size}")
# # Rank of Training Matrix X: ==> 94 <==
# #     required number of rows for full recon: 100
# #     for:    image matrix size (10, 10)
# #     projection angles: [-90. -72. -54. -36. -18.   0.  18.  36.  54.  72.]
# #     total number of projs: 10


# Fit the model
sk_lr.fit(X, y)

# Test the model using original input:
y_pred = sk_lr.predict(X)

# max_err = np.amax(np.absolute(y - y_pred))
# print(f"maximum error between y and y_pred: {max_err}")
# # maximum error between y and y_pred: 0.14141845703125
# # so we have errors of up to ~ 14% in reconstruction

# # plot y, y_pred side by side
# fig, axs = plt.subplots(nrows=1, ncols=2)
# im0 = axs[0].imshow(y, cmap=cm.get_cmap("plasma"))
# axs[0].set_title("y")
# im1 = axs[1].imshow(y_pred, cmap=cm.get_cmap("plasma"))
# axs[1].set_title("y_pred")
# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im1, cax=cax)
# plt.show()

# # plot Error
# fig, ax = plt.subplots()
# im = ax.imshow(y_pred - y, cmap=cm.get_cmap("plasma"))
# ax.set_title("Error: y_pred - y")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()

# # plot sample reconstructed image i, j = (2, 3)
# i, j = (2, 3)
# fig, ax = plt.subplots()
# im_recon_ij = y_pred.reshape(N, M, len(proj_angs), det_sz)[i, j]
# im = ax.imshow(im_recon_ij, cmap=cm.get_cmap("plasma"))
# ax.set_title(f"Error: one hot {(i, j)} reconstruction")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()

# In this setting, with 6% loss in rank we have upto 14% deviations in pixel value deviations.

# Looking at the root mean square error and mean absolute error:
rmse_train = mean_squared_error(y, y_pred, squared=False)
mae_train = mean_absolute_error(y, y_pred)

# print(f"{proj_sz} angles Linear OLS fit\nRoot Mean Square Error: {rmse_train:.4f}\nMean Absolute Error: {mae_train:.4f}")
# # Root Mean Square Error: 0.0220
# # Mean Absolute Error: 0.0143

# Next, we will look into regularized methods for a better reconstruction performance

# # LASSO regression:
# lasso = Lasso(alpha=0.1)
# lasso.fit(X, y)
# y_lasso = lasso.predict(X)

# rmse_lasso = mean_squared_error(y, y_lasso, squared=False)
# mae_lasso = mean_absolute_error(y, y_lasso)
# print(f"LASSO fit:\nRoot Mean Square Error: {rmse_lasso:.4f}\nMean Absolute Error: {mae_lasso:.4f}")
# # LASSO fit:
# # Root Mean Square Error: 0.0995
# # Mean Absolute Error: 0.0198
# # much larger RMSE error (10% vs 2.2% from OLS)

# # Ridge regression:
# ridge = Ridge(alpha=1.0)
# ridge.fit(X, y)
# y_ridge = ridge.predict(X)

# rmse_ridge = mean_squared_error(y, y_ridge, squared=False)
# mae_ridge = mean_absolute_error(y, y_ridge)
# print(f"Ridge fit:\nRoot Mean Square Error: {rmse_ridge:.4f}\nMean Absolute Error: {mae_ridge:.4f}")
# # Ridge fit:
# # Root Mean Square Error: 0.0490
# # Mean Absolute Error: 0.0295
# # Larger rmse (5%) and mae (2.6%) compared to OLS 

# # Elastic Net regression:
# el_net = ElasticNet(alpha=0.1, l1_ratio=0.001)
# el_net.fit(X, y)
# y_el_net = el_net.predict(X)

# rmse_el_net = mean_squared_error(y, y_el_net, squared=False)
# mae_el_net = mean_absolute_error(y, y_el_net)
# print(f"Elastic Net fit:\nRoot Mean Square Error: {rmse_el_net:.4f}\nMean Absolute Error: {mae_el_net:.4f}")
# # Elastic Net fit:
# # Root Mean Square Error: 0.0696
# # Mean Absolute Error: 0.0257
# # Similar to LASSO, poor rmse performance even with small alpha values


# Now let's increase the number of projections, to see if we can get a full rank X matrix:

proj_angs_12 = np.linspace(-90, +90, 12, endpoint=False)
proj_sz_12 = len(proj_angs_12)
# print(proj_angs_12)     # [-90. -75. -60. -45. -30. -15.   0.  15.  30.  45.  60.  75.]
# print(proj_sz_12)          # 12

# one_hot matrices are the same, let's generate one_hot projections for 12 angles:
proj_mats_o_h_12 = gen_one_hot_projs(one_hot_mats,
                                        proj_angs_12,
                                        det_sz,
                                        im_pix_sz=im_pix_sz,
                                        det_elm_sz=det_elm_sz)

# print("projections of the (2, 3) one-hot image:")
# print(np.squeeze(proj_mats_o_h_12[2, 3]))
# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]

X, y = prepare_one_hot_Xy(proj_mats_o_h_12, one_hot_mats, proj_sz_12, det_sz)

# print(f"X shape: {X.shape}, y shape: {y.shape}" )
# # X shape: (100, 120), y shape: (100, 100)

# X_rank = np.linalg.matrix_rank(X)
# print(f"Rank of Training Matrix X: ==> {X_rank} <==\n\
#     required number of rows for full recon: {N * M}\n\
#     image matrix size {im_mat_sz}\n\
#     projection angles: {proj_angs_12}\n\
#     total number of projs: {proj_angs_12.size}")
# # Rank of Training Matrix X: ==> 100 <==
# #     required number of rows for full recon: 100
# #     image matrix size (10, 10)
# #     projection angles: [-90. -75. -60. -45. -30. -15.   0.  15.  30.  45.  60.  75.]
# #     total number of projs: 12

# now let's see if we can get a better reconstruction now that we have a full rank X
sk_lr_12 = LinearRegression(fit_intercept=False, n_jobs=-1)

sk_lr_12.fit(X, y)

y_pred_12 = sk_lr_12.predict(X)

# Looking at the root mean square error and mean absolute error:
rmse_train_12 = mean_squared_error(y, y_pred_12, squared=False)
mae_train_12 = mean_absolute_error(y, y_pred_12)

print(f"Linear OLS fit {proj_sz_12} angles\nRoot Mean Square Error: {rmse_train_12:.4f}\nMean Absolute Error: {mae_train_12:.4f}")
# Linear OLS fit Root Mean Square Error: 0.0000
# Mean Absolute Error: 0.0000
# Encouraging results

# # Now let's plot the results:
# # plot y, y_pred side by side
# fig, axs = plt.subplots(nrows=1, ncols=2)
# im0 = axs[0].imshow(y, cmap=cm.get_cmap("plasma"))
# axs[0].set_title("y")
# im1 = axs[1].imshow(y_pred_12, cmap=cm.get_cmap("plasma"))
# axs[1].set_title("y_pred_12")
# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im1, cax=cax)
# plt.show()

# # plot the error map
# fig, ax = plt.subplots()
# im = ax.imshow(y_pred_12 - y, cmap=cm.get_cmap("plasma"))
# ax.set_title("Error: y_pred_12 - y")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()

# # plot sample reconstructed image i, j = (2, 3)
# i, j = (2, 3)
# fig, ax = plt.subplots()
# im_recon_ij = y_pred_12.reshape(N, M, len(proj_angs), det_sz)[i, j]
# im = ax.imshow(im_recon_ij, cmap=cm.get_cmap("plasma"))
# ax.set_title(f"Error: one hot {(i, j)} reconstruction. {proj_sz_12} angles")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()


# Now let's move to a larger matrix. Let's see how the linear system performs for
# a 100x100 matrix. Takes too long. Projections for 50x50 takes a couple minutes. 100x100 would be hours
# The Pipeline:
# (1) Define the 100x100 image matrix
im_pix_sz = 1.0                     # physical size of a pixel. Arbitrary units
im_mat_sz = (100, 100)                # matrix size: N, M
N, M = im_mat_sz                    # for convenient use

# (2) Define the detector array (1D)
det_elm_sz = 1.0                    # physical size of the detector elm. AU
det_sz = 100

# (3) Define the angles of projection
proj_sz = 120
proj_angs = np.linspace(-90, +90, proj_sz, endpoint=False)
print("parset initialization done.")

# (4) Build coordinate matrices:
mat_x_0, mat_y_0 = gen_mat_x_mat_y(im_mat_sz, 
                                    im_pix_sz)
print("building coord matrices done.")

# (5) Build one-hot matrices (dense)
one_hot_mats = build_one_hot_mats(im_mat_sz)
print("building o_h matrices done.")

# (6) one-hot projection matrices
proj_mats_one_hot = gen_one_hot_projs(one_hot_mats,
                                        proj_angs,
                                        det_sz,
                                        im_pix_sz=im_pix_sz,
                                        det_elm_sz=det_elm_sz)
print("building o_h projections done.")

# (7) Prepare Features and Targets (Xy)
X, y = prepare_one_hot_Xy(proj_mats_one_hot, 
                            one_hot_mats, 
                            proj_sz, 
                            det_sz)
print("X-y preparation done.")

# (8) Calculate the rank of features:
X_rank = np.linalg.matrix_rank(X)
print(f"Rank of Training Matrix X: ==> {X_rank} <==\n\
    required number of rows for full recon: {N * M}")
# Rank of Training Matrix X: ==> 10000 <==
#     required number of rows for full recon: 10000
# Note: Rank calculation took 3 minutes for the 100x100 matrix

# (9) Instantiate Linear Regressor
sk_lr = LinearRegression(fit_intercept=False, n_jobs=-1)

# (10) Fit the regressor:
sk_lr.fit(X, y)

# Evaluation Phase
# (11) Predict from one-hot projections:
y_pred = sk_lr.predict(X)

# (12) Calculate errors, RMS and MAE:
rmse_train = mean_squared_error(y, y_pred, squared=False)
mae_train = mean_absolute_error(y, y_pred)

print(f"Linear OLS Fit:\n{proj_sz} angles, {im_mat_sz} matrix, {det_sz} detector size\nRoot Mean Square Error: {rmse_train:.4f}\nMean Absolute Error: {mae_train:.4f}")
# Linear OLS Fit:
# 120 angles, (100, 100) matrix, 100 detector size
# Root Mean Square Error: 0.0000
# Mean Absolute Error: 0.0000
# Note: OLS fit for 100x100 reconstruction too 11 minutes
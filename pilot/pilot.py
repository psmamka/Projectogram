# The Pilot Project

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# # Print Versions:
# print("python version:", sys.version)   # python version: 3.10.6
# print("numpy version:", np.__version__) # numpy version: 1.23.0


# Define the image matrix (2D)
im_pix_sz = 1.0                     # physical size of a pixel. Arbitrary units
im_mat_sz = (10, 10)                # matrix size: N, M
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

# Define the angles of projection
proj_angs = np.linspace(-90, +90, 10, endpoint=False)
proj_sz = len(proj_angs)
# print(proj_angs)    # [-90. -72. -54. -36. -18.   0.  18.  36.  54.  72.]
# print(proj_sz)      # 10

# Define the projection matrix (2D)
proj_mat = np.zeros((proj_sz, det_sz))


# Project the matrix to the detector for each angle θ.
# 
# First, we need to get the (x,y) coordinates for each image pixel.
# for X, Y axes passing through the image center, the center locations (x, y) of
# a pixel in the (u, v) row/column is given by:

def pixel_coords(u, v, matrix_size=(10, 10), im_pix_size=1.0):
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

# initializing of original (0) x and y coordinate matrices:
mat_x_0 = np.zeros(im_mat_sz)
mat_y_0 = np.zeros(im_mat_sz)

N, M = im_mat_sz
for u in range(N):
    for v in range(M):
        mat_x_0[u, v] = -(M - 1) / 2.0 + v
        mat_y_0[u, v] = (N - 1) / 2.0 - u

mat_x_0 *= im_pix_sz    # 1.0 here. for completeness
mat_y_0 *= im_pix_sz

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
    mat_x_th = np.zeros((N, M))
    mat_y_th = np.zeros((N, M))

    cos_th = np.cos(theta * np.pi / 180)
    sin_th = np.sin(theta * np.pi / 180)

    for u in range(N):
        for v in range(M):
            mat_x_th[u, v] = cos_th * mat_x[u, v] - sin_th * mat_y[u, v]
            mat_y_th[u, v] = sin_th * mat_x[u, v] + cos_th * mat_y[u, v]

    return mat_x_th, mat_y_th

mat_x_45, mat_y_45 = rotate_xy(mat_x_0, mat_y_0, 45)

# print("After the 45deg rotation: x, y coords for the (0,0) and (9,9) elements :")
# print((mat_x_45[0, 0], mat_y_45[0, 0]), (mat_x_45[9, 9], mat_y_45[9, 9]))   # (-6.3639610306789285, 0.0) (6.3639610306789285, 0.0)
# print( 4.5 * np.sqrt(2))    # 6.3639610306789285


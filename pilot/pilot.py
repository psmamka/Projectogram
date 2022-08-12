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
im_mat_sz = (10, 10)                # matrix size
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
# Note: Rotating Detector by θ is equivalent to rotating the image matrix by -θ
# for the 2D case, a rotation by θ in the (x,y) plane is defined by:
# [ [ cos(θ) , -sin(θ)] , [sin(θ) , cos(θ)] ]
# 
# First, we need to get the (x,y) coordinates for each image pixel.
# for X, Y axes passing through the image center, the center locations (x, y) of
# a pixel in the (u, v) row/column is given by:

def pixel_coords(u, v, matrix_size=(10, 10), im_pix_size=1.0):
    # u, v are te row, column indices, respectively
    # N, M are the height and width of the image matrix (in pixels)
    # returns (x, y) coordinates of (u, v) in physical units
    N, M = matrix_size

    x = - (M - 1) / 2 + v
    y =   (N - 1) / 2 - u

    return (im_pix_size * x, im_pix_size * y)

# print("x, y physical coords for the (0,0) and (9,9) elements in a 10x10 matrix with unit pixel length:")
# print(pixel_coords(0, 0, (10,10), 1.0), pixel_coords(9, 9, (10,10), 1.0)) # (-4.5, 4.5) (4.5, -4.5)


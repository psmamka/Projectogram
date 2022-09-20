import sys
sys.path.append('../lib')

import numpy as np

try:
    from ..lib.projection import Projection2D
    from ..lib.inversion import Inversion2D
    from ..lib.phantom import pacman_mask, rectangle_mask
except:
    from projection import Projection2D
    from inversion import Inversion2D
    from phantom import pacman_mask, rectangle_mask

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# This is a pilot project to test for the idea of using projectogram as a set of classifiers, 
# with each row funcitoning as a decision tree, deterining the value of the pixel, proceeding
# iteratively until arriving at the final decision, i.e. the reconstructed image.
# The "Decision rate" or the "reconstruction rate" is the amount of adjustment we apply to the pixel 
# value after each decision

# Steps:
# () generate phantom projections
# () select rows that have non-zero min-product with the phantom projection
    # - min-product is defined (for now) as follows:
    # - select non-zeros elements of the p-gram row
    # - pointwise multiply by the same elements from the phantom projection (sinogram) to form a pixel-array
    # - select the minimum of the arrow
    # - if the minimum is non-zero (positive) the row in included
    # - These are the candidate, or contributing pixels. All else are zero
# () select a random permutation of the candidate pixels, and loop through it:
#   - calculate the min-product of the pixel with the (updated) phantom sinogram
#   - increase the pixel value by the recon rate times the min-product
#   - subtract the p-gram row normalized by the recon-rate from the phantom sinogram
#   - if this results in a negative value anywhere, go back and reduce the recon rate
# () Contiue the loop until one or more of the rows in the sinogram become all zero (or small)
# () For properly normalized projections, we should be able to find solutions satisfying all constraints (projections)
# 
# if we are interested in the degree of variability in the decision (image) result, we reconstruct the image with 
# different permutations of the candidate pixels array, then look at the stan.dev of pixel values across images


# par-set:
im_mat_sh = (20, 20)
det_len = im_mat_sh[0]
num_angs = 5
proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
pix_ph_sz = 1.0
det_elm_ph_sz = 1.0
det_ph_offs = 0.0

rect_ph = rectangle_mask(mat_sz=im_mat_sh, rect_sz=1)

proj_20 = Projection2D(im_mat_sh=im_mat_sh,
                        det_len=det_len, 
                        pix_ph_sz=pix_ph_sz, 
                        det_elm_ph_sz=det_elm_ph_sz, 
                        det_ph_offs=det_ph_offs, 
                        proj_angs=proj_angs)
    
single_pix_mats = proj_20.build_single_pixel_mats_dense()
proj_single_pix = proj_20.gen_single_pixel_projs(single_pixel_mats=single_pix_mats)
proj_mat = proj_20.single_mat_all_ang_proj(rect_ph, is_sparse=False)

# # plotting single-pixel
# # first pix, its projections
# fig, axs = plt.subplots(nrows=1, ncols=2)
# fig.set_size_inches(12, 6)
# im0 = axs[0].imshow(single_pix_mats[0, 0, :, :], cmap=cm.get_cmap("plasma"))
# im1 = axs[1].imshow(proj_single_pix[0, 0, :, :], cmap=cm.get_cmap("plasma"))
# print(proj_single_pix[0, 0, :, :])  # problem with cropping
# plt.show()

# plotting phantom projections
fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 6)
im0 = axs[0].imshow(rect_ph, cmap=cm.get_cmap("plasma"))
im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
plt.show()


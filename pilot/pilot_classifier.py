import sys
sys.path.append('../lib')

import numpy as np
from scipy import linalg

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
# () select rows that have non-zero min-product/masked-min-product with the phantom projection
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

# phantom = rectangle_mask(mat_sz=im_mat_sh, rect_sz=4, ul_corner=(4, 4))
# phantom = rectangle_mask(mat_sz=im_mat_sh, rect_sz=2, ul_corner=(4, 4)) \
#         + rectangle_mask(mat_sz=im_mat_sh, rect_sz=2, ul_corner=(4, 13)) \
#         + rectangle_mask(mat_sz=im_mat_sh, rect_sz=2, ul_corner=(13, 13))

phantom = pacman_mask(mat_sz=im_mat_sh, cent=(9.5,9.5), rad=7, direc=0, ang=60)

proj_20 = Projection2D(im_mat_sh=im_mat_sh,
                        det_len=det_len, 
                        pix_ph_sz=pix_ph_sz, 
                        det_elm_ph_sz=det_elm_ph_sz, 
                        det_ph_offs=det_ph_offs, 
                        proj_angs=proj_angs)
    
single_pix_mats = proj_20.build_single_pixel_mats_dense()
proj_single_pix = proj_20.gen_single_pixel_projs(single_pixel_mats=single_pix_mats)
proj_mat = proj_20.single_mat_all_ang_proj(phantom, is_sparse=False)


def plot_multiple_images(im_arr, nrows=1, ncols=2, fig_size=(12, 6), col_map="plasma", show_cbar=True):
    sz = len(im_arr)
    if (nrows * ncols != sz): nrows, ncols = (1, sz) 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(fig_size)
    plots, ims, cax = (im_arr, [None] * sz, [None] * sz)
    for ind in range(sz):
        ims[ind] = axs[ind].imshow(plots[ind], cmap=cm.get_cmap(col_map))
        divider = make_axes_locatable(axs[ind])
        if show_cbar:
            cax[ind] = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(ims[ind], cax=cax[ind])
    plt.show()

num_pix = im_mat_sh[0] * im_mat_sh[1]
num_elms = num_angs * det_len
sinpix_mats_rs = single_pix_mats.reshape(num_pix, num_pix)
proj_sinpix_rs = proj_single_pix.reshape(num_pix, num_elms)

# # plotting single-pixels + projectogram
# plot_multiple_images([sinpix_mats_rs, proj_sinpix_rs])
# # plotting phantom projections
# plot_multiple_images([phantom, proj_mat], col_map="ocean")



# now we go over pixels looking at masked-min-product with the phantom projection
# select the ones with non-zero masked min prod
pix_prod_list = []

# simple looping for now. refactor later
for i in range(im_mat_sh[0]):
    for j in range(im_mat_sh[1]):
        masked_idx = proj_single_pix[i, j] > 0
        prod_arr = proj_single_pix[i, j, masked_idx] * proj_mat[masked_idx]
        # if min(prod_arr) > 0: print(i, j, min(prod_arr))
        if min(prod_arr) > 0: pix_prod_list.append((i, j, min(prod_arr)))

print(pix_prod_list)
# [(4, 4, 2.0), (4, 5, 2.0), (4, 13, 1.0), (4, 14, 1.0), (5, 4, 2.0), (5, 5, 2.0), (5, 13, 1.0), 
# (5, 14, 1.0), (13, 13, 1.0), (13, 14, 1.0), (14, 13, 1.0), (14, 14, 1.0)]

# Most unimaginative approach: perform inversion on the masked pixels:
proj_sinpix_masked = np.zeros((len(pix_prod_list), num_elms))

# select the corresponding pixels:
for idx, (i, j, prd) in enumerate(pix_prod_list):
    proj_sinpix_masked[idx, :] = proj_single_pix[i, j, :, :].ravel()

# # plot phantom, projections, and the masked sinpix projections
# plot_multiple_images([phantom, proj_mat, proj_sinpix_masked])

# Now get the pseudo-inverse of the masked projectogram using scipy linalg:
def recon_with_masked_projectogram_psinv(proj_sinpix_masked, proj_mat, pix_prod_list):
    ps_inv, ps_rank = linalg.pinv(proj_sinpix_masked.T, atol=None, rtol=None, return_rank=True, check_finite=True)
    print(f"Pseudo-inverse shape: {ps_inv.shape}\nproj_mat shape: {proj_mat.shape}\nPseudo-inverse rank: {ps_rank}")

    recon_data_psinv = ps_inv.dot(proj_mat.ravel())

    im_recon_ps_inv = np.zeros(im_mat_sh)
    for idx, (i, j, _) in enumerate(pix_prod_list):
        im_recon_ps_inv[i, j] = recon_data_psinv[idx]
    
    return im_recon_ps_inv, ps_inv, ps_rank

# im_recon_ps_inv, ps_inv, ps_rank = recon_with_masked_projectogram_psinv(proj_sinpix_masked, proj_mat, pix_prod_list)
# plot_multiple_images([phantom, ps_inv.T, im_recon_ps_inv])


# steps for iterative reconstruction based on masked pizels and projectogram
# First: sort pix_prod_list based on min-prod
# Second: starting from k pixels with highest min-prod, update the pixel value by alpha (parameters of the recon)
# Third: update the proj_mat and the min-prod values
# Fourth: continue updating the value of the new highest min-prod pixels
# Fifth: Stop when the proj_mat equal to zero or below certain energy, or certain number of steps

im_recon = np.zeros(im_mat_sh)

k_par = 5           # recon parameter: number of pixels we update in each pass before re-sorting
k_par = min(k_par, len(pix_prod_list))
trm_eps = 0.01  # recon parameter: how close do we get to solution in proj_mat (epsilon) before terminating
upd_rt = 0.1    # update rate

cur_proj_mat = proj_mat.copy()

masked_proj = proj_single_pix > 0   # for fast calculation of min-prod

# print( max(pix_prod_list, key=lambda lst: lst[2])[2] )  

# # prep numpy arr:
# pix_len = len(pix_prod_list)
# pix_prod_arr = np.zeros((pix_len, 3))
# for idx in range(pix_len):
#     pix_prod_arr[idx] = [pix_prod_list[idx][0], pix_prod_list[idx][1], pix_prod_list[idx][2]]

# better way: structures in the numpy array:
dtype = [('i', int), ('j', int), ('minprod', float)]
pix_prod_arr = np.array(pix_prod_list, dtype=dtype)     # structured array
pix_prod_arr = np.sort(pix_prod_arr, order='minprod')[::-1] # reverse sort

while pix_prod_arr[0]['minprod'] > trm_eps:    # termination cond. first element is the largest min-prod after sort 
    # iteration logic:

    # update im_recon based on min_prod values
    for idx in range(k_par):
        pix = (pix_prod_arr[idx]['i'], pix_prod_arr[idx]['j'])
        # print(pix)
        pix_update = upd_rt * pix_prod_arr[idx][2]
        im_recon[pix] += upd_rt * pix_prod_arr[idx][2]
    
        # update the current proj_mat, by subtracting projections
        # Better way would be to add sin-pix matrices
        # cur_proj_mat = proj_20.single_mat_all_ang_proj(im_recon)
        cur_proj_mat -= proj_single_pix[pix] * pix_update
        # print(cur_proj_mat)

    # Re-calculate min-prod for the masked pixels (after the update loop)
    for idx in range(k_par):
        pix = (pix_prod_arr[idx]['i'], pix_prod_arr[idx]['j'])
        # print(pix)
        prod_arr = proj_single_pix[pix][masked_proj[pix]] * cur_proj_mat[masked_proj[pix]]
        pix_prod_arr[idx][2] = min(prod_arr)

    # re-sort the pix_prod_array (descending)
    pix_prod_arr = np.sort(pix_prod_arr, order='minprod')[::-1]

    print(pix_prod_arr)

plot_multiple_images([phantom, im_recon])
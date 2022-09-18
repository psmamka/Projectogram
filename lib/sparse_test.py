import numpy as np

from projection import Projection2D
from inversion import Inversion2D
from phantom import pacman_mask, rectangle_mask

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sparse_50(mat_sz=(50, 50), num_angs=5, inv_tech="ps-inv", plot_proj=True, plot_recon=True):
    # phantom
    rect_ph = rectangle_mask(mat_sz=mat_sz, rect_sz=4)
    
    # par-set:
    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
    im_mat_sh = mat_sz
    det_len = mat_sz[0]
    pix_ph_sz = 1.0
    det_elm_ph_sz = 1.0
    det_ph_offs = 0.0

    # forward:
    proj_50 = Projection2D(im_mat_sh=im_mat_sh, 
                            det_len=det_len, 
                            pix_ph_sz=pix_ph_sz, 
                            det_elm_ph_sz=det_elm_ph_sz, 
                            det_ph_offs=det_ph_offs, 
                            proj_angs=proj_angs)

    sinpix_50 = proj_50.build_single_pixel_mats_dense()
    projectogram_50 = proj_50.gen_single_pixel_projs(sinpix_50)
    proj_mat = proj_50.single_mat_all_ang_proj(rect_ph, is_sparse=False)

    if plot_proj:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)
        im0 = axs[0].imshow(rect_ph, cmap=cm.get_cmap("plasma"))
        im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
        plt.show()
    
    inverse_50 = Inversion2D(projectogram = projectogram_50,
                                single_pixel_mats = sinpix_50,
                                im_mat_sh=im_mat_sh,
                                det_len=det_len,
                                proj_angs=proj_angs,
                                pix_ph_sz=pix_ph_sz,
                                det_elm_ph_sz=det_elm_ph_sz,
                                det_ph_offs=det_ph_offs)

    if inv_tech == "ps-inv":
        ps_inv_50, rank_50 = inverse_50.get_projectrogram_pseudoinv(verbose=True)
        # Pseudo-inverse rank: 250, image pixels: 2500
        recon_50 = inverse_50.general_projection_recon_pseudoinv(proj_mat)
    elif inv_tech == "lin-reg":
        rank_50 = inverse_50.get_projectogram_rank(verbose=True)
        # Image Shape: (50, 50), Image Pixels: 2500, Features Rank: 250, Null Space: 2250
        inverse_50.train_lin_reg_model()
        recon_50 = inverse_50.general_projection_reconstruction(proj_mat)
    else:
        raise Exception(f"Set inverstion technique (inv_tech) to `ps-inv` or `lin-reg`. We got `{inv_tech}`.")

    if plot_recon:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(rect_ph, cmap=cm.get_cmap("plasma"))
        axs[0].set_title("pacman_20: Original")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(recon_50, cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"pacman_20: Reconstructed with rank {rank_50}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    return


sparse_50(mat_sz=(50, 50), num_angs=5, inv_tech="lin-reg", plot_proj=False, plot_recon=True) # inv_tech: "ps-inv" | "lin-reg"
# Pseudo-inverse rank: 250, image pixels: 2500
# Extreme streaking effects, for both pseudo-inv and lin-reg network
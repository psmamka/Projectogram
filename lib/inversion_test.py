import numpy as np

from projection import Projection2D
from inversion import Inversion2D
from phantom import pacman_mask

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# test instationaion of an inverse2D object using a 20x20x20x20 projectogram: 
# rank, training, errors
def instance_20(plot_results=True):

    proj_angs = np.linspace(start=-90, stop=90, num=20, endpoint=False)
    im_mat_sh = (20, 20)
    det_len = 20
    pix_ph_sz = 1.0
    det_elm_ph_sz = 1.0
    det_ph_offs = 0.0

    proj_20 = Projection2D(im_mat_sh=im_mat_sh, 
                            det_len=det_len, 
                            pix_ph_sz=pix_ph_sz, 
                            det_elm_ph_sz=det_elm_ph_sz, 
                            det_ph_offs=det_ph_offs, 
                            proj_angs=proj_angs)
    
    sinpix_20 = proj_20.build_single_pixel_mats_dense()
    projectogram_20 = proj_20.gen_single_pixel_projs(sinpix_20)

    inverse_20 = Inversion2D(projectogram = projectogram_20,
                                single_pixel_mats = sinpix_20,
                                im_mat_sh=im_mat_sh, 
                                det_len=det_len,
                                proj_angs=proj_angs, 
                                pix_ph_sz=pix_ph_sz, 
                                det_elm_ph_sz=det_elm_ph_sz, 
                                det_ph_offs=det_ph_offs)

    rank_20 = inverse_20.get_projectogram_rank(verbose=True)
    # Image Shape: (20, 20), Image Pixels: 400, Features Rank: 390, Null Space: 10

    lr_model_20 = inverse_20.train_lin_reg_model()

    sinpix_20_recon = inverse_20.single_pixel_lin_reg_reconstruct()

    # plot sin-pix original vs reconstruction
    if plot_results:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(sinpix_20.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[0].set_title("Single Pixels: Originals")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(sinpix_20_recon.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"Single Pixels: Reconstructed with rank {rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    rmse, mae = inverse_20.single_pixel_lin_reg_errors(verbose=True)
    # Root Mean Squared Error: 0.0074
    # Mean Absolute Error: 0.0056


# # simple projection and reconstruction of a 20x20 pacman
# def simple_20():
#     pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)

#     proj_20 = Projection2D(im_mat_sh=pacman_20.shape, 
#                             det_len=20, 
#                             pix_ph_sz=1.0, 
#                             det_elm_ph_sz=1.0, 
#                             det_ph_offs=0.0, 
#                             proj_angs=np.linspace(start=-90, stop=90, num=20, endpoint=False))


#     proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

#     fig, axs = plt.subplots(nrows=1, ncols=2)
#     fig.set_size_inches(12, 6)
#     im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
#     im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
#     plt.show()



instance_20(plot_results=True)
# simple_20()

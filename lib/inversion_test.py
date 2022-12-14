# Tests for the Inversion2D class

# Copyright (C) 2022  P. S. Mamkani

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from projection import Projection2D
from inversion import Inversion2D
from phantom import pacman_mask

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# utility plottig
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


# test instationaion of an inverse2D object using a 20x20x20x20 projectogram: 
# rank, training, errors
def train_instance_20(plot_sinpix=False, plot_projgram=True):

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
    if plot_sinpix:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(sinpix_20.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[0].set_title("Single Pixels: Originals")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(sinpix_20_recon.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"Reconstructogram: Reconstructed with rank {rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    if plot_projgram:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(projectogram_20.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[0].set_title("Projectogram: Originals")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        # project the reconstructed single-pixels (non-sparse projection)
        projectogram_proj = proj_20.gen_single_pixel_projs(sinpix_20_recon.reshape(20, 20, 20, 20), 
                                                            is_sparse=False)

        im1 = axs[1].imshow(projectogram_proj.reshape(400, 400), cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"Projectogram: After a proj-recon cycle of rank {rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    rmse, mae = inverse_20.single_pixel_lin_reg_errors(verbose=True)
    # Root Mean Squared Error: 0.0074
    # Mean Absolute Error: 0.0056


# simple projection and reconstruction of a 20x20 pacman
def train_pacman_20(num_angs=20, plot_proj=True, plot_recon=True):
    pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)

    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
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
    proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

    if plot_proj:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)
        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
        plt.show()

    inverse_20 = Inversion2D(projectogram = projectogram_20,
                                single_pixel_mats = sinpix_20,
                                im_mat_sh=im_mat_sh, 
                                det_len=det_len,
                                proj_angs=proj_angs, 
                                pix_ph_sz=pix_ph_sz, 
                                det_elm_ph_sz=det_elm_ph_sz, 
                                det_ph_offs=det_ph_offs)

    rank_20 = inverse_20.get_projectogram_rank(verbose=True)
    # num_angs=21: Image Shape: (20, 20), Image Pixels: 400, Features Rank: 399, Null Space: 1
    # num_angs=15: Image Shape: (20, 20), Image Pixels: 400, Features Rank: 300, Null Space: 100
    inverse_20.train_lin_reg_model()

    recon_20 = inverse_20.general_projection_reconstruction(proj_mat)

    if plot_recon:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        axs[0].set_title("pacman_20: Original")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(recon_20, cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"pacman_20: Reconstructed with rank {rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    # calculate phantom-vs-reconstruction errors:
    rmse = mean_squared_error(pacman_20, recon_20, squared=False)
    mae = mean_absolute_error(pacman_20, recon_20)
    print(f"pacman_20 reconstruction errors:\n RMSE: {rmse:.4f} MAE: {mae:.4f}")
    # RMSE: 0.0307 MAE: 0.0202
    # 15 angles: RMSE: 0.0722 MAE: 0.0563


# pseudoinverse sinpix and projectogram test performance
def pseudoinv_instance_20(num_angs=20, plot_sinpix=False, plot_projgram=True):
    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
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
    
    ps_inv_20, ps_rank_20 = inverse_20.get_projectrogram_pseudoinv(verbose=True)

    sinpix_20_recon = inverse_20.single_pixel_recon_pseudoinv()

    # plot sin-pix original vs reconstruction
    sinpix_sh = (im_mat_sh[0] * im_mat_sh[1] , im_mat_sh[0] * im_mat_sh[1])  # shape of the sinpix plots
    if plot_sinpix:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(sinpix_20.reshape(sinpix_sh), cmap=cm.get_cmap("plasma"))
        axs[0].set_title("Single Pixels: Originals")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(sinpix_20_recon.reshape(sinpix_sh), cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"Reconstructogram: PsInv Reconstruction Rank {ps_rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()
    
    # plot projectogram, orignal vas reconstruction
    projgram_sh = (im_mat_sh[0] * im_mat_sh[1], num_angs * det_len)
    if plot_projgram:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(projectogram_20.reshape(projgram_sh), cmap=cm.get_cmap("plasma"))
        axs[0].set_title("Projectogram: Originals")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        # project the reconstructed single-pixels (non-sparse projection)
        projectogram_proj = proj_20.gen_single_pixel_projs(sinpix_20_recon.reshape(20, 20, 20, 20), 
                                                            is_sparse=False)

        im1 = axs[1].imshow(projectogram_proj.reshape(projgram_sh), cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"Projectogram: PsInv proj-recon cycle of rank {ps_rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()


# projection and inversion of a 20x20 pacman using pseudo-inverse
def pseudoinv_pacman_20(num_angs=20, plot_proj=True, plot_recon=True):
    pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)

    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
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
    proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

    if plot_proj:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)
        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
        plt.show()

    inverse_20 = Inversion2D(projectogram = projectogram_20,
                                single_pixel_mats = sinpix_20,
                                im_mat_sh=im_mat_sh, 
                                det_len=det_len,
                                proj_angs=proj_angs, 
                                pix_ph_sz=pix_ph_sz, 
                                det_elm_ph_sz=det_elm_ph_sz, 
                                det_ph_offs=det_ph_offs)

    # rank_20 = inverse_20.get_projectogram_rank(verbose=True)

    ps_inv_20, ps_rank_20 = inverse_20.get_projectrogram_pseudoinv(verbose=True)
    # 20 angles: Pseudo-inverse rank: 390, image pixels: 400
    # 21 angles: 

    recon_20 = inverse_20.general_projection_recon_pseudoinv(proj_mat)

    if plot_recon:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        axs[0].set_title("pacman_20: Original")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(recon_20, cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"pacman_20: Reconstructed with rank {ps_rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    # calculate phantom-vs-reconstruction errors:
    rmse = mean_squared_error(pacman_20, recon_20, squared=False)
    mae = mean_absolute_error(pacman_20, recon_20)
    print(f"pacman_20 reconstruction errors:\n RMSE: {rmse:.4f} MAE: {mae:.4f}")
    # 20 angles: RMSE: 0.0000 MAE: 0.0000
    # 15 angles: RMSE: 0.0722 MAE: 0.0563
    return


# projection and inversion of a 20x20 pacman using masked pseudo-inverse (mask applied to recon image)
def pseudoinv_pacman_masked_20(num_angs=5, plot_proj=True, plot_recon=True):
    pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)

    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
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
    proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

    if plot_proj:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)
        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
        plt.show()

    inverse_20 = Inversion2D(projectogram = projectogram_20,
                                single_pixel_mats = sinpix_20,
                                im_mat_sh=im_mat_sh, 
                                det_len=det_len,
                                proj_angs=proj_angs, 
                                pix_ph_sz=pix_ph_sz, 
                                det_elm_ph_sz=det_elm_ph_sz, 
                                det_ph_offs=det_ph_offs)

    recon_20, ps_inv, ps_rank_20, _ = inverse_20.general_projection_recon_pseudoinv_masked(proj_mat, verbose=False)

    if plot_recon:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        axs[0].set_title("pacman_20: Original")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(recon_20, cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"pacman_20: Reconstructed with rank {ps_rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    # calculate phantom-vs-reconstruction errors:
    rmse = mean_squared_error(pacman_20, recon_20, squared=False)
    mae = mean_absolute_error(pacman_20, recon_20)
    print(f"pacman_20 masked reconstruction errors:\n RMSE: {rmse:.4f} MAE: {mae:.4f}")
    # num angs 4: RMSE: 0.0982 MAE: 0.0642

    return



# projection and inversion of a 20x20 pacman using masked pseudo-inverse (applied to projectogram)
def pseudoinv_pacman_masked_20_2(num_angs=5, plot_proj=True, plot_recon=True):
    pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)

    proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
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
    proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

    if plot_proj:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)
        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
        plt.show()

    inverse_20 = Inversion2D(projectogram = projectogram_20,
                                single_pixel_mats = sinpix_20,
                                im_mat_sh=im_mat_sh, 
                                det_len=det_len,
                                proj_angs=proj_angs, 
                                pix_ph_sz=pix_ph_sz, 
                                det_elm_ph_sz=det_elm_ph_sz, 
                                det_ph_offs=det_ph_offs)

    recon_20, ps_inv, ps_rank_20, _ = inverse_20.general_projection_recon_pseudoinv_masked_2(proj_mat, verbose=False)

    if plot_recon:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 6)

        im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
        axs[0].set_title("pacman_20: Original")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)

        im1 = axs[1].imshow(recon_20, cmap=cm.get_cmap("plasma"))
        axs[1].set_title(f"pacman_20: Reconstructed with rank {ps_rank_20}")
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        plt.show()

    # calculate phantom-vs-reconstruction errors:
    rmse = mean_squared_error(pacman_20, recon_20, squared=False)
    mae = mean_absolute_error(pacman_20, recon_20)
    print(f"pacman_20 masked reconstruction errors:\n RMSE: {rmse:.4f} MAE: {mae:.4f}")
    
    return

# train_instance_20(plot_sinpix=True, plot_projgram=True)
# train_pacman_20(num_angs=20, plot_proj=False, plot_recon=True)
# pseudoinv_instance_20(num_angs=10, plot_sinpix=True, plot_projgram=True)
# pseudoinv_instance_20(num_angs=4, plot_sinpix=True, plot_projgram=True)

# pseudoinv_pacman_20(num_angs=4, plot_proj=False, plot_recon=True)
# pseudoinv_pacman_masked_20(num_angs=4, plot_proj=False, plot_recon=True)

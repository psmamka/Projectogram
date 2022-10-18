# Tests for the Projection2D class

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
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import numpy as np
from projection import Projection2D
from phantom import pacman_mask
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# simple projection of a 20x20 pacman
def simple_20():
    pacman_20 = pacman_mask(20, (9.5,9.5), 7, direc=0, ang=60)
    # print(pacman_20.shape)

    # fig, ax = plt.subplots()
    # im = ax.imshow(pacman_20, cmap=cm.get_cmap("plasma"))
    # plt.show()

    proj_20 = Projection2D(im_mat_sh=pacman_20.shape, 
                            det_len=20, 
                            pix_ph_sz=1.0, 
                            det_elm_ph_sz=1.0, 
                            det_ph_offs=0.0, 
                            proj_angs=np.linspace(start=-90, stop=90, num=20, endpoint=False))


    proj_mat = proj_20.single_mat_all_ang_proj(pacman_20, is_sparse=False)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
    im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
    plt.show()


def one_single_pix_proj_20(im_shp=(20,20), det_len=20, num_angs=20):
    ''' Full projection set for one (the first) 20x20 single-pixel matrix
    '''

    proj_20 = Projection2D(im_mat_sh=im_shp,
                            det_len=det_len, 
                            pix_ph_sz=1.0, 
                            det_elm_ph_sz=1.0, 
                            det_ph_offs=0.0, 
                            proj_angs=np.linspace(start=-90, stop=90, num=num_angs, endpoint=False))

    sp_mat = np.zeros((20, 20))
    sp_mat[0, 0] = 1.0

    # proj_mat = proj_20.single_mat_all_ang_proj(sp_mat, is_sparse=False)
    proj_mat = proj_20.single_mat_all_ang_proj(sp_mat, is_sparse=True, sparse_idx=[(0, 0)])

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    im0 = axs[0].imshow(sp_mat, cmap=cm.get_cmap("plasma"))
    im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
    plt.show()


def all_single_pix_proj_20(im_shp=(20,20), det_len=20, num_angs=20):
    ''' Full projection set for all 20x20 single-pixel matrices 
    '''

    proj_20 = Projection2D(im_mat_sh=im_shp,
                            det_len=det_len, 
                            pix_ph_sz=1.0, 
                            det_elm_ph_sz=1.0, 
                            det_ph_offs=0.0, 
                            proj_angs=np.linspace(start=-90, stop=90, num=num_angs, endpoint=False))
    
    single_pix_mats = proj_20.build_single_pixel_mats_dense()

    proj_single_pix = proj_20.gen_single_pixel_projs(single_pixel_mats=single_pix_mats)

    # # plotting
    # # first pix, its projections
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # fig.set_size_inches(12, 6)
    # im0 = axs[0].imshow(single_pix_mats[0, 0, :, :], cmap=cm.get_cmap("plasma"))
    # im1 = axs[1].imshow(proj_single_pix[0, 0, :, :], cmap=cm.get_cmap("plasma"))
    # print(proj_single_pix[0, 0, :, :])  # problem with cropping
    # plt.show()

    # all pix all proj
    im_len = im_shp[0] * im_shp[1]
    proj_len = det_len * num_angs
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    im0 = axs[0].imshow(single_pix_mats.reshape(im_len, im_len), cmap=cm.get_cmap("plasma"))
    im1 = axs[1].imshow(proj_single_pix.reshape(im_len, proj_len), cmap=cm.get_cmap("plasma"))
    plt.show()
    
    return


# simple_20()
# one_single_pix_proj_20(im_shp=(20,20), det_len=24, num_angs=20)
# all_single_pix_proj_20(im_shp=(20,20), det_len=20, num_angs=20)
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


# utility plottig
def plot_multiple_images(im_arr, nrows=1, ncols=2, fig_size=(12, 6), col_map="plasma", titles_arr=None, show_cbar=True):
    sz = len(im_arr)
    if (nrows * ncols != sz): nrows, ncols = (1, sz) 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(fig_size)
    plots, ims, cax = (im_arr, [None] * sz, [None] * sz)
    if sz > 1:
        for ind in range(sz):
            ims[ind] = axs[ind].imshow(plots[ind], cmap=cm.get_cmap(col_map))
            if titles_arr is not None: axs[ind].set_title(titles_arr[ind])
            divider = make_axes_locatable(axs[ind])
            if show_cbar:
                cax[ind] = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ims[ind], cax=cax[ind])
    else:
        ims = axs.imshow(plots[0], cmap=cm.get_cmap(col_map))
        if titles_arr is not None: axs.set_title(titles_arr[0])
        divider = make_axes_locatable(axs)
        if show_cbar:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(ims, cax=cax)

    # plt.show()

    return fig, axs

class Parset():
    '''Simple way of containing the parameter set. Disctionaries are too clunky'''
    def __init__(self, num_angs=12,
                        proj_angs=None,
                        im_mat_sh=(20, 20),
                        det_len=20,
                        pix_ph_sz=1.0,
                        det_elm_ph_sz=1.0,
                        det_ph_offs=0.0):

        if proj_angs is None: 
            self.proj_angs = np.linspace(start=-90, stop=90, num=num_angs, endpoint=False)
        else:
            self.proj_angs = proj_angs,

        self.num_angs = num_angs
        self.im_mat_sh = im_mat_sh
        self.det_len = det_len
        self.pix_ph_sz = pix_ph_sz
        self.det_elm_ph_sz = det_elm_ph_sz
        self.det_ph_offs = det_ph_offs


def initialize_objs(num_angs=12,
                    proj_angs=None,
                    im_mat_sh=(20, 20),
                    det_len=20,
                    pix_ph_sz=1.0,
                    det_elm_ph_sz=1.0,
                    det_ph_offs=0.0):

    parset = Parset(num_angs=num_angs,
                    proj_angs=proj_angs,
                    im_mat_sh=im_mat_sh,
                    det_len=det_len,
                    pix_ph_sz=pix_ph_sz,
                    det_elm_ph_sz=det_elm_ph_sz,
                    det_ph_offs=det_ph_offs)

    print(parset.im_mat_sh)      
    
    proj_obj = Projection2D(im_mat_sh=parset.im_mat_sh, 
                            det_len=parset.det_len, 
                            pix_ph_sz=parset.pix_ph_sz, 
                            det_elm_ph_sz=parset.det_elm_ph_sz, 
                            det_ph_offs=parset.det_ph_offs, 
                            proj_angs=parset.proj_angs)

    return parset, proj_obj


def generate_figures():
    
    parset, proj_20 = initialize_objs(num_angs=12, im_mat_sh=(20, 20), det_len=20)

    sinpix_20 = proj_20.build_single_pixel_mats_dense()
    projectogram = proj_20.gen_single_pixel_projs(sinpix_20)

    img_orig = sinpix_20[2, 2]
    mat_proj = projectogram[2, 2]

    # plot single pixel image and its projections
    fig, axs = plot_multiple_images([img_orig, mat_proj], 
                                    titles_arr=["Single Pixel Image With Projection Angles", 
                                                f"Projections Along The {parset.num_angs} Angles"])
    axs[0].set_xticks(np.arange(0, 20, step=4))
    axs[0].set_yticks(np.arange(0, 20, step=4))
    axs[1].set_xticks(np.arange(0, 20, step=4))

    # draw projection direction lines
    l_len = 9.5  # line length
    x0, y0 = l_len, l_len
    for idx in range(parset.num_angs):
        theta = np.pi/parset.num_angs * idx
        xi = l_len * (1 - np.cos(theta))
        yi = l_len * (1 + np.sin(theta))
        # axs[0].plot([x0, xi], [y0, yi], color="w")
        axs[0].arrow(x0, y0, (xi-x0), (yi-y0), color="w", length_includes_head=True, head_length=1, head_width=0.4)
        
    plt.show()

    pg_rank = 0
    # plot the projectogram
    fig, axs = plot_multiple_images([projectogram.reshape((400, 240))], 
                                    nrows=1, ncols=1, fig_size=(10,10),
                                    titles_arr=["Projectogram for a 20x20 Image\n Projection Angles: 12, Detector Size: 20"])
    axs.set_xlabel("projection elements")
    axs.set_ylabel("pixels")

    # fig, axs = plot_multiple_images([np.zeros((20,20))])
    plt.show()

    return


generate_figures()


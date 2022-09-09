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
    im0 = axs[0].imshow(pacman_20, cmap=cm.get_cmap("plasma"))
    im1 = axs[1].imshow(proj_mat, cmap=cm.get_cmap("plasma"))
    plt.show()

simple_20()
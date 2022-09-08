from phantom import pacman_mask
import matplotlib.pyplot as plt
import matplotlib.cm as cm


fig, ax = plt.subplots()

mask_1 = pacman_mask((10, 10), (5,5), 4, 0, 60)
# im = ax.imshow(mask_1, cmap=cm.get_cmap("plasma"))
# plt.show()


mask_2 = pacman_mask((20, 20), (9.5,9.5), 9, 90, 60)
# im = ax.imshow(mask_2, cmap=cm.get_cmap("plasma"))
# plt.show()


mask_3 = pacman_mask(30, (20,20), 15, 90, 60)
# im = ax.imshow(mask_3, cmap=cm.get_cmap("plasma"))
# plt.show()
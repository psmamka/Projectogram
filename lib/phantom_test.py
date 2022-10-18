# Tests phantom generation functions

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

from phantom import pacman_mask, rectangle_mask
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

rect_1 = rectangle_mask(mat_sz=(40, 40), rect_sz=(10, 20))
# im = ax.imshow(rect_1, cmap=cm.get_cmap("plasma"))
# plt.show()

rect_2 = rectangle_mask(mat_sz=(55, 55), rect_sz=(15, 15), cent=(27, 27))
im = ax.imshow(rect_2, cmap=cm.get_cmap("plasma"))
plt.show()
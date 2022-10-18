# Simple shapes for reference images (i.e. phantoms)

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
import math


def pol2cart(r, theta):
    """theta in degrees

    returns tuple; (float, float); (x,y)
    """
    x = r * np.cos(math.radians(theta))
    y = r * np.sin(math.radians(theta))
    return x, y

def cart2pol(x, y):
    """returns r, theta(degrees)
    """
    r = np.sqrt(x * x + y * y)
    theta = math.degrees(np.arctan2(y, x)) % 360
    return r, theta



def pacman_mask(mat_sz, cent, rad, direc=0.0, ang=60.0):
    """ Creates a pac-man shape mask
    Parameters:
    mat_sz: (rows, cols): number of rows and columns of the matrix
    cent: geometric center of the phantom (u, v)
    rad: radius
    direc: 0 to 360 degrees direction of the face
    ang: angle of the 'mouth' opening in degrees

    output: NxM 2D array of 0's and 1's
    """
    if isinstance(mat_sz, int): mat_sz = (mat_sz, mat_sz) 
    
    rows, cols = mat_sz
    
    out_mask = np.ones((rows, cols))

    ang1, ang2 = ((direc - ang / 2.0) % 360, (direc + ang / 2.0) % 360)
    # print(ang1, ang2)
    for u in range(rows):
        for v in range(cols):
            x, y = (v - cent[1], -u + cent[0])
            r, th = cart2pol(x, y)
            # print(r, th)
            if r > rad: out_mask[u, v] = 0
            if ang2 > ang1:
                if th > ang1 and th < ang2: out_mask[u, v] = 0
            if ang2 < ang1: # crossing the zero angle
                if th > ang1 or th < ang2: out_mask[u, v] = 0
    
    return out_mask


def rectangle_mask(mat_sz, rect_sz, cent=None, ul_corner=None):
    """Creates a rectangular phantom
    Parameters:
    mat_sz: (rows, cols): number of rows and columns of the matrix
    rect_sz: (h, w): height inrows and width in columns of the rectangle
    cent: center of the rectangle (u_center, v_center) in row and column indices
    ul_corner: alternatively, we can give the location of the upper left corner of the rectangle in (u_ul, v_ul) indices
    """
    if isinstance(mat_sz, int): mat_sz = (mat_sz, mat_sz)
    if isinstance(rect_sz, int): rect_sz = (rect_sz, rect_sz)

    rows, cols = mat_sz
    rect_rows, rect_cols = rect_sz

    # validate inputs:
    if (ul_corner is None) and (cent is None):
        # raise Exception("cent and ul_corner can not both be None.") # or just center the center:
        ul_corner = (math.floor((rows - rect_rows) / 2), math.floor((cols - rect_cols) / 2))

    out_mat = np.zeros(mat_sz)
    rect_mat = np.ones(rect_sz)

    if ul_corner is None:
        ul_corner = (math.floor(cent[0] - (rect_rows - 1) / 2), math.floor(cent[1] - (rect_cols - 1) / 2))
        
    # print(ul_corner)
    out_mat[ul_corner[0]:ul_corner[0] + rect_rows , ul_corner[1]:ul_corner[1] + rect_cols] = rect_mat

    return out_mat
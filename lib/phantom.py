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


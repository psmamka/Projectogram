import numpy as np

class Projection2D:
    # Forward projection from a 2D image matrix to a 1D detector array
    # For now, pixels are all square and the same size, detector elements are linear and same size
    # for now projection beams are all parallel (like linear algebra projections)
    # im_mat_sh is the (N, M) 2-tuple for the shape of the image matrix
    # det_len is the length of the detector array
    # _ph_sz suffixes refer to physical length in chosen units
    def __init__(self, im_mat_sh, det_len, pix_ph_sz=1.0, det_elm_ph_sz=1.0):
        self.im_mat_sh = im_mat_sh
        self.N, self.M = im_mat_sh      # for convenience N rows, M columns
        self.im_mat = np.zeros(im_mat_sh)

        self.det_len = det_len

        self.pix_ph_sz = pix_ph_sz
        self.det_elm_ph_sz = det_elm_ph_sz

        # original physical coordinates for the entire matrix before any rotation
        self.mat_x_0, self.mat_y_0 = self.generate_x_y_mats()

    # for X, Y axes passing through the image center, the center locations (x, y) of
    # a pixel in the (u, v) row/column is given by:
    def pixel_coords(self, u, v):
        # u, v are te row, column indices, respectively
        # N, M are the height and width of the image matrix (in pixels)
        # returns (x, y) coordinates of (u, v) pixel in physical units
        x = - (self.M - 1) / 2.0 + v
        y =   (self.N - 1) / 2.0 - u

        return (self.im_pix_size * x, self.im_pix_size * y)

    # generateoriginal x and y physical coordinates for the entire image matrix, store in mat_x, mat_y
    def generate_x_y_mats(self):
        mat_x = np.zeros(self.N, self.M)
        mat_y = np.zeros(self.N, self.M)

        for u in range(self.N):
            for v in range(self.M):
                mat_x[u, v] = -(self.M - 1) / 2.0 + v
                mat_y[u, v] =  (self.N - 1) / 2.0 - u
        # normalize physical coordinates
        mat_x *= self.pix_ph_sz
        mat_y *= self.pix_ph_sz

        return mat_x, mat_y

    # Applying rotation theta to physical coordinates (in the right hand sense):
    # inputs are the physical coordinate matrices. 
    # outputs are new x and y rotated matrices
    # for the 2D case, a rotation by θ in the (x,y) plane is defined by:
    # [ [ cos(θ) , -sin(θ)]
    #   [sin(θ) ,   cos(θ)] ]
    # 
    # Note: Rotating detector by θ would be equivalent to rotating the image matrix by -θ
    # using degrees for theta
    def rotate_xy(self, theta):

        cos_th = np.cos(theta * np.pi / 180)
        sin_th = np.sin(theta * np.pi / 180)

        mat_x_th = self.mat_x * cos_th - self.mat_y * sin_th
        mat_y_th = self.mat_x * sin_th + self.mat_y * cos_th

        return mat_x_th, mat_y_th

    # Map the image matrix to detector elements usng physicial coordinates: matrix-detector-index (mat_det_idx)
    # 
    # First we define a simple "nearest-element-center" projection rule:
    # (1) Each pixel representative (center) is projected vertically on a detector plane below
    # (2) The detector element assigned to the pixel is the one where the pixel center lands
    # (3) Later, we can try other forms of dividing pixel information amonf nearby elements
    # (4) Since we are projecting along the y-axis, we only need the x coords.

    # First, a function to assign detector element index to each matrix element:
    # General formula for assigning detector element index for an matrix elm with xcoord:

    # k_x = floor(x/d) + det_sz/2                           : for det_sz even
    # k_x = floor(x/d - 0.5) + (det_sz + 1)/2               : for det_sz odd

    # inputs:
    # mat_x: physical x coords of each mtrix elm,
    # det_sz: number of detector elements
    # det_elm_len: physical length of the detector element (arb units)
    # crop_outside: indices outside the detector range, should we turn to None, or leave them as is
    # output:
    # mat_det_idx: matrix with the index of detector elm where each pixel is projected on
    def mat_det_idx_y(self, mat_x, crop_outside=True):
        mat_det_idx = np.zeros(mat_x.shape)
        
        if self.det_len % 2 == 0:
            mat_det_idx = np.floor(mat_x / self.det_elm_ph_sz) + self.det_len / 2
        else:
            mat_det_idx = np.floor(mat_x / self.det_elm_ph_sz - 0.5) + (self.det_len + 1) / 2

        # post processing of projection values:
        trans1 = lambda z: np.round(z).astype("int")
        if crop_outside:
            trans2 = lambda z: z if (0 <= z <= self.det_len - 1) else None
        else:
            trans2 = lambda z: z   # do nothing

        return np.array([trans2(trans1((z))) for z in mat_det_idx.ravel()]).reshape(mat_x.shape)
    
    # Projection for a single angle based on the detector element indices
    # For sparse matrices, it's much faster to sum over the nonzero elements.
    def mat_det_proj(self, img_mat, mat_det_idx, is_sparse=False, sparse_idx=[]):
        det_out = np.zeros(self.det_sz)
        N, M = img_mat.shape

        if not is_sparse:                   # for dense mats, sum over all elms
            for u in range(N):
                for v in range(M):
                    d_idx = mat_det_idx[u, v]
                    if (d_idx is not None) and (0 <= d_idx <= self.det_sz - 1):
                        det_out[d_idx] += img_mat[u, v]
        else:                               # for sparse mats, sum over nonzero indices:
            for s_idx in sparse_idx:
                d_idx = mat_det_idx[s_idx]  # detector elm index
                det_out[d_idx] += img_mat[s_idx]

        return det_out
        




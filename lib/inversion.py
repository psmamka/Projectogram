import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class Inversion2D:
    '''Finding the Inverse transformation using the single-pixel projections, i.e. the `projectogram`
    Similar naming convention of the variables to the Projection2D class
    '''

    def __init__(self, projectogram,
                        single_pixel_mats,
                        im_mat_sh=(100, 100), 
                        det_len=100,
                        proj_angs=[0.], 
                        pix_ph_sz=1.0, 
                        det_elm_ph_sz=1.0, 
                        det_ph_offs=0.):
        # input verification: projectogram should have the size (N, M, Angs, det_len)
        if (self.pg_sh[0] != im_mat_sh[0] or \
            self.pg_sh[1] != im_mat_sh[1] or \
            self.pg_sh[2] != len(proj_angs) or \
            self.pg_sh[3] != det_len):
             raise Exception(f"Constructor Parameters Inconsistent: \n\
             projectogram {projectogram.shape}, image: {im_mat_sh}, angles: {len(proj_angs)}, detector: {det_len}")

        self.projectoram = projectogram
        self.single_pixel_mats = single_pixel_mats

        self.pg_sh = projectogram.shape
        self.im_mat_sh = im_mat_sh
        self.N, self.M = im_mat_sh      # for convenience N rows, M columns (reverse order intended)
        self.proj_angs = proj_angs
        self.proj_ang_sz = len(proj_angs)
        self.det_len = det_len

        self.pix_ph_sz = pix_ph_sz
        self.det_elm_ph_sz = det_elm_ph_sz
        self.det_ph_offs = det_ph_offs

    # Use the X (features) and y (labels) terminology from machine learning
    # Here features X are the single pixel projections, or the projectogram
    # Labels y are the single pixel (one-hot) matrices
    # X and y are returned both in 2D form
    def prepare_single_pixel_Xy(self):
        X = self.projectoram.reshape(self.N * self.M, self.proj_ang_sz * self.det_len)
        y = self.single_pixel_mats.reshape(self.N * self.M, self.N * self.M)

        return X, y
        
    

        



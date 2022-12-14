# The Inversion2D class: contains methods for the inverse of projection operation (1D → 2D reconstruction)

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

from statistics import linear_regression
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

        self.pg_sh = projectogram.shape
        # input verification: projectogram should have the size (N, M, Angs, det_len)
        if (self.pg_sh[0] != im_mat_sh[0] or \
            self.pg_sh[1] != im_mat_sh[1] or \
            self.pg_sh[2] != len(proj_angs) or \
            self.pg_sh[3] != det_len):
            raise Exception(f"Constructor Parameters Inconsistent: \n\
            projectogram {projectogram.shape}, image: {im_mat_sh}, angles: {len(proj_angs)}, detector: {det_len}")

        self.projectogram = projectogram
        self.single_pixel_mats = single_pixel_mats

        self.im_mat_sh = im_mat_sh
        self.N, self.M = im_mat_sh      # for convenience N rows, M columns (reverse order intended)
        self.proj_angs = proj_angs
        self.proj_angs_sz = len(proj_angs)
        self.det_len = det_len

        self.pix_ph_sz = pix_ph_sz
        self.det_elm_ph_sz = det_elm_ph_sz
        self.det_ph_offs = det_ph_offs

        self.linear_model = None    # linear network from sklearn

        self.pseudoinv = None       # pseudo-inverse from scipy.linalg
        self.pseudorank = None

    # Use the X (features) and y (labels) terminology from machine learning
    # Here features X are the single pixel projections, or the projectogram
    # Labels y are the single pixel (one-hot) matrices
    # X and y are returned both in 2D form
    def prepare_single_pixel_Xy(self):
        X = self.projectogram.reshape(self.N * self.M, self.proj_angs_sz * self.det_len)
        y = self.single_pixel_mats.reshape(self.N * self.M, self.N * self.M)

        return X, y
        
    # Finding the rank of the projectogram, or the feature-space in the ML lingo
    def get_projectogram_rank(self, verbose=False):
        X, _ = self.prepare_single_pixel_Xy()

        X_rank = np.linalg.matrix_rank(X)

        if verbose:
            print(f"Image Shape: {self.im_mat_sh}, Image Pixels: {X.shape[0]}, Features Rank: {X_rank}, Null Space: {X.shape[0] - X_rank}")

        return X_rank
    

    # The pseudo-inverse method, using scipy.linalg's SVD-based generalized inverse technique
    # The idea is that the projectogram's (pseudo-) inverse can be used to reconstruct images:
    #       Projections = Projectogram * Image + epsilon
    # ===> (least squares)
    #       Image ~= Projectogram_ps_inv * Projections
    # Where pseudo-inverse of A is:
    #       A_ps_inv = (A^H * A)^-1 * A^H       (A^H being conjugate transpose of A) 
    def get_projectrogram_pseudoinv(self, verbose=True):
        X, _ = self.prepare_single_pixel_Xy()

        psinv, psrank = linalg.pinv(X, atol=None, rtol=None, return_rank=True, check_finite=True)
        self.pseudoinv = psinv
        self.pseudorank = psrank

        if (verbose):
            print(f"Pseudo-inverse rank: {psrank}, image pixels: {X.shape[0]}")

        return psinv, psrank
    

    # Pseudo-inverse reconstruction of single-pixel images from the projectogram
    def single_pixel_recon_pseudoinv(self):

        # validate that the peudo-inverse is already calculated
        if self.pseudoinv is None:
            self.get_projectrogram_pseudoinv()
        
        X, _ = self.prepare_single_pixel_Xy()
        sinpix_recon_all = (self.pseudoinv.T).dot(X.T)    # matrix multiplicaltion

        return sinpix_recon_all


    # Reconstruction of an image from projections matrix using the pseudoinverse method
    # projections are obtained in accordance with the __init__ parameters self.proj_angs_sz, self.det_len
    def general_projection_recon_pseudoinv(self, proj_mat):
        # validate shape
        if proj_mat.shape != (self.proj_angs_sz, self.det_len):
            raise Exception(f"Projection matrix shape {proj_mat.shape} not compatible with angles length {self.proj_angs_sz}, detector length {self.det_len}")
        
        # validate that the peudo-inverse is already calculated
        if self.pseudoinv is None:
            self.get_projectrogram_pseudoinv()

        X = proj_mat.reshape(-1, 1)
        y_pred = (self.pseudoinv.T).dot(X)    # matrix multiplicaltion

        return y_pred.reshape(self.N, self.M)


    # pseudo-inverse reconstruction whilt applying the mask to the reconstructed image
    def general_projection_recon_pseudoinv_masked(self, proj_mat, verbose=False):
        
        pix_mask = np.full(self.im_mat_sh, False)

        for i in range(self.im_mat_sh[0]):
            for j in range(self.im_mat_sh[1]):
                masked_idx = self.projectogram[i, j] > 0
                prod_arr = self.projectogram[i, j, masked_idx] * proj_mat[masked_idx]
                if min(prod_arr) > 0: pix_mask[i, j] = True
        
        im_recon = np.zeros(self.im_mat_sh)
        im_recon_psinv = self.general_projection_recon_pseudoinv(proj_mat)

        im_recon[pix_mask] = im_recon_psinv[pix_mask]

        return im_recon, self.pseudoinv, self.pseudorank, pix_mask

   
    # pseudo-inverse reconstruction whilt applying the mask to the projectogram
    def general_projection_recon_pseudoinv_masked_2(self, proj_mat, verbose=False):
        
        pix_mask = np.full(self.im_mat_sh, False)
        pix_prod_list = []

        for i in range(self.im_mat_sh[0]):
            for j in range(self.im_mat_sh[1]):
                masked_idx = self.projectogram[i, j] > 0
                prod_arr = self.projectogram[i, j, masked_idx] * proj_mat[masked_idx]
                # if min(prod_arr) > 0: print(i, j, min(prod_arr))
                # if min(prod_arr) > 0: pix_prod_list.append((i, j, min(prod_arr)))
                if min(prod_arr) > 0: 
                    pix_mask[i, j] = True
                    pix_prod_list.append((i, j, min(prod_arr)))
        
        if verbose: print("Nonzero Pixels Mask:\n", pix_prod_list)

        mask_len = len(pix_prod_list)
        projectogram_masked = np.zeros((mask_len, self.proj_angs_sz * self.det_len))

        for idx, (i, j, _) in enumerate(pix_prod_list):
            projectogram_masked[idx, :] = self.projectogram[i, j, :, :].ravel()
        
        ps_inv, ps_rank = linalg.pinv(projectogram_masked.T, atol=None, rtol=None, return_rank=True, check_finite=True)
        if verbose:
            print(f"Pseudo-inverse shape: {ps_inv.shape}\nproj_mat shape: {proj_mat.shape}\nPseudo-inverse rank: {ps_rank}")

        recon_data_psinv = ps_inv.dot(proj_mat.ravel())

        im_recon = np.zeros(self.im_mat_sh)
        for idx, (i, j, _) in enumerate(pix_prod_list):
            im_recon[i, j] = recon_data_psinv[idx]

        return im_recon, ps_inv, ps_rank, pix_mask
    

    # Use sk-learn OLS to calculate parameters of a linear model
    # fit_intercept is False, since we don't expect any bias in detector output (w/ or w/o noise)
    # n_jobs of -1 allows the machine to decide the number of cpu cores used for training
    def train_lin_reg_model(self):

        self.linear_model = LinearRegression(fit_intercept=False, n_jobs=-1)

        X, y = self.prepare_single_pixel_Xy()

        self.linear_model.fit(X, y)

        return self.linear_model

    # Use the lin_reg model to predict/reconstruct the single pixel matrices from the projectogram
    def single_pixel_lin_reg_reconstruct(self):

        if self.linear_model is None:
            # raise Exception(f"self.linear_model is None.")
            self.train_lin_reg_model()
        
        X, _ = self.prepare_single_pixel_Xy()
        y_pred = self.linear_model.predict(X)

        return y_pred
    

    # RMSE (root mean squared error), MAE (mean absolute error) of single pixel reconstructions from 
    # the projectogram using the lin_reg model
    # Later: write a method to get the distance as input and calc the err
    def single_pixel_lin_reg_errors(self, verbose=False):
        _, y = self.prepare_single_pixel_Xy()
        y_pred = self.single_pixel_lin_reg_reconstruct()

        rmse_single_pix = mean_squared_error(y, y_pred, squared=False)
        mae_single_pix = mean_absolute_error(y, y_pred)

        if verbose:
            print("Single-pixel reconstructions errors from the projectogram using the trained Linear Regression Model:")
            print(f"Root Mean Squared Error: {rmse_single_pix:.4f}")
            print(f"Mean Absolute Error: {mae_single_pix:.4f}")

        return rmse_single_pix, mae_single_pix


    # LinReg Reconstruction of an image from projections matrix obtained in accordance with the __init__ parameters
    # later: API with the Projection2D class
    def general_projection_reconstruction(self, proj_mat):
        # validate shape
        if proj_mat.shape != (self.proj_angs_sz, self.det_len):
            raise Exception(f"Projection matrix shape {proj_mat.shape} not compatible with angles length {self.proj_angs_sz}, detector length {self.det_len}")
        
        # validatethat the linear model is already trained
        if self.linear_model is None:
            self.train_lin_reg_model()

        X = proj_mat.reshape(1, -1)
        y_pred = self.linear_model.predict(X)

        return y_pred.reshape(self.N, self.M)
        


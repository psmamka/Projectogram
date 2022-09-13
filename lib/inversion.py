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

        self.projectoram = projectogram
        self.single_pixel_mats = single_pixel_mats

        self.im_mat_sh = im_mat_sh
        self.N, self.M = im_mat_sh      # for convenience N rows, M columns (reverse order intended)
        self.proj_angs = proj_angs
        self.proj_angs_sz = len(proj_angs)
        self.det_len = det_len

        self.pix_ph_sz = pix_ph_sz
        self.det_elm_ph_sz = det_elm_ph_sz
        self.det_ph_offs = det_ph_offs

        self.linear_model = None

    # Use the X (features) and y (labels) terminology from machine learning
    # Here features X are the single pixel projections, or the projectogram
    # Labels y are the single pixel (one-hot) matrices
    # X and y are returned both in 2D form
    def prepare_single_pixel_Xy(self):
        X = self.projectoram.reshape(self.N * self.M, self.proj_angs_sz * self.det_len)
        y = self.single_pixel_mats.reshape(self.N * self.M, self.N * self.M)

        return X, y
        
    # Finding the rank of the projectogram, or the feature-space in the ML lingo
    def get_projectogram_rank(self, verbose=False):
        X, _ = self.prepare_single_pixel_Xy()

        X_rank = np.linalg.matrix_rank(X)

        if verbose:
            print(f"Image Shape: {self.im_mat_sh}, Image Pixels: {X.shape[0]}, Features Rank: {X_rank}, Null Space: {X.shape[0] - X_rank}")

        return X_rank
    
    # Use sk-learn (or any OLS library) to calculate parameters of a linear model
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


    # Reconstruction of an image from projections matrix obtained in accordance with the __init__ parameters
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

    
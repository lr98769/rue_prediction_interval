import numpy as np
from tqdm.auto import tqdm
from scipy.stats import norm


class ConditionalGaussianDistribution:
    def __init__(self, Y, X):
        self.Y = Y # Assume shape = (num_samples, num_output)
        self.X = X # Assume shape = (num_samples, num_inputs)
        self.YX = np.concatenate((Y, X), axis=-1) # Assume shape = (num_samples, num_output+num_inputs)
        self.num_Y = self.Y.shape[1]
        self.num_X = self.X.shape[1]

        # Fitting
        self.esti_mean_YX = np.mean(self.YX, axis = 0) # (num_output+num_inputs,)
        self.esti_covariance_matrix_YX = np.cov(self.YX.T, ddof=1) # (num_output+num_inputs, num_output+num_inputs)
        
        # Useful stats
        self.mu_Y = self.esti_mean_YX[:self.num_Y]
        self.mu_X = self.esti_mean_YX[self.num_Y:]
        self.cov_XX = self.esti_covariance_matrix_YX[self.num_Y:, self.num_Y:]
        self.cov_YY = self.esti_covariance_matrix_YX[:self.num_Y, :self.num_Y]
        self.cov_YX = self.esti_covariance_matrix_YX[:self.num_Y, self.num_Y:]
        self.cov_XY = self.esti_covariance_matrix_YX[self.num_Y:, :self.num_Y]
        
    def get_conditional_mean(self, alpha):
        # alpha = (num_samples, self.num_X)
        return (
            self.mu_Y + 
            np.linalg.multi_dot((
                self.cov_YX,
                np.linalg.inv(self.cov_XX),
                (alpha - self.mu_X).T))
        ).flatten()
        
    def get_conditional_cov(self):
        return (
            self.cov_YY - 
            np.linalg.multi_dot((
                self.cov_YX,
                np.linalg.inv(self.cov_XX), 
                self.cov_XY))
        ).flatten()[0]

def cond_gauss_prediction_interval(
    df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, 
    seed, alpha=0.05):
    pi_label = "_cond_gauss"
    df_val, df_test = df_val.copy(), df_test.copy()

    unscaled_cols = [col+"_unscaled" for col in pred_cols]
    prediction_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
    unscaled_pred_cols = [col+"_unscaled" for col in prediction_cols]

    # Unscale the prediction columns
    df_val[unscaled_cols] = scaler.inverse_transform(df_val[pred_cols])
    df_test[unscaled_cols] = scaler.inverse_transform(df_test[pred_cols])
    df_val[unscaled_pred_cols] = scaler.inverse_transform(df_val[prediction_cols])
    df_test[unscaled_pred_cols] = scaler.inverse_transform(df_test[prediction_cols])

    # Get reconstruction errors
    reconstruction_cols = [col+"_reconstruction_"+regressor_label for col in predictors]
    valid_re = np.abs(df_val[predictors].values - df_val[reconstruction_cols].values)
    test_re = np.abs(df_test[predictors].values - df_test[reconstruction_cols].values)

    n_val = len(df_val)

    for col in tqdm(pred_cols):
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col+"_unscaled"].values.astype('float32')
        val_y_pred = df_val[col+pred_label+"_"+regressor_label+"_unscaled"].values.astype('float32')
        val_pe = np.abs(val_y-val_y_pred)

        # Stack both pe and reconstruction error to fit the cond gaussian model
        val_data = np.hstack((np.expand_dims(val_pe, axis=1), valid_re))
        
        # Parameter Estimation on Validation Set
        uncertainty_distribution = ConditionalGaussianDistribution(
            Y=np.expand_dims(val_pe, axis=1), 
            X= valid_re
        )
        esti_conditional_mean_Y = uncertainty_distribution.get_conditional_mean(test_re)
        esti_conditional_std_Y = np.sqrt(uncertainty_distribution.get_conditional_cov())
        
        # 1-alpha/2 -> Because Gaussian is symmetrical
        df_test[col+"_"+ue_col+pi_label] = norm.ppf(1-alpha, loc=esti_conditional_mean_Y, scale=esti_conditional_std_Y).flatten()
    
    return df_test

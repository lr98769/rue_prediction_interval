import numpy as np
from scipy import stats
from scipy.stats import norm
from tqdm.auto import tqdm
from src.pi_methods.cond_gaussian import ConditionalGaussianDistribution

EPSILON = np.finfo(np.float32).eps

class GaussianCopula:
    def __init__(self, X):
        self.X = X
        self.n_feat = X.shape[1]

        # Get all CDFs
        self.cdfs = []
        for i_feat in range(self.n_feat):
            self.cdfs.append(stats.ecdf(X[:, i_feat]))

    def add_y(self, y):
        self.y = y
        self.y_cdf = stats.ecdf(self.y)
        
    def X_to_V(self, cur_X):
        X_Vi = np.empty(cur_X.shape)
        for i_feat in range(self.n_feat):
            Fi = self.cdfs[i_feat].cdf.evaluate(cur_X[:, i_feat]).clip(EPSILON, 1 - EPSILON)
            X_Vi[:,i_feat] = norm.ppf(Fi)
        return X_Vi

    def Y_to_V(self, cur_y):
        y_Fi = self.y_cdf.cdf.evaluate(cur_y).clip(EPSILON, 1 - EPSILON)
        y_Vi = norm.ppf(y_Fi)
        return y_Vi

    def V_to_Y(self, cur_y_Vi):
        y_Fi = norm.cdf(cur_y_Vi)
        y = np.percentile(self.y, y_Fi*100, axis=0, method="higher")
        return y

def gauss_copula_prediction_interval(
    df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, 
    seed, alpha=0.05):
    pi_label = "_gauss_copula"
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
    reconstruction_cols = [col+"_reconstruction"+"_"+regressor_label for col in predictors]
    valid_re = np.abs(df_val[predictors].values - df_val[reconstruction_cols].values)
    test_re = np.abs(df_test[predictors].values - df_test[reconstruction_cols].values)

    # Convert reconstruction and prediction error to V
    gc = GaussianCopula(valid_re)
    valid_re = gc.X_to_V(valid_re)
    test_re = gc.X_to_V(test_re)

    n_val = len(df_val)

    for col in tqdm(pred_cols):
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col+"_unscaled"].values.astype('float32')
        val_y_pred = df_val[col+pred_label+"_"+regressor_label+"_unscaled"].values.astype('float32')
        val_pe =np.abs(val_y-val_y_pred)

        gc.add_y(val_pe)
        val_pe = gc.Y_to_V(val_pe)

        # Parameter Estimation on Validation Set
        uncertainty_distribution = ConditionalGaussianDistribution(
            Y=np.expand_dims(val_pe, axis=1), 
            X= valid_re
        )
        esti_conditional_mean_Y = uncertainty_distribution.get_conditional_mean(test_re)
        esti_conditional_std_Y = np.sqrt(uncertainty_distribution.get_conditional_cov())

        pi_V = norm.ppf(1-alpha, loc=esti_conditional_mean_Y, scale=esti_conditional_std_Y).flatten()
        # print(pi_V)

        # Convert from V space back to Y
        df_test[col+"_"+ue_col+pi_label] = gc.V_to_Y(pi_V)
    
    return df_test

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def weighted_percentile_matrix(values, weights, percentile):
    ix = np.argsort(values, axis=-1) # sort within each 
    values = np.take_along_axis(values, ix, axis=-1) # sort data
    weights = np.take_along_axis(weights, ix, axis=-1) # sort weights
    cdf = (np.cumsum(weights, axis=-1) - 0.5 * weights) / np.sum(weights, axis=-1, keepdims=True) # 'like' a CDF function
    return [np.interp(percentile, cdf[i], values[i]) for i in range(values.shape[0])]

def weighted_prediction_interval(
    df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, seed, 
    alpha=0.05):
    pi_label = "_weighted"
    df_val, df_test = df_val.copy(), df_test.copy()

    # Set K = sqrt of size of validation set
    n_val = len(df_val)
    n_feat = len(predictors)
    k = max(round(np.sqrt(n_val)), 80)

    # Get reconstruction errors
    reconstruction_cols = [col+"_reconstruction"+"_"+regressor_label for col in predictors]
    valid_re = np.abs(df_val[predictors].values - df_val[reconstruction_cols].values)
    test_re = np.abs(df_test[predictors].values - df_test[reconstruction_cols].values)

    # Mahalanobis distance = euclidan distance after pc with whitening
    # - https://stackoverflow.com/questions/69811792/mahalanobis-distance-not-equal-to-euclidean-distance-after-pca
    re_scaler = StandardScaler()
    pca = PCA(whiten=True)
    valid_re = re_scaler.fit_transform(valid_re)
    valid_re = pca.fit_transform(valid_re)
    test_re = re_scaler.transform(test_re)
    test_re = pca.transform(test_re)

    # Find neighbours
    sigma = 10
    kdtree = KDTree(valid_re)
    mahalanobis_dist, ind = kdtree.query(test_re, k=k, workers=-1, p=2)
    dist = mahalanobis_dist/np.sqrt(n_feat) # already sorted by distance (nearest first; ascending order of distance)
    weights = np.exp(-np.square(dist)/(2*sigma**2)) # descending order of weights
    print("Number of Zero Weights:", np.sum(np.sum(weights, axis=1)==0))
    print("Std of Weights:", np.mean(np.var(weights, axis=1)))
    modified_alpha = np.ceil((k+1)*(1-alpha))/k

    lb_cols, ub_cols = [], []
    for col in tqdm(pred_cols):
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col].values.astype('float32') # +"_unscaled"
        val_y_pred = df_val[col+pred_label+"_"+regressor_label].values.astype('float32') # "_unscaled"
        test_y_pred = df_test[col+pred_label+"_"+regressor_label].values.astype('float32') # "_unscaled"
        val_pe = np.abs(val_y-val_y_pred)

        # Nearbouring prediction errors
        neighbour_pe = val_pe[ind]

        # # PI = 0.95 Weighted Percentile of neigbouring errors
        # pis = []
        # for i in tqdm(range(n_test), total=n_test, leave=False):
        #     df = pd.DataFrame({'v': neighbour_pe[i], 'w': weights[i]})
        #     # display(df)
        #     calc = wc.Calculator('w')  # w designates weight
        #     pi = calc.quantile(df, 'v', modified_alpha)
        #     pis.append(pi)
        pi = weighted_percentile_matrix(values=neighbour_pe, weights=weights, percentile=modified_alpha)
        
        # Get Upper and Lower Bound
        pi_col = col+"_"+ue_col+pi_label
        lb_col, ub_col = pi_col+"_lb", pi_col+"_ub"
        df_test[lb_col] = test_y_pred-pi
        df_test[ub_col] = test_y_pred+pi
        lb_cols.append(lb_col)
        ub_cols.append(ub_col)
        
    # Unscaled Columns
    prediction_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
    unscaled_cols = [col+"_unscaled" for col in pred_cols]
    unscaled_pred_cols = [col+"_unscaled" for col in prediction_cols]
    unscaled_lb_cols = [col+"_unscaled" for col in lb_cols]
    unscaled_ub_cols = [col+"_unscaled" for col in ub_cols]
    
    # Unscale the prediction columns
    df_test[unscaled_cols] = scaler.inverse_transform(df_test[pred_cols])
    df_test[unscaled_pred_cols] = scaler.inverse_transform(df_test[prediction_cols])
    df_test[unscaled_lb_cols] = scaler.inverse_transform(df_test[lb_cols])
    df_test[unscaled_ub_cols] = scaler.inverse_transform(df_test[ub_cols])
    
    return df_test
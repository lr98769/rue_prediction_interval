import numpy as np
from scipy.spatial import KDTree

def knn_prediction_interval(
    df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, 
    seed, alpha=0.05):
    pi_label = "_knn"
    df_val, df_test = df_val.copy(), df_test.copy()

    unscaled_cols = [col+"_unscaled" for col in pred_cols]
    prediction_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
    unscaled_pred_cols = [col+"_unscaled" for col in prediction_cols]

    # Unscale the prediction columns
    df_val[unscaled_cols] = scaler.inverse_transform(df_val[pred_cols])
    df_test[unscaled_cols] = scaler.inverse_transform(df_test[pred_cols])
    df_val[unscaled_pred_cols] = scaler.inverse_transform(df_val[prediction_cols])
    df_test[unscaled_pred_cols] = scaler.inverse_transform(df_test[prediction_cols])

    # Set K = sqrt of size of validation set
    k = round(np.sqrt(len(df_val)))

    # Get reconstruction errors
    reconstruction_cols = [col+"_reconstruction_"+regressor_label for col in predictors]
    valid_re = np.abs(df_val[predictors].values - df_val[reconstruction_cols].values)
    test_re = np.abs(df_test[predictors].values - df_test[reconstruction_cols].values)

    # Find neighbours
    n_val = len(valid_re)
    kdtree = KDTree(valid_re)
    dist, ind = kdtree.query(test_re, k=k, workers=-1)

    for col in pred_cols:
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col+"_unscaled"].values.astype('float32')
        val_y_pred = df_val[col+pred_label+"_"+regressor_label+"_unscaled"].values.astype('float32')
        val_pe = np.abs(val_y-val_y_pred)

        # Nearbouring prediction errors
        neighbour_pe = val_pe[ind]

        # PI = 0.95 Percentile of neigbouring errors
        df_test[col+"_"+ue_col+pi_label] = np.quantile(neighbour_pe, np.ceil((k+1)*(1-alpha))/k, method='higher')
        #np.percentile(neighbour_pe, (1-alpha/2)*100, axis=1)
    
    return df_test

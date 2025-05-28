import numpy as np
from scipy.spatial import KDTree

def knn_prediction_interval(
    df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, 
    seed, alpha=0.05):
    pi_label = "_knn"
    df_val, df_test = df_val.copy(), df_test.copy()

    # Set K = sqrt of size of validation set
    k = max(round(np.sqrt(len(df_val))), 50)

    # Get reconstruction errors
    reconstruction_cols = [col+"_reconstruction_"+regressor_label for col in predictors]
    valid_re = np.abs(df_val[predictors].values - df_val[reconstruction_cols].values)
    test_re = np.abs(df_test[predictors].values - df_test[reconstruction_cols].values)

    # Find neighbours
    n_val = len(valid_re)
    kdtree = KDTree(valid_re)
    dist, ind = kdtree.query(test_re, k=k, workers=-1)

    lb_cols, ub_cols = [], []
    for col in pred_cols:
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col].values.astype('float32') # +"_unscaled"
        val_y_pred = df_val[col+pred_label+"_"+regressor_label].values.astype('float32') # "_unscaled"
        test_y_pred = df_test[col+pred_label+"_"+regressor_label].values.astype('float32') # "_unscaled"
        val_pe = np.abs(val_y-val_y_pred)

        # Nearbouring prediction errors
        neighbour_pe = val_pe[ind]

        # PI = 0.95 Percentile of neigbouring errors
        pi = np.quantile(neighbour_pe, np.ceil((k+1)*(1-alpha))/k, method='higher')
        
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

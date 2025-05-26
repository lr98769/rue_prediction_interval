import numpy as np

def conformal_prediction_interval(df_val, df_test, scaler, predictors, pred_cols, pred_label, regressor_label, ue_col, alpha=0.05, seed=seed):
    pi_label = "_conformal"
    df_val, df_test = df_val.copy(), df_test.copy()
    unscaled_cols = [col+"_unscaled" for col in pred_cols]
    prediction_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
    unscaled_pred_cols = [col+"_unscaled" for col in prediction_cols]
    df_val[unscaled_cols] = scaler.inverse_transform(df_val[pred_cols])
    df_test[unscaled_cols] = scaler.inverse_transform(df_test[pred_cols])
    df_val[unscaled_pred_cols] = scaler.inverse_transform(df_val[prediction_cols])
    df_test[unscaled_pred_cols] = scaler.inverse_transform(df_test[prediction_cols])

    for col in pred_cols:
        # 1. Val df
        # Get error for each variable
        val_y = df_val[col+"_unscaled"].values.astype('float32')
        val_y_pred = df_val[col+pred_label+"_"+regressor_label+"_unscaled"].values.astype('float32')
        val_error = np.abs(val_y-val_y_pred)
    
        # Get uncertainty 
        val_ue = df_val[ue_col].astype('float32')

        # Get scores
        val_scores = val_error/val_ue
        # Get the score quantile
        n = len(df_val)
        qhat = np.quantile(val_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')

        # 2. Test df
        # Get uncertainty 
        test_ue = df_test[ue_col].astype('float32')

        # Caclulate PI
        df_test[col+"_"+ue_col+pi_label] = test_ue*qhat
    
    return df_test
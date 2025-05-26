from src.misc import set_seed


import numpy as np


def model_test_predictions(
    ae_regressor, df_train, df_test, pred_cols, predictors, regressor_label, pred_min, seed, T=10):
    df_test = df_test.copy()
    set_seed(seed)
    test_X, test_y = (
        df_test[predictors].values.astype('float32'), df_test[pred_cols].values.astype('float32'))

    # MAE
    test_y_pred, test_X_reconstruction = ae_regressor.predict(inputs=test_X)
    # - Add to df
    predicted_colnames = [col+"_direct"+regressor_label for col in pred_cols]
    reconstruction_colnames = [col+"_reconstruction"+regressor_label for col in predictors]
    rue_colname = "rue"
    df_test[predicted_colnames] = test_y_pred
    df_test[reconstruction_colnames] = test_X_reconstruction
    df_test[rue_colname] = np.mean(np.abs(test_X_reconstruction-test_X), axis=1)

    # For MC Dropout
    # - Sample with dropout
    test_y_sample_preds = []
    for i in range(T):
        test_y_pred = ae_regressor.predict(inputs=test_X, dropout_activated=True, with_mae=False)
        test_y_sample_preds.append(test_y_pred)
    test_y_sample_preds = np.array(test_y_sample_preds)
    test_y_sample_preds =  test_y_sample_preds.transpose((1, 2, 0))
    # - Get mean and std of all sample predictions
    test_y_mean_pred = np.mean(test_y_sample_preds, axis=-1)
    test_y_std_pred = np.std(test_y_sample_preds, axis=-1, ddof=1)

    # - Add to df
    predicted_mean_colnames = [col+"_mean"+regressor_label for col in pred_cols]
    predicted_std_colnames = [col+"_std"+regressor_label for col in pred_cols]
    mcd_colname = "mcd"
    df_test[predicted_mean_colnames] = test_y_mean_pred
    df_test[predicted_std_colnames] = test_y_std_pred
    df_test[mcd_colname] = np.mean(test_y_std_pred, axis=-1)

    # if df_test['target_index'].dtype != "int64":
    #     df_test['target_index'] = df_test['target_index'].apply(lambda x: eval(x)[pred_min-1])

    return df_test
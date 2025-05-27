from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics import auc
import numpy as np
from IPython.display import display

from src.df_display.highlight_df import highlight_first_n_second_lowest, highlight_first_n_second_highest

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def calculate_aurc(ue, loss):
    num_samples = len(ue)
    ue_loss_df = pd.DataFrame({"ue":ue, "loss":loss})
    ue_loss_df = ue_loss_df.sort_values(by="ue", ascending=True)
    ue_loss_df["cumulative_loss"] = ue_loss_df["loss"].expanding().mean()
    ue_loss_df["coverage"] = (np.arange(num_samples)+1)/num_samples
    return auc(ue_loss_df["coverage"], ue_loss_df["cumulative_loss"].values)

def min_max_norm(vec):
    return (vec - vec.min()) / (vec.max() - vec.min())

def remove_outliers(vec):
    # vector within 3 std away from mean
    # data_mean, data_std = np.mean(vec), np.std(vec)
    # num_std = 3
    # return vec[(vec <= data_mean + num_std*data_std) & (vec >= data_mean - num_std*data_std)]

    Q1 = np.percentile(vec, 25, method= 'midpoint') 
    Q3 = np.percentile(vec, 75, method= 'midpoint') 
    IQR = Q3 - Q1 
    # low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    return vec[(vec <= up_lim )]
    
def min_max_norm_wo_outliers(vec):
    vec_wo_outliers = remove_outliers(vec)
    vec_min, vec_max = min(vec_wo_outliers), max(vec_wo_outliers)
    return (vec - vec_min) / (vec_max - vec_min)

def calculate_sigma_risk(ue, loss, thres):
    ue = min_max_norm_wo_outliers(ue)
    return loss[ue<=thres].mean()

def get_ue_performance_table(test_df_dict, ue_dict):
    perf_df_dict = []
    for ue_name, ue_info in ue_dict.items():
        ue_row_dict = {"Model": ue_name}
        pred_label = ue_info["pred_label"]
        ue_col = ue_info["ue_col"]
        for regressor_label, test_df_info in test_df_dict.items():
            test_df = test_df_info["test_df"]
            pred_cols = test_df_info["pred_cols"]
            y_pred_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
            
            y_true = test_df[pred_cols].values
            y_pred = test_df[y_pred_cols].values
            ues = test_df[ue_col].values
            
            mean_abs_errors = np.mean(np.abs(y_true-y_pred), axis=1)
            
            corr, p_value = pearsonr(ues, mean_abs_errors)
            aurc = calculate_aurc(ues, mean_abs_errors)

            ue_row_dict[regressor_label+" Corr"] = corr
            ue_row_dict[regressor_label+" Pval"] = p_value
            ue_row_dict[regressor_label+" AURC"] = aurc

            for thres in [0.1, 0.2, 0.3, 0.4]:
                ue_row_dict[regressor_label+f" Sigma={thres}"] = calculate_sigma_risk(ues, mean_abs_errors, thres)
            
        perf_df_dict.append(ue_row_dict)
    perf_df = pd.DataFrame(perf_df_dict)
    perf_df = perf_df.set_index("Model")
    return perf_df

def restructure_ue_df(ue_perf_df):
    ue_perf_df = ue_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    all_dfs = []
    for i in range(num_time):
        column_indices = list(range(i*num_metrics, (i+1)*num_metrics))
        cur_df = ue_perf_df.iloc[:,column_indices]
        cur_df.columns = cur_df.columns.str.split(" ").str[-1] # remove time label from column names
        cur_df.loc[:,"Time Horizon"] = f"t+{i+1}"
        all_dfs.append(cur_df)
    all_dfs = pd.concat(all_dfs)
    all_dfs = all_dfs.reset_index().set_index(["Time Horizon", "Model"])
    return all_dfs

def display_ue_perf(ue_perf_df, consolidated=False):
    ue_perf_df = ue_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    for time_horizon, cur_df in ue_perf_df.groupby(level="Time Horizon"):
        print(time_horizon)
        display(
            cur_df.style.apply(
                highlight_first_n_second_highest, subset=cur_df.columns[0], split=consolidated).apply(
                    highlight_first_n_second_lowest, subset=cur_df.columns[1:], split=consolidated
                )
        )


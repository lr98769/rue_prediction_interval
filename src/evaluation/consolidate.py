import pandas as pd
import numpy as np
from os.path import join
from IPython.display import display

def combine_mean_n_std_matrices(mean, std, sp=3):
    assert mean.shape == std.shape
    shape = mean.shape
    returned_list = []
    for i in range(shape[0]):
        cur_list = []
        for j in range(shape[1]):
            cur_list.append(f"{round(mean[i][j],sp)} ± {std[i][j]:.3f}")
        returned_list.append(cur_list)
    return returned_list

def get_mean_std_of_all_seed_csvs(seed_list, fp_folder, filename, sp=3, reindex=None):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        if reindex is not None:
            df = df.reset_index()
            df = df.set_index(reindex)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combine_mean_n_std_matrices(combined_mean, combined_std, sp=sp), 
        index=df.index, columns=df.columns)

def get_mean_of_all_seed_csvs(seed_list, fp_folder, filename):
    result_list = []
    for cur_seed in seed_list:
        fp_perf = join(fp_folder, str(cur_seed), filename)
        df = pd.read_csv(fp_perf, index_col=0)
        result_list.append(df.values)
    results = np.array(result_list)
    combined_mean = np.mean(results, axis=0)
    return pd.DataFrame(
        combined_mean, 
        index=df.index, columns=df.columns)
    
def consolidate_pred_perf(seed_list, fp_evaluation):
    return get_mean_std_of_all_seed_csvs(
        seed_list, fp_evaluation, filename="pred_perf.csv", sp=4)
    
def consolidate_ue_perf(seed_list, fp_evaluation, exclude_columns=None):
    output_df = get_mean_std_of_all_seed_csvs(
        seed_list, fp_evaluation, filename="ue_perf.csv", 
        reindex=["Time Horizon","Model"], sp=3)
    if exclude_columns is not None:
        output_df = output_df.drop(columns=exclude_columns)
    return output_df
    
def consolidate_pi_perf(seed_list, fp_evaluation, exclude_columns=None):
    output_df = get_mean_std_of_all_seed_csvs(
        seed_list, fp_evaluation, filename="pi_perf.csv", 
        reindex=["Time Horizon","Method"], sp=5)
    if exclude_columns is not None:
        output_df = output_df.drop(columns=exclude_columns)
    return output_df


    


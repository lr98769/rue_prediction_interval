import matplotlib.pyplot as plt
from copy import deepcopy
from os.path import exists, join
import pandas as pd
from tqdm.auto import tqdm

from src.evaluation.pi_eval import get_pi_performance_table
from src.file_processing.processing_predictions import load_pi_df_dict
from src.pi_methods.knn import knn_prediction_interval
from src.file_processing.processing_predictions import save_pi_df_dict

def knn_ablation(pred_df_dict, k_list, split_dict, scaler, seed, ue_dict, fp_cur_ablation_folder, metric="CWFDC"):
    ablation_result_df = {}
    # Get pred label and ue columns to use
    ue_label = "RUE"
    pred_label, ue_col = ue_dict[ue_label]["pred_label"], ue_dict[ue_label]["ue_col"]
    # pi to evaluate
    pi_dict = {
        "RUE KNN": {
            "pred_label": "_direct", "ue_col": "rue", "pi_label": "_knn"
        },
    }
    # iterate through each k to evaluate the PI at each prediction horizon
    with tqdm(k_list) as pbar:
        for k in pbar:
            pbar.set_description(str(k))
            fp_k_folder = join(fp_cur_ablation_folder, str(k))
            pi_df_dict = deepcopy(pred_df_dict)
            
            # Get PI
            fp_pi_pred_folder = join(fp_k_folder, "pi_pred")
            if not exists(fp_pi_pred_folder):
                for time_label, time_info in tqdm(pi_df_dict.items()):
                    val_df, test_df, pred_cols = time_info["valid_df"], time_info["test_df"], time_info["pred_cols"]
                    pi_df_dict[time_label]["test_df"] = knn_prediction_interval(
                        df_val=val_df, df_test=test_df, 
                        predictors=split_dict["feat_cols"], pred_cols=pred_cols, 
                        pred_label=pred_label, regressor_label=time_label, ue_col=ue_col, 
                        scaler=scaler, seed=seed, k=k
                    )
                save_pi_df_dict(pi_df_dict, fp_pi_pred_folder)
                
            # Evaluate PI
            fp_pi_perf = join(fp_k_folder, "pi_perf.csv")
            if not exists(fp_pi_perf):
                pi_df_dict = load_pi_df_dict(fp_pi_pred_folder)
                pi_perf_df = get_pi_performance_table(pi_df_dict, pi_dict)
                pi_perf_df.to_csv(fp_pi_perf)
            
            # Get metric
            # display(eval_df)
            eval_df = pd.read_csv(fp_pi_perf, index_col="Time Horizon")
            ablation_result_df[k] = eval_df[metric].tolist()
        
    knn_ablation_df = pd.DataFrame(ablation_result_df, index=eval_df.index)
    knn_ablation_df.loc["Mean",:] = knn_ablation_df.mean(axis=0).tolist()
    return knn_ablation_df


def display_knn_ablation(knn_ablation_df, dpi=300, metric="CWFDC"):
    knn_ablation_df = knn_ablation_df.copy()
    nrows = len(knn_ablation_df)
    ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*1), dpi=dpi, sharex=True)
    x = knn_ablation_df.columns
    axes[0].set_title(metric)
    for i, ax in enumerate(axes):
        ax.plot(x, knn_ablation_df.iloc[i, :], label=knn_ablation_df.index[i])
        ax.set_ylabel(knn_ablation_df.index[i])
    ax.set_xlabel("k")
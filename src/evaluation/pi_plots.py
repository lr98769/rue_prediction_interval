from math import ceil
import matplotlib.pyplot as plt
from os.path import join


from src.misc import create_folder

def plot_predictions_with_pi_across_methods(
    df_dict, pi_dict, 
    fp_cur_evaluation_folder, record,
    num_cols = 5, display_feature="ABPsys (mmHg)", regressor_label="t+3",
    dpi=300, save_fig=False, ylim=None, record_col="record_id",
    time_col="time"
    ):
    fp_fig_folder = join(fp_cur_evaluation_folder, "pi_line_graphs")
    create_folder(fp_fig_folder)
    
    num_methods = len(pi_dict)
    num_rows = ceil(num_methods/num_cols)
    
    # Plot for all methods
    fig, axes = plt.subplots(
        num_rows, num_cols, dpi=dpi, figsize=(4*num_cols, 3*num_rows),
        sharey=True, sharex="col"
    )
    if num_rows > 1:
        axes = axes.flatten()
        
    # Plots in multiples of three
    for i, (pi_name, pi_info) in enumerate(pi_dict.items()):
        ax = axes[i]
        # print(pi_name, ":")
        pred_label, ue_col, pi_label = pi_info["pred_label"], pi_info["ue_col"], pi_info["pi_label"]
        
        test_df_info = df_dict[regressor_label]

        test_df = test_df_info["test_df"]
        test_record_df = test_df.loc[test_df[record_col]==record]
        pred_cols = test_df_info["pred_cols"]
        
        # print(pred_cols)
        # print(features)

        #     # Sort columns for display
        pred_cols.sort()

        for j, pred_col in enumerate(pred_cols):
            feature = pred_col.split("_")[0]
            if feature != display_feature:
                continue
            # print(feature)
            # print(pred_col)
            y_pred_col = pred_col+pred_label+"_"+regressor_label

            index = test_record_df[time_col]
            y_true = test_record_df[pred_col+"_unscaled"].values
            y_pred = test_record_df[y_pred_col+"_unscaled"].values
            pi = test_record_df[pred_col+"_"+ue_col+pi_label].values
            
            # Plot predictions and their CI
            ax.plot(index, y_true, color="blue")
            ax.plot(index, y_pred, color="red", alpha=0.8)
            ax.fill_between(
                index, (y_pred-pi), (y_pred+pi), 
                color='red', alpha=0.3
            )  
            if i%num_cols==0:
                ax.set_ylabel(feature)
            ax.set_title(pi_name)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if ylim is not None:
                ax.set_ylim(*ylim)
    plt.tight_layout()
    if save_fig:
        plt.savefig(join(fp_fig_folder, f"pi_comparison_across_methods_{display_feature}_{regressor_label}.jpg"), bbox_inches="tight")
    plt.show()
    

from matplotlib.font_manager import FontProperties
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import numpy as np
from matplotlib.colors import PowerNorm

from src.evaluation.consolidate import get_mean_of_all_seed_csvs

def generate_pi_heatmap(
    seed_list, fp_evaluation, fp_consolidated, categories=["CovP", "PINAW", "PINAFD", "CWFDC"], methods=None, save_fig=False, 
    legend_ncol=5, legend_fs=16, title_fs=16, xlabel_fs=16, ylabel_fs=16, xtick_fs=16, ytick_fs=16,
    width=15, height=6, bolded_methods=[], pi_order=None, gamma=1
):
    
    df = get_mean_of_all_seed_csvs(seed_list, fp_evaluation, filename="pi_perf.csv", reindex=["Time Horizon","Method"]).reset_index()
    
    if pi_order is not None:
        df = df.set_index(["Time Horizon", "Method"])
        new_df = []
        for time, cur_df in df.groupby(level="Time Horizon"):
            new_df.append(df.loc[(time, pi_order), :])
        df = pd.concat(new_df).reset_index()
    
    df.columns = ["CovP" if col == "CP" else col for col in df.columns]
    if methods:
        df = df[df["Method"].isin(methods)]
    else:
        methods = df["Method"].unique()

    # Define bold font
    bold_font = FontProperties(weight='bold')
    
    df = pd.melt(
        df, 
        id_vars=['Time Horizon', "Method"], 
        value_vars=categories, 
        var_name="Metric", 
        value_name="Value"
    )
    df["Metric"] = df["Metric"].astype(str) + " (↓)"
    
    df["Value"] = np.log10(df["Value"])
    df["Value"] = df.groupby(["Metric", "Time Horizon"])["Value"].transform(
        lambda x:  (x - x.min()) / (x.max() - x.min()) # (x - x.mean()) / x.std()
    )
    df = df.sort_values(by=["Metric", "Time Horizon", "Value"])
    vmin, vmax = df["Value"].min(), df["Value"].max()

    # display(df)
    metrics = [col + " (↓)" for col in categories]
    
    # Plot subplots of heatmaps for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, height), dpi=300) # sharey=True
    
    for i, metric in enumerate(metrics):
        # Filter data for the current metric
        metric_data = df[df["Metric"] == metric]
        
        # display(metric_data)
        heatmap_data = metric_data.pivot_table(
            index="Method", 
            columns="Time Horizon", 
            values="Value"
        )
        heatmap_data = heatmap_data.reindex(methods)
    
        # Create heatmap for the current metric
        sns.heatmap(
            heatmap_data, 
            ax=axes[i], 
            annot=True, 
            fmt=".3f", 
            cmap='coolwarm',  # Use custom colormap
            linewidths=0.5, 
            vmin=vmin, 
            vmax=vmax,  # Ensure the same scale across subplots
            cbar=i == len(metrics) - 1,  # Add a color bar only to the last subplot
            # cbar_kws={"label": "Value"} if i == len(metrics) - 1 else None
            norm=PowerNorm(gamma=gamma) # skew towards smaller values
        )
       
        axes[i].set_title(metric, size=title_fs)
        axes[i].set_xlabel("Time Horizon", size=xlabel_fs)
        if i != 0:
            axes[i].set_ylabel("")
            axes[i].set_yticks([])
        else:
            axes[i].set_ylabel("Method", size=ylabel_fs)
            for label in axes[i].get_yticklabels():
                if label.get_text() in bolded_methods:
                    label.set_fontproperties(bold_font)

        axes[i].tick_params(axis='x', labelsize=xtick_fs)
        axes[i].tick_params(axis='y', labelsize=ytick_fs)
        if i == len(metrics)-1:
            axes[i].tick_params(axis="both", labelsize=legend_fs)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(join(fp_consolidated, f"pi_heatmap.jpg"), bbox_inches="tight")
    plt.show()
    
    
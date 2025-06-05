from matplotlib.font_manager import FontProperties
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.cm import ScalarMappable

from src.evaluation.consolidate import get_mean_of_all_seed_csvs, get_std_of_all_seed_csvs

def reorder_pi(df, pi_order):
    df = df.set_index(["Time Horizon", "Method"])
    new_df = []
    for time, cur_df in df.groupby(level="Time Horizon"):
        new_df.append(df.loc[(time, pi_order), :])
    df = pd.concat(new_df).reset_index()
    return df


def melt_n_min_max_norm(mean_df, std_df, metrics):
    mean_df = pd.melt(
        mean_df, 
        id_vars=['Time Horizon', "Method"], 
        value_vars=metrics, 
        var_name="Metric", 
        value_name="Value"
    )
    std_df = pd.melt(
        std_df, 
        id_vars=['Time Horizon', "Method"], 
        value_vars=metrics, 
        var_name="Metric", 
        value_name="Value"
    )
    mean_df["Metric"] = mean_df["Metric"].astype(str) + " (↓)"
    std_df["Metric"] = std_df["Metric"].astype(str) + " (↓)"
    new_mean_df, new_std_df = [], []
    mean_df_grp = mean_df.groupby(["Metric", "Time Horizon"])
    std_df_grp = std_df.groupby(["Metric", "Time Horizon"])
    for grp, cur_mean_df in mean_df.groupby(["Metric", "Time Horizon"]):
        # min max scale the mean
        cur_min, cur_max = cur_mean_df["Value"].min(), cur_mean_df["Value"].max()
        cur_mean_df["Value"] = (cur_mean_df["Value"]-cur_min)/(cur_max-cur_min)
        new_mean_df.append(cur_mean_df)
        
        # min max scale the std
        cur_std_df = std_df_grp.get_group(grp)
        cur_std_df["Value"] = (cur_std_df["Value"]) * (1/(cur_max-cur_min))
        new_std_df.append(cur_std_df)
    mean_df = pd.concat(new_mean_df)
    std_df = pd.concat(new_std_df)
    # mean_df["Value"] = mean_df.groupby(["Metric", "Time Horizon"])["Value"].transform(
    #     lambda x:  (x - x.min()) / (x.max() - x.min()) # (x - x.mean()) / x.std()
    # )
    return mean_df, std_df

def get_heatmap_annotations(metric_mean_df, metric_std_df, methods, dp):
    metric_mean_df = metric_mean_df.rename(columns={"Value":"mean"})
    metric_std_df = metric_std_df.rename(columns={"Value":"std"})
    joined_df = metric_mean_df.merge(metric_std_df, on=["Time Horizon", "Method", "Metric"])
    joined_df["annotation"] = (
        joined_df["mean"].apply(lambda x: f'{x:.{dp}f}') + "\n± " +
        joined_df["std"].apply(lambda x: f'{x:.{dp}f}')
    )
    annotation_data = joined_df.pivot_table(
            index="Method", 
            columns="Time Horizon", 
            values="annotation",
            aggfunc=lambda x: ' '.join(x)
        )
    annotation_data = annotation_data.reindex(methods)
    return annotation_data

def generate_pi_heatmap(
    seed_list, fp_evaluation, fp_consolidated, 
    metrics=["CovP", "PINAW", "PINAFD", "CWFDC"], methods_to_drop=None, save_fig=False, 
    legend_fs=16, title_fs=16, xlabel_fs=16, ylabel_fs=16, xtick_fs=16, ytick_fs=16, annot_fs=12, 
    width=15, height=6, bolded_methods=[], pi_order=None, gamma=1, dp=3
):
    
    mean_df = get_mean_of_all_seed_csvs(seed_list, fp_evaluation, filename="pi_perf.csv", reindex=["Time Horizon","Method"]).reset_index()
    std_df = get_std_of_all_seed_csvs(seed_list, fp_evaluation, filename="pi_perf.csv", reindex=["Time Horizon","Method"]).reset_index()
    
    # This is to reorder methods
    if pi_order is not None:
        mean_df = reorder_pi(mean_df, pi_order)
        std_df = reorder_pi(std_df, pi_order)
    
    # This is to remove any methods
    if methods_to_drop:
        mean_df = mean_df[~mean_df["Method"].isin(methods_to_drop)]
        std_df = std_df[~std_df["Method"].isin(methods_to_drop)]
    
    methods = mean_df["Method"].unique()

    # Define bold font
    bold_font = FontProperties(weight='bold')
    
    scaled_mean_df, scaled_std_df = melt_n_min_max_norm(mean_df, std_df, metrics)
    del mean_df
    vmin, vmax = scaled_mean_df["Value"].min(), scaled_std_df["Value"].max()

    # display(df)
    metrics = [col + " (↓)" for col in metrics]
    
    cmap = plt.get_cmap('coolwarm') 
    norm = PowerNorm(gamma=gamma) # skew colors towards smaller values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar
    
    # Plot subplots of heatmaps for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, height), dpi=300) # sharey=True
    
    for i, metric in enumerate(metrics):
        # Filter data for the current metric
        metric_mean_df = scaled_mean_df[scaled_mean_df["Metric"] == metric]
        metric_std_df = scaled_std_df[scaled_std_df["Metric"] == metric]
        
        # Get heatmap data to plot
        heatmap_data = metric_mean_df.pivot_table(
            index="Method", 
            columns="Time Horizon", 
            values="Value"
        )
        heatmap_data = heatmap_data.reindex(methods)

        # Get heatmap annotations
        annotation = get_heatmap_annotations(metric_mean_df, metric_std_df, methods, dp)

        # Create heatmap for the current metric
        sns.heatmap(
            heatmap_data, 
            ax=axes[i], 
            annot=False, 
            fmt="",
            cmap=cmap,  # Use custom colormap
            linewidths=0.5, 
            vmin=vmin, 
            vmax=vmax,  # Ensure the same scale across subplots
            cbar=False,
            # cbar_kws={"label": "Value"} if i == len(metrics) - 1 else None
            norm=norm 
        )
        
        # Manually add styled annotations
        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                mean_val = metric_mean_df[
                    (metric_mean_df["Method"] == methods[y]) &
                    (metric_mean_df["Time Horizon"] == heatmap_data.columns[x])
                ]["Value"].values[0]

                std_val = metric_std_df[
                    (metric_std_df["Method"] == methods[y]) &
                    (metric_std_df["Time Horizon"] == heatmap_data.columns[x])
                ]["Value"].values[0]

                # Get background color from colormap
                bg_color = cmap(mean_val)

                # Convert RGBA to perceived brightness (luminance)
                r, g, b, _ = bg_color
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'black' if luminance > 0.45 else 'white'

                # Plot mean
                axes[i].text(
                    x + 0.5, y+0.12 + 0.45, 
                    f"{mean_val:.{dp}f}",
                    ha='center', va='bottom',
                    fontsize=annot_fs + 2,
                    # fontweight='bold',
                    color=text_color
                )

                # Plot std
                axes[i].text(
                    x + 0.5, y+0.12 + 0.5, 
                    f"± {std_val:.{dp}f}",
                    ha='center', va='top',
                    fontsize=annot_fs - 4,
                    color=text_color
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
            
    # Add the single colorbar outside the subplots
    cbar_ax = fig.add_axes([1, 0.1375, 0.015, 0.79])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # Remove the outline (spines) of the colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(join(fp_consolidated, f"pi_heatmap.jpg"), bbox_inches="tight")
    plt.show()
    
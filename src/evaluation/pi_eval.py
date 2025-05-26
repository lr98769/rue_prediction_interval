import pandas as pd
import numpy as np
from IPython.display import display

from src.display.highlight_df import highlight_first_n_second_lowest, highlight_first_n_second_highest


def calculate_picp_for_a_feature(actual, lb, ub):
    # PICP (Prediction interval coverage probability)
    # - The percentage of points within the prediction interval; The higher the better. 
    points_within_pi = ((actual >= lb) & (actual <= ub))
    num_points_within_pi = points_within_pi.sum()
    total_num_points = len(actual)
    picp = num_points_within_pi/total_num_points
    return picp

def calculate_pinaw_for_a_feature(actual, lb, ub):
    # Prediction interval normalized average width (PINAW)
    # - The average size of the prediction interval. 
    range_of_underlying_target = actual.max() - actual.min()
    total_num_points = len(actual)
    width_of_pi = ub-lb
    sum_of_widths_of_pi = width_of_pi.sum()
    pinaw = sum_of_widths_of_pi/(total_num_points*range_of_underlying_target)
    return pinaw

def calculate_pinafd_for_a_feature(actual, lb, ub):
    # Prediction interval normalized average failure distance (PINAFD)
    # - The average distance of points outside of the prediction interval to the bounds of the prediction interval. 
    range_of_underlying_target = actual.max() - actual.min()
    points_outside_pi = ((actual < lb) | (actual > ub))
    num_points_outside_pi = points_outside_pi.sum()
    if num_points_outside_pi == 0:
        return 0
    else:
        dist_lb = (lb-actual).abs()
        dist_ub = (ub-actual).abs()
        shortest_dist_to_any_b = np.minimum(dist_lb, dist_ub)
        failure_dist = shortest_dist_to_any_b[points_outside_pi]
        pinafd = failure_dist.sum()/(num_points_outside_pi*range_of_underlying_target)
        return pinafd
    
def calculate_cp_for_a_feature(actual, lb, ub):
    picp = calculate_picp_for_a_feature(actual, lb, ub)
    alpha = 0.05
    delta = alpha/50
    cp = (1-alpha+delta-picp)**2
    return cp
    
    
def calculate_cwfdc_for_a_feature(actual, lb, ub):
    pinaw = calculate_pinaw_for_a_feature(actual, lb, ub)
    pinafd = calculate_pinafd_for_a_feature(actual, lb, ub)
    cp = calculate_cp_for_a_feature(actual, lb, ub)
    rho = 1
    beta = 1000
    cwfdc = pinaw + rho*pinafd + beta * cp
    return cwfdc

def aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func):
    total_metric = 0
    pred_cols, df_test = df_info["pred_cols"], df_info["test_df"]
    pred_label, ue_col,  pi_label = pi_info["pred_label"], pi_info["ue_col"], pi_info["pi_label"]
    num_feat = len(pred_cols)
    for pred_col in pred_cols:
        pi_col = f"{pred_col}_{ue_col}{pi_label}"
        predicted_col = f"{pred_col}{pred_label}_{time_label}_unscaled"
        pi, pred = df_test[pi_col], df_test[predicted_col]
        total_metric += metric_func(
            actual=df_test[pred_col+"_unscaled"], 
            lb=pred-pi, 
            ub=pred+pi
        )
    av_metric = total_metric/num_feat
    return av_metric

def get_pi_performance_table(df_dict, pi_dict):
    output_df_list = []
    for time_label, df_info in df_dict.items():
        for pi_label, pi_info in pi_dict.items():
            picp = aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func=calculate_picp_for_a_feature)
            pinaw = aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func=calculate_pinaw_for_a_feature)
            pinafd = aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func=calculate_pinafd_for_a_feature)
            cp = aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func=calculate_cp_for_a_feature)
            cwfdc = aggregate_metrics_across_feats(df_info, pi_info, time_label, metric_func=calculate_cwfdc_for_a_feature)
            output_df_list.append({
                "Time Horizon":time_label,
                "Method":pi_label,
                "PICP":picp,
                "PINAW":pinaw,
                "PINAFD":pinafd,
                "CovP":cp,
                "CWFDC":cwfdc
            })
    output_df = pd.DataFrame(output_df_list)
    return output_df

def display_pi_perf(
    pi_perf_df, highest_cols=["PICP"], lowest_cols=["PINAW", "PINAFD", "CovP"], consolidated=False):
    pi_perf_df = pi_perf_df.copy()
    # Split df into time label
    num_time, num_metrics = 3, 7
    for time_horizen, cur_df in pi_perf_df.groupby(level="Time Horizon"):
        print(f"{time_horizen}:")
        display(
            cur_df.style.apply(
                highlight_first_n_second_highest, subset=highest_cols, split=consolidated).apply(
                    highlight_first_n_second_lowest, subset=lowest_cols, split=consolidated
                )
        )
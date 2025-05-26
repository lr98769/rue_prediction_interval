import pandas as pd
from sklearn.metrics import mean_squared_error
from IPython.display import display

from src.display.highlight_df import highlight_first_n_second_lowest

def get_prediction_performance_table(test_df_dict, ue_dict):
    perf_df_dict = []
    for ue_name, ue_info in ue_dict.items():
        ue_row_dict = {"Model": ue_name}
        pred_label = ue_info["pred_label"]
        for regressor_label, test_df_info in test_df_dict.items():
            test_df = test_df_info["test_df"]
            pred_cols = test_df_info["pred_cols"]
            y_pred_cols = [col+pred_label+"_"+regressor_label for col in pred_cols]
            # ue_cols = [col+ue_label for col in predictors]
            ue_row_dict[regressor_label] = mean_squared_error(
                test_df[pred_cols], test_df[y_pred_cols])
        perf_df_dict.append(ue_row_dict)
    perf_df = pd.DataFrame(perf_df_dict)
    perf_df = perf_df.set_index("Model")
    return perf_df

def display_pred_perf(pred_perf_df, consolidated=False):
    display(pred_perf_df.style.apply(highlight_first_n_second_lowest, split=consolidated))


    
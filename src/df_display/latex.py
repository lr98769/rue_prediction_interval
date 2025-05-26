import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Function to bold the best
def bold_best(s, direction):
    s = s.map(lambda x: float(x.replace(r"\underline{", "").replace("}", "").split(" ")[0]))
    if direction == "max":
        best = s.drop_duplicates().nlargest(1).iloc[-1] # Find the best
    elif direction == "min":
        best = s.drop_duplicates().nsmallest(1).iloc[-1] # Find the best
    else:
        raise Exception(f"Invalid direction {direction}!")
    
    output = []
    for v in s:
        if v == best:
            output.append("textbf:--rwrap;")
        else:
            output.append("")
    return output

# Function to underline the second best
def underline_second_best(s, direction):
    ori_s = s.copy()
    s = s.map(lambda x: float(x.split(" ")[0]))
    if direction == "max":
        second_best = s.drop_duplicates().nlargest(2).iloc[-1]  # Find the second best
    elif direction == "min":
        second_best = s.drop_duplicates().nsmallest(2).iloc[-1]  # Find the second best
    else:
        raise Exception(f"Invalid direction {direction}!")
    
    output = []
    for ori_v, v in zip(ori_s, s):
        if v == second_best:
            output.append(r'\underline{'+ori_v+'}')
        else:
            output.append(ori_v)
    return output


def df_to_latex(df, column_format_dict):
    df = df.copy()
    for col, direction in column_format_dict.items():
        df[col] = underline_second_best(df[col], direction)
    styler = df.style
    # Bold column names
    styler.map_index(lambda v: "textbf:--rwrap;", axis="columns")
    # Bold best
    for col, direction in column_format_dict.items():
        styler.apply(bold_best, subset=[col], direction=direction)
    return styler.to_latex(column_format='c'*(df.shape[1]+df.index.nlevels))

# Function to bold the best
def bold_best_grouped(s, direction):
    s = s.map(lambda x: float(x.replace(r"\underline{", "").replace("}", "").split(" ")[0]))
    if direction == "max":
        best = s.drop_duplicates().nlargest(1).iloc[-1] # Find the best
    elif direction == "min":
        best = s.drop_duplicates().nsmallest(1).iloc[-1] # Find the best
    else:
        raise Exception(f"Invalid direction {direction}!")
    
    output = []
    for v in s:
        if v == best:
            output.append("textbf:--rwrap;")
        else:
            output.append("")
    return output

# Function to underline the second best
def bold_best_underline_second_best_grouped(s, direction, group_col):
    s = s.copy()
    output = []
    for group_name, group_s in s.groupby(level=group_col):
        ori_group_s = group_s.copy()
        group_s = group_s.map(lambda x: float(x.split(" ")[0]))
        if direction == "max":
            best = group_s.drop_duplicates().nlargest(1).iloc[-1] # Find the best
            second_best = group_s.drop_duplicates().nlargest(2).iloc[-1]  # Find the second best
        elif direction == "min":
            best = group_s.drop_duplicates().nsmallest(1).iloc[-1] # Find the best
            second_best = group_s.drop_duplicates().nsmallest(2).iloc[-1]  # Find the second best
        else:
            raise Exception(f"Invalid direction {direction}!")
        for ori_v, v in zip(ori_group_s, group_s):
            if v == best:
                output.append(r'\textbf{'+ori_v+'}')
            elif v == second_best:
                output.append(r'\underline{'+ori_v+'}')
            else:
                output.append(ori_v)
    return output

def df_to_latex_grouped(ue_perf_df, column_format_dict, group_col="Time Horizon"):
    for col, direction in column_format_dict.items():
        ue_perf_df[col] = bold_best_underline_second_best_grouped(
            ue_perf_df[col], direction , group_col=group_col)
    return ue_perf_df.to_latex(column_format='c'*(ue_perf_df.shape[1]+ue_perf_df.index.nlevels))


import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import pingouin as pg
from tqdm.auto import tqdm
from os.path import join
from scipy.stats import combine_pvalues, kstest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


from src.evaluation.consolidate import combine_mean_n_std_matrices
from src.file_processing.processing_predictions import load_pi_df_dict

def check_inter_point_distance(re):
    # Justify KNN
    return pdist(re).mean()

def check_distance_meaningfulness(re, pe):
    # Justify weighted
    re_scaler = StandardScaler()
    pca = PCA(whiten=True)
    re = re_scaler.fit_transform(re)
    re = pca.fit_transform(re)
    re_dist = squareform(pdist(re))
    all_corr = []
    for i in range(pe.shape[1]):
        cur_pe = pe[:, i]
        pe_dist = np.abs(cur_pe[:, None] - cur_pe[None, :])
        corr = pearsonr(re_dist.flatten(), pe_dist.flatten()).statistic
        all_corr.append(corr)
    return np.array(all_corr).mean()

def check_marginals_are_gaussian(re, pe):
    # Justify Gaussian
    X = np.concatenate((re, pe), axis=1)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    pvals = [kstest(X_std[:, i], 'norm').pvalue for i in range(X_std.shape[1])]
    return combine_pvalues(pvals).pvalue

def check_multivariate_normality(re, pe):
    # Justify Gaussian Copula
    X = np.concatenate((re, pe), axis=1)
    res = pg.multivariate_normality(X, alpha=0.05)
    return res.hz, res.pval

def investigate_re_n_pe(pred_df_dict, split_dict, pred_label="_direct"):
    predictors = split_dict["feat_cols"]
    all_dist = []
    index = []
    for time_label, time_info in tqdm(pred_df_dict.items()):
        actual_cols = split_dict["target_cols"][time_label]
        pred_cols = [col+pred_label+"_"+time_label for col in actual_cols]
        reconstruction_cols = [col+"_reconstruction"+"_"+time_label for col in predictors]
        
        cur_dict = {}
        for name, df in time_info.items():
            if "df" in name:
                re = np.abs(df[predictors].values - df[reconstruction_cols].values)
                pe = np.abs(df[actual_cols].values-df[pred_cols].values)
                cur_dict["dist\n"+name] = check_inter_point_distance(re)
                cur_dict["dist meaning\n"+name] = check_distance_meaningfulness(re, pe)
                cur_dict["marginal gauss\n"+name] = check_marginals_are_gaussian(re, pe)
                hz, pval = check_multivariate_normality(re, pe)
                cur_dict["normality pval "+name] = pval
                cur_dict["normality stat "+name] = hz
                
        all_dist.append(cur_dict)
        index.append(time_label)
    output_df = pd.DataFrame(all_dist, index=index)
    output_df = output_df[output_df.columns.sort_values()]
    return output_df

def investigate_re_n_pe_for_all_seeds(seed_list, split_dict, fp_pi_predictions):
    results = []
    for seed in tqdm(seed_list):
        fp_cur_pi_predictions_folder = join(fp_pi_predictions, str(seed))
        pi_df_dict = load_pi_df_dict(fp_cur_pi_predictions_folder)
        df = investigate_re_n_pe(pi_df_dict, split_dict=split_dict)
        results.append(df.values)
    combined_mean = np.mean(results, axis=0)
    combined_std = np.std(results, axis=0)
    return pd.DataFrame(
        combine_mean_n_std_matrices(combined_mean, combined_std, sp=5), 
        index=df.index, columns=df.columns)
    
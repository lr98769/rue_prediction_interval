ue_dict = {
    "RUE": {"pred_label": "_direct", "ue_col": "rue"},
    "MC Dropout": {"pred_label": "_mean", "ue_col": "mcd", },
    "GPR": {"pred_label": "_gpr", "ue_col": "gpr_std_mean", },
    "Infer-Noise": {"pred_label": "_infernoise", "ue_col": "infernoise_uncertainty", },
    # "BNN": {"pred_label": "_bnn", "ue_col": "bnn_uncertainty", },
    "DER": {"pred_label": "_der", "ue_col": "der_uncertainty", }
}   

pi_dict = {
    "RUE Gaussian Copula": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_gauss_copula"
    },
    "RUE Conditional Gaussian": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_cond_gauss"
    },
    "RUE Weighted": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_weighted"
    },
    "RUE KNN": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_knn"
    },
    "RUE Conformal": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_conformal"
    },
    "Infer-Noise Conformal": {
        "pred_label": "_infernoise", "ue_col": "infernoise_uncertainty", "pi_label": "_conformal"
    },
    "MC Dropout Conformal": {
        "pred_label": "_mean", "ue_col": "mcd", "pi_label": "_conformal"
    },
    "GPR Conformal": {
        "pred_label": "_gpr", "ue_col": "gpr_std_mean", "pi_label": "_conformal"
    },
    # "BNN Conformal": {
    #     "pred_label": "_bnn", "ue_col": "bnn_uncertainty", "pi_label": "_conformal"
    # },
    "DER Conformal": {
        "pred_label": "_der", "ue_col": "der_uncertainty", "pi_label": "_conformal"
    },
}   

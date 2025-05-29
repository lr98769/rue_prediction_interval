ue_dict = {
    "RUE": {"pred_label": "_direct", "ue_col": "rue"},
    "MC Dropout": {"pred_label": "_mean", "ue_col": "mcd", },
    "GPR": {"pred_label": "_gpr", "ue_col": "gpr_std_mean", },
    "Infer-Noise": {"pred_label": "_infernoise", "ue_col": "infernoise_uncertainty", },
    "BNN": {"pred_label": "_bnn", "ue_col": "bnn_uncertainty", },
    "DER": {"pred_label": "_der", "ue_col": "der_uncertainty", }
}   

pi_dict = {
    "RUE Gauss Copula": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_gauss_copula"
    },
    "RUE Cond Gauss": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_cond_gauss"
    },
    "RUE Weighted": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_weighted"
    },
    "RUE KNN": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_knn"
    },
    "RUE CP": {
        "pred_label": "_direct", "ue_col": "rue", "pi_label": "_conformal"
    },
    "IN CP": {
        "pred_label": "_infernoise", "ue_col": "infernoise_uncertainty", "pi_label": "_conformal"
    },
    "MCD CP": {
        "pred_label": "_mean", "ue_col": "mcd", "pi_label": "_conformal"
    },
    "GPR CP": {
        "pred_label": "_gpr", "ue_col": "gpr_std_mean", "pi_label": "_conformal"
    },
    "BNN CP": {
        "pred_label": "_bnn", "ue_col": "bnn_uncertainty", "pi_label": "_conformal"
    },
    "DER CP": {
        "pred_label": "_der", "ue_col": "der_uncertainty", "pi_label": "_conformal"
    },
}   

pi_order = [
    "RUE Gauss Copula", "RUE KNN", "RUE Cond Gauss", "RUE Weighted", 
    "RUE CP", "IN CP", "MCD CP", "GPR CP", "BNN CP", "DER CP"]

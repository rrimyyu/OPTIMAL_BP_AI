import numpy as np
import shap

from ._data_preprocessing import load_data, data_preprocessing

def plot_of_SHAP(model, X, y_bp_mrs):
    if 'multi' in X.columns:
        X = X.drop(columns=['multi'])

    X = np.array(X)
    explainer = shap.DeepExplainer(model, X)

    return explainer

def find_explainer(model):
    cs_cln_df, is_cln_df = load_data()
    cs_cln_df, is_cln_df, bp_cln_df = data_preprocessing(cs_cln_df, is_cln_df)

    x_df = bp_cln_df.drop(['multi', 'optimal_bp_reg_no', 'mRS_3months'], axis = 1)

    model_bp = model

    explainer = shap.DeepExplainer(model_bp, np.array(x_df))

    return explainer
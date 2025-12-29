import numpy as np
import shap
import pickle

def plot_of_SHAP(model, X, y_bp_mrs):
    if 'multi' in X.columns:
        X = X.drop(columns=['multi'])

    X = np.array(X)
    explainer = shap.DeepExplainer(model, X)

    return explainer

def find_explainer(model):
    model_bp = model

    summary_data = np.load("/home/ec2-user/OPTIMAL_BP_AI/optimal_ai_project/"
                           "optimal_ai_project/model/shap-background.npy")

    explainer = shap.DeepExplainer(model_bp, np.array(summary_data))

    return explainer
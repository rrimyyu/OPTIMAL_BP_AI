import numpy as np
import shap
import matplotlib.pyplot as plt

def plot_of_SHAP(model, _X, filename):
    if 'multi' in _X.columns:
        _X = _X.drop(columns=['multi'])

    X_cols = [c for c in _X.columns]

    if len(X_cols) == 20:
        feature_names = ['Age', 'Sex', 'Hypertension', 'Hyperlipidemia', 'Smoking', 'Previous Stroke',
                         'CAOD', 'Active Cancer', 'Congestive Heart Failure', 'PAOD', 'NIHSS Score',
                         'Onset to Registration', 'IV tPA', 'DM', 'Atrial Fibrillation', 'Antiplatelet',
                         'Anticoagulant', 'Hemoglobin', 'White Blood Cell', 'Body Mass Index']

    else :
        feature_names = ['Age', 'Sex', 'Hypertension', 'Hyperlipidemia', 'Smoking', 'Previous Stroke',
                         'CAOD', 'Active Cancer', 'Congestive Heart Failure', 'PAOD', 'NIHSS Score',
                         'Onset to Registration', 'IV tPA', 'SBP Enroll', 'DM', 'Atrial Fibrillation',
                         'Antiplatelet', 'Anticoagulant', 'Hemoglobin', 'White Blood Cell', 'Body Mass Index',
                         'Group', 'SBP Max', 'SBP Min', 'SBP Mean', 'SBP Time Rate', 'SBP Standard Deviation',
                         'SBP Coefficient of Variation', 'SBP Variation Independent of the Mean']

    X = np.array(_X)
    explainer = shap.DeepExplainer(model, X)

    shap_values = explainer.shap_values(X)
    shap_values = np.squeeze(shap_values)

    plt.figure(figsize=(5, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="violin", max_display=10, show=False)
    plt.savefig(filename, format='svg')
    plt.show()

    return explainer

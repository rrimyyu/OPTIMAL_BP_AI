import numpy as np
import pandas as pd

from data_preprocessing import load_data, data_preprocessing, remove_bp_features
from model_creation import create_deep_neural_network
from model_evaluation import evaluate_model
from SHAP import plot_of_SHAP
from compare_auc_delong import delong_roc_test

cs_cln_df, is_cln_df = load_data()
cs_cln_df, is_cln_df, bp_cln_df = data_preprocessing(cs_cln_df, is_cln_df)

x_df = cs_cln_df.drop(['multi', 'optimal_bp_reg_no', 'mRS_3months'], axis = 1)
y_mrs_df = cs_cln_df['mRS_3months']

shuffled_index = np.random.permutation(x_df.index)
n_train = int(0.7 * len(shuffled_index))
_X_train, _X_valid = x_df.loc[shuffled_index[:n_train]], x_df.loc[shuffled_index[n_train:]]
y_train_mrs, y_valid_mrs = y_mrs_df.loc[shuffled_index[:n_train]], y_mrs_df.loc[shuffled_index[n_train:]]

X_train = remove_bp_features(_X_train)
X_valid = remove_bp_features(_X_valid)

model = create_deep_neural_network(X_train)
history = model.fit(X_train, y_train_mrs, epochs=300, batch_size=64, validation_data=(X_valid, y_valid_mrs))
df, y_pred_proba_train, y_pred_proba_valid = evaluate_model(model, X_train, y_train_mrs, X_valid, y_valid_mrs)

X_bp_train = _X_train
X_bp_valid = _X_valid
y_bp_train_mrs = y_train_mrs
y_bp_valid_mrs = y_valid_mrs

model_bp = create_deep_neural_network(X_bp_train)
history_bp = model_bp.fit(X_bp_train, y_bp_train_mrs, epochs=300, batch_size=64, validation_data=(X_bp_valid, y_bp_valid_mrs))
df_bp, y_bp_pred_proba_train, y_bp_pred_proba_valid = evaluate_model(model_bp, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

p_value_train = delong_roc_test(y_train_mrs.to_numpy().reshape(len(y_train_mrs),), y_pred_proba_train, y_bp_pred_proba_train)
p_value_valid = delong_roc_test(y_valid_mrs.to_numpy().reshape(len(y_valid_mrs),), y_pred_proba_valid, y_bp_pred_proba_valid)

# visualize_roc_curve(model, X_train, y_train_mrs, model_bp, X_bp_train, p_value_train, y_bp_train_mrs, "ROC_CURVE_TRAIN_CONV.svg")
# visualize_roc_curve(model, X_valid, y_valid_mrs, model_bp, X_bp_valid, p_value_valid, y_bp_valid_mrs, "ROC_CURVE_VALID_CONV.svg")

plot_of_SHAP(model, X_valid, "CONV_SHAP_CLN.svg")
plot_of_SHAP(model_bp, X_bp_valid, "CONV_SHAP_ADDED.svg")

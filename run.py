import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from data_preprocessing import load_data, data_preprocessing, remove_bp_features
from model_creation import create_deep_neural_network
from model_evaluation import evaluate_model, k_fold_cross_validation, visualize_roc_curve, visualize_loss_and_accuracy, visualize_roc_comparison
from SHAP import plot_of_SHAP
from compare_auc_delong import delong_roc_test

cs_cln_df, is_cln_df = load_data()
cs_cln_df, is_cln_df, bp_cln_df = data_preprocessing(cs_cln_df, is_cln_df)

# Clinical only case :
x_df = bp_cln_df.drop(['multi', 'optimal_bp_reg_no', 'mRS_3months'], axis = 1)
y_mrs_df = bp_cln_df['mRS_3months']

shuffled_index = np.random.permutation(x_df.index)
n_train = int(0.7 * len(shuffled_index))
_X_train, _X_valid = x_df.loc[shuffled_index[:n_train]], x_df.loc[shuffled_index[n_train:]]
y_train_mrs, y_valid_mrs = y_mrs_df.loc[shuffled_index[:n_train]], y_mrs_df.loc[shuffled_index[n_train:]]

X_train = remove_bp_features(_X_train)
X_valid = remove_bp_features(_X_valid)

model = create_deep_neural_network(X_train)
history = model.fit(X_train, y_train_mrs, epochs=300, batch_size=64, validation_data=(X_valid, y_valid_mrs))
model.save("cln_only_model.h5")
df, y_pred_proba_train, y_pred_proba_valid = evaluate_model(model, X_train, y_train_mrs, X_valid, y_valid_mrs)

# Clinical and Systolic Blood Pressure case :
X_bp_train = _X_train
X_bp_valid = _X_valid
y_bp_train_mrs = y_train_mrs
y_bp_valid_mrs = y_valid_mrs

model_bp = create_deep_neural_network(X_bp_train)
history_bp = model_bp.fit(X_bp_train, y_bp_train_mrs, epochs=300, batch_size=64,
                          validation_data=(X_bp_valid, y_bp_valid_mrs))
model_bp.save("cln_sbp_model.h5")

df_bp, y_bp_pred_proba_train, y_bp_pred_proba_valid = evaluate_model(model_bp, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

# p-value between model and model_bp by validation dataset
p_value_train = delong_roc_test(y_train_mrs.to_numpy().reshape(len(y_train_mrs), ), y_pred_proba_train, y_bp_pred_proba_train)
p_value_valid = delong_roc_test(y_valid_mrs.to_numpy().reshape(len(y_valid_mrs), ), y_pred_proba_valid, y_bp_pred_proba_valid)

visualize_roc_curve(model, X_train, y_train_mrs, model_bp, X_bp_train, y_bp_train_mrs, p_value_train, "1_ROC_CURVE_TRAIN.svg")
visualize_roc_curve(model, X_valid, y_valid_mrs, model_bp, X_bp_valid, y_bp_valid_mrs, p_value_valid, "2_ROC_CURVE_VALID.svg")

visualize_loss_and_accuracy(history, history_bp)

plot_of_SHAP(model, X_valid, "5_SHAP_CLN.svg")
plot_of_SHAP(model_bp, X_bp_valid, "6_SHAP_ADDED.svg")

feature = np.array(X_train)
label = np.transpose(np.array(y_train_mrs, dtype = int))
k_fold_cross_validation(model, feature, label, "11_KFOLD_CLN.svg")

feature = np.array(X_bp_train)
label = np.transpose(np.array(y_bp_train_mrs, dtype=int))
k_fold_cross_validation(model_bp, feature, label, "12_KFOLD_ADDED.svg")

# machine learning models
model_tr = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=18, min_samples_split=8)
model_tr.fit(X_train, y_train_mrs)
df_tr, _, __ = evaluate_model(model_tr, X_train, y_train_mrs, X_valid, y_valid_mrs)

model_etr = tree.ExtraTreeClassifier(max_depth=8, min_samples_leaf=12, min_samples_split=20)
model_etr.fit(X_train, y_train_mrs)
df_etr, _, __ = evaluate_model(model_etr, X_train, y_train_mrs, X_valid, y_valid_mrs)

model_rf = RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=12, min_samples_split=8)
model_rf.fit(X_train, y_train_mrs)
df_rf ,_, __ = evaluate_model(model_rf, X_train, y_train_mrs, X_valid, y_valid_mrs)

model_xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=10)
model_xgb.fit(X_train, y_train_mrs)
df_xgb, _, __ = evaluate_model(model_xgb, X_train, y_train_mrs, X_valid, y_valid_mrs)

model_lgbm = LGBMClassifier(learning_rate=0.01, max_depth=6)
model_lgbm.fit(X_train, y_train_mrs)
df_lgbm, _, __ = evaluate_model(model_lgbm, X_train, y_train_mrs, X_valid, y_valid_mrs)

model_cb = CatBoostClassifier(learning_rate=0.05, max_depth=12)
model_cb.fit(X_train, y_train_mrs)
df_cb, _, __ = evaluate_model(model_cb, X_train, y_train_mrs, X_valid, y_valid_mrs)

list_of_models = [model, model_tr, model_etr, model_rf, model_xgb, model_lgbm, model_cb]
visualize_roc_comparison(list_of_models, X_valid, y_valid_mrs, "3_ROC_CURVE_CLN.svg")

# machine learning models
model_bp_tr = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=18, min_samples_split=8)
model_bp_tr.fit(X_bp_train, y_bp_train_mrs)
df_bp_tr, _, __ = evaluate_model(model_bp_tr, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

model_bp_etr = tree.ExtraTreeClassifier(max_depth=8, min_samples_leaf=12, min_samples_split=20)
model_bp_etr.fit(X_bp_train, y_bp_train_mrs)
df_bp_etr, _, __ = evaluate_model(model_bp_etr, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

model_bp_rf = RandomForestClassifier(n_estimators=10, max_depth=6, min_samples_leaf=12, min_samples_split=8)
model_bp_rf.fit(X_bp_train, y_bp_train_mrs)
df_bp_rf, _, __ = evaluate_model(model_bp_rf, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

model_bp_xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=10)
model_bp_xgb.fit(X_bp_train, y_bp_train_mrs)
df_bp_xgb, _, __ = evaluate_model(model_bp_xgb, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

model_bp_lgbm = LGBMClassifier(learning_rate=0.01, max_depth=6)
model_bp_lgbm.fit(X_bp_train, y_bp_train_mrs)
df_bp_lgbm, _, __ = evaluate_model(model_bp_lgbm, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

model_bp_cb = CatBoostClassifier(learning_rate=0.05, max_depth=12)
model_bp_cb.fit(X_bp_train, y_bp_train_mrs)
df_bp_cb, _, __ = evaluate_model(model_bp_cb, X_bp_train, y_bp_train_mrs, X_bp_valid, y_bp_valid_mrs)

list_of_models = [model_bp, model_bp_tr, model_bp_etr, model_bp_rf, model_bp_xgb, model_bp_lgbm, model_bp_cb]
visualize_roc_comparison(list_of_models, X_bp_valid, y_bp_valid_mrs, "4_ROC_CURVE_ADDED.svg")

df_to_excel = pd.concat([df, df_tr, df_etr, df_rf, df_xgb, df_lgbm, df_cb,
                         df_bp, df_bp_tr, df_bp_etr, df_bp_rf, df_bp_xgb, df_bp_lgbm, df_bp_cb])

df_to_excel.to_excel("output.xlsx")

import numpy as np

from data_preprocessing import load_data, data_preprocessing
from model_creation import create_deep_neural_network

from model_evaluation import (
    bootstrap_optimism_model_fn,
    visualize_calibration_curve,
    k_fold_cross_validation_5fold,
    bootstrap_auc_ci_valid,
    visualize_dca_valid,
    visualize_roc_curve_valid_with_delong,
)

from shap_analysis import plot_of_SHAP

# =========================================================
# Load & preprocess data
# =========================================================
conv_cln_df, int_cln_df = load_data()
conv_cln_df, int_cln_df, bp_cln_df = data_preprocessing(conv_cln_df, int_cln_df)

X = bp_cln_df.drop(["multi", "mRS_3months"], axis=1)
y = bp_cln_df["mRS_3months"]

top_features = [
    "NIHSS_IAT_just_before",
    "Group",
    "Hyperlipidemia",
    "Previous_stroke_existence",
    "pt_age",
    "DM",
    "Anticoagulant",
    "Hgb",
    "systolic_TR",
    "systolic_min",
]

X_reduced = X[top_features]

# =========================================================
# Train / Validation split (pre-saved index)
# =========================================================
rng = np.random.default_rng(42)
indices = rng.permutation(X_reduced.index)

n_train = int(0.7 * len(indices))
train_idx = indices[:n_train]
valid_idx = indices[n_train:]

X_train_full = X_reduced.loc[train_idx]
X_valid_full = X_reduced.loc[valid_idx]

y_train = y.loc[train_idx]
y_valid = y.loc[valid_idx]

group_train = X_train_full["Group"].values
group_valid = X_valid_full["Group"].values

drop_cols = ["Group", "systolic_min", "systolic_TR"]
X_train = X_train_full.drop(columns=drop_cols)
X_valid = X_valid_full.drop(columns=drop_cols)

# =========================================================
# Clinical-only model
# =========================================================
model_cln = create_deep_neural_network(X_train.shape[1])
model_cln.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_valid, y_valid),
)
model_cln.save("clinical-only_model.h5")

# =========================================================
# Clinical + SBP model
# =========================================================
X_bp_train = X_train_full
X_bp_valid = X_valid_full

model_bp = create_deep_neural_network(X_bp_train.shape[1])
model_bp.fit(
    X_bp_train,
    y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_bp_valid, y_valid),
)
model_bp.save("clinical-and-sbp_model.h5")

# =========================================================
# Validation ROC + DeLong test
# =========================================================
p_value_valid, auc_cln, auc_bp = visualize_roc_curve_valid_with_delong(
    model_cln,
    X_valid,
    y_valid,
    model_bp,
    X_bp_valid,
    y_valid,
    filename="ROC_DELONG_VALID.svg",
)
print("DeLong p-value (valid):", p_value_valid)

# =========================================================
# 5-fold Cross Validation (train set)
# =========================================================
res_cln_train = k_fold_cross_validation_5fold(
    model_fn=lambda: create_deep_neural_network(X_train.to_numpy()),
    feature=X_train.to_numpy(),
    label=y_train.to_numpy(dtype=int),
    filename="REDUCED_CLN_TRAIN.svg",
    group=group_train,
    color="c",
)

res_bp_train = k_fold_cross_validation_5fold(
    model_fn=lambda: create_deep_neural_network(X_bp_train.to_numpy()),
    feature=X_bp_train.to_numpy(),
    label=y_train.to_numpy(dtype=int),
    filename="REDUCED_CLN+SBP_TRAIN.svg",
    group=group_train,
    color="r",
)

# =========================================================
# Bootstrap optimism correction
# =========================================================
bootstrap_optimism_model_fn(
    model_fn=lambda: create_deep_neural_network(X_train.to_numpy()),
    X=X_train.to_numpy(),
    y=y_train.to_numpy(dtype=int),
    B=500,
)

bootstrap_optimism_model_fn(
    model_fn=lambda: create_deep_neural_network(X_bp_train.to_numpy()),
    X=X_bp_train.to_numpy(),
    y=y_train.to_numpy(dtype=int),
    B=500,
)

# =========================================================
# SHAP analysis
# =========================================================
plot_of_SHAP(model_cln, X_valid, filename="SHAP_CLN_VALID.svg")
plot_of_SHAP(model_bp, X_bp_valid, filename="SHAP_ADDED_VALID.svg")

# =========================================================
# Bootstrap AUC CI (validation)
# =========================================================
bootstrap_results = bootstrap_auc_ci_valid(
    model=model_cln,
    X_valid=X_valid,
    y_valid=y_valid,
    model_bp=model_bp,
    X_bp_valid=X_bp_valid,
    y_bp_valid=y_valid,
    n_bootstrap=2000,
    random_state=42,
)

# =========================================================
# Calibration curve
# =========================================================
calib_df = visualize_calibration_curve(
    model_cln=model_cln,
    X_cln=X_valid,
    y_cln=y_valid,
    model_bp=model_bp,
    X_bp=X_bp_valid,
    y_bp=y_valid,
    label_cln="Clinical only",
    label_bp="Clinical & SBP metrics",
    filename="calibration_curve.svg",
)

# =========================================================
# Decision Curve Analysis (DCA)
# =========================================================
dca_df_valid = visualize_dca_valid(
    model_cln,
    X_valid,
    y_valid,
    model_bp,
    X_bp_valid,
    y_valid,
    pt_min=0.05,
    pt_max=0.60,
    n_pts=100,
    filename="DCA_VALID.svg",
)

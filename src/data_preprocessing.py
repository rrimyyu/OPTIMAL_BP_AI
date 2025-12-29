import pandas as pd
import numpy as np
import joblib
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =============================
# VIM
# =============================
def func(x, beta, k):
    return beta * x - np.log(k)


def calc_vim(mean_bp, sd_bp, k, beta):
    return k * sd_bp / (mean_bp ** beta)


# =============================
# Utilities
# =============================
def linear_interpolate_rowwise(df, cols):
    """Row-wise linear interpolation for BP columns."""
    df[cols] = df[cols].apply(
        lambda row: row.astype(float).interpolate(method="linear", limit_direction="both"),
        axis=1,
    )
    return df


def scale_dataframe(df, column_groups):
    """Scale selected column groups with StandardScaler and return fitted scalers."""
    scalers = {}
    for cols in column_groups:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        scalers[tuple(cols)] = scaler
    return df, scalers


# =============================
# Load data
# =============================
def load_data():
    xlsx_path = (
        "C:/Users/ORI1/Desktop/RY/RESEARCH_PHD/DATA/OPTIMAL-BP_AI/"
        "Core - 20230728 OPTIMAL-BP - nam 42.xlsx"
    )

    cln_df = pd.read_excel(xlsx_path, sheet_name="305_ITT", engine="openpyxl")
    cln_df = cln_df[cln_df["Per_protocol"] == 1]

    conv_df = pd.read_excel(xlsx_path, sheet_name="conventional_systolic", engine="openpyxl")
    int_df = pd.read_excel(xlsx_path, sheet_name="intensive_systolic", engine="openpyxl")

    cln_df = cln_df[
        [
            "multi",
            "optimal_bp_reg_no",
            "ASPECTS_raw",
            "TICI_immediate_최종_raw",
            "ICA_최종",
            "MCA_최종",
            "pt_age",
            "pt_sex",
            "HiBP",
            "Hyperlipidemia",
            "Smoking",
            "Previous_stroke_existence",
            "CAOD합친것",
            "cancer_active",
            "CHF_onoff",
            "PAOD_existence",
            "NIHSS_IAT_just_before",
            "Onset_to_registration_min",
            "IV_tPA",
            "Systolic_enroll",
            "DM",
            "A_fib합친것",
            "Antiplatelet",
            "Anticoagulant",
            "Hgb",
            "WBC",
            "BMI",
            "mRS_3months",
        ]
    ]

    cln_df = cln_df.fillna(cln_df[["WBC", "BMI", "ASPECTS_raw"]].mean())

    # Sex encoding
    encoder = LabelEncoder()
    cln_df["pt_sex"] = encoder.fit_transform(cln_df["pt_sex"])

    # TICI mapping
    tici_map = {"2b": 0, "2c": 1, 3: 2}
    cln_df["TICI_immediate_최종_raw"] = cln_df["TICI_immediate_최종_raw"].map(tici_map)

    bp_cols = ["연구대상자ID"] + [
        c for c in conv_df.columns if c.startswith("Systolic_") and c != "Systolic_enroll"
    ]

    conv_df = conv_df[bp_cols].iloc[:-2].rename(columns={"연구대상자ID": "optimal_bp_reg_no"})
    int_df = int_df[bp_cols].iloc[:-2].rename(columns={"연구대상자ID": "optimal_bp_reg_no"})

    conv_cln_df = pd.merge(conv_df, cln_df, on="optimal_bp_reg_no", how="inner")
    int_cln_df = pd.merge(int_df, cln_df, on="optimal_bp_reg_no", how="inner")

    conv_cln_df["mRS_3months"] = conv_cln_df["mRS_3months"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    int_cln_df["mRS_3months"] = int_cln_df["mRS_3months"].map(lambda x: 0 if x in [0, 1, 2] else 1)

    return conv_cln_df, int_cln_df


# =============================
# Preprocessing (conv / int)
# =============================
def data_preprocessing(conv_cln_df, int_cln_df):
    BP_cols = ["systolic_max", "systolic_min", "systolic_mean"]
    BPV_cols = ["systolic_TR", "systolic_SD", "systolic_CV", "systolic_VIM"]
    hourly_cols = [f"Systolic_{i}h" for i in range(1, 25)]

    def process_group(df, group_label):
        df = df.copy()
        df["Group"] = group_label

        all_sbp_cols = [c for c in df.columns if c.startswith("Systolic_") and c != "Systolic_enroll"]

        # Linear interpolation
        df = linear_interpolate_rowwise(df, all_sbp_cols)

        df["systolic_max"] = df[all_sbp_cols].max(axis=1)
        df["systolic_min"] = df[all_sbp_cols].min(axis=1)
        df["systolic_mean"] = df[all_sbp_cols].mean(axis=1)

        df["systolic_SD"] = df[hourly_cols].std(axis=1)
        df["systolic_CV"] = df["systolic_SD"] / df[hourly_cols].mean(axis=1)

        diffs = df[hourly_cols].diff(axis=1).abs()
        df["systolic_TR"] = diffs.iloc[:, 1:].mean(axis=1) / 60.0

        df = df.dropna(subset=["systolic_mean", "systolic_SD"])
        df = df[(df["systolic_mean"] > 0) & (df["systolic_SD"] > 0)]

        x = np.log(df["systolic_mean"].astype(float))
        y = np.log(df["systolic_SD"].astype(float))
        popt, _ = curve_fit(func, x, y)
        beta, k = popt

        df["systolic_VIM"] = calc_vim(df["systolic_mean"], df["systolic_SD"], k, beta)

        df = df.drop(columns=all_sbp_cols)
        return df.dropna()

    conv_cln_df = process_group(conv_cln_df, group_label=0)
    int_cln_df = process_group(int_cln_df, group_label=1)

    columns_to_scale = [
        "ASPECTS_raw",
        "TICI_immediate_최종_raw",
        "pt_age",
        "NIHSS_IAT_just_before",
        "Onset_to_registration_min",
        "Hgb",
        "WBC",
        "BMI",
        "Systolic_enroll",
    ]
    columns_groups = [columns_to_scale, BP_cols, BPV_cols]

    conv_cln_df.to_excel("conv_data_before_scaling.xlsx", index=False)
    int_cln_df.to_excel("int_data_before_scaling.xlsx", index=False)

    conv_cln_df, scaler_conv = scale_dataframe(conv_cln_df, columns_groups)
    int_cln_df, scaler_int = scale_dataframe(int_cln_df, columns_groups)

    joblib.dump(scaler_conv, "scaler_conv.pkl")
    joblib.dump(scaler_int, "scaler_int.pkl")

    bp_cln_df = pd.concat([conv_cln_df, int_cln_df], ignore_index=True).fillna(0)

    return conv_cln_df, int_cln_df, bp_cln_df

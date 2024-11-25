import pandas as pd
import numpy as np
import joblib
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    cln_df = pd.read_excel("C:/Users/ORI1/Desktop/RESEARCH_PHD/DATA/Core - 20230728 OPTIMAL-BP - nam 42.xlsx", sheet_name="305_ITT", engine='openpyxl')
    cs_df = pd.read_excel("C:/Users/ORI1/Desktop/RESEARCH_PHD/DATA/Core - 20230728 OPTIMAL-BP - nam 42.xlsx", sheet_name="conventional_systolic", engine='openpyxl')
    is_df = pd.read_excel("C:/Users/ORI1/Desktop/RESEARCH_PHD/DATA/Core - 20230728 OPTIMAL-BP - nam 42.xlsx", sheet_name="intensive_systolic", engine='openpyxl')

    cln_df = cln_df[['multi', 'optimal_bp_reg_no', 'pt_age', 'pt_sex', 'HiBP', 'Hyperlipidemia', 'Smoking',
                     'Previous_stroke_existence', 'CAOD합친것', 'cancer_active', 'CHF_onoff', 'PAOD_existence',
                     'NIHSS_IAT_just_before', 'Onset_to_registration_min', 'IV_tPA', 'Systolic_enroll',
                     'DM', 'A_fib합친것', 'Antiplatelet', 'Anticoagulant', 'Hgb', 'WBC', 'BMI',
                     'mRS_3months']]

    bp_columns = ['연구대상자ID'] + [col for col in cs_df.columns if col.startswith('Systolic_')]
    bp_columns.remove('Systolic_enroll')

    cs_df = cs_df[bp_columns]
    is_df = is_df[bp_columns]

    encoder = LabelEncoder()
    encoder.fit(cln_df['pt_sex'])
    cln_df['pt_sex'] = encoder.transform(cln_df['pt_sex'])

    cs_df = cs_df.iloc[:-2]
    is_df = is_df.iloc[:-2]

    interpolate_data(cs_df)
    interpolate_data(is_df)

    cs_df = cs_df.rename(columns = {'연구대상자ID':'optimal_bp_reg_no'})
    cs_cln_df = pd.merge(cs_df, cln_df)
    cs_cln_df['mRS_3months'] = cs_cln_df['mRS_3months'].map(lambda x: 0 if x in [0, 1, 2] else 1)

    is_df = is_df.rename(columns = {'연구대상자ID':'optimal_bp_reg_no'})
    is_cln_df = pd.merge(is_df, cln_df)
    is_cln_df['mRS_3months'] = is_cln_df['mRS_3months'].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})

    return cs_cln_df, is_cln_df

def interpolate_data(df):
    main_cols = [f'Systolic_{i}h' for i in range(1, 25)]

    for i, col in enumerate(main_cols, start=1):

        if i == 1:
            df[col] = df[col].fillna(df["Systolic_45min"])
            df[col] = df[col].fillna(df["Systolic_30min"])
            df[col] = df[col].fillna(df["Systolic_15min"])

        elif 1 < i < 24 :
            df[col] = df[col].fillna(df[f"{col}_15min"])
            df[col] = df[col].fillna(df[f"{main_cols[i - 2]}_45min"])
            df[col] = df[col].fillna(df[f"{col}_30min"])

        elif i == 24:
            df[col] = df[col].fillna(df["Systolic_23h_45min"])
            df[col] = df[col].fillna(df["Systolic_23h_30min"])
            df[col] = df[col].fillna(df["Systolic_23h_15min"])

    return df

def calculate_time_rate(dataframe, minutes):
    tr_values = []

    for idx, row in dataframe.iterrows():
        sbp_values = row.filter(like='Systolic').dropna().values
        valid_index = np.where(~row.filter(like='Systolic').isna())[0]

        if len(valid_index) < 2:
            tr_values.append(np.nan)
            continue

        tr_changes = np.diff(valid_index) * minutes

        sbp_changes = np.diff(sbp_values)
        sbp_changes = np.abs(sbp_changes)

        tr = np.divide(sbp_changes, tr_changes)
        tr_sum = np.sum(tr)

        intervals = len(sbp_changes)
        tr_sum = tr_sum/intervals

        tr_values.append(tr_sum)

    return tr_values

def func(x, beta, k):
    return beta * x - np.log(k)

def calculate_vim(sd, mean, k, beta):
    return k * sd / (mean ** beta)

def generate_bp_columns(intervals):
    bp_15min_cols = ["Systolic_15min", "Systolic_30min", "Systolic_45min"]

    for i in range(1, 24):
        bp_15min_cols.append(f"Systolic_{i}h")
        bp_15min_cols.append(f"Systolic_{i}h_15min")
        bp_15min_cols.append(f"Systolic_{i}h_30min")
        bp_15min_cols.append(f"Systolic_{i}h_45min")
    bp_15min_cols.append("Systolic_24h")

    bp_30min_cols = ["Systolic_30min"]
    for i in range(1, 24):
        bp_30min_cols.append(f"Systolic_{i}h")
        bp_30min_cols.append(f"Systolic_{i}h_30min")
    bp_30min_cols.append("Systolic_24h")

    if intervals == 15:
        return bp_15min_cols

    elif intervals == 30:
        return bp_30min_cols

def data_preprocessing(cs_cln_df, is_cln_df):
    cs_cln_df = cs_cln_df
    is_cln_df = is_cln_df

    bp_15min_cols = ["Systolic_15min"] + generate_bp_columns(15)
    bp_30min_cols = ["Systolic_30min"] + generate_bp_columns(30)
    bp_60min_cols = [f"Systolic_{i}h" for i in range(1, 25)]

    BP_cols = ['systolic_max', 'systolic_min', 'systolic_mean']
    BPV_cols = ['systolic_TR', 'systolic_SD', 'systolic_CV', 'systolic_VIM']

    _bp_data = cs_cln_df.loc[:, bp_15min_cols]

    cs_cln_df["Group"] = 0
    cs_cln_df["systolic_max"] = np.max(_bp_data, axis=1)
    cs_cln_df["systolic_min"] = np.min(_bp_data, axis=1)

    _bp_data = cs_cln_df.loc[:, bp_60min_cols]

    cs_cln_df["systolic_mean"] = np.mean(_bp_data, axis=1)
    cs_cln_df["systolic_TR"] = calculate_time_rate(_bp_data, 60)
    cs_cln_df["systolic_SD"] = np.std(_bp_data, axis=1)
    cs_cln_df["systolic_CV"] = np.std(_bp_data, axis=1) / np.mean(_bp_data, axis=1)

    cs_cln_df.dropna(subset=['systolic_max'], inplace=True)
    cs_cln_df = cs_cln_df.drop(bp_15min_cols, axis=1)
    cs_cln_df = cs_cln_df.dropna()

    x = np.log(cs_cln_df["systolic_mean"])
    y = np.log(cs_cln_df["systolic_SD"])

    popt, pcov = curve_fit(func, x, y)
    beta, k = popt

    cs_cln_df["systolic_VIM"] = calculate_vim(cs_cln_df["systolic_mean"], cs_cln_df["systolic_SD"],
                                              k, beta)

    _bp_data = is_cln_df.loc[:, bp_15min_cols]

    is_cln_df["Group"] = 1
    is_cln_df["systolic_max"] = np.max(_bp_data, axis=1)
    is_cln_df["systolic_min"] = np.min(_bp_data, axis=1)

    _bp_data = is_cln_df.loc[:, bp_60min_cols]

    is_cln_df["systolic_mean"] = np.mean(_bp_data, axis=1)
    is_cln_df["systolic_TR"] = calculate_time_rate(_bp_data, 60)
    is_cln_df["systolic_SD"] = np.std(_bp_data, axis=1)
    is_cln_df["systolic_CV"] = np.std(_bp_data, axis=1) / np.mean(_bp_data, axis=1)

    is_cln_df.dropna(subset=['systolic_max'], inplace=True)
    is_cln_df = is_cln_df.drop(bp_15min_cols, axis=1)
    is_cln_df = is_cln_df.dropna()

    x = np.log((is_cln_df["systolic_mean"]))
    y = np.log((is_cln_df["systolic_SD"]))

    popt, pcov = curve_fit(func, x, y)
    beta, k = popt

    is_cln_df["systolic_VIM"] = calculate_vim(is_cln_df["systolic_mean"], is_cln_df["systolic_SD"],
                                              k, beta)

    columns_to_scale = ['pt_age', 'NIHSS_IAT_just_before', 'Onset_to_registration_min',
                        'Hgb', 'WBC', 'BMI', 'Systolic_enroll']

    columns_groups = [columns_to_scale, BP_cols, BPV_cols]

    cs_cln_df, scaler_cs = scale_dataframe(cs_cln_df, columns_groups)
    is_cln_df, scaler_is = scale_dataframe(is_cln_df, columns_groups)

    joblib.dump(scaler_cs, "scaler_cs.pkl")
    joblib.dump(scaler_is, "scaler_is.pkl")

    bp_cln_df = pd.concat([cs_cln_df, is_cln_df], ignore_index=True)
    bp_cln_df = bp_cln_df.fillna(0)

    return cs_cln_df, is_cln_df, bp_cln_df

def scale_dataframe(df, column_groups):
    scalers = {}
    for cols in column_groups:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        scalers[tuple(cols)] = scaler

    return df, scalers

def remove_bp_features(data):
    BP_cols = ['systolic_max', 'systolic_min', 'systolic_mean']
    BPV_cols = ['systolic_TR', 'systolic_SD', 'systolic_CV', 'systolic_VIM']
    group_cols = ["Group"]

    data = data.drop(["Systolic_enroll"], axis=1)
    data = data.drop(BP_cols, axis=1)
    data = data.drop(BPV_cols, axis=1)
    data = data.drop(group_cols, axis=1)

    return data
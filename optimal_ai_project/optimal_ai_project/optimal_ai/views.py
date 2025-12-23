import io
from django.shortcuts import render
from .forms import OptimalAIForm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import base64
import joblib
import pandas as pd
import pickle
from ._SHAP import find_explainer
import os

# new_columns = [
#     'pt_age', 'pt_sex', 'HiBP', 'Hyperlipidemia', 'Smoking',
#     'Previous_stroke_existence', 'CAOD합친것', 'cancer_active',
#     'CHF_onoff', 'PAOD_existence', 'NIHSS_IAT_just_before',
#     'Onset_to_registration_min', 'IV_tPA', 'Systolic_enroll',
#     'DM', 'A_fib합친것', 'Antiplatelet', 'Anticoagulant',
#     'Hgb', 'WBC', 'BMI', 'Group', 'systolic_max', 'systolic_min',
#     'systolic_mean', 'systolic_TR', 'systolic_SD', 'systolic_CV',
#     'systolic_VIM']

selected_cols = [
    'NIHSS_IAT_just_before', 'Group', 'Hyperlipidemia',
    'Previous_stroke_existence', 'pt_age', 'DM',
    'Anticoagulant', 'Hgb', 'systolic_TR', 'systolic_min']

# Disable CUDA for TensorFlow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load Scaler Objects
scalers_cs = joblib.load("/home/ec2-user/OPTIMAL_BP_AI/optimal_ai_project/optimal_ai_project/TRAINED_MODEL/scaler_cs.pkl")
scalers_is = joblib.load("/home/ec2-user/OPTIMAL_BP_AI/optimal_ai_project/optimal_ai_project/TRAINED_MODEL/scaler_is.pkl")

# def prepare_input_features(collected_data, keep_cols=None):
#     group = int(float(collected_data.get("Group")))
#
#     if group == 0.0:
#         scaler = scalers_cs
#         if scaler is None:
#             raise ValueError("Scaler for group 0 not found")
#     elif group == 1.0:
#         scaler = scalers_is
#         if scaler is None:
#             raise ValueError("Scaler for group 1 not found")
#     else:
#         raise ValueError("Invalid group value")
#
#     df = pd.DataFrame([collected_data])
#
#     for cols, scaler_instance in scaler.items():
#         cols_list = list(cols)
#         df[cols_list] = scaler_instance.transform(df[cols_list])
#
#     if keep_cols is not None:
#         missing = [c for c in keep_cols if c not in df.columns]
#         if missing:
#             raise KeyError(f"Missing columns in input data: {missing}")
#         df = df[keep_cols]
#
#     return df

def prepare_input_features(collected_data, keep_cols):
    group = collected_data.get("Group")

    if group == 0.0:
        scaler_dict = scalers_cs
    elif group == 1.0:
        scaler_dict = scalers_is
    else:
        raise ValueError("Invalid group value")

    df = pd.DataFrame([collected_data])

    # ✅ scaler가 가진 컬럼 그룹 중, df에 존재하는 것만 transform
    for cols, scaler_instance in scaler_dict.items():
        cols_list = list(cols)
        existing = [c for c in cols_list if c in df.columns]

        # 아무것도 없으면 스킵
        if not existing:
            continue

        # 그룹의 일부만 존재하면 위험하니(스케일러 fit 차원 불일치) "그 그룹 전체가 있을 때만" 변환
        if len(existing) != len(cols_list):
            # 부분 컬럼만 있으면 transform 차원이 안 맞아서 오류 → 스킵
            continue

        df[cols_list] = scaler_instance.transform(df[cols_list])

    # ✅ 최종적으로 우리가 쓰는 10개만 남김
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input data: {missing}")

    return df[keep_cols]

def optimal_ai_view(request):
    result = None
    result_prob = None
    shap_plot = None
    insights = None

    if request.method == 'POST':
        form = OptimalAIForm(request.POST)
        if form.is_valid():
            collected_data = form.cleaned_data

            scaled_features = prepare_input_features(collected_data, selected_cols)
            input_features = scaled_features.to_numpy().reshape(1, -1)

            model = tf.keras.models.load_model("/home/ec2-user/OPTIMAL_BP_AI/optimal_ai_project/optimal_ai_project/TRAINED_MODEL/cln_sbp_model.h5")
            prediction = model.predict(input_features)
            result_prob = int(round(prediction[0][0] * 100, 2))

            prediction = np.where(prediction > 0.5, 1, 0)

            result = f"{int(prediction[0][0])}."

            explainer = find_explainer(model=model)
            shap_values = explainer.shap_values(input_features)
            shap_values = np.squeeze(shap_values)
            shap_values_with_cols = pd.DataFrame(shap_values)
            shap_values_with_cols = shap_values_with_cols.transpose()
            shap_values_with_cols.columns = selected_cols

            # feature_names = ['Age', 'Sex', 'Hypertension', 'Hyperlipidemia', 'Smoking', 'Previous stroke',
            #                  'CAOD', 'Active cancer', 'Congestive heart failure', 'PAOD', 'NIHSS score',
            #                  'Onset to registration', 'IV tPA', 'SBP enroll', 'DM', 'Atrial fibrillation',
            #                  'Antiplatelet', 'Anticoagulant', 'Hemoglobin', 'White blood cell', 'Body mass index',
            #                  'Group', 'SBP max', 'SBP min', 'SBP mean', 'SBP time rate', 'SBP standard deviation',
            #                  'SBP coefficient of variation', 'SBP variation independent of the mean']

            feature_names = [
                'NIHSS score', 'Group', 'Hyperlipidemia', 'Previous stroke',
                'Age', 'DM', 'Anticoagulant', 'Hemoglobin', 'SBP time rate', 'SBP min'
            ]

            insights = extract_shap_insights(np.transpose(shap_values), feature_names, top_n=3)

            shap.initjs()

            shap_expl = shap.Explanation(shap_values, explainer.expected_value.numpy()[0], feature_names=feature_names)

            shap_plot_fig, ax = plt.subplots()
            shap.plots.waterfall(shap_expl, max_display=6, show=False)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(shap_plot_fig)  # Close the figure to free up memory
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            shap_plot = base64.b64encode(image_png).decode('utf-8')

        else:
            # Form is invalid; Bootstrap will handle the display of errors
            pass
    else:
        form = OptimalAIForm()
    return render(request, 'optimal_ai/form.html', {'form': form, 'result': result, 'result_prob': result_prob, 'shap_plot': shap_plot, 'insights': insights})

def extract_shap_insights(shap_values, feature_names, top_n=3):
    """
    Extracts the top N factors that worsen and improve the mRS score based on SHAP values.

    Parameters:
    - shap_values (numpy.ndarray): SHAP values for the features.
    - feature_names (list): List of feature names.
    - top_n (int): Number of top factors to extract.

    Returns:
    - dict: Contains worsen_factors, worst_factor, improve_factors, best_factor
    """

    shap_abs = np.abs(shap_values)
    sorted_indices = np.argsort(shap_abs)[::-1]  # Descending order

    # Factors that worsen the mRS score (positive SHAP values)
    worsen_indices = np.where(shap_values > 0)[0]
    worsen_shap = shap_values[worsen_indices]
    worsen_sorted_indices = worsen_indices[np.argsort(worsen_shap)[::-1]]
    worsen_factors = [feature_names[i] for i in worsen_sorted_indices[:top_n]]
    worst_factor = feature_names[worsen_sorted_indices[0]] if len(worsen_sorted_indices) > 0 else 'N/A'

    # Factors that improve the mRS score (negative SHAP values)
    improve_indices = np.where(shap_values < 0)[0]
    improve_shap = shap_values[improve_indices]
    improve_sorted_indices = improve_indices[np.argsort(improve_shap)]
    improve_factors = [feature_names[i] for i in improve_sorted_indices[:top_n]]
    best_factor = feature_names[improve_sorted_indices[0]] if len(improve_sorted_indices) > 0 else 'N/A'

    return {
        'worsen_factors': worsen_factors,
        'worst_factor': worst_factor,
        'improve_factors': improve_factors,
        'best_factor': best_factor
    }
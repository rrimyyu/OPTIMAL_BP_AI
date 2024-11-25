import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, precision_score, confusion_matrix
from sklearn.model_selection import KFold
from confidenceinterval import roc_auc_score
from sklearn.metrics import roc_curve

def evaluate_model(model, train_X, train_y, valid_X, valid_y):
    metrics = {
        'AUC': [],
        'Sensitivity': [],
        'Specificity': [],
        'PPV': [],
        'NPV': [],
        'CI': []
    }

    if isinstance(model, tf.keras.models.Sequential):
        _y_pred_proba_train = model.predict(train_X.to_numpy()).ravel()
        y_pred_proba = model.predict(train_X.to_numpy()).ravel()
    else :
        _y_pred_proba_train = model.predict(train_X.to_numpy()).ravel()
        y_pred_proba = model.predict_proba(train_X)[:, 1].ravel()

    fpr, tpr, thresholds = roc_curve(train_y, y_pred_proba)
    roc_auc_train = auc(fpr, tpr)
    _, ci_train = roc_auc_score(train_y, y_pred_proba, confidence_level=0.95)

    preds_1d = y_pred_proba.flatten()
    y_pred_proba = np.where(preds_1d > 0.5, 1, 0)
    cm = confusion_matrix(train_y, y_pred_proba)

    tn, fp, fn, tp = cm.ravel()
    ss_train = (tp / (tp + fn))
    sp_train = (tn / (fp + tn))
    ppv_train = precision_score(train_y, y_pred_proba)
    npv_train = (tn / (tn + fn))

    metrics['AUC'].append(round(roc_auc_train, 3))
    metrics['Sensitivity'].append(round(ss_train, 3))
    metrics['Specificity'].append(round(sp_train, 3))
    metrics['PPV'].append(round(ppv_train, 3))
    metrics['NPV'].append(round(npv_train, 3))
    metrics['CI'].append(ci_train)

    # AUC, Sensitivity, Specificity, PPV and NPV of valid set
    if isinstance(model, tf.keras.models.Sequential):
        _y_pred_proba_valid = model.predict(valid_X.to_numpy()).ravel()
        y_pred_proba = model.predict(valid_X.to_numpy()).ravel()
    else :
        _y_pred_proba_valid = model.predict_proba(valid_X)[:, 1].ravel()
        y_pred_proba = model.predict_proba(valid_X)[:, 1].ravel()

    fpr, tpr, thresholds = roc_curve(valid_y, y_pred_proba)
    roc_auc_valid = auc(fpr, tpr)

    _, ci_valid = roc_auc_score(valid_y, y_pred_proba, confidence_level=0.95)

    preds_1d = y_pred_proba.flatten()
    y_pred_proba = np.where(preds_1d > 0.5, 1, 0)
    cm = confusion_matrix(valid_y, y_pred_proba)

    tn, fp, fn, tp = cm.ravel()
    ss_valid = (tp / (tp + fn))
    sp_valid = (tn / (fp + tn))
    ppv_valid = precision_score(valid_y, y_pred_proba)
    npv_valid = (tn / (tn + fn))

    metrics['AUC'].append(round(roc_auc_valid, 3))
    metrics['Sensitivity'].append(round(ss_valid, 3))
    metrics['Specificity'].append(round(sp_valid, 3))
    metrics['PPV'].append(round(ppv_valid, 3))
    metrics['NPV'].append(round(npv_valid, 3))
    metrics['CI'].append(ci_valid)

    df = pd.DataFrame(metrics)

    return df, _y_pred_proba_train, _y_pred_proba_valid

# Clinical only vs Clinical & Systolic Blood Pressure
def visualize_roc_curve(model_cln, X_cln, y_cln, model_bp, X_bp, y_bp, p_value, filename):

    if isinstance(model_cln, tf.keras.models.Sequential):
        y_pred_proba = model_cln.predict(X_cln.to_numpy()).ravel()
    else:
        y_pred_proba = model_cln.predict_proba(X_cln)[:, 1].ravel()

    fpr, tpr, thresholds = roc_curve(y_cln, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    if isinstance(model_bp, tf.keras.models.Sequential):
        y_bp_pred_proba = model_bp.predict(X_bp.to_numpy()).ravel()
    else :
        y_bp_pred_proba = model_bp.predict_proba(X_bp)[:, 1].ravel()

    fpr_bp, tpr_bp, thresholds_bp = roc_curve(y_bp, y_bp_pred_proba)
    roc_auc_bp = auc(fpr_bp, tpr_bp)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')
    plt.plot(fpr_bp, tpr_bp, color='r', lw=1, label='Clinical & SBP parameters (AUC = %0.2f)' % (roc_auc_bp))
    plt.plot(fpr, tpr, color='c', lw=1, label='Clinical only (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], color='darkgrey', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1-Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.legend(loc="lower right", frameon=False)

    if p_value < 0.001 :
        plt.text(0.75, 0.5, r'$\it{P} < 0.001$', horizontalalignment='center', verticalalignment='center',
                 fontsize=16)
    else :
        p_value = p_value[0, 0]
        plt.text(0.75, 0.5, f'$\it{{P}}$ = {p_value:.3f}', horizontalalignment='center', verticalalignment='center',
                 fontsize=16)

    plt.savefig(filename, format='svg')
    plt.show()

def visualize_roc_comparison(list_of_models, X, y, filename):
    list_for_roc = []

    for model in list_of_models:
        if isinstance(model, tf.keras.models.Sequential):
            y_pred_proba = model.predict(X.to_numpy()).ravel()
        else:
            y_pred_proba = model.predict_proba(X)[:, 1].ravel()

        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        list_for_roc.append([fpr, tpr, roc_auc])

    models = ["DNN", "Decision trees", "Extra tree", "Random forest", "XGBoost", "LightGBM", "CatBoost"]
    colors = ['red', 'limegreen', 'limegreen', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue']
    linestyles = ['solid', 'solid', 'dotted', 'solid', 'dotted', 'dashed', 'dashdot']

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')

    for idx in range(len(models)):
        if idx == 0 :
            plt.plot(list_for_roc[idx][0], list_for_roc[idx][1], color=colors[idx], linestyle=linestyles[idx], lw=1,
                     label=models[idx] + " (AUC = %0.2f)" % (list_for_roc[idx][2]))
        else :
            plt.plot(list_for_roc[idx][0], list_for_roc[idx][1], color=colors[idx], linestyle=linestyles[idx], lw=1,
                     alpha=0.6, label= models[idx] + " (AUC = %0.2f)" % (list_for_roc[idx][2]))

    plt.plot([0, 1], [0, 1], color='darkgrey', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.xlabel('1-Specificity', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.legend(loc="lower right", frameon=False)

    plt.savefig(filename, format='svg')
    plt.show()


def k_fold_cross_validation(model, feature, label, filename) :
    kfold = KFold(n_splits=5)

    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')

    for i, (train_index, test_index) in enumerate(kfold.split(feature)):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]

        model.fit(x_train, y_train)
        y_score = model.predict(x_test)

        _, ci = roc_auc_score(y_test, y_score.ravel(), confidence_level=0.95)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = np.round(auc(fpr, tpr), 4)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.legend(frameon=False)
    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    plt.plot([0, 1], [0, 1], color='darkgrey', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('Sensitivity', fontsize=14)
    plt.xlabel('1-Specificity', fontsize=14)
    plt.savefig(filename, format='svg')
    plt.show()


def visualize_loss_and_accuracy(history, history_bp):
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("7_LOSS_CLN.svg", format='svg')
    plt.show()

    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("8_ACC_CLN.svg", format='svg')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.plot(history_bp.history['loss'], label='Training Loss')
    plt.plot(history_bp.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("9_LOSS_ADDED.svg", format='svg')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.plot(history_bp.history['accuracy'], label='Training Accuracy')
    plt.plot(history_bp.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("10_ACC_ADDED.svg", format='svg')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, ConfusionMatrixDisplay
import json

save_to = 'D:\\year 4\\semester 1\\BT\\BT 4033\\prediction\\'
testing_result_dir = 'D:\\year 4\\semester 1\\BT\\BT 4033\\prediction\\models\\'


def calculate_metrics(y_true, y_pred):
    """Calculate micro/macro AUPR and Fmax"""
    metrics = {}

    # **AUPR Calculations**
    metrics['micro_aupr'] = average_precision_score(y_true, y_pred, average='micro')
    metrics['macro_aupr'] = average_precision_score(y_true, y_pred, average='macro')

    # **Micro & Macro Fmax Calculation (CAFA-style)**
    thresholds = np.arange(0.0, 1.0, 0.01)
    
    fmax_micro, best_t_micro = 0, 0
    fmax_macro, best_t_macro = 0, 0

    for t in thresholds:
        y_pred_t = (y_pred > t).astype(int)

        # **Micro Fmax**
        f1_micro = f1_score(y_true, y_pred_t, average='micro')
        if f1_micro > fmax_micro:
            fmax_micro, best_t_micro = f1_micro, t

        # **Macro Fmax**
        f1_macro = f1_score(y_true, y_pred_t, average='macro')
        if f1_macro > fmax_macro:
            fmax_macro, best_t_macro = f1_macro, t

    metrics['micro_fmax'], metrics['best_t_micro'] = fmax_micro, best_t_micro
    metrics['macro_fmax'], metrics['best_t_macro'] = fmax_macro, best_t_macro

    return metrics, thresholds


def aggregate_confusion_matrix(y_true, y_pred, aspect):

    # Compute TP, FP, TN, FN for each class
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # Construct confusion matrix
    confusion_matrix = np.array([[TP, FP], 
                                 [FN, TN]])
    
    labels = ["Positive", "Negative"]
    disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {aspect}")
    plt.savefig(f"{save_to}confusion_matrix_{aspect}.png", dpi=300, bbox_inches="tight")


for aspect in ['bp', 'mf', 'cc']:
    results = np.load(f'{testing_result_dir}{aspect}_testing.npz')
    y_true = results['true']
    y_hat = results['predicted']
    aggregate_confusion_matrix(y_true, y_hat, aspect.upper())

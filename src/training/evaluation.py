import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import training_config as TC
from src.models.ensemble import predict_ensemble

__all__ = ["evaluate_ensemble", "find_best_threshold"]


def evaluate_ensemble(models, X_test, y_test, threshold=0.5):
    probs = predict_ensemble(list(models.values()), X_test)
    preds = (probs > threshold).astype(int)

    print("\n--- ENSEMBLE PERFORMANCE ---")
    print(classification_report(y_test, preds, target_names=["Normal", "Attack"]))
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall: {recall_score(y_test, preds):.4f}")
    print(f"F1-score: {f1_score(y_test, preds):.4f}")

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix – Ensemble")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.show()
    return probs


def find_best_threshold(probs, y_test):
    start, stop, step = TC.THRESH_GRID
    thresholds = np.arange(start, stop, step)
    best_f1, best_t = 0, 0.5
    f1_scores = []

    for t in thresholds:
        f1 = f1_score(y_test, (probs > t).astype(int))
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Threshold"); plt.ylabel("F1-score")
    plt.title("F1 vs Threshold"); plt.grid(); plt.show()
    return best_t, best_f1
import numpy as np

__all__ = ["predict_ensemble"]


def predict_ensemble(models, X):
    preds = [m.predict(X, verbose=0) for m in models]
    return np.mean(preds, axis=0)
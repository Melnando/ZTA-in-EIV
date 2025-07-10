import numpy as np
from sklearn.utils import resample
from tensorflow.keras.callbacks import EarlyStopping

from config import training_config as TC
from src.data.augmentation import jitter
from src.models.base_model import build_base_model
from src.utils.metrics import FocalLoss

__all__ = ["train_all_models"]


def _oversample(X_train, y_train):
    flat = X_train.reshape(X_train.shape[0], -1)
    majority = flat[y_train == 0]
    minority = flat[y_train == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced = np.vstack([majority, minority_up])
    labels = np.array([0] * len(majority) + [1] * len(minority_up))
    return balanced.reshape(-1, X_train.shape[1], X_train.shape[2]), labels


def _augment_attacks(X, y):
    attack_idx = np.where(y == 1)[0]
    X_aug = np.array([jitter(sample) for sample in X[attack_idx]])
    return np.concatenate([X, X_aug]), np.concatenate([y, np.ones(len(X_aug))])


def train_all_models(X_train, y_train, X_val, y_val):
    cb = EarlyStopping(monitor="val_loss", patience=TC.PATIENCE, restore_best_weights=True)

    # --- Standard ---
    std = build_base_model(X_train.shape[1:], loss="binary_crossentropy")
    std.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=TC.EPOCHS, batch_size=TC.BATCH_SIZE, callbacks=[cb], verbose=2)

    # --- Focal ---
    focal = build_base_model(X_train.shape[1:], loss=FocalLoss())
    focal.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=TC.EPOCHS, batch_size=TC.BATCH_SIZE, callbacks=[cb], verbose=2)

    # --- Attention ---
    att = build_base_model(X_train.shape[1:], loss="binary_crossentropy", attention=True)
    att.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=TC.EPOCHS, batch_size=TC.BATCH_SIZE, callbacks=[cb], verbose=2)

    # --- Augmented ---
    X_aug, y_aug = _augment_attacks(*_oversample(X_train, y_train))
    aug = build_base_model(X_train.shape[1:], loss="binary_crossentropy")
    aug.fit(X_aug, y_aug, validation_data=(X_val, y_val), epochs=TC.EPOCHS, batch_size=TC.BATCH_SIZE, callbacks=[cb], verbose=2)

    return {
        "standard": std,
        "focal": focal,
        "attention": att,
        "augmented": aug,
    }
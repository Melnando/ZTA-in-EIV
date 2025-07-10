# Hyper‑parameters and model‑specific options
SEED = 42

MODEL = {
    "conv_filters": 64,
    "kernel_size": 3,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dropout_conv": 0.2,
    "dropout_lstm1": 0.3,
    "dropout_dense": 0.2,
}

ENSEMBLE_MODELS = [
    "standard",
    "focal",
    "attention",
    "augmented",
]
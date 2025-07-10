import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import data_config as DC
from config import model_config as MC

__all__ = [
    "load_csv",
    "create_time_series",
    "scale_split",
]


def load_csv(path: str = DC.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path).fillna(0)
    df["Time"] = pd.to_datetime(df["Time"], unit="s")
    df = (
        df.sort_values("Time")
          .set_index("Time")
          .resample(DC.RESAMPLE_STR)
          .mean()
          .interpolate(method="cubic")
          .reset_index()
    )
    numeric_signals = [c for c in df.columns if "Signal" in c]
    df[numeric_signals] = df[numeric_signals].apply(pd.to_numeric, errors="coerce")
    return df


def create_time_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    signals = [c for c in df.columns if "Signal" in c]
    seqs, labels = [], []
    for i in range(0, len(df) - DC.WINDOW_SIZE, DC.STEP_SIZE):
        window = df.iloc[i : i + DC.WINDOW_SIZE]
        seqs.append(window[signals].values)
        labels.append(int(window["Label"].sum() > 0))
    return np.asarray(seqs), np.asarray(labels), signals


def scale_split(X: np.ndarray, y: np.ndarray):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DC.TEST_SPLIT, random_state=MC.SEED
    )

    # NaN‑safety & scaling
    X_train, X_test = map(lambda z: np.nan_to_num(z), (X_train, X_test))
    s = MinMaxScaler()
    X_train = s.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = s.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_test, y_train, y_test
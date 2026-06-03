from __future__ import annotations
import numpy as np
from .vendor_config import VendoredConfig

def shape_data(arr: np.ndarray, train: bool, config: VendoredConfig, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    l_s = config.l_s
    n_pred = config.n_predictions
    window_len = l_s + n_pred
    n_windows = len(arr) - window_len
    if n_windows <= 0:
        raise ValueError(f"Array length {len(arr)} too short for window_len={window_len}")
    data = np.zeros((n_windows, window_len, arr.shape[1]), dtype=arr.dtype)
    for i in range(n_windows):
        data[i] = arr[i:i + window_len]
    if train:
        if rng is None:
            np.random.shuffle(data)
        else:
            rng.shuffle(data)
    X = data[:, :-n_pred, :]
    y = data[:, -n_pred:, 0]
    return X, y


class Channel:
    def __init__(self, chan_id: str):
        self.id = chan_id
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.train: np.ndarray | None = None
        self.test: np.ndarray | None = None
        self.y_hat: np.ndarray = np.array([])

    @classmethod
    def from_arrays(cls,chan_id: str, train_arr: np.ndarray, test_arr: np.ndarray, config: VendoredConfig, seed: int = 42) -> "Channel":
        ch = cls(chan_id)
        ch.train = train_arr
        ch.test = test_arr
        rng = np.random.default_rng(seed)
        ch.X_train, ch.y_train = shape_data(train_arr, train=True,  config=config, rng=rng)
        ch.X_test,  ch.y_test  = shape_data(test_arr,  train=False, config=config, rng=rng)
        return ch
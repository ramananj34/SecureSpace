from __future__ import annotations
import numpy as np
import torch
from .channel import Channel
from .vendor_config import VendoredConfig

def aggregate_predictions(y_hat_so_far: np.ndarray, y_hat_batch: np.ndarray, config: VendoredConfig, method: str = "first") -> np.ndarray:
    n_predictions = config.n_predictions
    agg_y_hat_batch = np.array([])
    for t in range(len(y_hat_batch)):
        start_idx = max(0, t - n_predictions)
        y_hat_t = np.flipud(y_hat_batch[start_idx:t + 1]).diagonal()
        if method == "first":
            agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
        elif method == "mean":
            agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'first' or 'mean'.")

    return np.append(y_hat_so_far, agg_y_hat_batch)

@torch.no_grad()
def batch_predict(model: torch.nn.Module, channel: Channel, config: VendoredConfig, method: str = "first", device: torch.device | None = None ) -> Channel:
    if channel.X_test is None:
        raise ValueError(f"Channel {channel.id} has no X_test set.")
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    batch_size = config.batch_size
    n_test_windows = channel.X_test.shape[0]
    num_batches = max(1, n_test_windows // batch_size)
    y_hat = np.array([])
    for i in range(num_batches + 1):
        prior_idx = i * batch_size
        idx = (i + 1) * batch_size
        if i == num_batches:
            idx = n_test_windows
        if prior_idx >= n_test_windows:
            break

        X_batch = torch.from_numpy(channel.X_test[prior_idx:idx].astype(np.float32)).to(device)
        y_hat_batch = model(X_batch).cpu().numpy()
        y_hat = aggregate_predictions(y_hat, y_hat_batch, config, method=method)

    channel.y_hat = y_hat.reshape(-1)
    return channel
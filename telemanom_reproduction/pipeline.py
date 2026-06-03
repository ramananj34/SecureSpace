from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_THIS_DIR))
from smap_msl_dataset_api import SMAPMSLChannelDataset
from telemanom_lstm import TelemanomLSTM, TelemanomConfig
from VENDOR_telemanom import VendoredConfig, Channel, batch_predict, Errors

def build_channel(chan_id: str, data_dir: Path, vendored_config: VendoredConfig | None = None, seed: int = 42) -> Channel:
    cfg = vendored_config or VendoredConfig()
    train_ds = SMAPMSLChannelDataset(chan_id, split="train", data_dir=data_dir)
    test_ds = SMAPMSLChannelDataset(chan_id, split="test", data_dir=data_dir)
    train_features = np.concatenate([train_ds.telemetry_scaled[:, None], train_ds._commands], axis=1).astype(np.float32)
    test_features = np.concatenate([test_ds.telemetry_scaled[:, None], test_ds._commands], axis=1).astype(np.float32)
    return Channel.from_arrays(chan_id=chan_id,train_arr=train_features,test_arr=test_features,config=cfg,seed=seed)

def load_trained_model(model_path: Path, n_features: int,config: TelemanomConfig | None = None,device: torch.device | None = None) -> TelemanomLSTM:
    import json
    if config is None:
        config_path = Path(model_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            config = TelemanomConfig(**cfg_dict)
        else:
            config = TelemanomConfig(input_dim=n_features)
    if config.input_dim != n_features:
        from dataclasses import replace
        config = replace(config, input_dim=n_features)
    model = TelemanomLSTM(config)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model

@dataclass
class DetectionResult:
    chan_id: str
    predicted_sequences: list[tuple[int, int]]
    labeled_sequences: list[tuple[int, int]]
    n_test_windows: int
    normalized_error: float
    smoothed_errors: np.ndarray
    aggregated_predictions: np.ndarray

def detect_anomalies_for_channel(chan_id: str, model_path: Path, data_dir: Path, vendored_config: VendoredConfig | None = None, aggregation_method: str = "first", device: torch.device | None = None, verbose: bool = False) -> DetectionResult:
    cfg = vendored_config or VendoredConfig()
    channel = build_channel(chan_id, data_dir, cfg)
    n_features = channel.X_train.shape[2]
    model = load_trained_model(model_path, n_features=n_features, device=device)
    batch_predict(model, channel, cfg, method=aggregation_method, device=device)
    errors = Errors(channel, cfg, verbose=verbose)
    errors.process_batches(channel)
    test_ds = SMAPMSLChannelDataset(chan_id, split="test", data_dir=data_dir)
    mask = test_ds.anomaly_mask
    labeled = _mask_to_sequences(mask)

    return DetectionResult(chan_id=chan_id, predicted_sequences=list(errors.E_seq), labeled_sequences=labeled, n_test_windows=channel.X_test.shape[0], normalized_error=errors.normalized, smoothed_errors=errors.e_s.copy(), aggregated_predictions=channel.y_hat.copy())

def _mask_to_sequences(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))

def evaluate_anomalies(predicted: list[tuple[int, int]], labeled: list[tuple[int, int]]) -> dict:
    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])
    tp = sum(1 for p in predicted if any(overlaps(p, l) for l in labeled))
    fp = len(predicted) - tp
    fn = sum(1 for l in labeled if not any(overlaps(p, l) for p in predicted))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    if (beta ** 2 * precision + recall) > 0:
        f0_5 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    else:
        f0_5 = 0.0
    return {"tp": tp,"fp": fp,"fn": fn,"precision": precision,"recall": recall,"f0_5": f0_5,"n_predicted": len(predicted),"n_labeled": len(labeled)}
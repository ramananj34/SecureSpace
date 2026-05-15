#PyTorch Dataset adapter wrapping SMAPMSLChannelDataset for LSTM training.
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class TelemanomTrainingDataset(Dataset):
    #PyTorch Dataset returning (x_window, y_future) for telemanom LSTM training.

    def __init__(self, channel_dataset, sequence_length: int = 250, n_predictions: int = 10, stride: int = 1): #SMAPMSLChannelDataset, avoiding circular import
        self.cd = channel_dataset
        self.sequence_length = sequence_length
        self.n_predictions = n_predictions
        self.stride = stride

        #Build the feature matrix [telemetry_scaled, command_features] Shape: (n_timesteps, 1 + n_command_features)
        scaled_tele = self.cd.telemetry_scaled  # (n_timesteps,)
        commands = self.cd._commands  # (n_timesteps, 24)
        self._features = np.concatenate([scaled_tele[:, None], commands], axis=1).astype(np.float32)  # use float32 to play nice with torch

        #Verify we have enough timesteps for at least one window.
        min_needed = sequence_length + n_predictions
        if len(self._features) < min_needed + 1:
            raise ValueError(
                f"Channel {self.cd.chan_id} {self.cd.split} has {len(self._features)} "
                f"timesteps; need at least {min_needed + 1} for window+target."
            )

    def __len__(self) -> int:
        #Last valid starting index i: i + sequence_length + n_predictions <= n_timesteps so i ranges in [0, n_timesteps - sequence_length - n_predictions]
        max_i = len(self._features) - self.sequence_length - self.n_predictions
        return max_i // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.stride
        if i + self.sequence_length + self.n_predictions > len(self._features):
            raise IndexError(f"Index {idx} out of range")
        x = self._features[i : i + self.sequence_length]
        #Future telemetry values: feature 0 at timesteps [i+L, i+L+n_predictions)
        y = self._features[i + self.sequence_length : i + self.sequence_length + self.n_predictions, 0]
        return torch.from_numpy(x), torch.from_numpy(y)

def make_train_val_loaders(channel_dataset, sequence_length: int = 250, n_predictions: int = 10, batch_size: int = 64, validation_split: float = 0.2, num_workers: int = 0, seed: int = 42) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    full = TelemanomTrainingDataset(channel_dataset, sequence_length, n_predictions, stride=1)
    n_total = len(full)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))
    train_subset = torch.utils.data.Subset(full, train_indices)
    val_subset = torch.utils.data.Subset(full, val_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False )
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader
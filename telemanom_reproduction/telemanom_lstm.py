"""
Telemanom LSTM model. Exact reproduction of the architecture from Hundman et al. (KDD 2018) "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding."

Architecture:
  - 2 stacked LSTM layers, 80 units each.
  - Dropout 0.3 between LSTM layers and before the final dense.
  - Final dense layer outputs n_predictions = 10 scalar predictions (one per
    future timestep, t+1 through t+n_predictions).
  - Sequence length l_s = 250 timesteps.
  - Input dim = 25 features (telemetry + 24 one-hot command features).

Model parameter count (PyTorch's nn.LSTM uses two bias vectors per gate;
mathematically equivalent to one combined bias but consumes 2× bias parameters):
  - LSTM layer 1 (25 -> 80): 4 * 80 * (25 + 80) + 2 * 4 * 80 = 34,240 params
  - LSTM layer 2 (80 -> 80): 4 * 80 * (80 + 80) + 2 * 4 * 80 = 51,840 params
  - Dense (80 -> 10):        80 * 10 + 10                    =    810 params
                                                               --------
                                                               86,890 params

Source: https://github.com/khundman/telemanom/blob/master/config.yaml
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass(frozen=True)
class TelemanomConfig:
    #Architecture
    input_dim: int = 25 #1 telemetry + 24 command features (SMAP/MSL preprocessed)
    hidden_size: int = 80
    num_layers: int = 2
    dropout: float = 0.3
    sequence_length: int = 250
    n_predictions: int = 10
    #Training
    batch_size: int = 64
    epochs: int = 35
    validation_split: float = 0.2
    patience: int = 10
    min_delta: float = 0.0003
    learning_rate: float = 1e-3

class TelemanomLSTM(nn.Module):
    #Two-layer LSTM with dropout, outputting n_predictions future telemetry values. Input: (batch, sequence_length, input_dim). Output: (batch, n_predictions)
    def __init__(self, config: TelemanomConfig | None = None):
        super().__init__()
        cfg = config or TelemanomConfig()
        self.config = cfg
        self.lstm = nn.LSTM(input_size=cfg.input_dim, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_size, cfg.n_predictions)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def count_parameters(model: nn.Module) -> int:
    #Total number of trainable parameters in a model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
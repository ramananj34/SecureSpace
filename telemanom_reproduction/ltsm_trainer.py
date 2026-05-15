#LSTM training utility for telemanom-style anomaly-detection LSTMs

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

from telemanom_lstm import TelemanomConfig, TelemanomLSTM

@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False

class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0003):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False
    def step(self, val_loss: float, epoch: int) -> bool:
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(n_samples, 1)

@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(n_samples, 1)


def train(model: TelemanomLSTM, train_loader: DataLoader, val_loader: DataLoader, config: Optional[TelemanomConfig] = None, device: Optional[torch.device] = None, verbose: bool = True, on_epoch_end: Optional[Callable[[int, float, float], None]] = None) -> tuple[dict, TrainHistory]:
    cfg = config or model.config
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta)
    history = TrainHistory()
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        improved = stopper.step(val_loss, epoch)
        if improved:
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            history.best_epoch = epoch
            history.best_val_loss = val_loss
        if verbose:
            marker = " *" if improved else ""
            print(
                f"  epoch {epoch+1:3d}/{cfg.epochs}: train={train_loss:.6f}  "
                f"val={val_loss:.6f}{marker}"
            )
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss)
        if stopper.should_stop:
            history.stopped_early = True
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best was {stopper.best_epoch+1})")
            break
    if verbose:
        print(
            f"  done: best val_loss = {history.best_val_loss:.6f} at epoch {history.best_epoch+1}"
        )
    return best_state, history
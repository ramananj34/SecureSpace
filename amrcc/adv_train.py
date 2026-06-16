from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from telemanom_lstm import TelemanomConfig, TelemanomLSTM
from ltsm_trainer import TrainHistory, EarlyStopper, eval_one_epoch

def pgd_feat0_linf(model, x, y, criterion, eps, alpha, steps, clip_lo=-1.0, clip_hi=1.0, random_start=True, gen_mode="train"):
    if eps <= 0 or steps <= 0:
        return x.detach()
    was_training = model.training
    model.train() if gen_mode == "train" else model.eval()
    use_cudnn = (gen_mode == "train")
    x0 = x.detach()
    def _project(d0):
        d0 = d0.clamp(-eps, eps)
        return (x0[:, :, 0] + d0).clamp(clip_lo, clip_hi) - x0[:, :, 0]
    delta = torch.zeros_like(x0)
    if random_start:
        delta[:, :, 0].uniform_(-eps, eps)
        delta[:, :, 0] = _project(delta[:, :, 0])
    with torch.backends.cudnn.flags(enabled=use_cudnn):
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            loss = criterion(model(x0 + delta), y)
            grad, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                new0 = _project(delta[:, :, 0] + alpha * grad[:, :, 0].sign())
            delta = torch.zeros_like(x0)
            delta[:, :, 0] = new0
    model.train() if was_training else model.eval()
    return (x0 + delta).detach()

def train_one_epoch_adv(model, loader, optimizer, criterion, device, eps, alpha, steps, clip_lo, clip_hi, random_start, gen_mode):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = pgd_feat0_linf(model, x, y, criterion, eps, alpha, steps, clip_lo, clip_hi, random_start, gen_mode)
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_adv), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)

def eval_one_epoch_adv(model, loader, criterion, device, eps, alpha, steps, clip_lo, clip_hi, gen_mode):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = pgd_feat0_linf(model, x, y, criterion, eps, alpha, steps, clip_lo, clip_hi, random_start=False, gen_mode="eval")
        with torch.no_grad():
            loss = criterion(model(x_adv), y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)

def train_adv(model, train_loader, val_loader, config=None, device=None, eps=0.125, alpha=None, steps=40, monitor="adv", clip_lo=-1.0, clip_hi=1.0, random_start=True, gen_mode="train", seed=42, verbose=True):
    cfg = config or model.config
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = (eps / 4.0) if alpha is None else alpha
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta)
    history = TrainHistory()
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    for epoch in range(cfg.epochs):
        tr = train_one_epoch_adv(model, train_loader, optimizer, criterion, device, eps, alpha, steps, clip_lo, clip_hi, random_start, gen_mode)
        if monitor == "adv":
            va = eval_one_epoch_adv(model, val_loader, criterion, device, eps, alpha, steps, clip_lo, clip_hi, gen_mode)
        else:
            va = eval_one_epoch(model, val_loader, criterion, device)
        history.train_loss.append(tr)
        history.val_loss.append(va)
        improved = stopper.step(va, epoch)
        if improved:
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            history.best_val_loss = va
        if verbose:
            print(f"  epoch {epoch+1:3d}/{cfg.epochs}: train_adv={tr:.6f}  "
                  f"val_{monitor}={va:.6f}{' *' if improved else ''}")
        if stopper.should_stop:
            history.stopped_early = True
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best {stopper.best_epoch+1})")
            break
    if verbose:
        print(f"  done: best val_{monitor}={history.best_val_loss:.6f} at epoch {history.best_epoch+1}")
    return best_state, history
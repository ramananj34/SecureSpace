from dataclasses import dataclass

@dataclass(frozen=True)
class VendoredConfig:
    batch_size: int = 70
    window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    p: float = 0.13
    l_s: int = 250
    n_predictions: int = 10
    lstm_batch_size: int = 64
    epochs: int = 35
    patience: int = 10
    min_delta: float = 0.0003
    validation_split: float = 0.2
    layers: tuple = (80, 80)
    dropout: float = 0.3
    loss_metric: str = 'mse'
    optimizer: str = 'adam'
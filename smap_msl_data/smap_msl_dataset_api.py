from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import pandas as pd

#Default paths
DEFAULT_DATA_DIR = Path("")

#Channels excluded from the LSTM training corpus (M-6: flat_train AND extreme test-range outlier (test_max=258))
EXCLUDED_CHANNELS: frozenset[str] = frozenset({"M-6"})


#Manifest loading
def load_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    #Load the channel manifest
    path = data_dir / "channel_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Channel manifest not found at {path}.")
    return pd.read_csv(path)
def working_channels(spacecraft: Optional[str] = None, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    #Return the working set of channel IDs (usable, not excluded).
    m = load_manifest(data_dir)
    m = m[(m["status"] == "usable") & (~m["chan_id"].isin(EXCLUDED_CHANNELS))]
    if spacecraft is not None:
        m = m[m["spacecraft"] == spacecraft]
    return sorted(m["chan_id"].tolist())
def flat_anomaly_channels(spacecraft: Optional[str] = None, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    #Channels with flat training data but labeled test-set anomalies.
    m = load_manifest(data_dir)
    m = m[(m["status"] == "flat_train") & (~m["chan_id"].isin(EXCLUDED_CHANNELS))]
    if spacecraft is not None:
        m = m[m["spacecraft"] == spacecraft]
    return sorted(m["chan_id"].tolist())

#Quantizer
@dataclass(frozen=True)
class Quantizer:
    #Uniform b-bit signed quantizer mapping real-valued telemetry to F_2^b bits.
    x_min: float = -1.0
    x_max: float = 1.0
    b: int = 8
    @property
    def n_levels(self) -> int:
        return 1 << self.b #2^b
    @property
    def delta_lsb(self) -> float:
        return (self.x_max - self.x_min) / self.n_levels
    @property
    def delta_msb(self) -> float:
        return (1 << (self.b - 1)) * self.delta_lsb  #2^{b-1} * Δ_LSB
    def c_quant(self) -> float:
        #The c_quant constant: sum of (bit weight)^2 per coord. For uniform b-bit quantization: c_quant = (4^b - 1) / 3 * Δ_LSB^2. For b=8 with Δ_LSB = 2^{-7}: c_quant ≈ 1.333.
        return (self.n_levels ** 2 - 1) / 3.0 * self.delta_lsb ** 2
    def quantize(self, x: np.ndarray) -> np.ndarray:
        #Real to integer level in {0, ..., 2^b - 1}. Values outside [x_min, x_max] are clipped at the boundary.
        clipped = np.clip(x, self.x_min, self.x_max)
        #Map [x_min, x_max] -> [0, n_levels - 1]. Use floor and clip again to handle x = x_max exactly
        scaled = (clipped - self.x_min) / (self.x_max - self.x_min)
        levels = np.floor(scaled * self.n_levels).astype(np.int64)
        levels = np.clip(levels, 0, self.n_levels - 1)
        if self.b <= 8:
            return levels.astype(np.uint8)
        elif self.b <= 16:
            return levels.astype(np.uint16)
        else:
            return levels.astype(np.uint32)
    def dequantize(self, levels: np.ndarray) -> np.ndarray:
        #Integer level -> real value at the bin center. Mid-tread reconstruction: returns x_min + (level + 0.5) * Δ_LSB. This minimizes worst-case dequantization error to ±Δ_LSB/2 on in-range data.
        return self.x_min + (levels.astype(np.float64) + 0.5) * self.delta_lsb
    def levels_to_bits(self, levels: np.ndarray) -> np.ndarray:
        #Convert integer levels to a bit array with bits as the last axis.
        #Compute bits with broadcasting: shape (..., 1) >> shape (b,) -> (..., b)
        #Make sure we cast to a wide enough integer type for the shift.
        levels_int = levels.astype(np.int64)
        shifts = np.arange(self.b, dtype=np.int64)
        bits = ((levels_int[..., None] >> shifts) & 1).astype(np.uint8)
        return bits
    def bits_to_levels(self, bits: np.ndarray) -> np.ndarray:
        #Inverse of levels_to_bits.
        if bits.shape[-1] == self.b:
            #bits as last axis (canonical form)
            bits_grouped = bits
        elif bits.shape[-1] % self.b == 0:
            #flattened: split last axis into (n_coords, b)
            n_coords = bits.shape[-1] // self.b
            bits_grouped = bits.reshape(*bits.shape[:-1], n_coords, self.b)
        else:
            raise ValueError(f"bits last axis length {bits.shape[-1]} not compatible with b={self.b}")
        weights = (1 << np.arange(self.b, dtype=np.int64))
        levels = (bits_grouped.astype(np.int64) * weights).sum(axis=-1)
        if self.b <= 8:
            return levels.astype(np.uint8)
        elif self.b <= 16:
            return levels.astype(np.uint16)
        else:
            return levels.astype(np.uint32)

#Per-channel scaler
@dataclass(frozen=True)
class ChannelScaler:
    #Per-channel scaling and quantizer-range definition.
    chan_id: str
    train_min: float
    train_max: float
    def transform(self, x: np.ndarray) -> np.ndarray:
        #Scale x so that [train_min, train_max] maps to [-1, 1].
        if self.train_max - self.train_min < 1e-12:
            #Degenerate (flat_train) channel (just return zeros)
            return np.zeros_like(x, dtype=np.float64)
        return 2.0 * (x - self.train_min) / (self.train_max - self.train_min) - 1.0
    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        return self.train_min + 0.5 * (x_scaled + 1.0) * (self.train_max - self.train_min)

#Dataset
class SMAPMSLChannelDataset:
    #Sliding-window dataset for a single SMAP/MSL channel.
    def __init__(self,chan_id: str,split: str = "train",window: int = 250,stride: int = 1,data_dir: Path = DEFAULT_DATA_DIR):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.chan_id = chan_id
        self.split = split
        self.window = window
        self.stride = stride
        self.data_dir = Path(data_dir)
        #Load raw data
        raw_path = self.data_dir / split / f"{chan_id}.npy"
        if not raw_path.exists():
            raise FileNotFoundError(f"Channel data not found: {raw_path}")
        self._raw: np.ndarray = np.load(raw_path)  #shape (n_timesteps, 25)
        #Build the scaler from train statistics (always train, regardless of split)
        train_raw = np.load(self.data_dir / "train" / f"{chan_id}.npy")
        train_tele = train_raw[:, 0]
        self.scaler = ChannelScaler(chan_id=chan_id, train_min=float(train_tele.min()), train_max=float(train_tele.max()))
        #Pre-compute scaled telemetry stream
        self._scaled_tele: np.ndarray = self.scaler.transform(self._raw[:, 0]).astype(np.float64)
        #Command features pass through unchanged
        self._commands: np.ndarray = self._raw[:, 1:].astype(np.float64)
        #Length check
        if len(self._raw) <= self.window:
            raise ValueError(
                f"Channel {chan_id} {split} has {len(self._raw)} timesteps; "
                f"need > window={window}"
            )
    def __len__(self) -> int:
        #Number of (x_window, y_target) pairs available. Window i covers timesteps [i, i+window), target at i+window. So last valid i is len(raw) - window - 1.
        return (len(self._raw) - self.window) // self.stride
    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        i = idx * self.stride
        if i < 0 or i + self.window >= len(self._raw):
            raise IndexError(f"Index {idx} out of range")
        x_window = np.concatenate([self._scaled_tele[i : i + self.window, None], self._commands[i : i + self.window],],axis=1)
        y_target = float(self._scaled_tele[i + self.window])
        return x_window, y_target
    @property
    def n_timesteps(self) -> int:
        return len(self._raw)
    @property
    def telemetry_scaled(self) -> np.ndarray:
        #The scaled-telemetry stream (useful for inverse transforming predictions)
        return self._scaled_tele.copy()
    @property
    def telemetry_raw(self) -> np.ndarray:
        #The original (unscaled) telemetry stream
        return self._raw[:, 0].copy()
    @property
    def anomaly_mask(self) -> np.ndarray:
        #Boolean mask of length n_timesteps; True where a timestep falls inside any labeled anomaly sequence.
        mask = np.zeros(self.n_timesteps, dtype=bool)
        if self.split != "test":
            return mask
        labels_path = self.data_dir / "labeled_anomalies.csv"
        df = pd.read_csv(labels_path)
        rows = df[df["chan_id"] == self.chan_id]
        for _, row in rows.iterrows():
            sequences = ast.literal_eval(row["anomaly_sequences"])
            for s, e in sequences:
                #The ranges are inclusive [s, e].
                mask[s : e + 1] = True
        return mask
    def iter_windows(self) -> Iterator[tuple[np.ndarray, float]]:
        for i in range(len(self)):
            yield self[i]
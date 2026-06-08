from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FramePlan:
    channels: list
    window: int = 100
    b: int = 8
    header_bits: int = 400

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def slot_bits(self) -> int:
        return self.window * self.b

    @property
    def k(self) -> int:
        return self.header_bits + self.n_channels * self.slot_bits

def pack_frame(plan: FramePlan, windows: np.ndarray, quantizer, header_bits: np.ndarray | None = None) -> np.ndarray:
    windows = np.asarray(windows)
    assert windows.shape == (plan.n_channels, plan.window), \
        f"windows {windows.shape} != ({plan.n_channels}, {plan.window})"
    if header_bits is None:
        header_bits = np.zeros(plan.header_bits, dtype=np.int8)
    header_bits = np.asarray(header_bits, dtype=np.int8).ravel()
    assert header_bits.size == plan.header_bits, "header length mismatch"
    parts = [header_bits]
    for i in range(plan.n_channels):
        levels = quantizer.quantize(windows[i])
        bits = quantizer.levels_to_bits(levels).ravel()
        parts.append(bits.astype(np.int8))
    m_in = np.concatenate(parts)
    assert m_in.size == plan.k, f"packed {m_in.size} != k {plan.k}"
    return m_in

def unpack_frame(plan: FramePlan, m_in: np.ndarray, quantizer):
    m_in = np.asarray(m_in).ravel()
    assert m_in.size == plan.k, f"message {m_in.size} != k {plan.k}"
    header = m_in[:plan.header_bits].astype(np.int8)
    off = plan.header_bits
    windows = np.zeros((plan.n_channels, plan.window), dtype=np.float64)
    for i in range(plan.n_channels):
        bits = m_in[off:off + plan.slot_bits]
        off += plan.slot_bits
        levels = quantizer.bits_to_levels(bits)
        windows[i] = quantizer.dequantize(levels)
    return header, windows

def channel_slice(plan: FramePlan, channel_index: int) -> tuple[int, int]:
    start = plan.header_bits + channel_index * plan.slot_bits
    return start, start + plan.slot_bits
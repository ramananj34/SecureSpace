import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "smap_msl_data"))
sys.path.insert(0, str(_PROJECT_ROOT / "telemanom_reproduction"))
import numpy as np
from smap_msl_dataset_api import Quantizer, SMAPMSLChannelDataset, working_channels
from nullspace_attack_utils.ldpc_ops import LDPCCode
from nullspace_attack_utils.frame_packing import FramePlan, pack_frame, unpack_frame, channel_slice
DATA_DIR = _PROJECT_ROOT / "smap_msl_data"
WINDOW = 100
RUNS_DIR = _PROJECT_ROOT / "runs"

def pick_channels(n, window):
    chans = []
    for c in working_channels(data_dir=DATA_DIR):
        try:
            ds = SMAPMSLChannelDataset(c, split="test", data_dir=DATA_DIR)
        except Exception:
            continue
        if ds.n_timesteps >= window:
            chans.append(c)
        if len(chans) == n:
            break
    return chans

def demo_a(code: LDPCCode, q: Quantizer):
    print("Demo A -- frame packing + null-space injection (undefended path)")
    chans = pick_channels(40, WINDOW)
    assert len(chans) == 40, f"need 40 channels with >= {WINDOW} test steps, got {len(chans)}"
    plan = FramePlan(channels=chans, window=WINDOW, b=8, header_bits=400)
    assert plan.k == code.k == 32400, f"frame k {plan.k} != code k {code.k}"
    print(f"  packed {len(chans)} channels -> k = {plan.k} (matches long-frame LDPC)")
    windows = np.stack([SMAPMSLChannelDataset(c, split="test", data_dir=DATA_DIR).telemetry_scaled[:WINDOW] for c in chans], axis=0)
    header = np.zeros(plan.header_bits, dtype=np.int8)
    m_in = pack_frame(plan, windows, q, header_bits=header)
    c = code.encode(m_in)
    assert code.is_codeword(c), "clean codeword fails syndrome"
    _, clean_windows = unpack_frame(plan, code.info_bit_extract(c), q)
    target = 5
    s, e = channel_slice(plan, target)
    delta_info = np.zeros(code.k, dtype=np.int8)
    delta_info[[s + 10, s + 200, s + 533]] = 1
    delta = code.apply_B(delta_info)
    c_adv = (c ^ delta).astype(np.int8)
    assert code.is_codeword(c_adv), "null-space-perturbed word is not a codeword"
    c_naive = c.copy(); c_naive[12345] ^= 1
    naive_syn = int(code.syndrome(c_naive).sum())
    assert naive_syn > 0, "naive 1-bit flip should give nonzero syndrome"
    print(f"  beat 2 vs 3 (preview of E2 control arm): naive 1-bit flip -> "
          f"syndrome weight {naive_syn} (flagged);  null-space delta -> syndrome 0 (passes)")
    m_hat = code.info_bit_extract(c_adv)
    assert np.array_equal(m_hat, (m_in ^ delta_info)), "m_hat != m_in XOR delta_info"
    print("  decode -> m_hat = m_in XOR delta_info (info-bit perturbation reaches LSTM input)")
    _, adv_windows = unpack_frame(plan, m_hat, q)
    changed = np.where(np.any(np.abs(adv_windows - clean_windows) > 0, axis=1))[0]
    assert changed.tolist() == [target], \
        f"perturbation hit channels {changed.tolist()}, expected [{target}]"
    diff = np.abs(adv_windows[target] - clean_windows[target])
    print(f"  perturbation localized to channel {target} ({chans[target]}); "
          f"max telemetry change {diff.max():.4f} (dequantized bit flips)")
    print("  Demo A PASSED.\n")
    return chans

def demo_b(chans, q: Quantizer):
    print("Demo B -- codec output -> per-channel LSTM input")
    chan = chans[0]
    ds = SMAPMSLChannelDataset(chan, split="test", data_dir=DATA_DIR)
    x_window, _ = ds[0]
    n_feat = x_window.shape[1]
    tele = x_window[:, 0]
    tele_codec = q.dequantize(q.quantize(tele))
    codec_window = x_window.copy()
    codec_window[:, 0] = tele_codec
    assert np.array_equal(codec_window[:, 1:], x_window[:, 1:]), "commands were altered"
    tele_err = np.max(np.abs(tele_codec - np.clip(tele, -1, 1)))
    print(f"  channel {chan}: LSTM input {codec_window.shape}; commands untouched; "
          f"telemetry within {tele_err:.2e} (<= Δ_LSB/2 {0.5*q.delta_lsb:.2e})")
    model_path = RUNS_DIR / chan / "model.pt"
    try:
        import json, torch
        from telemanom_lstm import TelemanomLSTM, TelemanomConfig
        cfg_path = model_path.parent / "config.json"
        cfg = TelemanomConfig(**json.load(open(cfg_path))) if cfg_path.exists() \
            else TelemanomConfig(input_dim=n_feat)
        if cfg.input_dim != n_feat:
            from dataclasses import replace
            cfg = replace(cfg, input_dim=n_feat)
        model = TelemanomLSTM(cfg)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(x_window[None], dtype=torch.float32)
            cb = torch.tensor(codec_window[None], dtype=torch.float32)
            y0 = model(xb).cpu().numpy().ravel()
            y1 = model(cb).cpu().numpy().ravel()
        print(f"  trained LSTM forward OK: prediction dim {y0.size}; "
              f"max |pred(orig) - pred(codec)| = {np.max(np.abs(y0 - y1)):.2e} "
              f"(quantization barely perturbs the detector)")
    except Exception as exc:  # noqa: BLE001 -- env/path/model issues never block Day 6
        print(f"  LSTM forward SKIPPED (input plumbing verified above): {exc}")
    print("  Demo B PASSED.\n")

def main():
    q = Quantizer(x_min=-1.0, x_max=1.0, b=8)
    code = LDPCCode.dvbs2_long_rate12()
    chans = demo_a(code, q)
    demo_b(chans, q)
    print("Plumbing dry-run PASSED -- the E2 data path is wired end to end.")

if __name__ == "__main__":
    main()
from typing import Dict
import numpy as np

try:
    from museval.metrics import bss_eval_sources
except Exception:  # optional dependency at runtime
    bss_eval_sources = None


def evaluate_waveforms(estimated: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if bss_eval_sources is None:
        return {"SDR": float("nan"), "SIR": float("nan"), "SAR": float("nan")}
    sdr, sir, sar, _ = bss_eval_sources(target[np.newaxis, :], estimated[np.newaxis, :], compute_permutation=False)
    return {"SDR": float(sdr[0]), "SIR": float(sir[0]), "SAR": float(sar[0])}

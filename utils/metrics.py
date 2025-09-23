from typing import Dict
import numpy as np

try:
    from museval.metrics import bss_eval_sources  # type: ignore
except Exception:  # optional dependency at runtime
    bss_eval_sources = None


def _to_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    # If stereo or batched, pick the first channel/sample
    return np.squeeze(x)[..., 0]


def _si_sdr(estimated: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Scale-Invariant SDR (in dB) for 1D signals.

    Args:
        estimated: predicted waveform (1D numpy array)
        target: reference waveform (1D numpy array)
        eps: numerical stability constant
    Returns:
        SI-SDR in dB (float)
    """
    e = estimated.astype(np.float64)
    t = target.astype(np.float64)
    # Align lengths
    L = min(e.shape[-1], t.shape[-1])
    if L <= 0:
        return float("nan")
    e = e[:L]
    t = t[:L]

    t_energy = np.sum(t * t) + eps
    s_target = (np.sum(e * t) / t_energy) * t
    e_noise = e - s_target
    num = np.sum(s_target * s_target)
    den = np.sum(e_noise * e_noise) + eps
    return 10.0 * np.log10((num + eps) / den)


def evaluate_waveforms(
    estimated: np.ndarray,
    target: np.ndarray,
    sr: int | None = None,
    use_museval: bool = False,
    window_seconds: float = 3.0,
    hop_seconds: float | None = None,
) -> Dict[str, float]:
    """Evaluate separation quality between estimated and target waveforms.

    Defaults to SI-SDR to avoid the extremely large memory usage of museval's
    BSS eval on long signals. If use_museval=True and museval is available,
    runs a windowed BSS Eval with small windows to reduce memory pressure.

    Args:
        estimated: predicted waveform (1D numpy array)
        target: reference waveform (1D numpy array)
        sr: sample rate, only used to convert window_seconds to samples for museval
        use_museval: if True and museval is installed, compute (SDR/SIR/SAR) via
            bss_eval_sources with sliding windows. Otherwise, compute SI-SDR only.
        window_seconds: window length in seconds for museval (default 3.0s)
        hop_seconds: hop length in seconds for museval (default window/2)
    Returns:
        dict with keys: SDR, SIR, SAR (SIR/SAR are NaN when SI-SDR is used)
    """
    est = _to_1d(estimated)
    tgt = _to_1d(target)

    # Fast, memory-safe path
    if not use_museval or bss_eval_sources is None:
        sdr = _si_sdr(est, tgt)
        return {"SDR": float(sdr), "SIR": float("nan"), "SAR": float("nan")}

    # Museval windowed path
    try:
        if sr is None:
            # Fallback: infer a plausible sr from length to make windows modest
            # This only affects window sizing, not the signals themselves.
            sr = 44100
        w = max(256, int(window_seconds * sr))
        h = int(hop_seconds * sr) if hop_seconds is not None else max(128, w // 2)
        w = min(w, len(tgt), len(est))
        if w < 2:
            sdr = _si_sdr(est, tgt)
            return {"SDR": float(sdr), "SIR": float("nan"), "SAR": float("nan")}

        # Shape (nsrc, nsamples)
        ref = tgt[np.newaxis, :]
        hyp = est[np.newaxis, :]
        sdr, sir, sar, _ = bss_eval_sources(ref, hyp, compute_permutation=False, window=w, hop=h)
        # Aggregate across frames if needed
        def _agg(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return float(np.nanmean(x))
            # If museval returns shape (frames,)
            return float(np.nanmean(x))

        return {"SDR": _agg(sdr), "SIR": _agg(sir), "SAR": _agg(sar)}
    except Exception:
        # Fallback to SI-SDR on any museval failure
        sdr = _si_sdr(est, tgt)
        return {"SDR": float(sdr), "SIR": float("nan"), "SAR": float("nan")}

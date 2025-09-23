from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.no_grad()
def spectrogram_segment_infer(
    model,
    mix_norm: torch.Tensor,
    *,
    segment_frames: int,
    overlap_frames: int = 0,
    device: Optional[torch.device] = None,
    generate_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """Run model.generate_instrumental on spectrogram in time segments and stitch results.

    Args:
        model: CounterfactualDiffusion-like module with .generate_instrumental(mix, **kwargs)
        mix_norm: (1, 1, F, T) normalized magnitude spectrogram of the mixture
        segment_frames: window size in STFT frames
        overlap_frames: overlap between successive windows in frames (0 for no overlap)
        device: device for model inference; defaults to mix_norm.device
        generate_kwargs: extra kwargs passed to model.generate_instrumental

    Returns:
        instrumental_norm: (1, 1, F, T) predicted instruments spectrogram (normalized space)
    """
    assert mix_norm.dim() == 4 and mix_norm.size(0) == 1 and mix_norm.size(1) == 1, "mix_norm must be (1,1,F,T)"
    if device is None:
        device = mix_norm.device
    if generate_kwargs is None:
        generate_kwargs = {}

    _, _, F, T = mix_norm.shape
    seg = max(1, int(segment_frames))
    ov = max(0, int(overlap_frames))
    step = max(1, seg - ov)

    if seg >= T:
        # single-shot
        return model.generate_instrumental(mix_norm.to(device), **generate_kwargs).to(mix_norm.dtype).cpu() if device.type != "cpu" else model.generate_instrumental(mix_norm, **generate_kwargs)

    out = torch.zeros_like(mix_norm)
    weight = torch.zeros((T,), device=mix_norm.device, dtype=mix_norm.dtype)

    t = 0
    while t < T:
        end = min(t + seg, T)
        cur_len = end - t
        seg_tensor = mix_norm[..., t:end]
        # pad to seg on the right if needed
        if cur_len < seg:
            pad = seg - cur_len
            seg_tensor = torch.nn.functional.pad(seg_tensor, (0, pad))

        seg_tensor = seg_tensor.to(device)
        pred = model.generate_instrumental(seg_tensor, **generate_kwargs)
        pred = pred[..., :cur_len]  # crop padding
        out[..., t:end] += pred.to(out.dtype).to(out.device)
        weight[t:end] += 1.0

        if end == T:
            break
        t += step

    # Normalize overlap by counts
    weight = weight.clamp_min(1.0).view(1, 1, 1, T)
    out = out / weight
    return out

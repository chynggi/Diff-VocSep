# Diff-MSST: Diffusion-based Counterfactual Vocal Separation

Minimal runnable scaffold following `diffusion-vocal-separation-guide.md`.

## Setup

1) Create environment and install deps

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PyTorch install fails, install matching CUDA wheel first (example for CUDA 12.4):

```pwsh
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

2) Prepare MUSDB18 (optional for training)

```pwsh
python -c "import musdb; musdb.DB(root='data/musdb18', download=True)"
```

## Train (toy)

```pwsh
python train.py --config config.yaml
```

Training logs (TensorBoard):

```pwsh
tensorboard --logdir .\logs\tb --port 6006
```

## Inference

```pwsh
python inference.py --config config.yaml --model_path checkpoints/last.pt --input path\to\song.wav --output vocals.wav
```

Notes
- Shapes: (B, C, F, T) for spectrograms; model input is concat([x_t, mixture]).
- Normalization: log1p + min-max per-batch; keep stats for inverse before iSTFT.
- This scaffold aims for clarity, not SOTA performance. Tune configs, add DDIM, metrics, and logging as needed.
- Validation/Checkpoint: validates every 1000 steps and saves last/best checkpoints (configurable in config.yaml).

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

Use MUSDB18-HQ (local folders with mixture.wav and stems):

1) Folder layout per track

```
<musdbhq_root>/
	<artist> - <title>/
		mixture.wav
		drums.wav
		bass.wav
		vocals.wav
		other.wav
```

2) Set config

```yaml
data:
	dataset: musdbhq
	musdbhq_root: ./data/musdb18hq
```

3) Run train (same command)

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

## 원본 논문 출처

- **논문 제목:** 악기 종류에 무관한 보컬 분리를 위한 디퓨전 기반 반사실적 생성 기법 : A Diffusion based Counterfactual Generation Method for Instrument-Independent Vocal Separation: [논문 링크](https://s-space.snu.ac.kr/handle/10371/222058?mode=simple)
- **PDF:** [다운로드](https://dcollection.snu.ac.kr/public_resource/pdf/000000188545_20250923132518.pdf)
- **저자:** 강명오
- **기관:** 서울대학교 대학원

해당 논문을 참고하여 프로젝트를 개발하였습니다.
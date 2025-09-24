# Diff-VocSep: Diffusion-based Counterfactual Vocal Separation

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

During training you'll see two progress bars:
- An outer "Epochs" bar with average loss and best SDR.
- An inner per-epoch "Train e/E" bar with current loss and learning rate.

Validation also shows a short progress bar (up to `train.val_batches`) with the running SDR.

Key training options in `config.yaml`:
- train.loss_voc_weight: weight for auxiliary vocal loss (mixture − x0_pred vs GT vocals)
- diffusion.timesteps: total DDPM steps used in training
- diffusion.use_ddim + diffusion.ddim_steps + diffusion.eta: enable DDIM during validation/inference for faster sampling (eta=0.0 is deterministic)

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

Auto-download: If `dataset: musdbhq` and the folder is empty/missing, code will attempt to download a ZIP from Google Drive via gdown (id: 1ieGcVPPfgWg__BTDlIGi1TpntdOWwwdn) into `musdbhq_root` and extract it. You can pre-install gdown or it will raise an error asking to install it.

Training logs (TensorBoard):

```pwsh
tensorboard --logdir .\logs\tb --port 6006
```

## Inference

```pwsh
python inference.py --config config.yaml --model_path checkpoints/last.pt --input path\to\song.wav --output vocals.wav
```

Notes
- Set `diffusion.use_ddim: true` and tune `diffusion.ddim_steps` for a speed/quality trade-off (e.g., 20–100). `eta: 0.0` keeps it deterministic.
- Model infers instrumental counterfactual and subtracts from mixture in normalized spec space; outputs vocals via iSTFT with mixture phase.

Notes
- Shapes: (B, C, F, T) for spectrograms; model input is concat([x_t, mixture]).
- Normalization: log1p + min-max per-batch; keep stats for inverse before iSTFT.
- This scaffold aims for clarity, not SOTA performance. Tune configs, DDIM steps, metrics, and logging as needed.
- Validation/Checkpoint: validates every 1000 steps and saves last/best checkpoints (configurable in config.yaml).
 - Progress bars use `tqdm` (auto backend). Console prints during validation use `tqdm.write()` to avoid breaking the bars.

## TPU Training (torch-xla)

TPU는 로컬 Windows에서 직접 사용할 수 없고, Google Colab(TPU v2/v3) 또는 GCP TPU VM에서 사용하세요. 본 레포는 TPU 전용 학습 스크립트 `train_tpu.py`를 제공합니다.

1) TPU 환경에서 torch-xla 설치

```pwsh
pip install "torch==2.*" "torchvision==0.*" "torchaudio==2.*"
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

2) 학습 실행

```pwsh
python train_tpu.py --config config.yaml
```

메모
- 분산 학습은 XLA 프로세스별로 자동 스폰됩니다(`xmp.spawn`). 데이터셋은 `DistributedSampler`로 분할됩니다.
- AMP는 XLA의 `bfloat16` 자동 캐스트를 사용합니다.
- 검증/체크포인트는 마스터(rank 0)에서만 수행해 간단히 집계합니다.
- Colab에서는 런타임 유형을 TPU로 전환 후 위 명령을 실행하세요.

### Troubleshooting
- GroupNorm errors: channel counts are handled adaptively, but if you modify `channels_mult`, ensure at least three scales (e.g., `[1,2,4]`).
- Missing optional deps (museval, gdown, py7zr) will disable specific features; install them if needed.

## 원본 논문 출처

- **논문 제목:** 악기 종류에 무관한 보컬 분리를 위한 디퓨전 기반 반사실적 생성 기법 : A Diffusion based Counterfactual Generation Method for Instrument-Independent Vocal Separation: [논문 링크](https://s-space.snu.ac.kr/handle/10371/222058?mode=simple)
- **PDF:** [다운로드](https://dcollection.snu.ac.kr/public_resource/pdf/000000188545_20250923132518.pdf)
- **저자:** 강명오
- **기관:** 서울대학교 대학원

해당 논문을 참고하여 프로젝트를 개발하였습니다.

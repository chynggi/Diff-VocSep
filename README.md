# Diff-VocSep: Diffusion-based Counterfactual Vocal Separation

Diff-VocSep는 혼합 오디오에서 보컬을 분리하기 위해, 악기-only 스펙트럼을 생성한 뒤 혼합 스펙트럼에서 빼는(counterfactual) 전략을 사용하는 확산 모델 구현체입니다. 본 레포는 DDSP-SVC를 참조하여 얕은 디퓨전(shallow diffusion), 고급 샘플러(DDIM/UniPC/DPM-Solver 유사, 연속시간 버전 포함), Conformer 기반 노이즈 예측 백본 등을 제공합니다.

## 주요 특징
- Counterfactual 조건부 생성: 입력 혼합 스펙트럼을 조건으로 악기-only 스펙트럼을 생성 후 보컬=혼합−악기
- 얕은 디퓨전(Shallow diffusion): 초기 prior(x0 또는 x_k)에서 k스텝 역과정만 수행하여 빠른 추론
- 고급 샘플러:
	- DDPM, DDIM, UniPC(PC 방식), DPM-Solver(2차 근사) + 각 연속시간(CT) 변형 제공
	- beta 스케줄: linear | cosine, CT 임베딩: tau | logsnr
- 백본 선택: U-Net 또는 Conformer(시간/주파수/혼합축 처리)
- 세그먼트 추론: 긴 입력을 구간 단위로 처리해 메모리 사용량 절감
- 데이터 증강(옵션): Pitch shift(pedalboard)

## 설치

필요 패키지는 `requirements.txt` 참고:

```pwsh
# (선택) 가상환경 생성/활성화 후
pip install -r requirements.txt
```

- GPU 권장(PyTorch CUDA), CPU도 동작
- pitch 증강 사용 시 pedalboard가 설치되어야 합니다(이미 requirements에 포함)

## 구성 및 설정

설정 파일: `config.yaml`
- 데이터: MUSDB18/MUSDB18-HQ 경로, 세그먼트 길이, 피치 증강 옵션 등
- 오디오: STFT 파라미터
- 학습: 배치/에폭/스케줄/AMP/로그 주기
- 확산(diffusion):
	- `timesteps`, `beta_start`, `beta_end`, `beta_schedule: linear|cosine`
	- 샘플러 선택: `sampler: ddpm|ddim|unipc|dpm-solver|ddim-ct|unipc-ct|dpm-solver-ct`
	- 연속시간 임베딩: `ct_embed: tau|logsnr` (연속시간 샘플러 사용 시 유효)
	- `ddim_steps`: 고속 샘플러에서 사용할 스텝 수
- 모델(model):
	- `model_type: unet|conformer`
	- `model_kwargs`: conformer 하이퍼파라미터 및 축 옵션(axis: time|freq|mixed)

예시(config):
```yaml
diffusion:
	beta_schedule: cosine
	sampler: dpm-solver-ct
	ct_embed: logsnr
	ddim_steps: 20
model:
	model_type: conformer
	model_kwargs:
		d_model: 128
		n_heads: 4
		d_ff: 256
		num_layers: 6
		kernel_size: 31
		dropout: 0.0
		axis: mixed
```

## 학습

```pwsh
python train.py --config config.yaml
```
- 데이터셋: `data.dataset`에 따라 MUSDB 또는 MUSDB-HQ 사용
- 증강: `data.pitch_aug.enabled: true`로 활성화(훈련 전용 권장)

## 추론(보컬 분리)

단일 파일 추론:
```pwsh
python inference.py --input path\to\mix.wav --output vocals.wav
```

스테레오 출력(스테레오 입력 유지):
```pwsh
python inference.py --input path\to\mix_stereo.wav --output vocals_stereo.wav --stereo-output
```
메모
- 모델은 모노 스펙트럼으로 악기-only를 추정한 뒤, 모노 보컬 마그니튜드로 TF 마스크를 만들고 이를 각 채널의 복소 스펙트럼에 적용해 스테레오 보컬을 복원합니다(스테레오 이미지 유지).
- 입력이 모노면 출력도 모노입니다.

세그먼트 추론(긴 파일 권장):
```pwsh
python inference.py --input mix.wav --segment-seconds 8 --overlap-seconds 1
```

얕은 디퓨전(초기 prior 제공):
```pwsh
python inference.py `
	--input mix.wav `
	--init-instrumental coarse_instrumental.wav `
	--shallow-k 50 `
	--output vocals.wav
```
- prior가 이미 x_k라면 `--no-forward-noise` 추가
- prior가 없고 얕은 디퓨전만 원하면 `--shallow-k`만 지정(혼합 스펙을 약한 prior로 사용)

연속시간 샘플러와 조합:
- `diffusion.sampler: dpm-solver-ct` 또는 `unipc-ct`
- `diffusion.ct_embed: logsnr` 권장

## 권장값 & 팁
- 샘플러
	- DPM‑Solver‑CT: 20~30 스텝 빠르고 안정적
	- UniPC‑CT: 20~40 스텝, 결과 안정성 좋음
	- DDIM: 50~100 스텝, 기준선
- 얕은 디퓨전
	- k: 30~80 구간 탐색, prior 품질/목표 속도에 맞추어 조정
- Conformer
	- d_model: 128~256, n_heads: 4~8, layers: 4~8
	- axis: time부터 시작 → mixed로 확장

## 체크포인트
- `checkpoints/last.pt`, `checkpoints/best.pt` 저장(학습 중 설정에 따름)

## 라이선스
- 본 프로젝트는 MIT 라이선스로 제공됩니다(아래 LICENSE 참고).

---

## 참고
- MUSDB18: https://sigsep.github.io/datasets/musdb.html
- DDPM/DDIM/UniPC/DPM‑Solver 관련 원문 및 구현을 참고해 고속 샘플러를 경량 반영했습니다.


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

## Acknowledgement

[DDSP-SVC](https://github.com/yxlllc/ddsp-svc)

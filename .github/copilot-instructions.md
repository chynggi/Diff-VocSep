# Copilot Instructions: Diff-MSST (Diffusion-based Counterfactual Vocal Separation)

Goal: Build a Python system that separates vocals by generating an instrumental counterfactual via diffusion and subtracting it from the mixture.

Architecture (big picture)
- Audio I/O + STFT stack: load with torchaudio, compute STFT/iSTFT; keep phase to reconstruct audio. Normalize magnitudes with log scaling and batch min–max.
- Diffusion U-Net: ResBlocks (GroupNorm + Conv2d) with sinusoidal timestep embeddings (linear projection). Add self-attention on higher-channel feature maps.
- Gaussian diffusion: linear beta schedule; implement q(x_t|x_{t-1}) forward noising and reverse denoising loop; support faster sampling via DDIM.
- Counterfactual conditioning: concatenate [noisy input, mixture] along channel dim to condition the model to generate instruments-only spectrograms. Vocals = mixture − instruments.

Data & Dataloader
- Use musdb to load MUSDB18; sample fixed-length segments; convert stereo→mono consistently before STFT.
- Precompute/return: normalized magnitude spectrograms plus phase for reconstruction; ensure consistent hop/fft settings across train/val/infer.

Training
- Dual objectives: (1) counterfactual (instruments-only) generation loss on spectrograms, (2) vocal separation loss derived from mixture − instruments.
- Optimizer: AdamW; LR schedule: cosine annealing; enable gradient clipping and mixed precision (AMP) for efficiency.
- Logging: TensorBoard or Weights & Biases; track losses, SDR/SIR/SAR, and periodic audio/image samples (specs + reconstructions).
- Checkpointing/resume for long runs; validate regularly on held-out segments.

Inference
- Load audio → spectrogram → sample instrumental counterfactual via reverse diffusion (DDPM or DDIM) → vocals = mixture − instruments → iSTFT to waveform → save.

Conventions & patterns to follow
- Shapes: spectrogram tensors organized as (batch, channels, freq, time); condition by channel concat.
- Timestep embedding: sinusoidal → MLP/linear projection → added to ResBlocks.
- Normalization: log1p magnitude + batch min–max; retain stats if you need inverse scaling before iSTFT.
- Beta schedule configurable; expose total timesteps T and schedule in config for speed/quality trade-offs.
- Metrics: compute SDR/SIR/SAR on reconstructed waveforms (e.g., via museval-compatible utilities).

Dependencies & integration points
- Core: PyTorch, torchaudio, numpy; Data: musdb (MUSDB18); Logging: TensorBoard or Weights & Biases.
- Optional: DDIM sampler; mixed precision via torch.cuda.amp.

Notes for contributors (AI agents)
- Prefer efficient sampling during training previews (few steps/DDIM) and full-quality sampling for validation examples.
- Keep device handling explicit (cpu/cuda) and guard autocast blocks; clip gradients to stabilize diffusion training.
- When subtracting spectrograms, ensure consistent scaling and handle negative values safely before iSTFT.

Unclear/incomplete? If repo structure or configs are missing, generate minimal modules for: preprocessing (STFT/iSTFT), diffusion UNet, diffusion process, counterfactual sampler, MUSDB dataloader, train/validate/infer scripts, and logging.


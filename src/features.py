"""Stage 1 - the audio front-end: waveform -> log-mel spectrogram.

Numbers to memorise (Whisper convention, 16 kHz):
    n_fft      = 400 samples = 25 ms  (the analysis window)
    hop_length = 160 samples = 10 ms  (the frame shift)  -> 100 frames per second
    n_mels     = 80 (tiny/base/small/medium) or 128 (large-v3)
Whisper then pads/trims to 30 s => 3000 frames, and its two stride-2-in-total conv layers
halve that to 1500 vectors => 50 vectors per second reaching the transformer.

This file gives you two paths that must agree:
    hf_log_mel()     - what transformers' WhisperFeatureExtractor does
    manual_log_mel() - the same thing in ~15 lines of torch, so you can explain it
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoFeatureExtractor


def get_feature_extractor(name: str = "openai/whisper-tiny"):
    return AutoFeatureExtractor.from_pretrained(name)


def hf_log_mel(wav: np.ndarray, fe, pad_seconds: float | None = None) -> torch.Tensor:
    """-> (n_mels, n_frames). If pad_seconds is None, Whisper's own 30 s padding is used."""
    kwargs = dict(sampling_rate=fe.sampling_rate, return_tensors="pt")
    if pad_seconds is not None:
        n = int(pad_seconds * fe.sampling_rate)
        kwargs.update(padding="max_length", max_length=n, truncation=True)
    return fe(wav, **kwargs).input_features[0]


def manual_log_mel(wav: np.ndarray, fe) -> torch.Tensor:
    """Whisper's log-mel, written out. Compare with hf_log_mel to prove it is the same."""
    n_fft, hop = fe.n_fft, fe.hop_length
    x = torch.as_tensor(wav, dtype=torch.float32)

    # 1) short-time Fourier transform: sliding 25 ms windows, every 10 ms.
    window = torch.hann_window(n_fft)
    stft = torch.stft(x, n_fft, hop, window=window, center=True,
                      pad_mode="reflect", return_complex=True)
    # Whisper drops the last (incomplete) frame.
    power = stft[..., :-1].abs() ** 2                     # (n_fft/2+1, T) linear-freq energy

    # 2) squeeze 201 linear bins into 80 perceptual mel bins (triangular filterbank).
    mel_filters = torch.as_tensor(fe.mel_filters, dtype=torch.float32)   # (201, 80)
    mel = mel_filters.T @ power                            # (80, T)

    # 3) compress dynamic range (hearing is logarithmic) + Whisper's normalisation.
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)   # floor 80 dB below peak
    return (log_spec + 4.0) / 4.0                              # roughly zero-mean/unit-ish


def frames_to_encoder_tokens(n_frames: int) -> int:
    """Whisper's conv stack: conv1 stride 1, conv2 stride 2 -> halves the time axis."""
    return n_frames // 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--encoder", default="openai/whisper-tiny")
    ap.add_argument("--pad-seconds", type=float, default=None)
    args = ap.parse_args()

    from .config import DataConfig
    from .data import SpeechTextDataset

    fe = get_feature_extractor(args.encoder)
    ds = SpeechTextDataset(DataConfig(root=args.root))
    s = ds[0]
    wav = s["audio"]

    print(f"feature extractor : {type(fe).__name__}")
    print(f"  sampling_rate={fe.sampling_rate}  n_fft={fe.n_fft} ({fe.n_fft/fe.sampling_rate*1000:.0f} ms)"
          f"  hop={fe.hop_length} ({fe.hop_length/fe.sampling_rate*1000:.0f} ms)"
          f"  n_mels={fe.feature_size}  chunk={fe.chunk_length}s")
    print()
    print(f"waveform          : {wav.shape}  = {len(wav)/fe.sampling_rate:.2f} s")

    mel_hf = hf_log_mel(wav, fe, pad_seconds=args.pad_seconds)
    # compare like for like: the extractor pads the WAVEFORM before the STFT, so we pad too
    n_padded = mel_hf.shape[-1] * fe.hop_length
    wav_padded = np.pad(wav, (0, max(0, n_padded - len(wav))))[:n_padded]
    mel_manual = manual_log_mel(wav_padded, fe)
    print(f"log-mel (HF)      : {tuple(mel_hf.shape)}  (n_mels, frames)"
          f"   -> {frames_to_encoder_tokens(mel_hf.shape[-1])} encoder tokens")
    print(f"log-mel (manual)  : {tuple(mel_manual.shape)}")

    diff = (mel_hf - mel_manual).abs().max().item()
    print(f"max |HF - manual| : {diff:.2e}   <- must be ~0")
    T = int(np.ceil(len(wav) / fe.hop_length))   # frames that contain real audio
    print()
    print("compression chain :")
    n = len(wav)
    print(f"  {n:>8} samples  (1 vector/sample, 16000 per second)")
    print(f"  {T:>8} frames   x 80 mel bins        ({T/(n/16000):.0f} per second)")
    print(f"  {frames_to_encoder_tokens(T):>8} encoder tokens                 "
          f"({frames_to_encoder_tokens(T)/(n/16000):.0f} per second)")
    print(f"  {frames_to_encoder_tokens(T)//4:>8} LLM tokens after stack-4 projector "
          f"({frames_to_encoder_tokens(T)//4/(n/16000):.0f} per second)")


if __name__ == "__main__":
    main()

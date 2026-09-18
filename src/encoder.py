"""Stage 2 - the audio encoder.

We use the *encoder half* of Whisper and throw its decoder away: the LLM is our decoder now.

Whisper encoder, in order:
    conv1 (k=3, s=1) -> GELU        local context over mel frames
    conv2 (k=3, s=2) -> GELU        downsample time by 2  (100 fps -> 50 fps)
    + sinusoidal absolute position embeddings (fixed table of 1500 = 30 s)
    N transformer blocks, BIDIRECTIONAL self-attention (no causal mask - we see the whole clip)
    final layer_norm

Two things worth knowing for interviews:
  * the HF forward *requires* exactly 3000 mel frames (30 s) because of that position table.
    Here we also provide a variable-length forward that slices the table, so short CPU-friendly
    clips do not waste 90 % of the compute on silence.
  * the encoder takes NO attention mask. Padded frames are attended to like real audio.
    That is also true of stock Whisper (it always pads to 30 s); we handle padding downstream
    by telling the LLM how many audio tokens are real.
"""
from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel


def _dtype_kw(dtype):
    """transformers <4.56 uses `torch_dtype=`, v5 uses `dtype=`. Support both."""
    from transformers import __version__ as v
    major, minor = (int(x) for x in v.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


class WhisperAudioEncoder(nn.Module):
    def __init__(self, name: str = "openai/whisper-tiny", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.name = name
        full = WhisperModel.from_pretrained(name, **_dtype_kw(dtype))
        self.encoder = full.encoder          # decoder is dropped entirely
        self.config = self.encoder.config
        self.hidden_size = self.config.d_model
        self.max_source_positions = self.config.max_source_positions   # 1500

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def output_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
        """conv2 has stride 2 -> the time axis is halved."""
        return mel_lengths // 2

    def _run_layer(self, layer, hidden):
        """Call one WhisperEncoderLayer, tolerating signature changes across versions."""
        params = inspect.signature(layer.forward).parameters
        kwargs = {}
        if "attention_mask" in params:
            kwargs["attention_mask"] = None
        if "layer_head_mask" in params:
            kwargs["layer_head_mask"] = None
        out = layer(hidden, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    # ---------------------------------------------------------------- forwards
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_mel) with T_mel even and <= 3000. -> (B, T_mel//2, d_model)."""
        enc = self.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)                              # (B, T, d)
        t = x.shape[1]
        if t > self.max_source_positions:
            raise ValueError(f"{t} frames > encoder position table {self.max_source_positions}")
        x = x + enc.embed_positions.weight[:t].to(x.dtype)  # slice, do not interpolate
        for layer in enc.layers:
            x = self._run_layer(layer, x)
        return enc.layer_norm(x)

    def forward_hf(self, mel: torch.Tensor) -> torch.Tensor:
        """The stock path - only valid for exactly 3000 mel frames."""
        return self.encoder(mel).last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="openai/whisper-tiny")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    enc = WhisperAudioEncoder(args.encoder).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"{args.encoder}: d_model={enc.hidden_size} layers={len(enc.encoder.layers)} "
          f"heads={enc.config.encoder_attention_heads} params={n_params/1e6:.1f}M")
    print(f"position table  : {tuple(enc.encoder.embed_positions.weight.shape)} "
          f"(= {enc.max_source_positions/50:.0f} s at 50 fps)")
    print()
    print("module tree (top level):")
    for name, mod in enc.encoder.named_children():
        print(f"  {name:16s} {type(mod).__name__}")
    print()

    n_mels = enc.config.num_mel_bins
    with torch.no_grad():
        # 1) equivalence check on a full 30 s input
        mel30 = torch.randn(1, n_mels, 3000)
        a, b = enc(mel30), enc.forward_hf(mel30)
        print(f"30 s input  ours {tuple(a.shape)} vs HF {tuple(b.shape)}  "
              f"max|diff|={(a - b).abs().max().item():.2e}  <- must be ~0")

        # 2) the short input HF refuses
        t = int(args.seconds * 100)
        t -= t % 2
        mel = torch.randn(2, n_mels, t)
        out = enc(mel)
        print(f"{args.seconds:g} s input  mel {tuple(mel.shape)} -> encoder {tuple(out.shape)}  "
              f"({out.shape[1]/args.seconds:.0f} tokens/s)")
        try:
            enc.forward_hf(mel)
        except Exception as e:                                  # noqa: BLE001
            print(f"  stock HF forward on the same input: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:90]}")


if __name__ == "__main__":
    main()

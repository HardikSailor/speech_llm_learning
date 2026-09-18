"""Stage 4a - the bridge: encoder space (d_enc, 50 fps) -> LLM token space (d_llm, few fps).

Why a projector at all?
  * dimensions differ (384 vs 576 here),
  * more importantly the *rate* differs: 50 audio vectors/s against ~3 text tokens/s.
    Attention is O(T^2), so 30 s of audio at 50 fps = 1500 tokens would dominate the context.
    Every real system downsamples. This is the single biggest design knob in a speech-LLM.

Three implementations, all with the same signature:
    forward(hidden (B,T,d_enc), lengths (B,)) -> (B, T', d_llm), lengths' (B,)

  stack_mlp : concatenate k neighbouring frames, one MLP.  (SLAM-ASR / Qwen2-Audio style)
              cheap, keeps a monotonic time axis, T' = ceil(T/k).
  conv      : strided Conv1d. Same rate reduction, smoother, slightly more params.
  qformer   : N learned queries cross-attend to the audio.  (SALMONN / BLIP-2 style)
              T' = N, FIXED, i.e. constant cost but a bottleneck for long audio.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    ar = torch.arange(max_len, device=lengths.device)
    return ar[None, :] < lengths[:, None]           # True = keep


class StackMLPProjector(nn.Module):
    def __init__(self, d_enc: int, d_llm: int, k: int = 4, hidden_mult: int = 2):
        super().__init__()
        self.k = k
        hidden = d_llm * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_enc * k, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_llm),
        )
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, hidden, lengths):
        b, t, d = hidden.shape
        pad = (-t) % self.k
        if pad:
            hidden = torch.cat([hidden, hidden.new_zeros(b, pad, d)], dim=1)
        x = hidden.reshape(b, (t + pad) // self.k, d * self.k)   # the "stack"
        out_lengths = torch.div(lengths + self.k - 1, self.k, rounding_mode="floor")
        return self.norm(self.net(x)), out_lengths


class ConvProjector(nn.Module):
    def __init__(self, d_enc: int, d_llm: int, k: int = 4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d_enc, d_llm, kernel_size=2 * k - 1, stride=k, padding=k - 1)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_llm, d_llm)
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, hidden, lengths):
        x = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        out_lengths = torch.div(lengths + self.k - 1, self.k, rounding_mode="floor")
        out_lengths = out_lengths.clamp(max=x.shape[1])
        return self.norm(self.proj(self.act(x))), out_lengths


class QFormerProjector(nn.Module):
    def __init__(self, d_enc: int, d_llm: int, n_queries: int = 64,
                 n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_llm) * 0.02)
        self.in_proj = nn.Linear(d_enc, d_llm)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_llm, n_heads, batch_first=True),
                "cross_attn": nn.MultiheadAttention(d_llm, n_heads, batch_first=True),
                "ff": nn.Sequential(nn.Linear(d_llm, 4 * d_llm), nn.GELU(),
                                    nn.Linear(4 * d_llm, d_llm)),
                "n1": nn.LayerNorm(d_llm), "n2": nn.LayerNorm(d_llm), "n3": nn.LayerNorm(d_llm),
            }) for _ in range(n_layers)
        ])
        self.n_queries = n_queries

    def forward(self, hidden, lengths):
        b = hidden.shape[0]
        mem = self.in_proj(hidden)
        pad_mask = ~_mask_from_lengths(lengths, hidden.shape[1])   # True = ignore
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        for l in self.layers:
            h = l["n1"](q)
            q = q + l["self_attn"](h, h, h, need_weights=False)[0]
            h = l["n2"](q)
            q = q + l["cross_attn"](h, mem, mem, key_padding_mask=pad_mask,
                                    need_weights=False)[0]
            q = q + l["ff"](l["n3"](q))
        out_lengths = torch.full((b,), self.n_queries, dtype=torch.long, device=hidden.device)
        return q, out_lengths


def build_projector(kind: str, d_enc: int, d_llm: int, cfg) -> nn.Module:
    if kind == "stack_mlp":
        return StackMLPProjector(d_enc, d_llm, cfg.stack_k, cfg.proj_hidden_mult)
    if kind == "conv":
        return ConvProjector(d_enc, d_llm, cfg.stack_k)
    if kind == "qformer":
        return QFormerProjector(d_enc, d_llm, cfg.qformer_queries,
                                cfg.qformer_layers, cfg.qformer_heads)
    raise ValueError(f"unknown projector {kind!r}")


if __name__ == "__main__":
    from .config import ModelConfig

    cfg = ModelConfig()
    d_enc, d_llm, b, t = 384, 576, 2, 250          # 250 enc frames = 5 s
    h = torch.randn(b, t, d_enc)
    lens = torch.tensor([t, t // 2])
    print(f"input {tuple(h.shape)}  lengths={lens.tolist()}  (5 s and 2.5 s of audio)")
    for kind in ("stack_mlp", "conv", "qformer"):
        p = build_projector(kind, d_enc, d_llm, cfg)
        out, ol = p(h, lens)
        print(f"  {kind:10s} -> {tuple(out.shape)} lengths={ol.tolist():}"
              f"  params={sum(x.numel() for x in p.parameters())/1e6:.2f}M")

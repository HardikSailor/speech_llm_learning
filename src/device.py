"""Device + precision helpers. The model code never hard-codes a device: every tensor it
creates is placed on `audio_embeds.device` / `lengths.device`. So moving to a GPU is only:

    1. model.to(device)                       - weights
    2. move_batch(batch, device)              - inputs (the collator always builds on CPU,
                                                in DataLoader worker processes)
    3. optionally autocast(bf16)              - ~2x faster matmuls, half the activation memory

bf16 needs no GradScaler: it has fp32's exponent range, so gradients do not underflow the
way they can in fp16. The weights (and the AdamW state) stay fp32; only the matmuls run
in bf16. That is "mixed precision" in one sentence.
"""
from __future__ import annotations

import contextlib

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    """Tensors go to `device`; lists of Segments / strings stay where they are.

    non_blocking=True overlaps the host->GPU copy with compute when the DataLoader uses
    pin_memory=True (pinned = page-locked host memory the GPU can DMA from directly).
    """
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def autocast(device: torch.device, precision: str):
    """Context manager for the forward pass. fp32 = no-op. (No fp16: it would need a
    GradScaler, and every GPU worth using for this has bf16.)"""
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def resolve_precision(precision: str, device: torch.device) -> str:
    """'auto' = bf16 on a GPU that supports it, else fp32."""
    if precision != "auto":
        return precision
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"cuda ({p.name}, {p.total_memory/2**30:.0f} GB)"
    return device.type

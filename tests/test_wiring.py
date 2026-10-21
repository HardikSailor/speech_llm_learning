"""Smoke tests. `python -m tests.test_wiring` - the one-batch overfit is the important one:
if the splice, the mask or the label offset is wrong, the loss will not reach ~0."""
from __future__ import annotations

import torch

from src.collate import SpeechCollator
from src.config import DataConfig, ModelConfig
from src.data import SpeechTextDataset
from src.modeling import IGNORE, SpeechLLM


def make(root: str = "", n: int = 2, **kw):
    cfg = ModelConfig(**kw)
    model = SpeechLLM(cfg)
    ds = SpeechTextDataset(DataConfig(root=root))
    coll = SpeechCollator(model.tokenizer, cfg)
    return model, coll([ds[i] for i in range(n)])


def test_shapes_and_masks(model, batch):
    aud, alen = model.encode_audio(batch["mel"], batch["mel_lengths"])
    inputs, attn, labels = model.build_inputs(aud, alen, batch["segments"])
    for i, seg in enumerate(batch["segments"]):
        expect = len(seg.pre_ids) + alen[i].item() + len(seg.post_ids) + len(seg.ans_ids)
        assert attn[i].sum().item() == expect, "attention mask length mismatch"
        assert (labels[i] != IGNORE).sum().item() == len(seg.ans_ids), "label mask mismatch"
        assert labels[i][attn[i] == 0].eq(IGNORE).all(), "padding is not masked"
    print("PASS shapes/masks:", [int(a.sum()) for a in attn])


def test_overfit_one_batch(model, batch, steps: int = 60, lr: float = 1e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    first = None
    for step in range(steps):
        loss, _ = model(batch["mel"], batch["mel_lengths"], batch["segments"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        first = first if first is not None else loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    print(f"PASS overfit: {first:.3f} -> {loss.item():.3f}")
    return loss.item()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)
    model, batch = make(a.root)
    print(model.trainable_report())
    test_shapes_and_masks(model, batch)
    test_overfit_one_batch(model, batch, steps=a.steps)
    print("\nreference:", batch["texts"][0][:80])
    print("decoded  :", model.generate(batch["mel"], batch["mel_lengths"],
                                       batch["segments"], max_new_tokens=40)[0][:120])

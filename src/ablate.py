"""Stage 5b - "is the model even listening?"

The most common silent failure in a speech-LLM: the projector output is ignored and the LLM
just models English text priors from the prompt. Teacher-forced loss still goes down, because
predicting plausible English is most of the CE.

The test: corrupt the audio and re-measure. If the loss barely moves, the audio is not being
used. Run this on every checkpoint before you believe a loss curve.

    $PY -m src.ablate --root $LIBRISPEECH/dev-clean --ckpt runs/stage5_30min/projector.pt
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .collate import SpeechCollator
from .config import DataConfig, ModelConfig
from .data import SpeechTextDataset
from .device import describe, move_batch, pick_device
from .modeling import SpeechLLM


MODES = {
    "real":     "the actual audio",
    "zeros":    "all audio embeddings set to 0",
    "noise":    "audio embeddings replaced by gaussian noise of the same scale",
    "shuffled": "each sample paired with another sample's audio",
}


@torch.no_grad()
def run(model, loader, mode: str, max_batches: int = 8,
        device: torch.device = torch.device("cpu")) -> float:
    tot, n_tok = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        aud, alen = model.encode_audio(batch["mel"], batch["mel_lengths"])
        if mode == "zeros":
            aud = torch.zeros_like(aud)
        elif mode == "noise":
            aud = torch.randn_like(aud) * aud.std()
        elif mode == "shuffled":
            perm = torch.roll(torch.arange(aud.shape[0], device=aud.device), 1)
            aud, alen = aud[perm], alen[perm]
        inputs, attn, labels = model.build_inputs(aud, alen, batch["segments"])
        out = model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
        n = sum(len(s.ans_ids) for s in batch["segments"])
        tot += out.loss.item() * n
        n_tok += n
    return tot / max(1, n_tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    mcfg = ModelConfig()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**sd["model_config"])
    device = pick_device(args.device)
    print(f"device: {describe(device)}")
    model = SpeechLLM(mcfg).eval()
    if args.ckpt:
        model.projector.load_state_dict(sd["projector"])
        print(f"loaded projector from {args.ckpt}")
    else:
        print("no checkpoint: using a RANDOM projector (the control case)")
    model.to(device)

    ds = SpeechTextDataset(DataConfig(root=args.root, max_seconds=mcfg.max_audio_seconds),
                           split_ratio=0.9, split="val", limit=args.batches * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=SpeechCollator(model.tokenizer, mcfg))

    base = None
    print(f"\n{'mode':10s} {'val loss':>9s} {'delta':>8s}   what it tests")
    for mode, desc in MODES.items():
        l = run(model, loader, mode, args.batches, device=device)
        base = l if base is None else base
        print(f"{mode:10s} {l:9.4f} {l-base:+8.4f}   {desc}")
    print("\n`shuffled` is the control that matters. zeros/noise push the LLM off-distribution"
          "\n(it never saw such vectors in training), so their delta overstates real audio use."
          "\n`shuffled` keeps the input distribution identical and destroys only the"
          "\ncorrespondence between THIS audio and THIS transcript. A small shuffled delta means"
          "\nthe projector passes generic speech-ness but little utterance-specific content.")


if __name__ == "__main__":
    main()

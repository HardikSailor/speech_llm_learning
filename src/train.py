"""Stage 5 - the training loop, written out by hand (no Trainer, no Accelerate).

Everything a Trainer would hide is visible here:
    the forward, the loss, the backward, gradient accumulation, clipping, the LR schedule,
    the optimizer step, the eval pass, the checkpoint.

Stage 6 replaces ~10 lines of this with Accelerate and nothing else changes - which is the
point: you will be able to say exactly what Accelerate does for you.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .collate import SpeechCollator
from .config import DataConfig, ModelConfig
from .data import SpeechTextDataset
from .device import autocast, describe, move_batch, pick_device, resolve_precision
from .modeling import SpeechLLM
from .sampler import LengthBucketBatchSampler, padding_waste


# ----------------------------------------------------------------- LR schedule
def lr_lambda(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 10 % of peak.

    Warmup exists because Adam's second-moment estimate is garbage for the first few dozen
    steps; a full-size step then can wreck a pretrained model. Cosine decay just anneals.
    """
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0)))


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, max_batches: int = 8, decode_batches: int = 1,
             device: torch.device = torch.device("cpu"), precision: str = "fp32"):
    model.eval()
    tot_loss, tot_tok, refs, hyps = 0.0, 0, [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast(device, precision):
            loss, _ = model(batch["mel"], batch["mel_lengths"], batch["segments"])
            if i < decode_batches:
                hyps += model.generate(batch["mel"], batch["mel_lengths"], batch["segments"],
                                       max_new_tokens=100)
        n = sum(len(s.ans_ids) for s in batch["segments"])
        tot_loss += loss.item() * n
        tot_tok += n
        if i < decode_batches:
            refs += batch["texts"]
    model.train()
    val_loss = tot_loss / max(1, tot_tok)
    wer = float("nan")
    if refs:
        try:
            import jiwer
            norm = lambda s: " ".join(s.lower().strip().split()) or "<empty>"   # noqa: E731
            wer = jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])
        except Exception:                                                        # noqa: BLE001
            pass
    return {"loss": val_loss, "ppl": math.exp(min(20, val_loss)), "wer": wer,
            "refs": refs, "hyps": hyps}


# ----------------------------------------------------------------- tensorboard
def make_writer(logdir: Path, enabled: bool):
    """A SummaryWriter, or None if disabled / tensorboard is not installed.

    Every `tb.add_*` call below is guarded by `if tb:` so the loop runs identically without it.
    View with:  tensorboard --logdir runs/  (then ssh -L 6006:localhost:6006 <login node>)
    """
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed - skipping TB logging")
        return None
    return SummaryWriter(log_dir=str(logdir))


def log_eval(tb, m: dict, step: int, prefix: str = "val") -> None:
    tb.add_scalar(f"{prefix}/loss", m["loss"], step)
    tb.add_scalar(f"{prefix}/ppl", m["ppl"], step)
    if not math.isnan(m["wer"]):
        tb.add_scalar(f"{prefix}/wer", m["wer"], step)
    # Markdown table in the TEXT tab: watch the hypotheses go from babble to transcripts.
    rows = "\n".join(f"| {r[:120]} | {h[:120]} |" for r, h in zip(m["refs"][:4], m["hyps"][:4]))
    tb.add_text(f"{prefix}/samples", "| ref | hyp |\n|---|---|\n" + rows, step)


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LibriSpeech-style directory")
    ap.add_argument("--minutes", type=float, default=20.0, help="training audio budget")
    ap.add_argument("--val-utts", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = run all epochs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--projector", default="stack_mlp")
    ap.add_argument("--stack-k", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/stage5")
    ap.add_argument("--no-tensorboard", action="store_true", help="disable TB logging")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16"],
                    help="auto = bf16 autocast on a bf16-capable GPU, else fp32")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        # TF32 for any matmul left in fp32 (Ampere+): 10-bit mantissa, ~free speedup.
        torch.set_float32_matmul_precision("high")
    print(f"device: {describe(device)}   precision: {precision}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tb = make_writer(out / "tb", enabled=not args.no_tensorboard)

    # ---- data ---------------------------------------------------------------
    dcfg = DataConfig(root=args.root, max_seconds=args.max_seconds)
    train_ds = SpeechTextDataset(dcfg, split_ratio=0.9, split="train",
                                 budget_minutes=args.minutes, seed=args.seed)
    val_ds = SpeechTextDataset(dcfg, split_ratio=0.9, split="val",
                               limit=args.val_utts, seed=args.seed)

    mcfg = ModelConfig(projector=args.projector, stack_k=args.stack_k,
                       max_audio_seconds=args.max_seconds)
    model = SpeechLLM(mcfg).to(device)
    coll = SpeechCollator(model.tokenizer, mcfg)

    sampler = LengthBucketBatchSampler(train_ds.durations, args.batch_size, seed=args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=coll,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=coll,
                            num_workers=0, shuffle=False, pin_memory=pin)

    naive = [list(range(i, min(i + args.batch_size, len(train_ds))))
             for i in range(0, len(train_ds), args.batch_size)]
    print(f"train : {len(train_ds)} utts  {train_ds.total_hours*60:.1f} min audio  "
          f"{len(sampler)} batches/epoch")
    print(f"val   : {len(val_ds)} utts")
    print(f"padding waste: {padding_waste(train_ds.durations, naive)*100:.1f} % naive -> "
          f"{padding_waste(train_ds.durations, list(sampler))*100:.1f} % bucketed")
    print()
    print(model.trainable_report())

    # ---- optimizer ----------------------------------------------------------
    # Only parameters with requires_grad go in. Anything else silently wastes optimizer
    # state (2 floats per parameter for Adam) and can un-freeze a tower through weight decay.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(sampler) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, warmup))
    print(f"\noptimizer steps: {total_steps} (warmup {warmup}), "
          f"effective batch = {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size*args.grad_accum}\n")

    # Frozen towers stay in eval mode: dropout/LayerNorm-stat behaviour should not change
    # just because the projector is training.
    model.train()
    model.encoder.eval()
    model.llm.eval()

    if tb:
        tb.add_text("config", "\n".join(f"    {k}: {v}" for k, v in vars(args).items()), 0)
        tb.add_text("model", "    " + model.trainable_report().replace("\n", "\n    "), 0)

    hist, step, micro, t0, seen_audio = [], 0, 0, time.time(), 0.0
    accum_loss = 0.0     # mean loss over the micro-batches of one optimizer step
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = move_batch(batch, device)
            with autocast(device, precision):
                loss, _ = model(batch["mel"], batch["mel_lengths"], batch["segments"])
            # Scale by 1/accum so that N micro-batches sum to the mean loss of one big batch.
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum
            seen_audio += batch["mel_lengths"].sum().item() / 100
            micro += 1
            if micro % args.grad_accum:
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if tb:
                # Every optimizer step: TB smooths the curve itself, so no need to subsample.
                el = time.time() - t0
                tb.add_scalar("train/loss", accum_loss, step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                tb.add_scalar("train/grad_norm", float(gnorm), step)
                tb.add_scalar("train/realtime_factor", seen_audio / el, step)
                tb.add_scalar("train/audio_minutes_seen", seen_audio / 60, step)
                tb.add_scalar("train/epoch", epoch, step)
            step_loss, accum_loss = accum_loss, 0.0

            if step % args.log_every == 0 or step == 1:
                el = time.time() - t0
                print(f"step {step:4d}/{total_steps}  loss {step_loss:7.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  |g| {gnorm:6.3f}  "
                      f"{seen_audio/el:5.1f}x realtime  {el:5.0f}s")
                hist.append({"step": step, "loss": step_loss,
                             "lr": sched.get_last_lr()[0], "gnorm": float(gnorm)})
            if step % args.eval_every == 0 or step == total_steps:
                m = evaluate(model, val_loader, device=device, precision=precision)
                if tb:
                    log_eval(tb, m, step)
                    tb.flush()
                print(f"  [eval] val_loss {m['loss']:.4f}  ppl {m['ppl']:.1f}  "
                      f"WER {m['wer']*100:.1f} %")
                for r, h in zip(m["refs"][:2], m["hyps"][:2]):
                    print(f"    ref: {r[:90]}")
                    print(f"    hyp: {h[:90]}")
            if step >= total_steps:
                done = True
                break

    # ---- checkpoint ---------------------------------------------------------
    # Save ONLY what we trained. The encoder and the LLM are byte-identical to the hub
    # weights, so storing them again would waste 145M x 4 bytes per checkpoint.
    # .cpu() so the checkpoint loads on a machine without a GPU.
    torch.save({"projector": {k: v.cpu() for k, v in model.projector.state_dict().items()},
                "model_config": vars(mcfg)}, out / "projector.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    m = evaluate(model, val_loader, max_batches=8, decode_batches=2,
                 device=device, precision=precision)
    (out / "final_eval.json").write_text(json.dumps(
        {k: v for k, v in m.items() if k in ("loss", "ppl", "wer")}, indent=2))
    if tb:
        log_eval(tb, m, step, prefix="final")
        # HPARAMS tab: compare runs (e.g. projector / minutes / lr) against final WER.
        hp = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
        tb.add_hparams(hp, {"hparam/final_wer": m["wer"], "hparam/final_val_loss": m["loss"]},
                       run_name=".")
        tb.close()
    print(f"\nsaved {out/'projector.pt'}  "
          f"({(out/'projector.pt').stat().st_size/1e6:.1f} MB)")
    print(f"final: val_loss {m['loss']:.4f}  WER {m['wer']*100:.1f} %")


if __name__ == "__main__":
    main()

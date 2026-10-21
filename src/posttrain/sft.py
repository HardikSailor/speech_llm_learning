"""Stage 10.1 - SFT: supervised fine-tuning, the baseline every other method starts from.

    loss = - mean over completion tokens of log pi(y_t | prompt, y_<t)

That is the whole method: maximise the likelihood of a demonstration, with the prompt masked
out (exactly the -100 trick from stage 3, written here as a mask).

What SFT cannot do, and why the rest of stage 10 exists:
  * it only sees GOOD examples, so it never learns what to avoid;
  * it imitates the demonstration token by token (teacher forcing), so it never sees its own
    mistakes (exposure bias);
  * it needs someone to WRITE the ideal answer. Preferences ("A is better than B") and
    rewards ("this scores 0.5") are much cheaper to get.

    $PY -m src.posttrain.sft --steps 40 --save runs/posttrain_sft
"""
from __future__ import annotations

import argparse

import torch

from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, completion_ids, evaluate, fmt,
                     load_model, make_problems, prompt_ids, sequence_logprobs)


def sft_loss(model, ids, attn, comp_mask):
    logps = sequence_logprobs(model, ids, attn)                 # (B, T-1)
    return -(logps * comp_mask).sum() / comp_mask.sum()         # mean NLL over answer tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    torch.manual_seed(0)

    model, tok = load_model(args.model)
    print("before:", fmt(evaluate(model, tok, EVAL_PROBLEMS)))

    # the demonstrations: the ideal answer is the bare number
    data = make_problems(args.steps * args.batch_size, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        chunk = data[step * args.batch_size:(step + 1) * args.batch_size]
        ids, attn, mask = build_batch([prompt_ids(tok, q) for q, _ in chunk],
                                      [completion_ids(tok, str(a)) for _, a in chunk],
                                      tok.pad_token_id)
        loss = sft_loss(model, ids, attn, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    print("after: ", fmt(evaluate(model, tok, EVAL_PROBLEMS)))
    if args.save:
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        print("saved to", args.save)


if __name__ == "__main__":
    main()

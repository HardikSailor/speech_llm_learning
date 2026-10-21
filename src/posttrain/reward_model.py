"""Stage 10.2 - Reward model: turn preference pairs into a scalar score.

Classic RLHF (InstructGPT) is SFT -> reward model -> PPO. Humans can't reliably give an
answer a score like "7.3", but they can say "A is better than B". The Bradley-Terry model
links the two: assume each answer has a hidden score r, and

    P(A preferred over B) = sigmoid(r_A - r_B)
    loss = -log sigmoid(r_chosen - r_rejected)

Only DIFFERENCES between scores are learned, so the scale and offset of r are arbitrary (one
reason PPO normalises rewards). DPO reuses this exact loss, with r replaced by the policy's
implicit reward.

Architecture: the LLM body without its LM head, plus Linear(d_model -> 1) read at the LAST
real token of prompt+answer. That token has attended to everything before it, so it can
"see" the whole answer.

    $PY -m src.posttrain.reward_model --steps 40
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..encoder import _dtype_kw
from .common import (MODEL_NAME, build_batch, completion_ids, make_preference_pairs,
                     make_problems, prompt_ids)


class RewardModel(nn.Module):
    def __init__(self, name: str = MODEL_NAME):
        super().__init__()
        self.body = AutoModel.from_pretrained(name, **_dtype_kw(torch.float32))
        self.head = nn.Linear(self.body.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)                    # every answer starts at score 0
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, attn):
        h = self.body(input_ids=ids, attention_mask=attn).last_hidden_state   # (B, T, d)
        last = attn.sum(-1) - 1                             # index of last real token (right pad)
        return self.head(h[torch.arange(len(ids)), last]).squeeze(-1)          # (B,)


def score(rm, tok, pairs):
    """Score chosen and rejected answers for a list of (q, chosen, rejected)."""
    p = [prompt_ids(tok, q) for q, _, _ in pairs]
    c = [completion_ids(tok, yc) for _, yc, _ in pairs]
    r = [completion_ids(tok, yr) for _, _, yr in pairs]
    ids, attn, _ = build_batch(p + p, c + r, tok.pad_token_id)
    s = rm(ids, attn)
    return s[:len(pairs)], s[len(pairs):]


@torch.no_grad()
def pair_accuracy(rm, tok, pairs) -> float:
    rm.eval()
    sc, sr = score(rm, tok, pairs)
    rm.train()
    return (sc > sr).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rm = RewardModel()
    train = make_preference_pairs(make_problems(args.steps * args.batch_size, seed=4), seed=4)
    held_out = make_preference_pairs(make_problems(64, seed=5), seed=5)
    print(f"held-out pair accuracy before: {pair_accuracy(rm, tok, held_out):.2f}  (ties = 0)")

    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr)
    for step in range(args.steps):
        chunk = train[step * args.batch_size:(step + 1) * args.batch_size]
        sc, sr = score(rm, tok, chunk)
        loss = -F.logsigmoid(sc - sr).mean()                # Bradley-Terry
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}  "
                  f"train pair-acc {(sc > sr).float().mean():.2f}  gap {(sc - sr).mean():+.3f}")

    print(f"held-out pair accuracy after:  {pair_accuracy(rm, tok, held_out):.2f}")

    # what did it learn? score four answers to one question
    q = "What is 37 + 48? Answer with just the number."
    answers = ["85", "37 + 48 = 85", "86", "I like turtles"]
    ids, attn, _ = build_batch([prompt_ids(tok, q)] * 4, [completion_ids(tok, a) for a in answers],
                               tok.pad_token_id)
    with torch.no_grad():
        for a, s in zip(answers, rm.eval()(ids, attn)):
            print(f"    r({a!r:16}) = {s.item():+.3f}")


if __name__ == "__main__":
    main()

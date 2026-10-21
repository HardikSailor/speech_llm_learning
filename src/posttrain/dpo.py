"""Stage 10.3 - DPO: learn from preference pairs, with no reward model and no RL.

RLHF wants   max_pi  E[r(x,y)] - beta * KL(pi || pi_ref).
That objective has a closed-form optimum, pi*(y|x) ∝ pi_ref(y|x) * exp(r(x,y)/beta), which can
be flipped around to write the reward in terms of the policy:

    r(x,y) = beta * log( pi(y|x) / pi_ref(y|x) )  + (a term that depends only on x)

Put that "implicit reward" into the Bradley-Terry preference loss (see reward_model.py).
The x-only term cancels between chosen and rejected, and what remains is DPO:

    loss = -log sigmoid( beta * [ (log pi(yc) - log pi_ref(yc)) - (log pi(yr) - log pi_ref(yr)) ] )

So DPO is a classification loss on pairs: push the chosen answer's log-ratio above the
rejected one's. pi_ref is what stops the policy drifting arbitrarily far. Lower beta lets it
drift further (a weaker KL leash).

Variants, selected with --loss:
  sigmoid  the DPO loss above
  ipo      squared loss toward a fixed margin 1/(2 beta). DPO keeps pushing the margin to
           infinity on separable data (it overfits); IPO stops at the target.
  simpo    no reference model at all: reward = beta * average per-token log pi, minus a
           margin gamma. Averaging also removes DPO's bias toward long answers.

--sft-weight > 0 adds alpha * (per-token NLL of the chosen answer) to the loss (RPO /
"DPO + SFT"). Without it, the run below shows the classic failure: the margin grows, but
mostly by crushing the rejected answer. log pi(chosen) falls too, and the freed probability
mass goes to an answer that is in NEITHER side of the pairs.

    $PY -m src.posttrain.dpo                         # from the base model
    $PY -m src.posttrain.dpo --model runs/posttrain_sft   # the usual recipe: SFT then DPO
    $PY -m src.posttrain.dpo --sft-weight 1.0              # DPO + SFT term on chosen
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, completion_ids, evaluate, fmt,
                     frozen_copy, load_model, make_preference_pairs, make_problems, prompt_ids,
                     sequence_logprobs)


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float, loss_type: str = "sigmoid",
             len_c=None, len_r=None, simpo_gamma: float = 0.5):
    """Inputs are SUMMED completion log-probs, one number per pair, shape (B,)."""
    if loss_type == "simpo":
        # reference-free, length-normalised
        logits = beta * (pol_c / len_c - pol_r / len_r) - simpo_gamma
        return -F.logsigmoid(logits).mean(), beta * pol_c / len_c, beta * pol_r / len_r
    chosen_reward = beta * (pol_c - ref_c)            # implicit rewards
    rejected_reward = beta * (pol_r - ref_r)
    margin = chosen_reward - rejected_reward
    if loss_type == "sigmoid":
        loss = -F.logsigmoid(margin)
    elif loss_type == "ipo":
        # IPO's target is on the log-ratio gap itself (margin / beta), not the beta-scaled one
        loss = (margin / beta - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(loss_type)
    return loss.mean(), chosen_reward, rejected_reward


def summed_logps(model, ids, attn, mask):
    return (sequence_logprobs(model, ids, attn) * mask).sum(-1)      # (2B,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--loss", default="sigmoid", choices=["sigmoid", "ipo", "simpo"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--sft-weight", type=float, default=0.0)
    args = ap.parse_args()
    torch.manual_seed(0)

    policy, tok = load_model(args.model)
    ref = frozen_copy(policy)                 # pi_ref = the starting point, never updated
    print("before:", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))

    pairs = make_preference_pairs(make_problems(args.steps * args.batch_size, seed=2))
    print("example pair:", pairs[0])
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    policy.train()
    B = args.batch_size
    for step in range(args.steps):
        chunk = pairs[step * B:(step + 1) * B]
        # one forward over [chosen_1..chosen_B, rejected_1..rejected_B]
        p = [prompt_ids(tok, q) for q, _, _ in chunk]
        c = [completion_ids(tok, yc) for _, yc, _ in chunk]
        r = [completion_ids(tok, yr) for _, _, yr in chunk]
        ids, attn, mask = build_batch(p + p, c + r, tok.pad_token_id)

        pol = summed_logps(policy, ids, attn, mask)
        with torch.no_grad():
            refl = summed_logps(ref, ids, attn, mask)
        lens = mask.sum(-1)
        loss, rc, rr = dpo_loss(pol[:B], pol[B:], refl[:B], refl[B:], args.beta, args.loss,
                                lens[:B], lens[B:])
        if args.sft_weight > 0:          # keep the chosen answer likely, not just "more likely than rejected"
            loss = loss + args.sft_weight * (-pol[:B] / lens[:B]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 5 == 0 or step == args.steps - 1:
            # the four numbers to watch in any DPO run
            print(f"step {step:3d}  loss {loss.item():.4f}  "
                  f"pair-acc {(rc > rr).float().mean():.2f}  margin {(rc - rr).mean():+.3f}  "
                  f"logp chosen {pol[:B].mean():.2f}  rejected {pol[B:].mean():.2f}")

    print("after: ", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))


if __name__ == "__main__":
    main()

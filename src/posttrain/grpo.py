"""Stage 10.6 - GRPO: PPO without a critic, using a reward you can check by rule.

For each prompt, sample a GROUP of G completions, score each one, and use the group itself
as the baseline:

    advantage_i = (r_i - mean(r_group)) / std(r_group)

"Better than your siblings" -> push up, "worse" -> push down. No value network is needed
(PPO needs one, see ppo.py), which is why DeepSeek-R1-style reasoning training uses it. The
rest is PPO's clipped objective plus a KL penalty to the reference model, per token:

    loss_t = -min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A)  +  beta * KL_t
    ratio_t = pi(y_t) / pi_old(y_t)            pi_old = the policy that generated the samples
    KL_t    = pi_ref/pi - log(pi_ref/pi) - 1   the "k3" estimator: unbiased, always >= 0

The things to watch while it runs:
  * reward going up (the point);
  * `no-signal` = share of groups where all G samples got the same reward -> advantage 0 ->
    no gradient. If it is near 1.0 the task is too easy or too hard for the current model;
  * KL to ref: slowly rising is fine; exploding means the leash is too loose (raise beta);
  * completion length: a sudden change is often the first sign of reward hacking.

    $PY -m src.posttrain.grpo --steps 30
"""
from __future__ import annotations

import argparse

import torch

from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, evaluate, fmt, frozen_copy,
                     load_model, make_problems, prompt_ids, reward_fn, sample,
                     sequence_logprobs)


def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards (P, G) -> advantages (P*G,), normalised within each prompt's group."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return ((rewards - mean) / (std + 1e-4)).flatten()


def grpo_loss(logps, old_logps, ref_logps, adv, mask, clip_eps: float, beta: float):
    """All (N, T-1) except adv (N,). Returns the scalar loss and the mean KL."""
    ratio = torch.exp(logps - old_logps)
    a = adv.unsqueeze(1)
    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    per_token = pg + beta * kl
    # average over each completion's tokens, then over completions (the original GRPO paper;
    # "Dr. GRPO" divides by a constant instead, to remove a bias toward short answers)
    loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    mean_kl = ((kl * mask).sum() / mask.sum()).item()
    return loss, mean_kl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=4, help="prompts per step")
    ap.add_argument("--group", type=int, default=8, help="G completions per prompt")
    ap.add_argument("--inner-epochs", type=int, default=2, help="updates per batch of samples")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()
    torch.manual_seed(0)

    policy, tok = load_model(args.model)
    ref = frozen_copy(policy)
    print("before:", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))

    problems = make_problems(args.steps * args.prompts, seed=3)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    G = args.group
    for step in range(args.steps):
        chunk = problems[step * args.prompts:(step + 1) * args.prompts]
        prompts = [prompt_ids(tok, q) for q, _ in chunk]

        # 1. rollout: G samples per prompt from the current policy
        comps = sample(policy, tok, prompts, n=G, temperature=args.temperature)
        texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
        rewards = torch.tensor([reward_fn(t, chunk[i // G][1]) for i, t in enumerate(texts)])
        rewards = rewards.view(len(chunk), G)

        # 2. advantages from the group, no critic
        adv = grpo_advantages(rewards)

        # 3. freeze the log-probs of the sampling policy and of the reference
        ids, attn, mask = build_batch([p for p in prompts for _ in range(G)], comps,
                                      tok.pad_token_id)
        with torch.no_grad():
            old_logps = sequence_logprobs(policy, ids, attn)
            ref_logps = sequence_logprobs(ref, ids, attn)

        # 4. a few clipped policy-gradient updates on the same samples
        for _ in range(args.inner_epochs):
            logps = sequence_logprobs(policy, ids, attn)
            loss, kl = grpo_loss(logps, old_logps, ref_logps, adv, mask, args.clip, args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        no_signal = (rewards.std(dim=1) == 0).float().mean()
        print(f"step {step:3d}  reward {rewards.mean():.3f}  format {(rewards == 1).float().mean():.2f}"
              f"  no-signal {no_signal:.2f}  kl {kl:.4f}  len {mask.sum(-1).mean():.1f}"
              f"  e.g. {texts[0]!r}")

    print("after: ", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))


if __name__ == "__main__":
    main()

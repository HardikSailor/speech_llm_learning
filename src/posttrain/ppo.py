"""Stage 10.5 - PPO for LLMs (the RLHF algorithm), cut down to its moving parts.

Treat generation as an RL episode: state = prompt + tokens so far, action = next token,
reward = arrives once, at the end. PPO needs four things, and these four are why it is
painful (classic RLHF keeps four models in memory):

  policy     pi        the LLM being trained
  reference  pi_ref    frozen copy; the KL to it is a per-token PENALTY inside the reward
  reward     r         here the rule-based reward_fn; in RLHF the model from reward_model.py
  critic     V         predicts future reward at every token; real setups use a full LLM
                       (often initialised from the RM). Here it is a single Linear on the
                       policy's hidden states, kept detached so it cannot disturb the policy.

Per token t of a completion:
    reward_t  = -beta * (log pi_old - log pi_ref)          (+ task reward on the LAST token)
    delta_t   = reward_t + V(t+1) - V(t)                    TD error, gamma = 1
    A_t       = delta_t + lam * A_(t+1)                     GAE: smooth A over future deltas
    loss      = -min(ratio*A, clip(ratio)*A)  +  c_v * (V - (A + V_old))^2

Compare with grpo.py: GRPO deletes the critic and GAE, and gets its baseline from G
samples of the same prompt. It also moves the KL out of the reward and into the loss.

    $PY -m src.posttrain.ppo --steps 30
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, evaluate, fmt, frozen_copy,
                     load_model, make_problems, prompt_ids, reward_fn, sample,
                     sequence_logprobs)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def gae(rewards, values, mask, gamma: float = 1.0, lam: float = 0.95):
    """Generalised Advantage Estimation, walking backwards over the completion tokens.
    All tensors (N, T-1). Positions outside the completion are masked out."""
    adv = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0])
    T = rewards.shape[1]
    for t in reversed(range(T)):
        nxt = mask[:, t + 1] if t + 1 < T else torch.zeros_like(running)   # 0 = episode over
        next_v = values[:, t + 1] * nxt if t + 1 < T else 0.0
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = delta + gamma * lam * nxt * running
        adv[:, t] = running
    return adv * mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--rollouts", type=int, default=16, help="prompts per step, 1 sample each")
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.05, help="KL penalty in the reward")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()
    torch.manual_seed(0)

    policy, tok = load_model(args.model)
    ref = frozen_copy(policy)
    value_head = nn.Linear(policy.config.hidden_size, 1)
    print("before:", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))

    problems = make_problems(args.steps * args.rollouts, seed=6)
    opt = torch.optim.AdamW([{"params": policy.parameters(), "lr": args.lr},
                             {"params": value_head.parameters(), "lr": 1e-3}])
    for step in range(args.steps):
        chunk = problems[step * args.rollouts:(step + 1) * args.rollouts]
        prompts = [prompt_ids(tok, q) for q, _ in chunk]

        # 1. rollout
        comps = sample(policy, tok, prompts, n=1)
        texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
        scores = torch.tensor([reward_fn(t, a) for t, (_, a) in zip(texts, chunk)])
        ids, attn, mask = build_batch(prompts, comps, tok.pad_token_id)

        # 2. score the rollout with the frozen pieces: old logps, ref logps, old values
        with torch.no_grad():
            old_logps, h = sequence_logprobs(policy, ids, attn, output_hidden_states=True)
            old_values = value_head(h).squeeze(-1) * mask
            ref_logps = sequence_logprobs(ref, ids, attn)

        # 3. per-token reward: KL penalty everywhere, task score on the last token
        kl = (old_logps - ref_logps) * mask
        rewards = -args.beta * kl
        last = (mask.cumsum(-1) == mask.sum(-1, keepdim=True)) & (mask > 0)
        rewards = rewards + last.float() * scores.unsqueeze(1)

        # 4. advantages and value targets
        adv = gae(rewards, old_values, mask)
        returns = adv + old_values
        adv = (adv - masked_mean(adv, mask)) / (adv[mask > 0].std() + 1e-8) * mask   # whiten

        # 5. clipped policy update + value regression
        for _ in range(args.inner_epochs):
            logps, h = sequence_logprobs(policy, ids, attn, output_hidden_states=True)
            values = value_head(h.detach()).squeeze(-1)
            ratio = torch.exp(logps - old_logps)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv)
            pg_loss = masked_mean(pg, mask)
            vf_loss = 0.5 * masked_mean((values - returns) ** 2, mask)
            loss = pg_loss + args.vf_coef * vf_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        clipped = masked_mean(((ratio - 1).abs() > args.clip).float(), mask)
        print(f"step {step:3d}  reward {scores.mean():.3f}  kl {masked_mean(kl, mask):.4f}  "
              f"value-loss {vf_loss.item():.4f}  clip-frac {clipped:.2f}  e.g. {texts[0]!r}")

    print("after: ", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))


if __name__ == "__main__":
    main()

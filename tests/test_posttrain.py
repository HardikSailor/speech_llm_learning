"""Checks on the post-training math. `python -m tests.test_posttrain` - no model, ~1 s.

Each test is a fact you should be able to state in an interview:
  * DPO starts at loss log(2): policy == reference -> every implicit reward is 0.
  * GRPO advantages are zero-mean per group, and all-equal rewards give zero gradient.
  * GAE with gamma = lam = 1 and a zero critic is just the reward-to-go.
  * The completion mask lines up with the TARGETS (shifted by one), not the inputs.
"""
from __future__ import annotations

import math

import torch

from src.posttrain.common import build_batch, reward_fn
from src.posttrain.dpo import dpo_loss
from src.posttrain.grpo import grpo_advantages, grpo_loss
from src.posttrain.ppo import gae


def test_reward():
    assert reward_fn(" 85 ", 85) == 1.0
    assert reward_fn("37 + 48 = 85", 85) == 0.5
    assert reward_fn("86", 85) == 0.0
    assert reward_fn("no idea", 85) == 0.0


def test_dpo_starts_at_log2():
    lp = torch.tensor([-5.0, -9.0])
    loss, rc, rr = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert rc.abs().max() == 0 and rr.abs().max() == 0
    # chosen up / rejected down relative to the reference -> loss drops below log 2
    loss2, _, _ = dpo_loss(lp + 1, lp - 2, lp, lp - 1, beta=0.1)
    assert loss2.item() < math.log(2)


def test_grpo_advantages():
    r = torch.tensor([[1.0, 0.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]])
    a = grpo_advantages(r).view(2, 4)
    assert torch.allclose(a[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert a[1].abs().max() == 0                     # no signal when the group agrees
    # at ratio = 1 and no KL, the loss is -mean(advantage) = 0 for a zero-mean group
    lp = torch.zeros(4, 3)
    loss, _ = grpo_loss(lp, lp, lp, a[0], torch.ones(4, 3), clip_eps=0.2, beta=0.04)
    assert abs(loss.item()) < 1e-6


def test_gae_is_reward_to_go():
    rewards = torch.tensor([[0.0, 0.1, 0.2, 1.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])      # first position = prompt, masked
    adv = gae(rewards * mask, torch.zeros(1, 4), mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[0.0, 1.3, 1.2, 1.0]]))


def test_mask_alignment():
    ids, attn, comp = build_batch([[10, 11, 12]], [[20, 21]], pad_id=0)
    assert ids.tolist() == [[10, 11, 12, 20, 21]]
    # targets are ids[:, 1:] = [11, 12, 20, 21]; only 20 and 21 are completion tokens
    assert comp.tolist() == [[0.0, 0.0, 1.0, 1.0]]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)

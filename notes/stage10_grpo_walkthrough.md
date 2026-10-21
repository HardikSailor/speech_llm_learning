# GRPO, line by line (`src/posttrain/grpo.py`)

Companion to `stage10_posttraining.md`. This file walks through every line of `grpo.py` in
order, with shapes and worked numbers, so you can rebuild it on a whiteboard.

Running example used throughout: `--prompts 4 --group 8`, so **P = 4** prompts per step,
**G = 8** samples per prompt, **N = P x G = 32** completions per step.

---

## 0. The idea in four sentences
1. For each prompt, sample G answers from the current model.
2. Score each answer with a rule (`reward_fn`): 1.0 / 0.5 / 0.0.
3. An answer's *advantage* is how much better it scored than its siblings:
   `(r - mean) / std` within the group. No critic network is needed.
4. Raise the log-prob of above-average answers and lower it for below-average ones, using
   PPO's clipped update, with a KL penalty that keeps the model near the reference.

---

## 1. Imports
```python
from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, evaluate, fmt, frozen_copy,
                     load_model, make_problems, prompt_ids, reward_fn, sample,
                     sequence_logprobs)
```
GRPO itself is short because all the plumbing is shared in `common.py`:

| helper | what it does |
|---|---|
| `load_model` | SmolLM2-135M-Instruct + tokenizer (pad = eos) |
| `frozen_copy` | deep copy with `requires_grad=False` -> the reference model |
| `make_problems` | `("What is a + b? Answer with just the number.", a+b)` pairs |
| `prompt_ids` | chat template + assistant header, as token ids |
| `sample` | batched generation, n per prompt, cut at first EOS |
| `reward_fn` | 1.0 exact / 0.5 right-but-chatty / 0.0 wrong |
| `build_batch` | prompt+completion, right-padded, plus a completion mask |
| `sequence_logprobs` | log pi(token_t \| tokens_<t) at every position, (B, T-1) |
| `evaluate` | greedy decode the 64 held-out problems, report reward/accuracy/format |

---

## 2. `grpo_advantages`: the "group relative" part
```python
def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return ((rewards - mean) / (std + 1e-4)).flatten()
```
- **Input** `rewards` is `(P, G)`: one row per prompt, one column per sample of it.
- `mean(dim=1, keepdim=True)` -> `(P, 1)`: each prompt's average score. This is the
  **baseline**. In PPO a critic network predicts it; here it is simply measured from the G
  samples. That is the whole trick of GRPO.
- `keepdim=True` keeps the shape `(P, 1)` so `rewards - mean` broadcasts across the G
  columns.
- `std` -> `(P, 1)`. Dividing by it puts every prompt on the same scale: an easy prompt
  where scores differ by 0.1 gets as much push as a hard one where they differ by 1.0.
  (Dr. GRPO argues this over-weights near-unanimous groups and drops the std.)
- `torch.std` is the **unbiased** (n-1) estimate by default. That's a small detail, but it
  explains why the numbers below are not exactly what you would get with n.
- `+ 1e-4`: if all G rewards are equal, std = 0. The epsilon avoids 0/0, and the numerator
  is 0 anyway, so the advantage is exactly 0. **Such a group contributes no gradient.**
- `.flatten()` -> `(N,)` = `(32,)`, in the same order as the samples: p0's 8 samples first,
  then p1's, and so on.

**Worked example** (one group, G = 4 for brevity):
```
rewards   = [1.0,   0.5,   0.0,   0.0 ]       mean = 0.375, std(n-1) = 0.479
advantage = [+1.31, +0.26, -0.78, -0.78]
```
The exact answer is pushed up hard. The chatty-but-right one gets a small push up, because
it beat the average. Both wrong ones are pushed down.

---

## 3. `grpo_loss`: clipped policy gradient + KL
```python
def grpo_loss(logps, old_logps, ref_logps, adv, mask, clip_eps: float, beta: float):
```
Shapes: `logps`, `old_logps`, `ref_logps`, `mask` are all `(N, T-1)`, one value per token
position. `adv` is `(N,)`, one value per completion.

```python
    ratio = torch.exp(logps - old_logps)
```
- `pi_theta(token) / pi_old(token)`, computed in log space (subtract, then exp) for numerical
  safety.
- `old_logps` came from the model **that generated the samples** (frozen before the update).
  On the first inner epoch, policy == old, so `ratio == 1` everywhere. It moves away from 1
  only on the second and later updates over the same samples (`--inner-epochs 2`).

```python
    a = adv.unsqueeze(1)
```
- `(N,)` -> `(N, 1)`, so it broadcasts over the T-1 token positions. **Every token in a
  completion gets the same advantage.** GRPO has no per-token credit assignment; PPO's GAE
  does.

```python
    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
```
This is PPO's clipped surrogate. It is the most-asked line, so read it slowly:
- `ratio * a` is the plain importance-weighted policy-gradient objective (to maximise).
- `clamp(ratio, 0.8, 1.2) * a` is the same with the ratio clamped to the trust region.
- `min` of the two = a **pessimistic** bound: we never take credit for moving more than
  20 % away from the sampling policy.
- The leading `-` turns "maximise objective" into "minimise loss".

Worked example, `clip_eps = 0.2`, token log-prob moved from -1.20 to -1.00 -> ratio = 1.22:

| advantage | `ratio*a` | `clamp*a` | min picks | gradient? |
|---|---|---|---|---|
| A = +1 | 1.22 | 1.20 | clamped 1.20 | **no**: already pushed up enough |
| A = -1 | -1.22 | -1.20 | unclipped -1.22 | **yes**: keep pushing it down |

So clipping only removes the gradient when the update has already moved *in the direction
the advantage wants* by more than eps. It never stops you from undoing a bad move.

```python
    kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
```
- The **k3 estimator** of KL(pi_theta || pi_ref), per token. With `d = log pi_ref - log pi`:
  `KL_t = e^d - d - 1`.
- Why this form: it is always >= 0 (since `e^d >= 1 + d`). It is unbiased in expectation
  under samples from pi. It has lower variance than the naive `-d` (k1), which can go negative
  on single samples.
- Example: `log pi_ref = -1.5`, `log pi = -1.0` -> d = -0.5 -> 0.607 + 0.5 - 1 = **0.107**.
- `ref_logps` come from the frozen copy of the starting model. This is the leash.

```python
    per_token = pg + beta * kl
```
- GRPO puts the KL **directly in the loss**. PPO instead subtracts it inside the reward
  (see `ppo.py`). That's a classic interview question.
- `beta = 0.04` (from the DeepSeekMath paper). Larger means the model stays closer to the
  reference.

```python
    loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
```
Read it inside-out:
1. `per_token * mask` zeroes the prompt and padding positions. Only completion tokens count.
   (`mask` is already shifted to line up with the targets; see `build_batch`.)
2. `.sum(-1) / mask.sum(-1)` gives the **mean over this completion's tokens** -> `(N,)`.
   `clamp(min=1)` guards against an empty completion.
3. `.mean()` averages over the N completions -> a scalar.

The consequence: a short wrong answer spreads its negative advantage over few tokens, so
each token gets pushed down hard. A long wrong answer gets diluted. This is the length
bias that **Dr. GRPO** removes by dividing by a constant (e.g. max length) instead.

```python
    mean_kl = ((kl * mask).sum() / mask.sum()).item()
    return loss, mean_kl
```
The KL is returned only for logging: the average over all real completion tokens in the
batch.

---

## 4. `main()`: arguments
```python
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=4, help="prompts per step")
    ap.add_argument("--group", type=int, default=8, help="G completions per prompt")
    ap.add_argument("--inner-epochs", type=int, default=2, help="updates per batch of samples")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
```
| knob | effect if you raise it |
|---|---|
| `--group` G | better baseline estimate, fewer all-equal groups, more generation cost |
| `--inner-epochs` | more reuse of each (expensive) batch of samples; ratio drifts, clipping kicks in |
| `--beta` | stays closer to the reference: safer, learns slower |
| `--clip` | bigger allowed step per update |
| `--temperature` | more diverse samples = more exploration, also more junk |
| `--lr` | faster and less stable. RL on a full LLM uses LRs ~10x below SFT |

---

## 5. `main()`: setup
```python
    torch.manual_seed(0)
    policy, tok = load_model(args.model)
    ref = frozen_copy(policy)
    print("before:", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))
```
- Seed for reproducible sampling.
- `policy` is the model we train. `ref` is a frozen snapshot taken **before** any update.
  That's two full copies of the model in memory: still half of PPO's four.
- Evaluate first, so you have a baseline to compare against.

```python
    problems = make_problems(args.steps * args.prompts, seed=3)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    G = args.group
```
- Pre-generate exactly enough prompts: 30 steps x 4 = 120. `seed=3` keeps them disjoint
  from the eval set (seed 12345).
- Note there are **no answers to imitate**, only the integer used by the reward. This is the
  key difference from SFT.

---

## 6. The training step
```python
    for step in range(args.steps):
        chunk = problems[step * args.prompts:(step + 1) * args.prompts]
        prompts = [prompt_ids(tok, q) for q, _ in chunk]
```
- Take this step's 4 problems and turn each into chat-templated token ids.

### 6.1 Rollout
```python
        comps = sample(policy, tok, prompts, n=G, temperature=args.temperature)
```
- 4 prompts x 8 samples = **32 completions**, as lists of token ids, grouped by prompt
  (`p0 x8, p1 x8, ...`).
- **Sampling, not greedy.** If the 8 samples were identical, every group would have
  std = 0 and there would be no learning signal at all. Diversity *is* the signal.
- Inside `sample`: the model is put in `eval()` mode, prompts are LEFT-padded so all rows
  generate in lockstep, `no_grad` is on, and each completion is cut after its first EOS.

```python
        texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
        rewards = torch.tensor([reward_fn(t, chunk[i // G][1]) for i, t in enumerate(texts)])
        rewards = rewards.view(len(chunk), G)
```
- Decode to text, because the reward is a rule on strings.
- `chunk[i // G][1]`: completion `i` belongs to prompt `i // G` (0-7 -> prompt 0, 8-15 ->
  prompt 1, ...). `[1]` is that prompt's correct answer.
- `.view(P, G)` -> `(4, 8)`: rows = prompts, columns = siblings. That is the layout
  `grpo_advantages` expects.

### 6.2 Advantages
```python
        adv = grpo_advantages(rewards)
```
- `(4, 8)` -> `(32,)`. See section 2. Rewards are numbers here: no gradient flows through
  them (and none could, since the reward is a Python rule).

### 6.3 Freeze the "before" log-probs
```python
        ids, attn, mask = build_batch([p for p in prompts for _ in range(G)], comps,
                                      tok.pad_token_id)
```
- Repeat each prompt G times so it lines up with its completions, then concatenate
  prompt + completion and RIGHT-pad.
- `ids`, `attn`: `(32, T)` where T = longest prompt+completion. `mask`: `(32, T-1)`, 1 only
  where a completion token is the target.
- Why right-pad now, when generation left-padded? Both give the same positions (HF derives
  positions from the attention mask), and right-padding makes the mask arithmetic simple.

```python
        with torch.no_grad():
            old_logps = sequence_logprobs(policy, ids, attn)
            ref_logps = sequence_logprobs(ref, ids, attn)
```
- `old_logps`: the policy's log-probs **right now, before this step's updates**. This is the
  denominator of the ratio. It has to be frozen, otherwise ratio == 1 forever.
- `ref_logps`: the starting model's log-probs, for the KL term.
- `no_grad`: neither is trained, so storing activations for them would waste memory.
- Why not reuse the log-probs from `generate`? You could (vLLM-based trainers do). Here a
  clean forward pass is simpler and guarantees the exact same numerics as the training
  forward.

### 6.4 Update, a few times on the same samples
```python
        for _ in range(args.inner_epochs):
            logps = sequence_logprobs(policy, ids, attn)
```
- The forward pass **with** gradients. On the first pass `logps == old_logps` (same
  weights). On the second, the weights have changed once, so the ratio differs from 1 and
  clipping can matter.

```python
            loss, kl = grpo_loss(logps, old_logps, ref_logps, adv, mask, args.clip, args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
```
- The standard step from stage 5. Gradient-norm clipping (a cap on the update's size) is a
  different thing from ratio clipping (a cap on how far the policy may move per token). Both
  are used.
- With `--inner-epochs 1` the ratio is always 1, clipping never fires, and GRPO reduces to
  **REINFORCE with a group-mean baseline, plus KL**. Many recent setups run exactly that.

### 6.5 Logging
```python
        no_signal = (rewards.std(dim=1) == 0).float().mean()
        print(f"step {step:3d}  reward {rewards.mean():.3f}  format {(rewards == 1).float().mean():.2f}"
              f"  no-signal {no_signal:.2f}  kl {kl:.4f}  len {mask.sum(-1).mean():.1f}"
              f"  e.g. {texts[0]!r}")
```
| column | meaning | healthy |
|---|---|---|
| `reward` | mean reward of the 32 samples | trends up (noisy: only 4 prompts) |
| `format` | share of samples that got exactly 1.0 | up, if the model ever finds it |
| `no-signal` | share of groups with all-equal rewards -> zero gradient | well below 1 |
| `kl` | mean per-token KL to the reference | slow rise; spikes = trouble |
| `len` | mean completion length in tokens | stable; a jump = possible reward hacking |

```python
    print("after: ", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))
```
Greedy eval on the same 64 held-out problems as "before".

---

## 7. What the real run showed (`runs/posttrain_logs/grpo.log`)
```
before: reward 0.242  accuracy 0.484  format 0.000
step   0  reward 0.062  format 0.03  no-signal 0.50  kl 0.0038  e.g. 'The solution starts with 40 and 2 times 85, which'
step   2  reward 0.141  format 0.00  no-signal 0.25  kl 0.0203  e.g. 'A doughnut is formed by adding one whole doughnut hole to a circle.'
step  13  reward 0.297  format 0.00  no-signal 0.00  kl 0.2827  e.g. '88 + 15 equals 103.'
step  29  reward 0.406  format 0.00  no-signal 0.25  kl 0.1728  e.g. '38 + 25 = 63.'
after:  reward 0.344  accuracy 0.688  format 0.000
```
How to read it:
- **Sampled reward starts far below greedy** (0.06 vs 0.24). At temperature 1.0 the small
  model rambles ("doughnut"). RL first removes the junk, then the samples collapse onto the
  `a + b = c.` template.
- **Accuracy 48 % -> 69 %** on held-out problems. Learned from a rule, with no demonstrations.
- **Format stayed at 0.** The bare-number answer was sampled ~3 % of the time at step 0 and
  then never again, so the 1.0 reward was almost never seen. **RL can only amplify what the
  policy already samples.** That is the argument for SFT before RL.
- The flip side, from `sft_then_grpo.log`: starting from SFT, reward rises 0.406 -> 0.430,
  but `no-signal` sits at **0.75**. The SFT model answers with a bare number almost every
  time, so most groups score unanimously and give zero gradient. GRPO needs prompts where
  the model *sometimes* succeeds.
- `no-signal` around 0.25: about one group in four gave no gradient. With G = 8 and a
  3-valued reward that is expected. DAPO-style "dynamic sampling" would replace those groups.
- KL rising from 0.004 to ~0.2 while reward rises is normal. One spike (0.79 at step 19)
  did not derail the run; repeated spikes would mean beta is too small or the LR too high.

---

## 8. Five-line summary to say out loud
1. Sample G completions per prompt, score with a verifiable reward.
2. Advantage = reward standardised within the group: the group mean replaces PPO's critic.
3. Freeze `log pi_old` and `log pi_ref` for the batch.
4. Loss per token = `-min(ratio*A, clip(ratio)*A) + beta * k3_KL`, averaged per completion
   then across completions.
5. Take a few gradient steps on the same samples, then sample again with the new policy.

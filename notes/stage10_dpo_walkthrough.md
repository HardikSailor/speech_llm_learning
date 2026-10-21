# DPO, line by line (`src/posttrain/dpo.py`)

Companion to `stage10_posttraining.md` (section 3). This file walks through every line of
`dpo.py` in order, with shapes and worked numbers, so you can rebuild it on a whiteboard.
Its twin for RL is `stage10_grpo_walkthrough.md`.

Running example used throughout: `--batch-size 8 --beta 0.1`, so **B = 8** preference pairs
per step, and the model sees **2B = 16** sequences per forward pass (8 chosen + 8 rejected).

---

## 0. The idea in four sentences
1. Each training example is a triple: (prompt, **chosen** answer, **rejected** answer).
2. The *implicit reward* of an answer is `beta * (log pi(y|x) - log pi_ref(y|x))`: how much
   more likely the policy makes it than the frozen starting model does.
3. The loss is plain logistic regression on pairs: push the chosen answer's implicit reward
   above the rejected one's, `loss = -log sigmoid(r_chosen - r_rejected)`.
4. There is no sampling, no reward model and no critic: it is a supervised loss on a fixed
   dataset, and pi_ref is both the KL leash and the zero point of the reward.

---

## 1. Where the loss comes from (the derivation interviewers ask for)
Start from the same objective PPO and GRPO optimise:
```
max_pi   E_{y ~ pi}[ r(x, y) ]  -  beta * KL( pi(.|x) || pi_ref(.|x) )
```
**Step 1: it has a closed-form optimum.** For a fixed r, the maximiser is
```
pi*(y|x) = pi_ref(y|x) * exp( r(x,y) / beta ) / Z(x)        Z(x) = sum_y pi_ref(y|x) exp(r/beta)
```
(Reference re-weighted by exponentiated reward. Z(x) is the normaliser, and it is
intractable: it sums over every possible answer.)

**Step 2: flip it around.** Take logs and solve for r:
```
r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) )  +  beta * log Z(x)
```
Any policy *defines* a reward this way. That is the "your language model is secretly a
reward model" line from the paper title.

**Step 3: plug into Bradley-Terry.** Human preferences are modelled as
`P(y_c > y_r | x) = sigmoid( r(x, y_c) - r(x, y_r) )` (the same loss `reward_model.py`
trains). In the *difference*, `beta * log Z(x)` appears once with + and once with -, so it
**cancels**. Replace pi* with the trainable pi_theta and maximise the log-likelihood of the
observed preferences:
```
loss = -log sigmoid( beta * [ (log pi(y_c) - log pi_ref(y_c)) - (log pi(y_r) - log pi_ref(y_r)) ] )
```
Why it matters: Z(x) is the reason you can't just "solve" RLHF. DPO only ever compares two
answers *to the same prompt*, so the intractable part drops out.

**The gradient** (worth memorising, it explains every behaviour in section 8):
```
grad loss = -beta * sigmoid(r_rejected - r_chosen) * [ grad log pi(y_c) - grad log pi(y_r) ]
            \____________ weight ______________/     \_______ direction ________/
```
- Direction: raise log pi(chosen) and lower log pi(rejected). **Only the difference is
  constrained.** Nothing says the chosen one must go up.
- Weight: `sigmoid(-margin)`. A pair the model already ranks correctly by a wide margin gets
  almost no gradient; a wrongly ranked pair gets close to beta. It's a built-in hard-example
  weighting.

---

## 2. Imports
```python
from .common import (EVAL_PROBLEMS, MODEL_NAME, build_batch, completion_ids, evaluate, fmt,
                     frozen_copy, load_model, make_preference_pairs, make_problems, prompt_ids,
                     sequence_logprobs)
```
Most helpers are the same ones GRPO uses (see the table in the GRPO walkthrough). The two
that are new or matter differently here:

| helper | what it does in DPO |
|---|---|
| `make_preference_pairs` | turns problems into (question, chosen, rejected) triples by a rule |
| `completion_ids` | a fixed answer string -> token ids **+ EOS** (DPO never generates during training) |
| `build_batch` | prompt+completion, right-padded, plus the completion mask (same as GRPO) |
| `sequence_logprobs` | per-token log-probs `(2B, T-1)`; DPO sums them per sequence |

Notice what is **not** imported: `sample` and `reward_fn`. DPO training never calls
`generate` and never scores anything. Both appear only inside `evaluate`.

---

## 3. The data: `make_preference_pairs` (in `common.py`)
```python
if rng.random() < 0.5:
    rejected = f"{a} + {b} = {ans}"                              # right number, wrong format
else:
    rejected = str(ans + rng.choice([-10, -2, -1, 1, 2, 10]))     # right format, wrong number
pairs.append((q, str(ans), rejected))
```
- **Chosen** is always the bare correct number (the reward-1.0 answer).
- **Rejected** is half the time chatty-but-correct (reward 0.5), half the time
  bare-but-wrong (reward 0.0). So the pairs try to teach both format and arithmetic.
- In real RLHF these come from human labelers or an LLM judge comparing two samples. Here a
  rule writes them. Important: **they are not samples from the model.** That makes this
  *offline* DPO, and section 8 shows what that costs.

What the pairs look like as tokens (SmolLM2 splits digits one per token):
```
prompt   46 tokens  "<|im_start|>system ... <|im_start|>user\nWhat is 17 + 21? ...<|im_start|>assistant\n"
chosen    3 tokens  ['3', '8', '<|im_end|>']
rejected  3 tokens  ['3', '9', '<|im_end|>']                                      (wrong number)
rejected 11 tokens  ['1','7',' +',' ','2','1',' =',' ','3','8','<|im_end|>']      (chatty)
```
A subtle consequence: for `38` vs `39` the first token `3` is identical with an identical
context, so its log-prob appears in both sums and **cancels exactly**. The whole learning
signal of that pair sits on one digit (and the EOS after it). DPO only learns from where
the two answers differ.

---

## 4. `dpo_loss`: the whole method in ~15 lines
```python
def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float, loss_type: str = "sigmoid",
             len_c=None, len_r=None, simpo_gamma: float = 0.5):
```
Inputs are **one number per pair**, shape `(B,)`: the *summed* completion log-prob
`log pi(y|x)` under the policy (`pol_*`) and the reference (`ref_*`), for chosen (`_c`) and
rejected (`_r`). Compare with GRPO, whose loss works per token `(N, T-1)`. DPO is a
**sequence-level** loss.

### 4.1 The SimPO branch (reference-free)
```python
    if loss_type == "simpo":
        logits = beta * (pol_c / len_c - pol_r / len_r) - simpo_gamma
        return -F.logsigmoid(logits).mean(), beta * pol_c / len_c, beta * pol_r / len_r
```
- Reward = `beta * average per-token log pi`. Two changes from DPO:
  1. **No pi_ref.** Half the memory and half the forward passes. The price: nothing anchors
     the policy except the data and the LR.
  2. **Divided by length.** A summed log-prob gets more negative with every token, so
     summed rewards are biased toward short answers (or long ones, depending on which side
     the data put them on). Averaging removes that.
- `- simpo_gamma`: the sigmoid only stops pushing once the chosen side wins by at least
  gamma. It's a target margin, like an SVM's.
- It returns the two rewards as well, so the logging below works for every branch.
- Caution: the SimPO paper uses beta around 2-2.5. With this script's default beta = 0.1,
  `beta * (avg-logp gap)` is tiny next to gamma = 0.5, so run it as `--loss simpo --beta 2`.
  (No SimPO run is in `posttrain_all.sh`.)

### 4.2 The implicit rewards
```python
    chosen_reward = beta * (pol_c - ref_c)
    rejected_reward = beta * (pol_r - ref_r)
    margin = chosen_reward - rejected_reward
```
- `pol_c - ref_c` = `log( pi(y_c) / pi_ref(y_c) )`: how many nats more likely the policy
  has made the chosen answer since training started. Times beta = the implicit reward from
  section 1.
- At step 0, `policy == ref`, so **both rewards are exactly 0 and margin = 0**.
- `ref_c`, `ref_r` are constants (computed under `no_grad`). The gradient flows only through
  `pol_c` and `pol_r`.

### 4.3 The DPO loss
```python
    if loss_type == "sigmoid":
        loss = -F.logsigmoid(margin)
```
- `-log sigmoid(m)` = `log(1 + e^-m)`: logistic regression where the "logit" is the margin.
- Use `F.logsigmoid`, not `torch.log(torch.sigmoid(m))`: for very negative m the sigmoid
  underflows to 0 and the log gives -inf. `logsigmoid` stays finite.
- **Starts at log 2 = 0.6931** because margin = 0. That is the first sanity check of any DPO
  run, and `tests/test_posttrain.py::test_dpo_starts_at_log2` tests it. If your step-0 loss
  isn't 0.693, the reference, masking or batching is wrong.

Worked example, beta = 0.1:
```
log pi(y_c) - log pi_ref(y_c) = +2.0  -> chosen_reward   = +0.2
log pi(y_r) - log pi_ref(y_r) = -3.0  -> rejected_reward = -0.3
margin = 0.5   loss = log(1 + e^-0.5) = 0.474   gradient weight = sigmoid(-0.5) = 0.378
```
Now push the margin to 3.1 (where the real run ends): loss = 0.044, weight = 0.043. Those
pairs have nearly stopped teaching. Note that a margin of 3.1 at beta = 0.1 means a
**31-nat** gap in log-ratio. That's a large move for 320 examples.

### 4.4 The IPO branch
```python
    elif loss_type == "ipo":
        loss = (margin / beta - 1 / (2 * beta)) ** 2
```
- `margin / beta` undoes the scaling: it's the raw log-ratio gap
  `h = (log pi_c - log ref_c) - (log pi_r - log ref_r)`.
- IPO regresses h to a **fixed target** `1/(2 beta)` = 5 nats at beta = 0.1, with a squared
  loss.
- Why: on separable data the sigmoid loss is never zero, so DPO keeps pushing h toward
  infinity (the deterministic-preference overfitting argument from the IPO paper). The square
  pushes back once h > 5, so the margin is bounded.
- Step-0 sanity value: `(0 - 5)^2 = 25.0`, which is exactly the first line of `dpo_ipo.log`.

```python
    else:
        raise ValueError(loss_type)
    return loss.mean(), chosen_reward, rejected_reward
```
- `.mean()` over the B pairs -> scalar. The per-pair rewards come back for logging.

---

## 5. `summed_logps`
```python
def summed_logps(model, ids, attn, mask):
    return (sequence_logprobs(model, ids, attn) * mask).sum(-1)      # (2B,)
```
- `sequence_logprobs` -> `(2B, T-1)` per-token log-probs.
- `* mask` keeps only completion tokens; `.sum(-1)` -> `log pi(completion | prompt)`, one
  number per row.
- **Sum, not mean.** The derivation in section 1 is about the probability of the *whole*
  answer, which is a product of token probabilities, i.e. a sum of log-probs. Using the mean
  gives you SimPO's reward instead, which is a different method.
- Does masking the prompt matter? For plain DPO, **not mathematically**: the prompt is the
  same in the chosen and rejected rows, so its log-probs cancel in `pol_c - pol_r` (and in
  the ref terms). It does matter for SimPO's length division and for the SFT term below,
  and it saves you from padding bugs. Always mask.

---

## 6. `main()`: arguments
```python
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--loss", default="sigmoid", choices=["sigmoid", "ipo", "simpo"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--sft-weight", type=float, default=0.0)
```
| knob | effect if you raise it |
|---|---|
| `--model` | start point = pi_ref. `runs/posttrain_sft` gives the standard SFT -> DPO recipe |
| `--beta` | stronger leash, smaller implicit rewards; the sigmoid saturates later in log-ratio terms. 0.1 is the common default |
| `--steps` x `--batch-size` | number of pairs (40 x 8 = 320); each pair is seen **once** (one epoch) |
| `--lr` | faster drift. DPO on a full model is typically 5e-7 .. 5e-6, well below SFT |
| `--sft-weight` | adds an NLL term on the chosen answer (RPO); keeps chosen likely in absolute terms |

---

## 7. `main()`: setup and the training step
```python
    torch.manual_seed(0)
    policy, tok = load_model(args.model)
    ref = frozen_copy(policy)                 # pi_ref = the starting point, never updated
    print("before:", fmt(evaluate(policy, tok, EVAL_PROBLEMS)))
```
- Same two-model setup as GRPO: trainable `policy` + frozen `ref`. Two copies of the
  weights, but only one gets gradients and Adam state.
- `ref` is a copy of **whatever you start from**. After SFT, pi_ref is the SFT model, not the
  base model. That's the standard recipe: the KL leash should hold you near a model that
  already answers in the right format.

```python
    pairs = make_preference_pairs(make_problems(args.steps * args.batch_size, seed=2))
    print("example pair:", pairs[0])
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    policy.train()
    B = args.batch_size
```
- 320 pairs, generated up front: the dataset is **fixed** before training starts. `seed=2`
  keeps them away from the eval set (seed 12345) and from GRPO's prompts (seed 3).
- `policy.train()` vs `ref.eval()`: SmolLM2 has no dropout, so this changes nothing here.
  With dropout it would matter: the policy and ref would then disagree at step 0 and the
  loss wouldn't start at log 2. Real DPO trainers turn dropout off for that reason.

### 7.1 Build one batch of 2B sequences
```python
    for step in range(args.steps):
        chunk = pairs[step * B:(step + 1) * B]
        p = [prompt_ids(tok, q) for q, _, _ in chunk]
        c = [completion_ids(tok, yc) for _, yc, _ in chunk]
        r = [completion_ids(tok, yr) for _, _, yr in chunk]
        ids, attn, mask = build_batch(p + p, c + r, tok.pad_token_id)
```
- `p + p` and `c + r` are list concatenation, so the rows are
  `[chosen_1 .. chosen_8, rejected_1 .. rejected_8]`. Row `i` and row `i + B` are the two
  sides of the same pair. This "concatenated forward" is what TRL does too: one forward pass
  instead of two, and identical padding/numerics for both sides.
- Shapes for the 38-vs-chatty example above: prompt 46 + longest completion 11 -> `ids`
  `(16, 57)`, `mask` `(16, 56)`, mask sums 3 for the bare answer and 11 for the chatty one.

### 7.2 Four log-probs per pair
```python
        pol = summed_logps(policy, ids, attn, mask)
        with torch.no_grad():
            refl = summed_logps(ref, ids, attn, mask)
        lens = mask.sum(-1)
```
- `pol`: `(16,)`, **with** gradients. `refl`: `(16,)`, constant.
- The reference forward can be done once for the whole dataset and cached. Real pipelines
  often precompute `ref_logps` offline, so training holds only one model in memory. That
  trick is possible only because the data is fixed. GRPO can't do it, since its samples
  change every step.
- `lens`: completion lengths incl. EOS, used only by SimPO and the SFT term.

### 7.3 The loss
```python
        loss, rc, rr = dpo_loss(pol[:B], pol[B:], refl[:B], refl[B:], args.beta, args.loss,
                                lens[:B], lens[B:])
```
- `[:B]` = chosen rows, `[B:]` = rejected rows. That's the reason for the concatenation order.

```python
        if args.sft_weight > 0:
            loss = loss + args.sft_weight * (-pol[:B] / lens[:B]).mean()
```
- `-pol / lens` = per-token NLL of the chosen answer: the ordinary SFT loss, restricted to
  the chosen side. This is RPO / "DPO + NLL" (Llama 3 and others use a version of it).
- Why: section 1's gradient only constrains the *difference*. This term adds a direct
  "make the chosen answer likely" force, so the margin can't be won only by crushing the
  rejected side.
- Scale check: at step 0 of `dpo_plus_sft.log`, loss = 3.39 = 0.693 (DPO) + ~2.7 (NLL per
  token). The NLL term dominates the total, which is intended at weight 1.0.

```python
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```
- The same step as SFT and GRPO. There are no inner epochs and no ratio clipping: there are
  no samples to reuse, so there's no `pi_old` and no trust region. **The only thing keeping
  updates small is beta (through the sigmoid's saturation) and the LR.**

### 7.4 Logging: the four numbers to watch in any DPO run
```python
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}  "
                  f"pair-acc {(rc > rr).float().mean():.2f}  margin {(rc - rr).mean():+.3f}  "
                  f"logp chosen {pol[:B].mean():.2f}  rejected {pol[B:].mean():.2f}")
```
| column | meaning | healthy |
|---|---|---|
| `loss` | starts at 0.693 (sigmoid) or 25 (IPO, beta 0.1) | goes down |
| `pair-acc` | share of pairs with `r_chosen > r_rejected` | up. **0.00 at step 0** because 0 > 0 is false |
| `margin` | mean `r_c - r_r` (beta-scaled) | up, but not without bound |
| `logp chosen` | mean summed log pi of chosen answers | **should not fall**; falling = pathology |
| `logp rejected` | the same for rejected | falls |

The last two columns are the ones that loss and pair accuracy hide. Also note that each
step uses a *new* batch of 8 pairs, so the columns are noisy and are not tracking the same
pairs over time.

---

## 8. What the real runs showed (`runs/posttrain_logs/dpo*.log`)
Base model before any training: `reward 0.242  accuracy 0.484  format 0.000`. It answers
`"63 + 11 = 74"` (chatty, usually right).

### 8.1 Plain DPO (`dpo.log`): wins its own metric, loses everything else
```
step   0  loss 0.6931  pair-acc 0.00  margin +0.000  logp chosen  -9.26  rejected  -7.95
step  25  loss 0.0990  pair-acc 1.00  margin +3.422  logp chosen -10.50  rejected -41.00
step  39  loss 0.2426  pair-acc 0.88  margin +3.103  logp chosen -15.20  rejected -43.51
after:  reward 0.000  accuracy 0.000  format 0.000
   'What is 63 + 11?' -> 'To add 63 and 11, we need to follow the order'
```
- Loss starts at exactly log 2: the plumbing is correct.
- At step 0 the base model prefers the *rejected* side (-7.95 > -9.26): it likes being
  chatty. DPO has real work to do.
- **The margin was won by crushing the rejected answers** (-8 -> -43) while the chosen ones
  *also* fell (-9 -> -15). Both log-probs falling is the textbook DPO failure, predicted by
  the gradient in section 1: only the difference is constrained.
- The freed probability mass went to an answer that is in **neither** side of any pair:
  `"To add 63 and 11, we need to..."`. Greedy decoding picks that, never reaches a number,
  and scores 0. The pairs say "bare number > chatty equation" and "right number > wrong
  number", but nothing about "To add..." text, so nothing stops it.

### 8.2 IPO (`dpo_ipo.log`): bounded margin, same collapse
```
step   0  loss 25.0000  pair-acc 0.00  margin +0.000  logp chosen -9.26  rejected -7.95
step  39  loss 10.2404  pair-acc 0.88  margin +0.398  logp chosen -5.77  rejected -7.03
after:  reward 0.023  accuracy 0.047  format 0.000
```
- Step-0 loss 25 = (0 - 5)^2, as computed in 4.4.
- The margin stayed small (0.4 = a 4-nat gap, near the 5-nat target) and the chosen logp
  even *rose* (-9.3 -> -5.8). IPO did fix the "margin to infinity" problem.
- It still collapsed to `"To add 63 and 11, we need to combine the two"`. A chosen logp of
  -5.8 over ~3 tokens is still well below what the first token "To" gets. **Rising is not
  enough: the chosen answer has to become the argmax for greedy decoding to output it.**
  Bounding the margin doesn't control where the rest of the mass goes.

### 8.3 SFT -> DPO (`sft_then_dpo.log`): the standard recipe
```
before: reward 0.406  accuracy 0.422  format 0.391           (the SFT model)
step   0  loss 0.6931  pair-acc 0.00  margin +0.000  logp chosen -2.58  rejected  -8.17
step  39  loss 0.2767  pair-acc 1.00  margin +3.250  logp chosen -2.34  rejected -40.10
after:  reward 0.328  accuracy 0.328  format 0.328
```
- After SFT the model already strongly prefers the chosen side (-2.6 vs -8.2), and
  **chosen stays put** (-2.3 at the end). With pi_ref = the SFT model, the leash holds the
  model near a policy that already answers with a bare number. That's why SFT comes first.
- The format survived (every correct answer is now a bare number: reward == accuracy ==
  format).
- But arithmetic got *worse* (42 % -> 33 %): `63 + 11 -> 178`. The wrong-number rejections
  are only near-misses (+-1, +-2, +-10). The model learned "not 73, 75, 64, 84" but nothing
  in the data says where that mass should go instead. Again the pairs describe two points
  per prompt, and the policy is free everywhere else.

### 8.4 DPO + SFT term, lr 1e-6 (`dpo_plus_sft.log`)
```
step   0  loss 3.3919  pair-acc 0.00  margin +0.000  logp chosen -9.26  rejected -7.95
step  39  loss 1.6208  pair-acc 0.88  margin +0.423  logp chosen -4.10  rejected -5.62
after:  reward 0.211  accuracy 0.250  format 0.172
   'What is 63 + 11?' -> '166'
```
- The fix works on the axis it targets: **chosen logp now rises** (-9.3 -> -4.1), and the
  output is a bare number (format 0 % -> 17 %), with no "To add..." collapse.
- Accuracy still dropped below the base model (48 % -> 25 %). A likely reason, also seen in
  plain SFT (42 %): the chatty `63 + 11 = ` prefix works as a tiny scratchpad. Writing the
  operands right before the answer helps a 135M model add. Forcing the bare number removes
  it. That's the chain-of-thought effect in miniature: the "preferred" format is also the
  harder one.
- Caveat: this run changed two things at once (SFT term **and** LR 5e-6 -> 1e-6), so it does
  not isolate the SFT term.

### 8.5 The overall lesson
Offline DPO optimises **pair ranking**, not task quality. On all four runs, pair-acc
reached 0.88-1.00 while held-out accuracy fell. Fixes, strongest first:
1. **On-policy / iterative DPO**: sample answers from the *current* model, rank them with
   the reward, and train on those pairs. The rejected side is then whatever the model
   actually does wrong (e.g. "To add..."), so the mass can't hide there.
2. **SFT first, then DPO** (pi_ref = SFT model), plus an **SFT/NLL term** on chosen.
3. Larger beta, lower LR, fewer steps, early stopping on a real metric (not the DPO loss).

Compare GRPO on the same task: it reached 69 % accuracy because it learns from its own
samples, scored by the real reward.

---

## 9. DPO vs GRPO side by side
| | DPO | GRPO |
|---|---|---|
| data | fixed (x, y_c, y_r) pairs | prompts only; answers sampled every step |
| signal | which of two answers is better | a scalar reward per sample |
| on/off-policy | offline | on-policy |
| models in memory | policy + ref (ref logps can be precomputed) | policy + ref |
| `generate` during training | never | every step (the expensive part) |
| loss granularity | sequence: summed log-probs | per token, averaged per completion |
| baseline | the other answer in the pair | the mean of the G siblings |
| trust region | none: only beta and the LR | ratio clipping + KL term |
| KL to ref | implicit, built into the reward | explicit k3 term in the loss |
| starting loss | log 2 = 0.693 | ~0 (advantages have mean 0) |
| typical failure | both logps fall; mass goes off-data | no-signal groups; reward hacking; length drift |

Loosely: DPO is GRPO with G = 2, a binary reward, a baseline equal to the
midpoint of the two, no clipping, and the samples written in advance by someone else.

---

## 10. For the speech model
Decode an n-best list per utterance (beam search or sampling), score each hypothesis with
WER against the reference, and take (lowest WER, highest WER) as (chosen, rejected). That's
**on-policy** preference data for free, with no labelers. The audio embeddings sit in the
prompt part, so they are masked out of the log-probs like any other prompt token, and the
same `dpo_loss` applies unchanged. Add the SFT term on the chosen side so the model doesn't
learn "lower the bad hypothesis" without also "raise the good one".

---

## 11. Five-line summary to say out loud
1. The RLHF objective has a closed-form optimum, so reward = `beta * log(pi / pi_ref)` up to
   a prompt-only term, which cancels in a pairwise comparison.
2. Loss = `-log sigmoid(beta * [log-ratio(chosen) - log-ratio(rejected)])`, on summed
   completion log-probs. It starts at log 2.
3. One concatenated forward over chosen + rejected for the policy, one `no_grad` forward
   for the reference (or cached), then an ordinary optimizer step: no sampling, no critic.
4. It only constrains the *gap*, so watch `logp chosen`: if it falls, add an SFT term, start
   from SFT, or use IPO / a larger beta.
5. Offline pairs say nothing about answers outside them. On-policy pairs ranked by a real
   reward are the strongest fix.

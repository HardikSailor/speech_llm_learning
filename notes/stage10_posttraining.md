# Stage 10 - Post-training: SFT, reward models, DPO, PPO, GRPO

Run (sequential; the login node gives you 4 CPUs / 16 GB total):
```
source scripts/env.sh
./scripts/posttrain_all.sh            # everything, logs in runs/posttrain_logs/
$PY -m tests.test_posttrain           # the math, no model, ~1 s
$PY -m src.posttrain.grpo --steps 30  # or any single method
```

## The setup: one toy task, every method
Text-only SmolLM2-135M-Instruct (the speech model's LLM) on
`"What is 37 + 48? Answer with just the number."`. Reward is a rule (`common.reward_fn`):

| output            | reward | why                    |
|-------------------|--------|------------------------|
| `85`              | 1.0    | right and obeys format |
| `37 + 48 = 85`    | 0.5    | right, chatty          |
| `86` / rambling   | 0.0    | wrong                  |

The base model is ~48 % correct and **0 %** on format, so there is a visible thing to fix.
The same toy reward applies to ASR later: `reward = -WER` is a verifiable reward too.

## The one function everything shares
```python
logps = log_softmax(model(ids).logits[:, :-1]).gather(ids[:, 1:])   # (B, T-1)
log_pi(y|x) = (logps * completion_mask).sum(-1)
```
Every method is "do something with `log pi(y|x)`":

| method | uses log pi(y\|x) as | needs |
|---|---|---|
| SFT  | `-mean(logps)` on demonstrations | good answers |
| RM   | (not the policy) BT loss on a scalar head | pairs |
| DPO  | `beta*(log pi - log pi_ref)` = implicit reward, BT loss on pairs | pairs, pi_ref |
| PPO  | ratio `pi/pi_old` x GAE advantage, clipped | reward, pi_ref, critic |
| GRPO | ratio x group-normalised advantage, clipped, + KL | reward, pi_ref |

## 1. SFT (`sft.py`)
Masked next-token CE on the ideal answer. Limits: only positive examples; teacher forcing, so
it never sees its own errors (exposure bias); someone must write the ideal answer.
**Gotcha seen here:** training to answer *only* the number removed the model's "scratchpad"
(`a + b = c`). Format went up while accuracy went *down* for a while. The output format is
part of how the model computes. This is the toy version of why reasoning models are allowed to
think before answering.

## 2. Reward model (`reward_model.py`)
LLM body + `Linear(d, 1)` read at the last real token. Bradley-Terry:
`P(A > B) = sigmoid(r_A - r_B)`, `loss = -log sigmoid(r_c - r_r)`.
Only differences are identified, so absolute scores mean nothing (normalise before PPO).
Failure mode: the policy finds inputs where the RM is wrong = **reward hacking**. That is
the reason for the KL leash.

## 3. DPO (`dpo.py`) - line-by-line walkthrough: `stage10_dpo_walkthrough.md`
Derivation in 3 lines (whiteboard this):
1. RLHF objective: `max E[r] - beta KL(pi || pi_ref)`.
2. Its optimum: `pi*(y|x) = pi_ref(y|x) exp(r/beta) / Z(x)`, so `r = beta log(pi*/pi_ref) + beta log Z(x)`.
3. Put it in Bradley-Terry: `Z(x)` cancels between chosen and rejected ->
   `loss = -log sigmoid(beta[(log pi_c - log ref_c) - (log pi_r - log ref_r)])`.

Things to know:
- **Loss starts at exactly log 2 = 0.693** (pi == pi_ref -> margin 0). If it doesn't, your
  reference or masking is wrong. `tests/test_posttrain.py` checks this.
- **beta**: small = weak leash, policy drifts far; large = stays near ref.
- **Both log-probs often fall.** Watch `logp chosen` vs `rejected` in the log: DPO only
  needs the *gap* to grow. It often gets there by crushing the rejected answer while the
  chosen one also drops. This is a known DPO pathology, and it's why people add an SFT term
  (RPO) or use on-policy pairs.
- **Offline**: pairs are fixed, not sampled from the current policy. That's cheap and stable,
  but the policy can move to places the data never covers. "Online/iterative DPO" re-samples
  the pairs.
- **Length bias**: summed log-probs favour certain lengths -> SimPO averages per token.

Variants in one line each:
- **IPO**: squared loss to a fixed margin; stops DPO from pushing margins to infinity.
- **KTO**: no pairs, just "good"/"bad" labels per answer, prospect-theory weighting.
- **ORPO**: no ref model; SFT loss + odds-ratio penalty on rejected, in one stage.
- **SimPO**: no ref model; length-normalised log-prob as reward, plus margin gamma.
- **CPO**: DPO without the ref term + SFT term; from MT.

## 4. PPO (`ppo.py`)
Token = action, reward arrives at the last token, KL to ref is subtracted **per token inside
the reward**. GAE: `delta_t = r_t + V_{t+1} - V_t`, `A_t = delta_t + lam A_{t+1}`.
Loss = clipped ratio objective + value regression. Why it is painful: 4 models (policy, ref,
RM, critic), a critic that must learn first (early advantages are noise), many
hyperparameters, and generation in the training loop. Numbers to watch: `clip-frac` (0.1-0.3
is healthy; 0 = steps too small; high = too big), value loss, KL.

## 5. GRPO (`grpo.py`) - line-by-line walkthrough: `stage10_grpo_walkthrough.md`
Delete the critic. Sample G answers per prompt and use `A_i = (r_i - mean) / std` within
the group. KL is moved out of the reward and into the loss (k3 estimator
`ref/pi - log(ref/pi) - 1`, always >= 0).
- `no-signal`: share of groups where all G got the same reward -> zero gradient. This is the
  main efficiency killer (DAPO's "dynamic sampling" drops those groups and re-samples).
- Per-sequence length normalisation biases toward short correct / long wrong answers;
  **Dr. GRPO** normalises by a constant. Std-normalisation over-weights easy/hard prompts.
- With 1 inner epoch, `ratio == 1` and the clip does nothing: GRPO is then just
  REINFORCE with a group baseline + KL.

## 6. Failure modes and how you detect them
| failure | symptom in logs | fix |
|---|---|---|
| reward hacking | reward up, eval/human quality down; length jumps | KL up, better reward, held-out eval |
| KL blow-up | KL grows fast, gibberish samples | larger beta, lower LR |
| entropy collapse | all G samples identical, `no-signal` -> 1 | higher temperature, entropy bonus, clip-higher (DAPO) |
| DPO over-optimisation | margin -> inf, both logps fall | IPO, larger beta, early stop, SFT term |

## Results on this box (`runs/posttrain_logs`, 64 held-out problems, greedy)
| run | reward | accuracy | format | one-line reading |
|---|---|---|---|---|
| base model | 0.242 | 48 % | 0 % | right half the time, never obeys format |
| SFT (40 steps) | 0.406 | 42 % | 39 % | learns format, loses the `a + b =` scratchpad |
| reward model | - | - | - | held-out pair accuracy 0.00 (all tied) -> 0.77 |
| DPO | 0.000 | 0 % | 0 % | **collapse**: both logps fall, mass goes to "To add 63 and 11, we..." |
| IPO | 0.023 | 5 % | 0 % | same collapse: the bounded margin did not save it |
| DPO + SFT term, lr 1e-6 | 0.211 | 25 % | 17 % | chosen logp now RISES (-9.3 -> -4.1), but still below base |
| SFT -> DPO | 0.328 | 33 % | 33 % | format kept, pairs "won" (pair-acc 1.0), accuracy fell |
| PPO | 0.320 | 64 % | 0 % | +16 pts accuracy, settles for the chatty 0.5 reward |
| GRPO | 0.344 | **69 %** | 0 % | best accuracy; never finds the bare-number format |
| SFT -> GRPO | **0.430** | 48 % | 38 % | best reward, small gain: `no-signal` 0.75 |

What the table teaches (better interview material than any win):
1. **RL only amplifies what the policy already samples.** The base model sampled the bare
   format ~3 % of the time, so PPO/GRPO almost never saw reward 1.0 and optimised the 0.5
   they could reach. This is the reason for SFT -> RL (DeepSeek-R1's "cold start" data).
2. **But after SFT the groups agree, so GRPO starves.** In 75 % of SFT->GRPO groups all 8
   samples scored the same -> advantage 0 -> no gradient. Too easy and too hard both kill
   the signal. That's why DAPO filters for groups with mixed rewards, and why curricula target
   ~50 % pass rate.
3. **Offline DPO can win on its own metric and lose on the real one.** Pair accuracy -> 1.0
   and margin -> 3, while greedy accuracy drops. The pairs only describe two points per
   prompt; the policy is free to move mass anywhere else. Fixes, in order of strength:
   on-policy pairs (sample from the current model, rank by reward) > SFT term > larger beta /
   lower LR / fewer steps.
4. **Verifiable reward + group baseline (GRPO) was the most reliable method here**, with no
   learned reward model to hack and no critic to warm up.

## Interview Q&A
**Q: Why do you need a reference model?** Without the KL leash the policy maximises the
reward by any means: it drifts off the language manifold and exploits reward-model errors.
In DPO pi_ref also defines the implicit reward itself.

**Q: DPO vs PPO in one sentence?** DPO solves the same KL-regularised objective in closed
form on a fixed set of pairs (supervised, offline). PPO optimises it online with sampling and
a critic: more expensive, but it explores and can use any reward.

**Q: Why did GRPO win for reasoning?** With verifiable rewards there's no RM to hack.
Dropping the critic halves memory, and a group baseline is a good, cheap variance reducer
when you sample many answers per prompt anyway.

**Q: Where does the KL go in PPO vs GRPO?** PPO: subtracted from the per-token reward
(it flows through GAE). GRPO: added directly to the loss.

**Q: What does beta do in DPO?** It's the inverse temperature of the implicit reward. It also
sets the strength of the KL leash: small beta = large deviations allowed.

**Q: How would you do this for the speech model?** Sample n-best transcripts per utterance
and score them with -WER. Then either DPO on (best, worst) or GRPO with reward = -WER on the
group. That's free preference data, with no humans.

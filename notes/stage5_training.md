# Stage 5 - The training loop, by hand

Run:
```
source scripts/env.sh
$PY -m src.train --root $LIBRISPEECH/dev-clean --minutes 30 --epochs 6 \
    --batch-size 8 --grad-accum 2 --lr 1e-3 --out runs/stage5_30min
```

## What stage-1 training actually is
Encoder frozen, LLM frozen, **only the projector learns** (2.4M of 145.2M = 1.68 %).
The task is not "learn ASR" - Whisper already hears and the LLM already writes English.
The task is **modality alignment**: learn a map from encoder space into a region of the LLM's
input space where the frozen LLM reads the audio as if it were text.

## The loop, line by line
```python
loss, _ = model(mel, mel_lengths, segments)   # CE on answer tokens only
(loss / grad_accum).backward()                # scale so N micro-batches == one big batch
if micro % grad_accum == 0:
    gnorm = clip_grad_norm_(params, 1.0)      # clip BEFORE step, AFTER the last accumulation
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
```
- **Gradient accumulation** = a larger batch you cannot fit in memory. `loss/accum` is
  required: `.backward()` *sums* into `.grad`, so without it your effective LR is N x larger.
  Classic bug: clipping inside the accumulation loop (clips partial gradients, N times).
- **`set_to_none=True`** frees the gradient tensors instead of writing zeros - less memory,
  slightly faster, and it makes "which parameters got no gradient" visible (`p.grad is None`).
- **Warmup then cosine.** Adam's second-moment estimate is unreliable for the first tens of
  steps; a full-size step there can wreck a pretrained model. 6 % warmup is a normal default.
- **Grad-norm is your best diagnostic.** Steadily rising = LR too high or a data bug.
  Collapsing to ~0 = the loss is saturated or something upstream is detached.
- **`betas=(0.9, 0.98)`** rather than 0.999: standard for transformer/ASR training, adapts
  faster on short runs.
- **Only trainable params go in the optimizer.** Otherwise Adam allocates 2 extra floats per
  parameter for tensors that never change, and weight decay would quietly modify a "frozen"
  tower.
- **Frozen towers stay in `.eval()`** while the projector is in `.train()`. Dropout in the
  frozen LLM would otherwise inject noise into the only signal the projector gets.

## Two efficiency things that matter here
1. **The frozen encoder runs under `no_grad`** (`SpeechLLM.encode_audio`): no activations
   are stored for its 4 layers. The **LLM cannot** - gradients must flow *through* it to
   reach the projector. Freezing saves weight-gradient computation and optimizer state, not
   the backward pass. This surprises people in interviews.
2. **Length bucketing** (`src/sampler.py`). Measured on this data:
   `padding waste: 36.4 % naive -> 5.1 % bucketed`. That is a ~1.5x speedup for free.
   We bucket inside a pool of ~50 batches, not globally, so batch content is not perfectly
   correlated with utterance length.

## Reading the log
```
step  1/6  loss 6.8471  lr 1.00e-03  |g| 10.884   7.4x realtime   8s
```
`Nx realtime` = seconds of audio processed per wall-clock second. It is the throughput number
an ASR person expects to hear, and it is comparable across machines and model sizes.

## Evaluation
`evaluate()` returns teacher-forced val loss (cheap, low variance) **and** free-running WER
(expensive, but the only number that reflects what the model actually does). They diverge, and in this run they diverged
*in opposite directions*: between step 80 and 120, val loss improved (4.55 -> 4.51) while WER
got worse (172 % -> 349 %). Teacher-forced loss measures "given the gold prefix, is the next
token right"; WER measures what happens when the model feeds itself. Never report only one.
WER > 100 % is normal and just means insertions - the model is rambling.

## Interview Q&A
**Q: Why is CE loss on the answer only?** Loss on the prompt teaches the model to regenerate
your own instruction, spends capacity on a fixed string, and dilutes the gradient that
actually concerns the audio.

**Q: How do you know the model is using the audio at all?** `src/ablate.py`. Corrupt the audio
and re-measure. Measured on `runs/stage5_30min` (30 min of audio, 120 steps):

| mode | val loss | delta |
|---|---|---|
| real | 4.4423 | - |
| zeros | 5.5077 | +1.07 |
| noise | 5.6098 | +1.17 |
| **shuffled** | 4.6095 | **+0.17** |

**`shuffled` is the control that matters**, and reading this table correctly is the whole
lesson. zeros/noise look like a big dependence on audio, but they push the LLM off
distribution - it has never seen a zero or noise vector in that slot - so most of that +1.07
is distribution shock, not information use. `shuffled` swaps in *another real utterance's*
audio: same distribution, same scale, only the correspondence is destroyed. It costs just
**+0.17**.

Conclusion: after 30 minutes of audio the projector transmits generic "this is read English
speech" information (register, prosody, roughly how long the utterance is) but almost no
utterance-specific content. That matches the decodes exactly:

```
ref: lectures                        hyp: i am a man of good heart
ref: plain buns seventeen twenty nine hyp: the man was a man of no end
```

Fluent LibriSpeech-register prose, unrelated to the waveform. Always run the shuffled control
before believing a loss curve.

**Q: What LR for a randomly initialised projector on a frozen model?** ~1e-3, i.e. pretraining
LRs, not finetuning LRs. Nothing pretrained is at risk. Once you unfreeze the LLM (Stage 8)
you drop to ~1e-4/2e-5.

**Q: What would you change to scale this up?** bf16 autocast, flash-attention-2, more data
before more epochs, and a real validation set. Those are Stage 6.

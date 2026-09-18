# Stage 4 - The bridge (the part that is actually "the model")

Run:
```
python -m src.projector            # the three adapter designs, shapes only
python -m src.modeling             # the full tensor journey + loss
python -m tests.test_wiring        # masks + one-batch overfit + decode
```

## 4a. The projector
Encoder gives 50 vectors/s at d_enc. The LLM wants ~a few tokens/s at d_llm.
The projector fixes **both the dimension and the rate**. Rate is the important one:
attention is O(T^2) and the KV cache is O(T), so 30 s of audio at 50 fps = 1500 tokens would
swamp a text context.

| design | output length | notes |
|---|---|---|
| `stack_mlp` | ceil(T/k) | concat k neighbouring frames, 2-layer MLP. SLAM-ASR / Qwen2-Audio style. Cheap, monotonic, strong ASR baseline. **default** |
| `conv` | ceil(T/k) | strided Conv1d, smoother than hard stacking |
| `qformer` | N (fixed!) | N learned queries cross-attend to the audio. SALMONN / BLIP-2 style. Constant cost, but a fixed-capacity bottleneck: bad for long audio, fine for audio *understanding* tasks |

Measured on 5 s of audio (250 encoder frames): stack_mlp 2.44M params, conv 1.88M,
qformer(64) 10.90M.

**The k ablation is your best interview story.** k=1 (50 tok/s) gives the best WER and the
worst cost; k=8 (6 tok/s) is cheap and starts dropping short words. Run it and quote numbers.

## 4b. Sequence assembly - where multimodal training actually goes wrong
```
[ pre_ids ][ AUDIO TOKENS ][ post_ids ][ answer_ids ]   <- inputs_embeds
[  -100   ][     -100     ][   -100   ][ answer_ids ]   <- labels
[    1    ][       1      ][     1    ][      1     ][ 0 0 0 ]  <- attention_mask + padding
```
Real numbers from `python -m src.modeling` on a 4.32 s clip:
`mel (2,80,432)` -> `encoder (2,216,384)` @50/s -> `projector (2,54,576)` @13/s ->
`inputs_embeds (2,105,576)`, of which only **11 positions carry loss**.

Details that matter:
- **Right padding for training** (positions stay aligned, pads are masked),
  **left padding for generation** (every prompt must end in the same column so `generate`
  appends at the true end).
- Audio tokens have **no token ids**. They never touch the embedding matrix or the softmax.
- Prompt *before* audio vs *after* audio is a real ablation: instruction-after-audio usually
  wins for ASR (the model has heard everything before being told what to do), instruction-
  first wins when you want one encoder pass reused across tasks with KV caching.

## The one-batch overfit test (do this before any real training run)
`tests/test_wiring.py` trains **only the 2.4M-param projector** on a single batch:
```
step 0 loss 6.4635 -> step 59 loss 0.0002
reference: just thrust and parry and victory to the stronger
decoded  : just thrust and parry and victory to the stronger
```
A frozen 8M encoder and a frozen 135M LLM, connected by 2.4M new parameters, reproduce the
transcript exactly. If this test cannot reach ~0, your splice, mask or label offset is wrong
and no amount of data will save you. It catches off-by-one label shifts, unmasked padding and
reversed left/right padding - which is 90 % of multimodal wiring bugs.

## Interview Q&A
**Q: Why freeze both towers first?** The projector starts random, so its outputs are noise in
the LLM's embedding space. Backpropagating that noise into a pretrained LLM damages it before
anything useful is learned. Stage 1 = alignment only (1.68 % of parameters trainable here).

**Q: How do the audio embeddings end up "in" the LLM's space?** Nothing constrains them to
look like real token embeddings - they only have to be *readable* by the frozen transformer.
In practice they land off the token-embedding manifold; adding a LayerNorm at the projector
output (we do) keeps their scale comparable, which matters a lot for training stability.

**Q: Continuous features vs discrete audio tokens?** The alternative is to quantise audio into
discrete units (EnCodec/RVQ, SpeechTokenizer, HuBERT k-means) and extend the LLM vocabulary.
Discrete = one uniform token stream, lets you also *generate* speech, but loses information at
quantisation and needs vocabulary surgery + retraining. Continuous = lossless and cheap to
attach, but text-output only. That fork is the central design decision in the field.

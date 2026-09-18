# Stage 2 - The audio encoder

Run: `python -m src.encoder --seconds 5`

We keep the **encoder half of Whisper** and delete its decoder. The LLM is the decoder now.

```
mel (B,80,T)
  conv1  k=3 s=1  + GELU      local smoothing, 80 -> d_model channels
  conv2  k=3 s=2  + GELU      time downsample x2   (100 fps -> 50 fps)
  + sinusoidal position embeddings (fixed table, 1500 rows = 30 s)
  N x TransformerEncoderLayer  BIDIRECTIONAL self-attention (no causal mask)
  final LayerNorm
-> (B, T/2, d_model)
```
whisper-tiny: d=384, 4 layers, 6 heads, 8.2M params. base/small/medium/large-v3:
512/768/1024/1280 dims, 6/12/24/32 layers.

## Two things we did here on purpose
1. **Variable-length forward.** HF's `WhisperEncoder.forward` hard-errors unless you give it
   exactly 3000 frames. We reimplement the forward and *slice* the position table
   (`embed_positions.weight[:T]`). Verified bit-identical to HF at 3000 frames
   (`max|diff| = 0.00e+00`). This is legitimate because the table is absolute and learned
   per index - slicing keeps every index in its trained position. (You could NOT do this with
   a table that was only ever seen fully populated, and you must not interpolate it.)
2. **No attention mask.** The Whisper encoder has none, so padded frames are attended to as
   if they were audio. Stock Whisper has the same property (it pads to 30 s with silence).
   We handle it downstream: `mel_lengths` -> how many audio tokens are real -> the LLM
   attention mask covers only those. The leakage into the padded region is harmless because
   we never read those positions.

## Interview Q&A
**Q: Why a supervised ASR encoder rather than wav2vec2/HuBERT/WavLM?**
SSL encoders are trained on masked-prediction of *acoustic* units; their features carry a lot
of speaker/channel detail and little semantics or language identity. A Whisper encoder was
trained through a text objective on 680k h, so its top layer is already "text-ish" - which is
exactly what an LLM wants to read. Empirically Whisper-encoder + LLM beats SSL + LLM on ASR
with a frozen front-end. SSL encoders win when you have lots of unlabelled target-domain audio.

**Q: Why not CTC?** CTC assumes conditional independence between output frames and a
monotonic alignment. That is fine for ASR, useless for QA/translation, and it cannot emit
free-form text. But CTC is still used *inside* speech-LLMs as a compressor (blank frames get
dropped before the LLM) - "CTC compression" is a real adapter design.

**Q: Which encoder layer should feed the LLM?** Not always the last. Final layers are
specialised toward the original decoder's output distribution; mid-layers can generalise
better for non-ASR tasks. Concatenating/weighting several layers (a learned weighted sum,
like SUPERB) is a cheap ablation worth running.

**Q: Do you unfreeze the encoder?** Only late, with a small LR (or LoRA). Unfreezing early
destroys the representation before the projector has learned to read it.

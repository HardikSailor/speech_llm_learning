# Stage 1 - Waveform to log-mel

Run: `python -m src.features` (proves our 15-line torch version == HF's extractor, exactly)

## The pipeline
1. **Frame**: 25 ms window (400 samples), 10 ms hop (160 samples), Hann window.
   Windows overlap 60 %, so a frame boundary never falls inside a phoneme unobserved.
   10 ms is roughly the shortest phonetic event worth resolving (a stop burst).
2. **STFT** -> 201 linear frequency bins, take `|.|^2` (power). Phase is discarded -
   the ear is largely phase-deaf and discarding it makes the representation shift-robust.
3. **Mel filterbank**: 201 linear bins -> 80 triangular mel bins. Mel spacing is ~linear
   below 1 kHz and logarithmic above, matching human pitch perception; it also concentrates
   resolution where formants live.
4. **log10** + dynamic-range floor (80 dB below peak) + Whisper's `(x+4)/4` normalisation.
   Log because loudness perception is logarithmic and because it turns multiplicative
   channel/gain effects into additive offsets that a network can subtract away.

## Whisper's fixed 30 s window
Whisper always pads/trims to 30 s -> 3000 frames -> 1500 encoder vectors, because its
positional embedding table has exactly 1500 rows. Consequence: transcribing 2 s of audio
costs the same as 30 s. In `collate.py` we instead pad to the batch maximum and slice the
position table (see Stage 2) - same math, far less wasted compute on CPU.

## Interview Q&A
**Q: Why not feed raw waveform?** You can - wav2vec2/HuBERT use a strided CNN front-end
(7 conv layers, 20 ms stride -> also 50 fps). Mel is a strong hand-designed prior that
saves parameters and data; raw-waveform front-ends need more of both but can learn features
mel throws away.

**Q: 80 vs 128 mel bins?** large-v3 moved to 128 for finer spectral detail with a bigger
model. Front-end and encoder must agree - a checkpoint mismatch here fails silently-ish.

**Q: What does one encoder token "see"?** 20 ms of stride, but the receptive field is the
whole clip after the first self-attention layer (the encoder is bidirectional).

**Q: MFCC vs log-mel?** MFCC = DCT of log-mel, decorrelates for GMM-era diagonal covariances.
Neural nets do not need decorrelation and the DCT throws away information, so log-mel won.

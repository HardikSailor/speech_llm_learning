# Stage 0 - Data

Run: `python -m src.data --root $SPEECH_LLM_DATA --n 3`

## What a speech dataset is
`(waveform, sample_rate, reference_text)`. Everything else (speaker, chapter, book) is metadata.

- **16 kHz mono** is the universal ASR sample rate. Nyquist says it captures up to 8 kHz;
  speech intelligibility lives below ~4 kHz, so 16 kHz is plenty. Telephony is 8 kHz,
  music/TTS is 22.05/24/44.1 kHz. Mismatched sample rate is the #1 silent ASR bug.
- float32 in [-1, 1]. 1 second = 16 000 numbers.
- LibriSpeech text is **uppercase, no punctuation**. That matters: a model trained on it will
  never emit punctuation, and WER against punctuated references will look terrible unless you
  normalise. We lowercase in `DataConfig`.

## Rates worth memorising
| thing | rate |
|---|---|
| samples | 16 000 / s |
| mel frames | 100 / s |
| Whisper encoder outputs | 50 / s |
| after stack-4 projector | 12.5 / s |
| English text tokens | ~3-4 / s |

The gap between 50 and 3-4 is why a projector exists.

## Interview Q&A
**Q: How much data do real systems use?** Whisper: 680k h weakly supervised web audio
(v3: 5M h incl. pseudo-labels). LibriSpeech: 960 h. Our toy run: minutes. The plan is to
understand the machinery, not to beat a benchmark.

**Q: Why is weak supervision enough?** Scale beats label quality past a point; noisy
(audio, transcript) pairs from the web cover far more acoustic/domain variety than clean
read speech, which is what makes Whisper robust zero-shot.

**Q: What preprocessing do you skip and why?** No VAD, no speed perturbation, no SpecAugment
here - they are regularisers that matter when you train the encoder. We freeze it.

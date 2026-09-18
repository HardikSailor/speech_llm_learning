# Speech-LLM, built step by step

Speech in, text out. A frozen Whisper encoder and a frozen small LLM, joined by a trainable
projector. Each stage is a runnable module that prints what it is doing.

See **PLAN.md** for the full 12-stage curriculum and **notes/** for the explanation +
interview Q&A of each stage.

## Setup
```bash
source scripts/env.sh     # sets $PY, $HF_HOME, $SPEECH_LLM_DATA, thread count
```
Env: `/scratch/users/astar/ares/sailorhb/envs/speechllm` (torch 2.14+cpu, transformers 4.57.1,
peft 0.21, trl 1.13, jiwer). Do **not** use the `py2hf` conda env - its torchvision is built
against a different torch and `from transformers import WhisperModel` fails outright.

Data: `/data/projects/13003558/speech_data/LibriSpeech`
(`dev-clean` 2703 utts, `test-clean` 2620, `train-clean-100` 28539).

## Stages 0-4 (done)
```bash
$PY -m src.data     --root $SPEECH_LLM_DATA   # 0: waveform + text
$PY -m src.features                           # 1: log-mel, ours vs HF (exact match)
$PY -m src.encoder  --seconds 5               # 2: Whisper encoder, variable length
$PY -m src.llm                                # 3: inputs_embeds, label shifting, chat template
$PY -m src.projector                          # 4a: three adapter designs
$PY -m src.modeling --root $SPEECH_LLM_DATA   # 4b: the whole tensor journey
$PY -m tests.test_wiring --root $SPEECH_LLM_DATA   # masks + one-batch overfit
```

## Stage 5 (done): train the projector
```bash
./scripts/train_stage5.sh          # MINUTES=30 EPOCHS=6 by default
```
Encoder and LLM frozen; only the 2.4M-param projector learns. Length bucketing cuts padding
waste from 36 % to 5 %. Logs loss / lr / grad-norm / throughput-vs-realtime, evaluates
teacher-forced loss and free-running WER, and checkpoints **only the projector** (9.7 MB).

## Layout
```
src/config.py      all knobs (encoder, llm, projector, freezing)
src/data.py        LibriSpeech-style loader (+ HF fallback)
src/features.py    log-mel, hand-written and HF, proven identical
src/encoder.py     Whisper encoder with a variable-length forward
src/llm.py         LLM loading + prompt/segment building
src/projector.py   stack_mlp | conv | qformer
src/collate.py     batching, padding, mel lengths
src/sampler.py     length-bucketing batch sampler
src/modeling.py    SpeechLLM: splice, mask, loss, generate
src/train.py       stage-5 training loop, no Trainer and no Accelerate
tests/             wiring + one-batch overfit
notes/             one markdown per stage: what, why, interview Q&A
```

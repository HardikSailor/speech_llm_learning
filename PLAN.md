# Speech-LLM from scratch — learning + interview plan

Goal: build, train and post-train a **speech-in / text-out LLM** (ASR first, then speech-QA),
using PyTorch + HF Transformers + Accelerate + PEFT + TRL, where every stage is small enough
to run, inspect and explain in an interview.

## The architecture we are building (fixed target)

```
waveform 16 kHz
   │  feature extractor (log-mel, 80/128 bins, 10 ms hop)
   ▼
[ Audio encoder ]   Whisper-base encoder (frozen at first)   -> T_a x 512, 50 fps
   │
   ▼
[ Projector / adapter ]  stack-k + Linear/MLP (or Conv1d, or Q-Former)  -> T_a/k x d_llm
   │                     this is the ONLY thing trained in stage 1
   ▼
[ LLM ]  Qwen2.5-0.5B-Instruct, inputs_embeds = concat(prompt_emb, audio_emb, instr_emb)
   │                     labels masked to -100 except on the answer
   ▼
text: transcript / answer
```

Everything else in the plan is a variation on this: change the adapter, unfreeze with LoRA,
change the prompt, change the loss (CE -> DPO -> GRPO).

---

## Stage 0 — Environment, data, repo skeleton
- Conda env: torch, transformers, accelerate, peft, trl, datasets, soundfile, librosa, jiwer, tensorboard.
- Data: LibriSpeech `dev-clean` / `test-clean` subset (~1–5 h) for ASR. Cached locally (cluster may be offline).
- Repo layout:
  ```
  src/       data.py  modeling.py  collate.py  train_*.py  evaluate.py
  scripts/   run_*.sh (single GPU / multi-GPU / job submission)
  notes/     one .md per stage: what happens, why, interview Q&A
  configs/   yaml for each training stage
  ```
- **Deliverable:** `python -m src.data --peek` prints one sample: waveform shape, sr, duration, text.

## Stage 1 — Audio front-end (you must be able to draw this)
- waveform -> log-mel: window 25 ms, hop 10 ms, 80 mels, log, normalization.
- Whisper's quirk: pad/trim to 30 s -> always 3000 frames -> encoder conv stride 2 -> 1500 vectors = 50 fps.
- **Interview:** why mel and not raw? why 10 ms? what does a frame "see"? what changes for wav2vec2/HuBERT (raw-waveform CNN, 20 ms = 50 fps too)?
- **Deliverable:** script printing shapes end-to-end + a saved mel picture.

## Stage 2 — Audio encoder deep-dive
- Load Whisper encoder alone; print module tree, param count, output shape.
- What each part does: 2 conv layers (downsample + local context), sinusoidal pos-emb, N transformer blocks (bidirectional self-attention, no causal mask).
- Encoder-only vs encoder-decoder vs CTC vs SSL (wav2vec2/HuBERT/WavLM) — when to pick which as the speech-LLM front-end.
- **Interview:** why is a *supervised ASR* encoder usually a better LLM front-end than a raw SSL encoder? (semantic vs acoustic features, robustness, language ID).

## Stage 3 — LLM side deep-dive
- Tokenizer, chat template, special tokens; `embed_tokens` lookup; causal mask; KV cache; logits -> loss.
- Key trick: `model(inputs_embeds=...)` instead of `input_ids` — this is what makes multimodality possible at all.
- Teacher forcing, label shifting (HF shifts internally), `-100` masking, why we do NOT train on the prompt.
- **Deliverable:** feed hand-made embeddings, generate text; prove `inputs_embeds` path == `input_ids` path.

## Stage 4 — The bridge (most important stage)
- Projector options, implement at least two:
  1. **stack-k + MLP** (k=4/5 → 12.5/10 audio tokens per second) — the Qwen2-Audio / SLAM-ASR style baseline.
  2. Conv1d stride-k, and optionally a small **Q-Former** (fixed 64 query vectors, cross-attention) — SALMONN style.
- Sequence assembly + label masking, done explicitly and printed as a table:
  `<|im_start|>user\n <AUDIO tokens> Transcribe the audio.<|im_end|>\n<|im_start|>assistant\n TEXT <|im_end|>`
- Attention mask & left/right padding pitfalls, position ids.
- **Interview:** why downsample audio tokens? cost is quadratic in seq len; 30 s of audio = 1500 tokens = as long as the whole text context.
- **Deliverable:** `modeling.py` with `SpeechLLM(nn.Module)` returning loss; a unit test that overfits **one batch** to ~0 loss.

## Stage 5 — Training loop v1, raw PyTorch (no Trainer)
- Dataset/collator: dynamic padding of audio + labels, bucketing by length, `attention_mask`.
- Loop by hand: forward, CE loss, backward, clip, AdamW, warmup+cosine schedule, grad accumulation, bf16 autocast.
- Freeze policy: encoder ❄, LLM ❄, projector 🔥 (~5–15 M params). Print trainable %.
- Logging: loss, lr, grad-norm, tokens/s, sample decode every N steps.
- **Deliverable:** train on 1 h of LibriSpeech, loss curve goes down, sample decodes look like English.

## Stage 6 — Same loop with Accelerate (and the distributed story)
- `accelerator.prepare`, `accumulate`, `backward`, `gather_for_metrics`, checkpoint save/load.
- Then the concepts: DDP (grad all-reduce) vs FSDP / DeepSpeed ZeRO-1/2/3 (shard optimizer/grad/params),
  gradient checkpointing, mixed precision bf16 vs fp16+GradScaler, flash-attention-2, seq packing.
- Memory math: params + grads + Adam states = 16 bytes/param in fp32-ish terms; why LoRA kills most of it.
- **Deliverable:** same training runs on 1 and 2 GPUs with identical loss curve; a note on what changed.

## Stage 7 — Inference + evaluation
- Generation with `inputs_embeds`: greedy vs beam vs sampling, `max_new_tokens`, repetition/hallucination on silence.
- WER with `jiwer`; text normalization (Whisper normalizer); why WER without normalization lies.
- **Deliverable:** `evaluate.py` → WER on test-clean subset; error analysis table (ins/del/sub).

## Stage 8 — Stage-2 training: LoRA on the LLM
- PEFT LoRA: rank, alpha, dropout, target modules (q,k,v,o + MLP), how ΔW = BA works and why it merges for free.
- Two-stage recipe: (1) projector only → (2) projector + LoRA, lower LR. Why not everything at once.
- Catastrophic forgetting: check the LLM can still do plain text tasks after tuning; mitigation (LoRA, replay data, low LR).
- **Deliverable:** WER improves vs stage-5; a forgetting check.

## Stage 9 — Multi-task: ASR + speech QA (+ optional AST)
- Instruction formatting; one prompt per task; task mixing ratios; why a single model beats task-specific heads.
- Build a tiny speech-QA set (e.g. questions generated over LibriSpeech transcripts, or Spoken-SQuAD subset).
- Evaluate per task; watch for task interference and "the model always transcribes" failure mode.
- **Deliverable:** one checkpoint answering both "Transcribe this" and "What did the speaker say about X?".

## Stage 10 — Post-training / alignment (the interview-heavy part)
Each sub-step: 10-line from-scratch loss **and** the TRL version, on our ASR model.
1. **SFT recap** — what it optimizes, its limits (only positive examples, exposure bias).
2. **Reward models** — Bradley–Terry pairwise loss, why RLHF needed one.
3. **DPO** — from RLHF objective to the closed form; implicit reward `β log πθ/πref`; the loss;
   role of β and of the frozen reference; length bias.
   *Our data:* sample n-best hypotheses from our model, score with WER → chosen = best, rejected = worst. Free preference data, no humans.
4. **Variants in one line each:** IPO (fixes DPO overfit), KTO (unpaired), ORPO (no ref model), SimPO (length-normalized, ref-free), CPO.
5. **PPO** — policy gradient, advantage, ratio clipping, KL-to-ref penalty, value head; why it's painful.
6. **GRPO** — sample G completions per prompt, advantage = (r − mean)/std within group, no critic, KL term;
   **reward = −WER is a verifiable reward**, so ASR is an ideal RLVR playground. Implement it.
7. Reward hacking, KL collapse, entropy collapse, and how you'd detect them.
- **Deliverable:** DPO run and GRPO run that each measurably cut WER over the SFT checkpoint, plus a notes page you could whiteboard from.

## Stage 11 — Systems & scaling topics (asked in every interview)
- Throughput math, KV-cache size, quantization (int8/int4, QLoRA), vLLM/serving, streaming ASR with chunked attention,
- Data scaling: how the big systems get 100k+ h; pseudo-labeling; dedup; curriculum.

## Stage 12 — Landscape + drill
- Read/compare and be able to contrast: Whisper, SLAM-ASR, SALMONN, Qwen2-Audio / Qwen3-Omni, AudioPaLM, SpeechGPT, SeamlessM4T, Moshi (full-duplex), speech *output* via discrete tokens.
- Continuous features vs discrete audio tokens (RVQ/EnCodec/SpeechTokenizer) — the central design fork.
- Final artifact: `notes/interview_qa.md` — ~60 Q&A, and 3–4 ablations you actually ran (your best interview material).

---

## Suggested additions beyond your original ask
- **Ablations you run yourself** (downsample factor 1/2/5/10; prompt before vs after audio; frozen vs LoRA encoder;
  MLP vs Q-Former). "I ran it and here's the curve" is worth more than any recited answer.
- **A one-batch overfit test** as a permanent smoke test — catches 90 % of multimodal wiring bugs (label offsets, mask errors).
- **Shape-printing mode** on every module, so you can narrate the tensor journey from memory.

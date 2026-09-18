"""Stage 4c - the collator: a list of samples -> one padded batch.

Decisions made here, all of them interview-relevant:
  * We pad the batch to its own longest clip (rounded to a multiple of 320 samples so the
    frame count stays even for the stride-2 conv), instead of Whisper's fixed 30 s.
    30 s padding is simpler and is what stock Whisper does, but on a 4 s batch it makes you
    pay 7x the compute for silence.
  * We carry per-sample `mel_lengths` so the model knows how many audio tokens are real.
  * Text targets are kept as per-sample `Segments`; padding of the text side happens inside
    the model, after the audio embeddings are spliced in.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .config import DEFAULT_INSTRUCTION, ModelConfig
from .features import get_feature_extractor
from .llm import PromptBuilder

HOP = 160          # samples per mel frame at 16 kHz


class SpeechCollator:
    def __init__(self, tokenizer, cfg: ModelConfig, instruction: str = DEFAULT_INSTRUCTION,
                 pad_to_30s: bool = False):
        self.fe = get_feature_extractor(cfg.encoder_name)
        self.prompt = PromptBuilder(tokenizer)
        self.cfg = cfg
        self.instruction = instruction
        self.pad_to_30s = pad_to_30s

    def __call__(self, samples: List[dict]) -> dict:
        wavs = [s["audio"] for s in samples]
        cap = int(self.cfg.max_audio_seconds * self.fe.sampling_rate)
        wavs = [w[:cap] for w in wavs]

        if self.pad_to_30s:
            feats = self.fe(wavs, sampling_rate=self.fe.sampling_rate, return_tensors="pt")
        else:
            n = max(len(w) for w in wavs)
            n = int(np.ceil(n / (2 * HOP)) * (2 * HOP))        # even number of mel frames
            feats = self.fe(wavs, sampling_rate=self.fe.sampling_rate, return_tensors="pt",
                            padding="max_length", max_length=n, truncation=True)
        mel = feats.input_features                              # (B, n_mels, T_mel)
        mel_lengths = torch.tensor(
            [min(int(np.ceil(len(w) / HOP)), mel.shape[-1]) for w in wavs], dtype=torch.long)

        segments = [
            self.prompt.build(samples[i].get("instruction", self.instruction),
                              answer=samples[i]["text"])
            for i in range(len(samples))
        ]
        return {
            "mel": mel,
            "mel_lengths": mel_lengths,
            "segments": segments,
            "texts": [s["text"] for s in samples],
            "uids": [s["uid"] for s in samples],
        }

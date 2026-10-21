"""Stage 4b - the whole model.

    mel --[frozen Whisper encoder]--> (T, d_enc)
        --[trainable projector]-----> (T', d_llm)   "audio tokens"
    [ pre_ids | AUDIO TOKENS | post_ids | answer_ids ]  -> embeddings -> LLM -> CE loss
      -100      -100           -100       answer_ids       (labels)

Everything hard about multimodal training lives in `build_inputs`: getting the splice, the
attention mask and the label offsets right. If any of those is off by one you get a model
that trains to a plausible-looking loss and decodes garbage - which is why tests/ has a
one-batch overfit check.
"""
from __future__ import annotations

import argparse
from typing import List, Sequence

import torch
import torch.nn as nn

from .config import ModelConfig
from .encoder import WhisperAudioEncoder
from .llm import PromptBuilder, Segments, load_llm
from .projector import build_projector

IGNORE = -100


class SpeechLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        self.encoder = WhisperAudioEncoder(cfg.encoder_name, dtype=dtype)
        self.llm, self.tokenizer = load_llm(cfg.llm_name, dtype=dtype)
        self.prompt = PromptBuilder(self.tokenizer)
        self.projector = build_projector(
            cfg.projector, self.encoder.hidden_size, self.llm.config.hidden_size, cfg)
        self.set_trainable(freeze_encoder=cfg.freeze_encoder, freeze_llm=cfg.freeze_llm)

    # ------------------------------------------------------------------ freezing
    def set_trainable(self, freeze_encoder: bool = True, freeze_llm: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze_encoder
        for p in self.llm.parameters():
            p.requires_grad = not freeze_llm
        for p in self.projector.parameters():
            p.requires_grad = True

    def trainable_report(self) -> str:
        rows, total, train = [], 0, 0
        for name, mod in (("encoder", self.encoder), ("projector", self.projector),
                          ("llm", self.llm)):
            n = sum(p.numel() for p in mod.parameters())
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            total += n
            train += t
            rows.append(f"  {name:10s} {n/1e6:8.1f}M total  {t/1e6:8.1f}M trainable")
        rows.append(f"  {'TOTAL':10s} {total/1e6:8.1f}M total  {train/1e6:8.1f}M trainable "
                    f"({100*train/total:.2f} %)")
        return "\n".join(rows)

    # ------------------------------------------------------------------ audio path
    def encode_audio(self, mel: torch.Tensor, mel_lengths: torch.Tensor):
        """(B, n_mels, T_mel) -> (B, T', d_llm), lengths.

        When the encoder is frozen we run it under no_grad: no graph is kept for it, which
        saves the activation memory of every encoder layer. We CANNOT do that for the LLM
        even when it is frozen - gradients must flow *through* the LLM to reach the
        projector. Freezing saves weight-gradient computation, not the backward pass.
        """
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                hidden = self.encoder(mel)
        else:
            hidden = self.encoder(mel)
        enc_lengths = self.encoder.output_lengths(mel_lengths).clamp(max=hidden.shape[1])
        return self.projector(hidden, enc_lengths)

    # ------------------------------------------------------------------ the splice
    def build_inputs(self, audio_embeds, audio_lengths, segments: Sequence[Segments],
                     for_generation: bool = False):
        """Concatenate text and audio embeddings per sample, then pad the batch.

        Training pads on the RIGHT (padding is masked out, position ids stay aligned).
        Generation pads on the LEFT so that every sequence ends at the same column and
        `generate` appends to the true end of each prompt.
        """
        embed = self.llm.get_input_embeddings()
        device = audio_embeds.device
        seqs, labs = [], []
        for i, seg in enumerate(segments):
            pre = embed(torch.tensor(seg.pre_ids, device=device))
            post = embed(torch.tensor(seg.post_ids, device=device))
            aud = audio_embeds[i, : audio_lengths[i]]
            parts = [pre, aud, post]
            label = [torch.full((len(seg.pre_ids),), IGNORE, device=device, dtype=torch.long),
                     torch.full((aud.shape[0],), IGNORE, device=device, dtype=torch.long),
                     torch.full((len(seg.post_ids),), IGNORE, device=device, dtype=torch.long)]
            if not for_generation and seg.ans_ids:
                ans_ids = torch.tensor(seg.ans_ids, device=device)
                parts.append(embed(ans_ids))
                label.append(ans_ids)            # <- the only place loss is computed
            seqs.append(torch.cat(parts, dim=0))
            labs.append(torch.cat(label, dim=0))

        b, maxlen, d = len(seqs), max(s.shape[0] for s in seqs), seqs[0].shape[1]
        inputs = torch.zeros(b, maxlen, d, device=device, dtype=seqs[0].dtype)
        attn = torch.zeros(b, maxlen, device=device, dtype=torch.long)
        labels = torch.full((b, maxlen), IGNORE, device=device, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            sl = slice(maxlen - n, maxlen) if for_generation else slice(0, n)
            inputs[i, sl], attn[i, sl], labels[i, sl] = s, 1, l
        return inputs, attn, labels

    # ------------------------------------------------------------------ forward
    def forward(self, mel, mel_lengths, segments):
        audio_embeds, audio_lengths = self.encode_audio(mel, mel_lengths)
        inputs, attn, labels = self.build_inputs(audio_embeds, audio_lengths, segments)
        out = self.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, mel, mel_lengths, segments, max_new_tokens: int = 128, **kw):
        audio_embeds, audio_lengths = self.encode_audio(mel, mel_lengths)
        inputs, attn, _ = self.build_inputs(audio_embeds, audio_lengths, segments,
                                            for_generation=True)
        ids = self.llm.generate(inputs_embeds=inputs, attention_mask=attn,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=self.tokenizer.pad_token_id, **kw)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="print the tensor journey end to end")
    ap.add_argument("--root", default="")
    ap.add_argument("--projector", default="stack_mlp")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    from .collate import SpeechCollator
    from .config import DataConfig
    from .data import SpeechTextDataset

    cfg = ModelConfig(projector=args.projector)
    model = SpeechLLM(cfg).eval()
    ds = SpeechTextDataset(DataConfig(root=args.root))
    coll = SpeechCollator(model.tokenizer, cfg)
    batch = coll([ds[i] for i in range(min(args.batch, len(ds)))])

    print("trainable parameters")
    print(model.trainable_report())
    print()
    mel, mlen, segs = batch["mel"], batch["mel_lengths"], batch["segments"]
    with torch.no_grad():
        hidden = model.encoder(mel)
        enc_len = model.encoder.output_lengths(mlen)
        aud, alen = model.projector(hidden, enc_len)
        inputs, attn, labels = model.build_inputs(aud, alen, segs)
        loss, logits = model(mel, mlen, segs)

    secs = mlen.float() / 100
    print("tensor journey (batch of %d)" % mel.shape[0])
    print(f"  waveform          {[round(s, 2) for s in secs.tolist()]} s")
    print(f"  mel               {tuple(mel.shape)}      lengths={mlen.tolist()}")
    print(f"  encoder out       {tuple(hidden.shape)}      lengths={enc_len.tolist()}"
          f"   ({(enc_len/secs).mean():.0f} tok/s)")
    print(f"  projector out     {tuple(aud.shape)}      lengths={alen.tolist()}"
          f"   ({(alen/secs).mean():.0f} tok/s)")
    print(f"  llm inputs_embeds {tuple(inputs.shape)}  attn sum={attn.sum(1).tolist()}")
    print(f"  labels != -100    {(labels != IGNORE).sum(1).tolist()} positions "
          f"(answer tokens only)")
    print(f"  logits            {tuple(logits.shape)}")
    print(f"  loss              {loss.item():.4f}   (ln(vocab)={torch.log(torch.tensor(float(model.llm.config.vocab_size))):.2f} = random)")
    print()
    print("sequence layout of sample 0:")
    s = segs[0]
    print(f"  [{len(s.pre_ids):>3} text] {model.tokenizer.decode(s.pre_ids)!r}")
    print(f"  [{alen[0].item():>3} AUDIO] <- projected speech, no token ids exist for these")
    print(f"  [{len(s.post_ids):>3} text] {model.tokenizer.decode(s.post_ids)!r}")
    print(f"  [{len(s.ans_ids):>3} ans ] {model.tokenizer.decode(s.ans_ids)!r}   <- loss here only")


if __name__ == "__main__":
    main()

"""Stage 3 - the text side.

Three facts this file demonstrates, all of which you should be able to state cold:

1. `model(input_ids=...)` is just `embed_tokens[input_ids]` followed by
   `model(inputs_embeds=...)`. THAT is the hook multimodality hangs on: anything we can
   turn into a (T, d_model) tensor can be spliced into the token stream.
2. When you pass `labels`, HF shifts them internally: the loss at position i predicts
   token i+1. So `labels` is just a copy of the inputs with -100 where you do not want loss.
3. The chat template is plain text with special tokens. We split it around a sentinel and
   drop the audio embeddings into the hole.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AUDIO_SENTINEL = "<|AUDIO|>"


@dataclass
class Segments:
    """One training example, already split into the four pieces of the sequence."""
    pre_ids: List[int]     # chat prefix, up to where the audio goes
    post_ids: List[int]    # instruction + assistant header
    ans_ids: List[int]     # target text + EOS (empty at inference time)

    def describe(self, tok) -> str:
        return (f"pre({len(self.pre_ids)}) {tok.decode(self.pre_ids)!r}\n"
                f"post({len(self.post_ids)}) {tok.decode(self.post_ids)!r}\n"
                f"ans({len(self.ans_ids)}) {tok.decode(self.ans_ids)!r}")


class PromptBuilder:
    """Turns (instruction, answer) into ids around an audio hole, using the model's own
    chat template so the special tokens are exactly what the LLM was trained on."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def build(self, instruction: str, answer: str | None = None,
              system: str | None = None) -> Segments:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": f"{AUDIO_SENTINEL}\n{instruction}"})
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if AUDIO_SENTINEL not in prompt:
            raise RuntimeError("chat template swallowed the sentinel")
        pre_text, post_text = prompt.split(AUDIO_SENTINEL)

        enc = lambda s: self.tok(s, add_special_tokens=False).input_ids   # noqa: E731
        ans_ids: List[int] = []
        if answer is not None:
            ans_ids = enc(answer) + [self.tok.eos_token_id]
        return Segments(pre_ids=enc(pre_text), post_ids=enc(post_text), ans_ids=ans_ids)


def load_llm(name: str, dtype: torch.dtype = torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    from .encoder import _dtype_kw
    model = AutoModelForCausalLM.from_pretrained(name, **_dtype_kw(dtype))
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = ap.parse_args()

    model, tok = load_llm(args.llm)
    model.eval()
    emb = model.get_input_embeddings()
    print(f"{args.llm}")
    print(f"  params        : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  d_model       : {model.config.hidden_size}   layers: {model.config.num_hidden_layers}")
    print(f"  vocab         : {model.config.vocab_size}  -> embedding matrix {tuple(emb.weight.shape)}")
    print(f"  eos/pad       : {tok.eos_token_id} / {tok.pad_token_id}")
    print()

    # ---- 1. input_ids path == inputs_embeds path ----------------------------
    ids = tok("The quick brown fox", return_tensors="pt").input_ids
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(inputs_embeds=emb(ids)).logits
    print(f"logits from input_ids   {tuple(a.shape)}")
    print(f"logits from inputs_embeds {tuple(b.shape)}  max|diff|={(a-b).abs().max():.2e}  <- must be ~0")
    print()

    # ---- 2. labels are shifted for you --------------------------------------
    labels = ids.clone()
    labels[:, :2] = -100                      # no loss on the first two tokens
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss
        logits = model(input_ids=ids).logits
        manual = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
            ignore_index=-100)
    print(f"HF loss {loss.item():.6f} vs hand-shifted CE {manual.item():.6f}  <- must match")
    print()

    # ---- 3. the prompt layout ------------------------------------------------
    pb = PromptBuilder(tok)
    seg = pb.build("Transcribe the audio into text.", answer="hello world")
    print("prompt segments (audio embeddings get spliced between pre and post):")
    print(seg.describe(tok))


if __name__ == "__main__":
    main()

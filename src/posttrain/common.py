"""Stage 10 - shared pieces for every post-training method.

Post-training methods look very different on a whiteboard, but in code they all use the
same three building blocks, and they all live in this file:

1. A task with a *checkable* answer, so we can score completions without humans:
   "What is 37 + 48? Answer with just the number."
   reward 1.0 = exactly "85", 0.5 = right number but extra text ("37 + 48 = 85"),
   0.0 = wrong. The base model is ~60 % correct but almost never obeys the format, so
   every method has something visible to fix.

2. `sequence_logprobs`: log pi(completion | prompt), per token. SFT, DPO, PPO and GRPO all
   differ only in what they DO with this number.

3. `sample` + `evaluate`: generate completions and score them.

The LLM is the same SmolLM2-135M-Instruct the speech model uses, as a text-only model, so
every run fits on CPU in minutes.
"""
from __future__ import annotations

import copy
import random
import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..encoder import _dtype_kw

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ----------------------------------------------------------------- the task
def make_problems(n: int, seed: int, lo: int = 10, hi: int = 99) -> List[Tuple[str, int]]:
    """n addition questions as (question, answer). Different seeds = different splits."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        out.append((f"What is {a} + {b}? Answer with just the number.", a + b))
    return out


def reward_fn(text: str, answer: int) -> float:
    """A *verifiable* reward: a rule, not a learned model. This is what RLVR means."""
    text = text.strip()
    if text == str(answer):
        return 1.0                                   # correct AND follows the format
    nums = re.findall(r"-?\d+", text)
    if nums and int(nums[-1]) == answer:
        return 0.5                                   # correct, but chatty
    return 0.0


def make_preference_pairs(problems, seed: int = 0) -> List[Tuple[str, str, str]]:
    """(question, chosen, rejected) triples, written by a rule instead of a human labeler.

    The chosen answer is always the bare correct number. The rejected answer is either
    chatty-but-correct ("37 + 48 = 85") or bare-but-wrong ("86"), so the preferences teach
    both the format and the arithmetic.
    """
    rng = random.Random(seed)
    pairs = []
    for q, ans in problems:
        a, b = re.findall(r"\d+", q)[:2]
        if rng.random() < 0.5:
            rejected = f"{a} + {b} = {ans}"
        else:
            rejected = str(ans + rng.choice([-10, -2, -1, 1, 2, 10]))
        pairs.append((q, str(ans), rejected))
    return pairs


# ----------------------------------------------------------------- model + tokens
def load_model(name_or_path: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **_dtype_kw(torch.float32))
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def frozen_copy(model):
    """The reference model pi_ref: a frozen snapshot of the policy before training.
    DPO, PPO and GRPO all measure how far the policy has moved away from it."""
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def prompt_ids(tok, question: str) -> List[int]:
    """Chat-templated prompt ending with the assistant header, ready to be continued."""
    msgs = [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def completion_ids(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]


def build_batch(prompts: List[List[int]], completions: List[List[int]], pad_id: int):
    """Concatenate prompt + completion per row and right-pad.

    Returns input_ids (B,T), attention_mask (B,T) and completion_mask (B,T-1). The last one is
    already aligned with the *targets* (input_ids[:, 1:]): it is 1 exactly where a completion
    token is being predicted. That one shifted mask is what keeps the prompt out of the loss.
    """
    seqs = [p + c for p, c in zip(prompts, completions)]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), T, dtype=torch.long)
    comp = torch.zeros(len(seqs), T, dtype=torch.float)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        attn[i, :len(p) + len(c)] = 1
        comp[i, len(p):len(p) + len(c)] = 1
    return ids, attn, comp[:, 1:]


def sequence_logprobs(model, ids, attn, output_hidden_states: bool = False):
    """log pi(token_t | tokens_<t) for every target position: shape (B, T-1).

    Position t of the logits predicts token t+1, hence logits[:, :-1] vs ids[:, 1:].
    Sum it under the completion mask and you get log pi(completion | prompt).
    """
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=output_hidden_states)
    logps = torch.gather(F.log_softmax(out.logits[:, :-1].float(), dim=-1), 2,
                         ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    if output_hidden_states:
        return logps, out.hidden_states[-1][:, :-1]      # PPO's value head reads these
    return logps


# ----------------------------------------------------------------- generation + eval
@torch.no_grad()
def sample(model, tok, prompts: List[List[int]], n: int = 1, temperature: float = 1.0,
           max_new_tokens: int = 16) -> List[List[int]]:
    """Generate n completions per prompt (rows grouped by prompt: p0,p0,..,p1,p1,..).

    Generation pads on the LEFT so every row's next token lands at the same position; HF
    derives position ids from the attention mask, so the positions match those of the
    right-padded training batch. Each completion is cut after its first EOS.
    """
    model.eval()
    P = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), P), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros(len(prompts), P, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p)
        attn[i, P - len(p):] = 1
    greedy = temperature == 0
    out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                         do_sample=not greedy, temperature=None if greedy else temperature,
                         top_p=None, top_k=None, num_return_sequences=n,
                         pad_token_id=tok.pad_token_id)
    model.train()
    comps = []
    for row in out[:, P:].tolist():
        if tok.eos_token_id in row:
            row = row[:row.index(tok.eos_token_id) + 1]
        comps.append(row)
    return comps


def evaluate(model, tok, problems, show: int = 3) -> dict:
    """Greedy decode every problem and report mean reward, accuracy and format rate."""
    prompts = [prompt_ids(tok, q) for q, _ in problems]
    comps = []
    for i in range(0, len(prompts), 32):
        comps += sample(model, tok, prompts[i:i + 32], temperature=0)
    texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
    rewards = [reward_fn(t, a) for t, (_, a) in zip(texts, problems)]
    for t, (q, a) in list(zip(texts, problems))[:show]:
        print(f"    {q[:16]}... -> {t!r:28} (answer {a})")
    return {"reward": sum(rewards) / len(rewards),
            "accuracy": sum(r > 0 for r in rewards) / len(rewards),
            "format": sum(r == 1.0 for r in rewards) / len(rewards)}


def fmt(m: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in m.items())


EVAL_PROBLEMS = make_problems(64, seed=12345)       # held out: no training seed uses this

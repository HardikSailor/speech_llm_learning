# Stage 3 - The text side

Run: `python -m src.llm` (proves all three claims below numerically)

## 1. `input_ids` is sugar for `inputs_embeds`
`model(input_ids=x)` == `model(inputs_embeds=embed_tokens(x))`. Verified: `max|diff| = 0`.
**This is the entire trick of multimodality.** The LLM does not consume text; it consumes a
sequence of d_model vectors. Anything you can turn into d_model vectors can be spliced in -
speech, images, video, whatever. There is no new architecture to invent.

## 2. `labels` are shifted for you
HF computes `CE(logits[:, :-1], labels[:, 1:])`. So `labels` is a *copy of the inputs*, with
`-100` wherever you do not want a loss. Verified against a hand-written shifted CE.
Two standard bugs: (a) shifting yourself as well (off by one, model learns to copy),
(b) forgetting to mask the prompt (the model spends capacity learning to generate your own
instruction back).

## 3. The chat template is just text
```
<|im_start|>system\n ... <|im_end|>\n<|im_start|>user\n {AUDIO} \n{instruction}<|im_end|>\n<|im_start|>assistant\n
```
We render it with `apply_chat_template` around a sentinel, split on the sentinel, and drop the
audio embeddings into the hole. Using the model's own template matters: an instruct model's
behaviour is conditioned on those exact special tokens.

## Other LLM facts you will be asked
- **Teacher forcing**: at train time the model always sees the gold prefix, at inference its
  own output. The mismatch is *exposure bias* - one of the motivations for RL post-training.
- **KV cache**: at generation step t you only compute the new query; K/V of the prefix are
  cached. Cache size = `2 * layers * kv_heads * head_dim * seq_len * bytes`. Audio tokens sit
  in that cache too - another reason to downsample them.
- **Why a small instruct model here?** SmolLM2-135M-Instruct (d=576, 30 layers) trains on CPU.
  Swap `ModelConfig.llm_name` to `Qwen/Qwen2.5-0.5B-Instruct` or larger on GPU; nothing else
  in the repo changes.

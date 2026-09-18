"""Central config. Everything in the repo reads from here so the stages stay consistent."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- Stage 2: audio encoder -------------------------------------------------
    encoder_name: str = "openai/whisper-tiny"   # 4 layers, d=384, 50 frames/s out
    # --- Stage 3: the LLM -------------------------------------------------------
    llm_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"   # d=576, 30 layers
    # --- Stage 4: the bridge ----------------------------------------------------
    projector: str = "stack_mlp"     # stack_mlp | conv | qformer
    stack_k: int = 4                 # 50 fps / 4 = 12.5 audio tokens per second
    proj_hidden_mult: int = 2
    qformer_queries: int = 64
    qformer_layers: int = 2
    qformer_heads: int = 8
    # --- what is trainable ------------------------------------------------------
    freeze_encoder: bool = True
    freeze_llm: bool = True
    # --- audio budget -----------------------------------------------------------
    max_audio_seconds: float = 20.0  # pad/trim window; keeps CPU runs cheap
    dtype: str = "float32"


@dataclass
class DataConfig:
    root: str = ""                   # path to a LibriSpeech-style directory
    hf_fallback: str = "hf-internal-testing/librispeech_asr_dummy"
    sample_rate: int = 16000
    min_seconds: float = 1.0
    max_seconds: float = 20.0
    lowercase: bool = True


DEFAULT_INSTRUCTION = "Transcribe the audio into English text."

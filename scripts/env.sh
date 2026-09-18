# source scripts/env.sh
export SPEECHLLM_ENV=/scratch/users/astar/ares/sailorhb/envs/speechllm
export PY=$SPEECHLLM_ENV/bin/python
export HF_HOME=/scratch/users/astar/ares/sailorhb/.cache/huggingface
export LIBRISPEECH=/data/projects/13003558/speech_data/LibriSpeech
export SPEECH_LLM_DATA=$LIBRISPEECH/dev-clean
export OMP_NUM_THREADS=16          # CPU box has 96 cores; 16 is a sane per-run share
unset TRANSFORMERS_CACHE           # deprecated, and it shadows HF_HOME
export PYTHONUNBUFFERED=1        # otherwise piping training output to grep/tee buffers it

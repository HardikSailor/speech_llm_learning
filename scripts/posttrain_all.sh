#!/usr/bin/env bash
# Stage 10: every post-training method on the toy addition task, one after another.
# Sequential on purpose: the login node gives each user 4 CPUs / 16 GB in total, and two
# runs in parallel (policy + reference + Adam each) get OOM-killed without a traceback.
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG=runs/posttrain_logs
mkdir -p "$LOG"
run() { name=$1; shift; echo "=== $name"; $PY -m "$@" 2>&1 | grep -v -i warn | tee "$LOG/$name.log"; }

$PY -m tests.test_posttrain
run sft          src.posttrain.sft --steps 40 --save runs/posttrain_sft
run reward_model src.posttrain.reward_model
run dpo          src.posttrain.dpo
run dpo_ipo      src.posttrain.dpo --loss ipo
run sft_then_dpo src.posttrain.dpo --model runs/posttrain_sft
run dpo_plus_sft src.posttrain.dpo --sft-weight 1.0 --lr 1e-6
run ppo          src.posttrain.ppo
run grpo         src.posttrain.grpo
run sft_then_grpo src.posttrain.grpo --model runs/posttrain_sft
grep -H "before:\|after:\|accuracy after" "$LOG"/*.log

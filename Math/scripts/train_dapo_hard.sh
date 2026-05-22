#!/bin/bash
# Train Qwen2.5-Math-7B on the dapo_hard dataset, optionally with the
# overconfidence-penalty probe enabled.
#
# Usage:
#   bash scripts/train_dapo_hard.sh <overconf_coeff>
#
# Examples (the two experiments from the paper):
#   bash scripts/train_dapo_hard.sh 0.0
#   bash scripts/train_dapo_hard.sh 1.0
#
# Required env vars:
#   MODEL_PATH                — path to a Qwen2.5-Math-7B checkpoint
#   DATA_DIR                  — directory containing dapo_hard/{train,test}.parquet
#   WANDB_API_KEY (optional)  — pass through to wandb logger

set -euo pipefail

OVERCONF_COEFF=${1:-0.0}
shift || true

: "${MODEL_PATH:?Set MODEL_PATH to a Qwen2.5-Math-7B checkpoint path}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing dapo_hard/train.parquet}"

MODEL_NAME=${MODEL_NAME:-Qwen2.5-Math-7B}
PROJECT_NAME=${PROJECT_NAME:-Qwen2.5_Math_7B_PRM_dapo_hard}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_${MODEL_NAME}_oc${OVERCONF_COEFF}}

NUM_GPUS=${NUM_GPUS:-8}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${USER}}

TRAIN_DATA=${DATA_DIR}/train.parquet
VAL_DATA=${VAL_DATA:-${DATA_DIR}/test.parquet}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
# Some AMD-ROCm-aware launchers set this even on NVIDIA hosts; verl rejects
# both being set so we drop it before Ray spawns the workers.
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir="${RAY_TMPDIR}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size=64 \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=multi_thread \
  probe.enable=True \
  probe.num_truncations=5 \
  "probe.suffix='... Thus, the final answer is: \\boxed{'" \
  probe.num_splits=1 \
  probe.overconf_coeff=${OVERCONF_COEFF} \
  trainer.balance_batch=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=${NUM_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.test_freq=50 \
  trainer.total_training_steps=1099 \
  trainer.total_epochs=5 \
  trainer.default_local_dir="${EXPERIMENT_DIR}" \
  "$@"

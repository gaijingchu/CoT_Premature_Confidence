#!/bin/bash
# Train one Countdown run.
#
# Usage:
#   bash scripts/train_countdown.sh <dataset_dir> <overconf_coeff>
#
# Examples (the four experiments from the paper):
#   bash scripts/train_countdown.sh countdown-data/countdown-4-30-100 0.0
#   bash scripts/train_countdown.sh countdown-data/countdown-4-30-100 1.0
#   bash scripts/train_countdown.sh countdown-data/countdown-4-10-50  0.0
#   bash scripts/train_countdown.sh countdown-data/countdown-4-10-50  1.0
#
# Required env vars:
#   BASE_MODEL  — path to a Qwen2.5-3B (base) checkpoint
#   WANDB_API_KEY, WANDB_ENTITY (or set trainer.logger=['console'] below)

set -euo pipefail

DATA_DIR=${1:?"Usage: bash scripts/train_countdown.sh <dataset_dir> <overconf_coeff>"}
OVERCONF_COEFF=${2:-0.0}

: "${BASE_MODEL:?Set BASE_MODEL to a Qwen2.5-3B base checkpoint path}"

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"countdown_$(basename "$DATA_DIR")_oc_${OVERCONF_COEFF}"}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
# Some AMD-ROCm-aware launchers set this even on NVIDIA hosts; verl rejects
# both being set so we drop it before Ray spawns the workers.
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=10 \
    trainer.project_name=premature-confidence-countdown \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=402 \
    probe.enable=True \
    probe.num_truncations=5 \
    probe.mc_samples=10 \
    probe.mc_max_tokens=32 \
    probe.num_splits=1 \
    probe.overconf_coeff="$OVERCONF_COEFF"

#!/bin/bash
# Setup prm conda environment based on install_vllm_sglang_mcore.sh
set -e

export MAX_JOBS=32
VERL_DIR="/home/jgai/Probe-PRM"
LOG="$VERL_DIR/scripts/setup_prm_env.log"

echo "=== Creating conda env 'prm' with Python 3.10 ===" | tee -a $LOG

conda create -n prm python=3.10 -y 2>&1 | tee -a $LOG

echo "=== Installing aria2 ===" | tee -a $LOG
conda run -n prm conda install -c conda-forge -y aria2 2>&1 | tee -a $LOG

echo "=== 1. Installing vllm and pytorch ===" | tee -a $LOG
conda run -n prm pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata 2>&1 | tee -a $LOG

echo "=== 2. Installing basic packages ===" | tee -a $LOG
conda run -n prm pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff 2>&1 | tee -a $LOG

conda run -n prm pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" 2>&1 | tee -a $LOG

echo "=== 3. Installing FlashAttention and FlashInfer ===" | tee -a $LOG
cd /tmp

if [ ! -f flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl ]; then
    aria2c -x 16 https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>&1 | tee -a $LOG
fi
conda run -n prm pip install --no-cache-dir /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>&1 | tee -a $LOG

if [ ! -f flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl ]; then
    aria2c -x 16 https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl 2>&1 | tee -a $LOG
fi
conda run -n prm pip install --no-cache-dir /tmp/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl 2>&1 | tee -a $LOG

echo "=== 4. Installing opencv ===" | tee -a $LOG
conda run -n prm pip install opencv-python 2>&1 | tee -a $LOG
conda run -n prm pip install opencv-fixer 2>&1 | tee -a $LOG
conda run -n prm python -c "from opencv_fixer import AutoFix; AutoFix()" 2>&1 | tee -a $LOG

echo "=== 5. Installing peft and math_verify ===" | tee -a $LOG
conda run -n prm pip install peft==0.17.0 2>&1 | tee -a $LOG
conda run -n prm pip install math_verify 2>&1 | tee -a $LOG

echo "=== 6. Installing verl in editable mode ===" | tee -a $LOG
conda run -n prm pip install -v -e "$VERL_DIR" 2>&1 | tee -a $LOG

echo "=== 7. Pinning transformers version ===" | tee -a $LOG
conda run -n prm pip install transformers==4.56 2>&1 | tee -a $LOG

echo "=== Done! prm env ready ===" | tee -a $LOG

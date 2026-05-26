<div align="center">

# Understanding and Mitigating Premature Confidence for Better LLM Reasoning

</div>

Implementation repository for the paper "Understanding and Mitigating Premature Confidence for Better LLM Reasoning" ([arXiv](https://arxiv.org/abs/2605.24396)).

<div align="center">
<a href="https://arxiv.org/abs/2605.24396">
    <img src="https://img.shields.io/badge/Paper-%23FF2442?style=for-the-badge"></a>
<a href="https://github.com/gaijingchu/CoT_Premature_Confidence">
    <img src="https://img.shields.io/badge/Code-%2300B4D8?style=for-the-badge"></a>
<a href="https://huggingface.co/guanning03/CoT_Premature_Confidence">
    <img src="https://img.shields.io/badge/Weights-%236C5CE7?style=for-the-badge"></a>
</div>

## Citation

If you found our work helpful, please consider citing this paper:

```
@misc{gai2026understandingmitigatingprematureconfidence,
      title={Understanding and Mitigating Premature Confidence for Better LLM Reasoning}, 
      author={Jingchu Gai and Guanning Zeng and Christina Baek and Chen Wu and J. Zico Kolter and Andrej Risteski and Aditi Raghunathan},
      year={2026},
      eprint={2605.24396},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.24396}, 
}
```

## Countdown Reasoning

### Setup

```bash
cd Countdown
conda create -y -n tinyzero python=3.9
conda activate tinyzero
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
pip install 'tensordict<0.6' 'transformers<4.48' \
    accelerate codetiming datasets dill hydra-core numpy pandas pybind11 wandb
pip install -e . --no-deps --no-build-isolation
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

### Run

Prepare the two datasets:

```bash
python examples/data_preprocess/countdown.py \
    --hf_dataset gaijingchu/countdown-4-30-100 \
    --local_dir countdown-data/countdown-4-30-100
python examples/data_preprocess/countdown.py \
    --hf_dataset gaijingchu/countdown-4-10-50 \
    --local_dir countdown-data/countdown-4-10-50
```

Launch the experiments:

```bash
export BASE_MODEL=/path/to/Qwen2.5-3B
bash scripts/train_countdown.sh countdown-data/countdown-4-30-100 0.0
bash scripts/train_countdown.sh countdown-data/countdown-4-30-100 1.0
```

In countdown task, due to the format reward, we implemented progressive confidence shaping before advantage computation, in order to avoid complexity.

Our checkpoints are available at [here](https://huggingface.co/guanning/CoT_Premature_Confidence/tree/main/countdown).

## Math Reasoning

### Setup

```bash
cd Math
conda create -y -n verl python=3.12
conda activate verl
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0 ray "transformers>=4.50,<5" math-verify
pip install -e .
pip install flash-attn --no-build-isolation
```

### Run

Prepare the dataset:

```bash
python examples/data_preprocess/prepare_dapo_hard.py --output_dir data/dapo_hard
```

Launch the experiments:

```bash
export MODEL_PATH=/path/to/Qwen2.5-Math-7B
export DATA_DIR=data/dapo_hard
bash scripts/train_dapo_hard.sh 0.0
bash scripts/train_dapo_hard.sh 1.0
```

Our checkpoints are available at [here](https://huggingface.co/guanning/CoT_Premature_Confidence/tree/main/math).

## Acknowledgement

The code is built on top of [Verl](https://github.com/ultralytics/yolov), [TinyZero](https://github.com/open-mmlab/mmdetection), and [Stream-of-Search](https://github.com/kanishkg/stream-of-search). We thank all the authors for their great work.

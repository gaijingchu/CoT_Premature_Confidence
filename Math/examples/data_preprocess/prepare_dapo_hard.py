"""Materialise the ``guanning/dapo_hard`` HuggingFace dataset into
``train.parquet`` / ``test.parquet`` in the schema expected by the
``multi_thread`` reward manager (``data_source`` column drives routing).

Usage:
    python examples/data_preprocess/prepare_dapo_hard.py \
        --output_dir data/dapo_hard [--seed 42] [--test_size 500]
"""

import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_dataset

COT_SUFFIX = "\n Let's think step by step and output the final answer within \\boxed{}."
DATA_SOURCE = "dapo_hard"


def _augment_prompt(messages):
    messages = list(messages)
    last = dict(messages[-1])
    last["content"] = last["content"] + COT_SUFFIX
    messages[-1] = last
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="guanning/dapo_hard")
    parser.add_argument("--output_dir", default="data/dapo_hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=500)
    args = parser.parse_args()

    df = load_dataset(args.hf_dataset, split="train").to_pandas()
    print(f"Loaded {len(df)} rows from {args.hf_dataset}")

    df["prompt"] = df["prompt"].apply(_augment_prompt)

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(df))
    test_idx = indices[: args.test_size]
    train_idx = indices[args.test_size :]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    def _build(subset: pd.DataFrame, split: str) -> pd.DataFrame:
        n = len(subset)
        out = pd.DataFrame()
        out["data_source"] = [DATA_SOURCE] * n
        out["prompt"] = subset["prompt"].values
        out["ability"] = subset["ability"].values
        out["reward_model"] = subset["reward_model"].values
        out["extra_info"] = [{"index": i, "split": split} for i in range(n)]
        return out

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    _build(df_train, "train").to_parquet(train_path, index=False)
    _build(df_test, "test").to_parquet(test_path, index=False)

    print(f"Saved {len(df_train)} train rows -> {train_path}")
    print(f"Saved {len(df_test)} test  rows -> {test_path}")


if __name__ == "__main__":
    main()

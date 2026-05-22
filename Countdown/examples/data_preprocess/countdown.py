"""Materialise the released Countdown HF dataset into train/test parquet files.

The released datasets (``gaijingchu/countdown-4-30-100`` and
``gaijingchu/countdown-4-10-50``) already ship pre-templated rows with the
prompt, ground truth, and metadata fields the trainer expects. This script
just streams those rows to ``train.parquet`` / ``test.parquet`` so the
trainer sees the *exact* same inputs as the original wandb runs.

If you point ``--hf_dataset`` at a dataset that does not ship a ``prompt``
field (only raw ``target`` / ``nums``), the script will apply the original
TinyZero prompt template on the fly.
"""

import argparse
import os

from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only "
    "be used once. Show your work in <think> </think> tags. And return the final "
    "answer in <answer> </answer> tags, for example <answer> (1 + 2) * 3 </answer>.\n"
    "Please verbalize your thinking process aloud as if you are brainstorming.\n"
    "Assistant: Let me solve this step by step.\n<think>"
)


def _materialise_raw(row, idx, split):
    """Apply the TinyZero prompt template to a raw (target, nums) row."""
    content = PROMPT_TEMPLATE.format(numbers=row['nums'], target=row['target'])
    return {
        'data_source': 'countdown',
        'prompt': [{'role': 'user', 'content': content}],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': {'target': row['target'], 'numbers': row['nums']},
        },
        'extra_info': {'split': split, 'index': idx},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_dataset', default='gaijingchu/countdown-4-30-100',
        help='HuggingFace dataset id. Use "gaijingchu/countdown-4-30-100" or '
             '"gaijingchu/countdown-4-10-50" to reproduce the paper experiments.')
    parser.add_argument('--local_dir', required=True,
                        help='Output directory; train.parquet / test.parquet are written here.')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='Last N rows are held out as the validation split.')
    parser.add_argument('--hdfs_dir', default=None,
                        help='Optional HDFS-style dst to copy results to.')
    args = parser.parse_args()

    raw = load_dataset(args.hf_dataset, split='train')
    n = len(raw)
    assert n > args.test_size, f'dataset too small ({n} rows) for test_size={args.test_size}'

    n_test = args.test_size
    train_ds = raw.select(range(n - n_test))
    test_ds = raw.select(range(n - n_test, n))

    # If the dataset already ships a `prompt` column we pass rows straight through
    # (this is the case for the released `gaijingchu/countdown-*` variants and
    # is required for exact wandb reproduction). Otherwise we apply the template.
    if 'prompt' in raw.column_names:
        print(f'[preprocess] {args.hf_dataset} ships baked prompts; passing through.')
    else:
        print(f'[preprocess] {args.hf_dataset} has raw target/nums; applying TinyZero template.')
        train_ds = train_ds.map(
            lambda row, idx: _materialise_raw(row, idx, 'train'),
            with_indices=True, remove_columns=raw.column_names)
        test_ds = test_ds.map(
            lambda row, idx: _materialise_raw(row, idx, 'test'),
            with_indices=True, remove_columns=raw.column_names)

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, 'train.parquet')
    test_path = os.path.join(args.local_dir, 'test.parquet')
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f'[preprocess] wrote {len(train_ds)} train rows -> {train_path}')
    print(f'[preprocess] wrote {len(test_ds)} test rows  -> {test_path}')

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)


if __name__ == '__main__':
    main()

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import random
import re
import shutil
import tempfile
from typing import Union

import numpy as np
import torch
import torch.distributed
from filelock import FileLock
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import get_device_name, get_torch_device


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: DictConfig = None,
    ):
        checkpoint_load_contents = checkpoint_contents.get("load_contents", None) if checkpoint_contents else None
        checkpoint_save_contents = checkpoint_contents.get("save_contents", None) if checkpoint_contents else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def local_mkdir(path):
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):  # Add timeout
                # make a new dir
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            # Even if the lock is not acquired, try to create the directory
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def is_checkpoint_valid(ckpt_path):
    """Check whether a checkpoint directory contains the required files.

    A checkpoint is considered valid if its ``actor/`` subdirectory contains at
    least one ``model_world_size_*_rank_*.pt`` file AND matching
    ``extra_state_world_size_*_rank_*.pt`` files.  Incomplete checkpoints
    (e.g. only model/optimizer saved before a preemption) will be detected and
    rejected.

    Args:
        ckpt_path (str): Path to a ``global_step_*`` checkpoint directory.

    Returns:
        bool: ``True`` if the checkpoint looks complete, ``False`` otherwise.
    """
    actor_dir = os.path.join(ckpt_path, "actor")
    if not os.path.isdir(actor_dir):
        return False
    model_files = glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt"))
    if len(model_files) == 0:
        return False
    extra_files = glob.glob(os.path.join(actor_dir, "extra_state_world_size_*_rank_*.pt"))
    if len(extra_files) == 0:
        print(f"WARNING: Checkpoint {ckpt_path} has model files but missing extra_state files, marking as invalid")
        return False
    if len(extra_files) != len(model_files):
        print(f"WARNING: Checkpoint {ckpt_path} has {len(model_files)} model files but {len(extra_files)} extra_state files, marking as invalid")
        return False
    return True


def _find_all_ckpt_steps(path):
    """Return a sorted (descending) list of checkpoint step numbers found under *path*."""
    if not os.path.isdir(path):
        return []
    steps = []
    for entry in os.listdir(path):
        m = re.match(r"^global_step_(\d+)$", entry)
        if m and os.path.isdir(os.path.join(path, entry)):
            steps.append(int(m.group(1)))
    steps.sort(reverse=True)
    return steps


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent *valid* checkpoint directory.

    First checks the tracker file for the latest iteration.  If that checkpoint
    is missing or corrupted (incomplete model files), falls back to scanning all
    ``global_step_*`` directories in descending order until a valid one is found.
    If no valid checkpoint exists, returns ``None`` so training starts from
    scratch.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest valid checkpoint directory, or
        None if no valid checkpoint is found.
    """
    if path is None:
        return None

    # 1. Try the tracker file first (fast path)
    tracker_file = get_checkpoint_tracker_filename(path)
    if os.path.exists(tracker_file):
        with open(tracker_file, "rb") as f:
            iteration = int(f.read().decode())
        ckpt_path = os.path.join(path, directory_format.format(iteration))
        if os.path.exists(ckpt_path) and is_checkpoint_valid(ckpt_path):
            print(f"Found valid checkpoint: {ckpt_path}")
            return ckpt_path
        else:
            print(f"WARNING: Checkpoint from tracker (global_step_{iteration}) is missing or corrupted, scanning for earlier valid checkpoints...")
    else:
        print(f"Checkpoint tracker file does not exist: {tracker_file}")

    # 2. Fallback: scan all global_step_* directories in descending order
    all_steps = _find_all_ckpt_steps(path)
    for step in all_steps:
        candidate = os.path.join(path, directory_format.format(step))
        if is_checkpoint_valid(candidate):
            print(f"Found valid fallback checkpoint: {candidate}")
            return candidate
        else:
            print(f"WARNING: Skipping corrupted checkpoint: {candidate}")

    print("No valid checkpoint found, will train from scratch")
    return None


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")

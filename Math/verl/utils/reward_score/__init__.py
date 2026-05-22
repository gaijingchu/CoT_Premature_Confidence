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

from verl.utils.import_utils import deprecated


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """Dispatch ``compute_score`` based on ``data_source``.

    For this code release only math sources are wired up — the dapo_hard
    dataset used by the paper experiments is routed to ``math_verify``.
    Code / search / mcq / maze datasets are not included.
    """
    if data_source == "openai/gsm8k" or data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"] or (
        data_source == "math_dapo"
        or data_source.startswith("aime")
        or data_source == "amc23"
        or data_source.startswith("dapo")
        or data_source.startswith("deepmath-103k")
        or data_source == "polaris"
        or data_source == "openr1"
        or data_source == "minerva"
        or data_source == "olympiadbench"
        or data_source == "simplelr_qwen"
        or data_source == "beyondaime"
        or data_source == "hmmt_feb_2025"
        or data_source == "hmmt_nov_2025"
        or data_source == "jeebench"
    ):
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """Legacy alias retained for backwards-compatible imports."""
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


__all__ = ["default_compute_score"]

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

from collections import defaultdict
from typing import Optional
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics, reverse_auc_rewards_for_group, build_embedder)


class TTRLRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", compute_score=None, n_votes_per_prompt=1, n_samples_per_prompt=1, mode="eval", eval_n_samples=1, penalty_reward: float = -1, sim_threshold: float = 0.6, cluster_backend: str = "embedding", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embed_device: Optional[str] = None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.penalty_reward = penalty_reward
        self.sim_threshold = sim_threshold
        self.cluster_backend = cluster_backend
        self.embedding_model_name = embedding_model_name
        self.embed_device = embed_device
        self.embedder = None
        if self.cluster_backend == "embedding":
            # Initialize embedder once
            self.embedder = build_embedder(self.embedding_model_name, device=self.embed_device)
            # Light warm-up to trigger lazy initialization and CUDA graph
            try:
                if self.embedder is not None:
                    _ = self.embedder.encode(["warmup", "预热"])
            except Exception:
                pass
        assert n_votes_per_prompt >= n_samples_per_prompt, f"For TTRL settings, n_votes_per_prompt {n_votes_per_prompt} should be greater than or equal to n_samples_per_prompt {n_samples_per_prompt}"

        embedder_backend = getattr(self.embedder, 'backend', None) if self.embedder is not None else None
        print("="*60)
        print(f"TTRLRewardManager initialized with n_votes_per_prompt {n_votes_per_prompt}, n_samples_per_prompt {n_samples_per_prompt}, eval_n_samples {eval_n_samples}, penalty_reward {penalty_reward}, sim_threshold {sim_threshold}, cluster_backend {cluster_backend}, embed_model {embedding_model_name}, embed_backend {embedder_backend}")
        print("="*60)

    def _data_source_to_task(self, data_source):
        # Normalize potential path-like sources to base name and uppercase for matching
        try:
            base = Path(str(data_source)).name
        except Exception:
            base = str(data_source).split("/")[-1]
        name = base.strip().upper()

        # Exact matches first
        if name in ["MATH-TTT", "AIME-TTT", "AMC-TTT"]:
            return "math"
        if name in ["GPQA-TTT", "GPQA"]:
            return "gpqa"

        # Fallback: substring-based mapping
        if any(k in name for k in ["MATH", "AIME", "AMC"]):
            return "math"
        if "GPQA" in name:
            return "gpqa"

        raise NotImplementedError(f"Data source {data_source} is not supported for TTRLRewardManager")

    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post TTRL metrics for the given data.
        """
        assert len(data) % self.n_samples_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_samples_per_prompt}"
        prompt_num = len(data) // self.n_samples_per_prompt

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
                group_vote_rewards = []
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None

                for i in range(self.n_samples_per_prompt):
                    data_item = data[prompt_i * self.n_samples_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    vote_reward = data_item.batch["acc"]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_vote_rewards.append(vote_reward)
                    group_extra_info.append(extra_info)
                
                post_ttrl_metrics = post_test_time_train_metrics(group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info)
                for k, v in post_ttrl_metrics.items():
                    post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
            
            prompt_num = len(data) // self.n_votes_per_prompt

            reward_tensor = torch.zeros_like(data.batch["responses"][:prompt_num*self.n_samples_per_prompt], dtype=torch.float32)

            already_print_data_sources = {}

            all_ttrl_metrics = defaultdict(list)

            scores = [0.0 for _ in range(len(data))]
            
            for prompt_i in range(prompt_num):
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None

                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)
                
                # Custom step-based reverse AUC rewards (no need for ground truth here)
                rewards, ttrl_metrics = reverse_auc_rewards_for_group(
                    group_pred_outputs,
                    penalty_reward=self.penalty_reward,
                    sim_threshold=self.sim_threshold,
                    cluster_backend=self.cluster_backend,
                    embedding_model_name=self.embedding_model_name,
                    embed_device=self.embed_device,
                    embedder=self.embedder,
                )

                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)

                for i in range(self.n_votes_per_prompt):
                    if i < self.n_samples_per_prompt:
                        reward_tensor[prompt_i * self.n_samples_per_prompt + i, valid_response_length - 1] = rewards[i]
                    scores[prompt_i * self.n_votes_per_prompt + i] = rewards[i]

                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                        print("[score]", rewards[i])

                # Print group-level cluster count once per group
                if "cluster_count" in ttrl_metrics:
                    print("[cluster_count_group]", ttrl_metrics["cluster_count"])

            data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v
            return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            already_print_data_sources = {}
            task = None
            for i in range(len(data)):
                data_item = data[i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]
                
                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)

                if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

            # Use custom step-based reverse AUC rewards for eval as well
            rewards, group_metrics = reverse_auc_rewards_for_group(
                group_pred_outputs,
                penalty_reward=self.penalty_reward,
                sim_threshold=self.sim_threshold,
                cluster_backend=self.cluster_backend,
                embedding_model_name=self.embedding_model_name,
                embed_device=self.embed_device,
                embedder=self.embedder,
            )
            if "cluster_count" in group_metrics:
                print("[cluster_count_group]", group_metrics["cluster_count"])
            verify_extra_info = {}

            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v

            for i in range(len(data)):
                reward_tensor[i, valid_response_length - 1] = rewards[i]

            # Compute group-level metrics using reverse AUC scheme
            all_ttrl_metrics = defaultdict(list)
            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                group_pred_outputs_ttrl = []
                group_labels_ttrl = []
                group_extra_info_ttrl = []
                task = None

                for i in range(self.eval_n_samples):
                    data_item = data[prompt_i * self.eval_n_samples + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels_ttrl.append(ground_truth)
                    group_pred_outputs_ttrl.append(response_str)
                    group_extra_info_ttrl.append(extra_info)
                
                _, ttrl_metrics = reverse_auc_rewards_for_group(
                    group_pred_outputs_ttrl,
                    penalty_reward=self.penalty_reward,
                    sim_threshold=self.sim_threshold,
                    cluster_backend=self.cluster_backend,
                    embedding_model_name=self.embedding_model_name,
                    embed_device=self.embed_device,
                    embedder=self.embedder,
                )
                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v
            
            return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False):

        if self.mode == "train":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for TTRLRewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "ttrl_info": ttrl_info,
                }
        else:
            return reward_tensor
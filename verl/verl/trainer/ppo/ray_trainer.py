#ray_trainer.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import copy
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import (get_response_mask, masked_mean,
                                         pad_2d_list_to_length,
                                         pad_sequence_to_length)
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.enable_edit = self.config.actor_rollout_ref.rollout.get("edit_config", {}).get("enable_edit", False)
        self.enable_select = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("enable_select", False)
        self.select_fun = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("select_fun", None)
        self.select_fun_args = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("select_fun_args", None)
        self.save_data = self.config.actor_rollout_ref.rollout.get("save_data", True)
        self.save_path = self.config.actor_rollout_ref.rollout.get("save_path", "")
        self.edit_sample_prob = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("edit_sample_prob", 0.1)

        print("---------------enable_edit---------------")
        print(self.enable_edit)
        print("---------------select_fun---------------")
        print(self.select_fun)

        
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % int(n_gpus) == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        print("=== 开始验证 ===")
        print(f"验证数据集大小: {len(self.val_dataloader)}")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        for test_data in self.val_dataloader:
            print("=== 处理验证批次 ===")
            test_batch = DataProto.from_single_dict(test_data)

            print(f"批次大小: {test_batch.batch.batch_size[0]}")
            print(f"reward_model配置: {self.config.reward_model.enable}")
            if "reward_model" in test_batch[0].non_tensor_batch:
                print(f"reward_model style: {test_batch[0].non_tensor_batch['reward_model']['style']}")
               
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=False)
        
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                print("使用模型奖励，直接返回空字典")
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        self.save_counter = self.global_steps
        print(f"Setting save_counter as global_steps: {self.save_counter}")

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        import traceback

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self.save_counter = 0
        self.final_save_counter = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                print(f'ray_trainer set batch.meta_info global_steps as {self.global_steps}')
 

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):

                    # --- Reference Model Update ---
                    if self.use_reference_policy:
                        ref_update_freq = self.config.trainer.get('ref_update_freq', -1)
                        print("------------------ref_update_freq----------------------------")
                        print(ref_update_freq)
                        if ref_update_freq > 0 and self.global_steps % ref_update_freq == 0:
                            print(f"\n[Step {self.global_steps}] Updating Reference Model Weights from Actor...")
                            try:

                                self.reset_actor_states_to_ref(synchronize_ref=True)
                                print(f"[Step {self.global_steps}] Reference Model Weights and Actor Optimizer State Updated.")

                            except Exception as sync_e:
                                print(f"ERROR during reference model sync at step {self.global_steps}: {sync_e}")
                                traceback.print_exc()

                    # generate a batch
                    # with _timer("gen", timing_raw):
                    #     if not self.async_rollout_mode:
                    #         gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    #     else:
                    #         self.async_rollout_manager.wake_up()


                    with _timer("gen", timing_raw):  
                        if not self.async_rollout_mode:  
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  
                        else:  
                            self.async_rollout_manager.wake_up()  
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)  
                            self.async_rollout_manager.sleep()  
                        
                        # metrics：{}
                        for key, value in gen_batch_output.meta_info.items():  
                            if key.startswith("feedback/"):  
                                # metrics[key] = value  
                                metrics.update({
                                    key: value,
                                }) # TODO actually not 4, maybe count/8 by mistake
                        # metrics：{'feedback/total_count': 4, 'feedback/success_count': 4, 'feedback/total_attempts': 4, 'feedback/total_response_length': 1164, 'feedback/success_rate': 1.0, 'feedback/avg_attempts': 1.0, 'feedback/avg_response_length': 291.0}
                        

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    print(f"=== DEBUG POINT 1: 第一次repeat之后 ===")
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # 128 * 2 = 256
                    # here has not merged with gen_batch_output
                    # (if has been edited, then it has been doubled) 

                    if not self.enable_edit:
                        batch = batch.union(gen_batch_output)

                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n

                        print('verl/recipe/vapo/vapo_ray_trainer.py L701')

                        if len(batch)>traj_bsz and self.config.actor_rollout_ref.rollout.get("filter_overlen_ngram",False) :
                            # filter 
                            assert False, 'deprecated'
                            print('enter filter_overlen_ngram from not edit')

                            # batch_??(traj_bsz=traj_bsz,max_len=,n_gram_selection_dic,)
                            from verl.utils.select import filter_batch_by_ngram_maxlen
                            ngram_thresholds = self.config.actor_rollout_ref.rollout.get("ngram_thresholds", {2: 0.2, 3: 0.12})  
                            print(f'{ngram_thresholds=}')
                            batch = filter_batch_by_ngram_maxlen(batch, traj_bsz=traj_bsz, max_len=self.config.data.get("max_response_length", None), ngram_thresholds=ngram_thresholds)
                            if len(batch)!=traj_bsz:
                                print(f'{len(batch)=}')
                                raise ValueError("len(batch)!=traj_bsz")
                    else: # self.enable_edit:
                        ''' batch.batch.keys()
                        gen_batch_output.batch.keys() #_StringKeys(dict_keys(['responses', 'position_ids', 'input_ids', 'is_original', 'prompts', 'is_edited_mask', 'attention_mask', 'is_edited', 'edited_log_probs']))

                        '''
                        batch=self._apply_selection_strategy_in_gen(batch, gen_batch_output)
                        

                        # 在合并edit后，为original和edited配对关系添加uid2标记
                        if "is_original" in batch.batch and "is_edited" in batch.batch:
                            original_indices = torch.where(batch.batch["is_original"])[0]
                            edited_indices = torch.where(batch.batch["is_edited"])[0]
                            

                            pair_uids = []
                            num_rollouts = self.config.actor_rollout_ref.rollout.n
                            half_size = len(original_indices)
                            
                            if half_size % num_rollouts != 0:
                                raise ValueError(
                                    f"The number of original samples ({half_size}) "
                                    f"is not divisible by rollout.n ({num_rollouts})."
                                )
                            num_prompts = half_size // num_rollouts
                            
                            for _ in range(num_prompts):
                                pair_uids.extend( [str(uuid.uuid4()) for _ in range(num_rollouts)]*2)
                            
                            batch.non_tensor_batch["uid2"] = np.array(pair_uids, dtype=object)
                        
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo

                    if self.config.trainer.balance_batch:
                        # Ensure non_tensor_batch sizes match batch size
                        batch_size = batch.batch["attention_mask"].shape[0]
                        for key, val in batch.non_tensor_batch.items():
                            if isinstance(val, np.ndarray) and val.shape[0] != batch_size:
                                print(f"Warning: non_tensor_batch[{key}] size ({val.shape[0]}) doesn't match batch_size ({batch_size})")

                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)


                        batch, reward_tensor, reward_extra_infos_dict = self._apply_selection_strategy_in_using_adv(batch, gen_batch_output, reward_tensor, reward_extra_infos_dict)  

                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)


                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # if self.config.get("NLL_loss", False) is True:  
                        use_NLL_loss=self.config.algorithm.get("NLL_loss", False)
                        if use_NLL_loss:
                            # Apply threshold-based filtering for positive examples  
                            threshold = self.config.algorithm.get("NLL_threshold", False)  
                            _nll_rate = self.config.algorithm.get("NLL_rate", False)  

                            if threshold:
                                print(f"vapo NLL pos {threshold=}")
                                # positive_mask = (batch.batch["reward"][:,-1] > threshold).float()  
                                token_level_scores = batch.batch["token_level_scores"]  
                                # token_level_scores2 = batch.batch["token_level_rewards"]  
                                # (token_level_scores2 * response_mask).sum(dim=-1)  

                                response_mask = batch.batch["response_mask"]
                                final_scores = (token_level_scores * response_mask).sum(dim=-1)  
                                NLL_loss_mask = (final_scores > threshold).float()
                                batch.batch["NLL_loss_mask"] = NLL_loss_mask
                                # tensor([0., 1., 0., 0.])


                            elif _nll_rate:
                                print(f"vapo NLL rate {_nll_rate=}")
                                
                                _nll_num_to_select = int(len(batch) * _nll_rate)                            
                                token_level_scores = batch.batch["rm_scores"]    
                                # torch.Size([bsz, seq_len])
                                # torch.Size([32, 2048])
                                # token_level_scores = batch.batch["token_level_scores"]  

                                response_mask = batch.batch["response_mask"]
                                final_scores = (token_level_scores * response_mask).sum(dim=-1)  
                                # bsz: torch.Size([32])

                                # sort by score in descending order
                                sorted_indices = torch.argsort(final_scores, descending=True)
                                
                                # create a zero mask
                                NLL_loss_mask = torch.zeros_like(final_scores)
                                
                                # set the top samples to 1
                                NLL_loss_mask[sorted_indices[:_nll_num_to_select]] = 1.0
                                
                                # add the mask to the batch
                                batch.batch["NLL_loss_mask"] = NLL_loss_mask


                            else:
                                assert False,'should have least one to use nll'
                                

                                
                            positive_ratio = NLL_loss_mask.mean().item()  
                            print(f'{positive_ratio=}')
                            metrics.update({  
                                "vapo_NLL/positive_ratio": positive_ratio,
                                })  

                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _apply_selection_strategy_in_using_adv(self, batch, gen_batch_output, reward_tensor, reward_extra_infos_dict):
        if self.select_fun == "sample_edit_reward":
            print("select the highest scored edited answer for each question")
                            
            self._original_batch_size = len(batch)
                            
            if not ("is_original" in gen_batch_output.batch and "is_edited" in gen_batch_output.batch):
                assert False,'should not happen'
            
            else:
                print("--------------self.config.trainer.balance_batch-------------------------")
                print(self.config.trainer.balance_batch)
                new_batch = TensorDict(
                                    {
                                        **batch.batch,
                                        "is_original": batch.batch["is_original"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_original"],
                                        "is_edited": batch.batch["is_edited"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_edited"]                                     
                                    },
                                    batch_size=batch.batch.batch_size
                                )
                maybe_balanced_full_batch = copy.deepcopy(new_batch)
                batch.batch = new_batch

                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
                raw_prompts = batch.non_tensor_batch.get("raw_prompts", [None] * len(prompts_text))
                responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
                feedback_texts = batch.non_tensor_batch.get("feedback_texts", [None] * len(prompts_text)) if hasattr(self, 'enable_feedback') and self.enable_feedback else [None] * len(prompts_text)


                token_level_scores = reward_tensor
                response_mask = batch.batch["response_mask"]
                final_scores = (token_level_scores * response_mask).sum(dim=-1)

                final_scores_list = final_scores.cpu().tolist()
                                
                question_uids = batch.non_tensor_batch["question_uid"]
                unique_question_uids = np.unique(question_uids)

                all_original_indices = torch.where(batch.batch["is_original"])[0]
                print(f'all_original_indices {len(all_original_indices)=}')
                selected_indices = all_original_indices.tolist() # Start with all original responses
                                
                # 计算每个question_uid需要选择的编辑回答数量
                rollout_per_query = self.config.actor_rollout_ref.rollout.n
                edits_per_uid = int(np.ceil(rollout_per_query * self.edit_sample_prob))
                
                best_edited_candidates = []
                
                for question_uid in unique_question_uids: #len(unique_question_uids)=8
                    group_mask = (question_uids == question_uid)
                    is_edited_mask = batch.batch["is_edited"].cpu().numpy() # is_original_mask=batch.batch["is_original"].cpu().numpy()
                    # edited_indices_in_group = np.where(group_mask & batch.batch["is_original"].cpu().numpy())[0] # original_indices_in_group = np.where(group_mask & is_original_mask)[0]
                    edited_indices_in_group = np.where(group_mask & is_edited_mask)[0]
                    # breakpoint()# prompts, attention_mask,position_ids # batch.batch['prompts'][original_indices_in_group][-110:105]
                    # batch.batch['prompts'][edited_indices_in_group][-110:105]
                                    
                    if not len(edited_indices_in_group) > 0:# len(edited_indices_in_group):4
                        assert False, 'should not happen'
                    else:
                        group_scores = final_scores.cpu().numpy()[edited_indices_in_group]
                        # 按分数排序，选择top-k个
                        if self.select_fun_args=='sample_edit_reward_low_reward':
                            sorted_indices = np.argsort(group_scores) 
                        else:
                            sorted_indices = np.argsort(group_scores)[::-1]  # 降序排列
                        num_to_select_in_group = min(edits_per_uid, len(sorted_indices))
                        
                        for i in range(num_to_select_in_group):
                            best_local_idx = sorted_indices[i]
                            best_global_idx = edited_indices_in_group[best_local_idx]
                            best_edited_candidates.append(best_global_idx)

                print(f'best_edited_candidates {len(best_edited_candidates)=}')
                print(f'edits_per_uid {edits_per_uid=}')
                if not best_edited_candidates:
                    assert False, 'should not happen'
                else:
                    # edit_sample_prob * original_batch_size, align with other methods
                    num_to_select = int(len(all_original_indices) * self.edit_sample_prob)
                    m = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    if m > 0:
                        num_to_select = (num_to_select // m) * m
                                    
                    # if the number of candidates is greater than the number needed, select randomly
                    if len(best_edited_candidates) > num_to_select:
                        perm = torch.randperm(len(best_edited_candidates))
                        final_selected_edited_indices = [best_edited_candidates[i] for i in perm[:num_to_select]]
                    else:
                        final_selected_edited_indices = best_edited_candidates
                    
                    selected_indices.extend(final_selected_edited_indices)

                print(f'final_selected_edited_indices {len(final_selected_edited_indices)=}')

                final_indices_tensor = torch.tensor(sorted(list(set(selected_indices))), dtype=torch.long)
                self._selected_indices = final_indices_tensor.cpu().numpy()

                new_batch = TensorDict({}, batch_size=[len(final_indices_tensor)])
                for key in batch.batch.keys():
                    if isinstance(batch.batch.get(key), torch.Tensor):
                        selected_data = batch.batch.get(key)[final_indices_tensor]
                        new_batch.set(key, selected_data)
                    else:
                        new_batch.set(key, batch.batch.get(key))
                                
                if batch.non_tensor_batch:
                    new_non_tensor_batch = {}
                    original_non_tensor_size = self._original_batch_size
                    for key, value in batch.non_tensor_batch.items():
                        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == original_non_tensor_size:
                            selected_data = value[self._selected_indices]
                            new_non_tensor_batch[key] = selected_data
                        else:
                            new_non_tensor_batch[key] = value
                    batch.non_tensor_batch = new_non_tensor_batch
                                
                batch.batch = new_batch

                # is_originals==list(torch.where(batch.batch["is_original"])[0].cpu().numpy()) 
                is_originals = [maybe_balanced_full_batch["is_original"][i].item() for i in range(len(prompts_text))]
                is_editeds = [maybe_balanced_full_batch["is_edited"][i].item() for i in range(len(prompts_text))]
                is_selected = [i in self._selected_indices for i in range(len(prompts_text))]

                self._save_selection_data(prompts_text, raw_prompts, responses_text, feedback_texts, final_scores_list,is_originals,is_editeds,is_selected)



        elif self.select_fun == "sample_edit":
            prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
            raw_prompts = batch.non_tensor_batch.get("raw_prompts", [None] * len(prompts_text))
            responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
            feedback_texts = batch.non_tensor_batch.get("feedback_texts", [None] * len(prompts_text)) if hasattr(self, 'enable_feedback') and self.enable_feedback else [None] * len(prompts_text)

                            
            original_indices = torch.where(batch.batch["is_original"])[0]
            edited_indices = torch.where(batch.batch["is_edited"])[0]

                            # Calculate scores
            token_level_scores = reward_tensor
            response_mask = batch.batch["response_mask"]
            final_scores = (token_level_scores * response_mask).sum(dim=-1)

            final_scores_list = final_scores.cpu().tolist()
                            # self._selected_indices = selected_indices.cpu().numpy()


            # 提前计算所需数据
            is_originals = [batch.batch["is_original"][i].item() for i in range(len(prompts_text))]
            is_editeds = [batch.batch["is_edited"][i].item() for i in range(len(prompts_text))]
            # 在sample_edit中，所有样本都被选中
            is_selected = [True] * len(prompts_text)  # 或者直接传入None，默认全选

            # 调用辅助函数保存数据
            self._save_selection_data(
                prompts_text, 
                raw_prompts, 
                responses_text, 
                feedback_texts, 
                final_scores_list,
                is_originals,
                is_editeds,
                is_selected  
            )


        elif self.select_fun == "rank_by_reward":
            print("sort the edited answer by the reward score in a batch, and select the top-k edited answer")
            self._original_batch_size = len(batch)

            if "is_original" in gen_batch_output.batch and "is_edited" in gen_batch_output.batch:
                new_batch = TensorDict(
                    {
                        **batch.batch,
                        "is_original": batch.batch["is_original"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_original"],
                        "is_edited": batch.batch["is_edited"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_edited"]                                     
                    },
                    batch_size=batch.batch.batch_size
                )
                maybe_balanced_full_batch = copy.deepcopy(new_batch)
                batch.batch = new_batch

                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
                raw_prompts = batch.non_tensor_batch.get("raw_prompts", [None] * len(prompts_text))
                responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
                feedback_texts = batch.non_tensor_batch.get("feedback_texts", [None] * len(prompts_text)) if hasattr(self, 'enable_feedback') and self.enable_feedback else [None] * len(prompts_text)
                # 计算分数
                token_level_scores = reward_tensor
                response_mask = batch.batch["response_mask"]
                final_scores = (token_level_scores * response_mask).sum(dim=-1)
                final_scores_list = final_scores.cpu().tolist()

                # 获取原始回答和编辑后回答的索引
                original_indices = torch.where(batch.batch["is_original"])[0]
                edited_indices = torch.where(batch.batch["is_edited"])[0]
                
                # 获取编辑后回答的分数
                edited_scores = final_scores[edited_indices]
                
                # 计算要选择的编辑后回答数量
                num_to_select = int(len(edited_indices) * self.edit_sample_prob)
                m = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                if m > 0:
                    num_to_select = (num_to_select // m) * m
                
                # 按照分数排序选择top-k
                if num_to_select > 0:
                    _, top_indices = torch.topk(edited_scores, num_to_select)
                    selected_edited_indices = edited_indices[top_indices]
                else:
                    selected_edited_indices = torch.tensor([], dtype=torch.long, device=original_indices.device)
                
                # 最终选择的索引 = 所有原始回答 + 选定的编辑后回答
                selected_indices = torch.cat([original_indices, selected_edited_indices])
                
                print(f"原始批次大小: {len(original_indices)}。增加 {len(selected_edited_indices)} 个按reward排序的编辑后回答。最终批次大小: {len(selected_indices)}。")
                
                # 过滤批次
                new_batch = TensorDict({}, batch_size=[len(selected_indices)])
                for key in batch.batch.keys():
                    if isinstance(batch.batch.get(key), torch.Tensor):
                        selected_data = batch.batch.get(key)[selected_indices]
                        new_batch.set(key, selected_data)
                    else:
                        new_batch.set(key, batch.batch.get(key))
                
                if batch.non_tensor_batch:
                    new_non_tensor_batch = {}
                    for key, value in batch.non_tensor_batch.items():
                        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == self._original_batch_size:
                            selected_data = value[selected_indices.cpu().numpy()]
                            new_non_tensor_batch[key] = selected_data
                        else:
                            new_non_tensor_batch[key] = value
                    batch.non_tensor_batch = new_non_tensor_batch
                
                batch.batch = new_batch
                self._selected_indices = selected_indices.cpu().numpy()

                is_originals = [maybe_balanced_full_batch["is_original"][i].item() for i in range(len(prompts_text))]
                is_editeds = [maybe_balanced_full_batch["is_edited"][i].item() for i in range(len(prompts_text))]
                is_selected = [i in self._selected_indices for i in range(len(prompts_text))]

                # 保存选择数据
                self._save_selection_data(
                    prompts_text, 
                    raw_prompts, 
                    responses_text, 
                    feedback_texts, 
                    final_scores_list,
                    is_originals,
                    is_editeds,
                    is_selected
                )

        elif self.select_fun == "rank_by_improvement":
            print("sort the edited answer by the difference between the edited score and the original score in a batch, and select the answer with the biggest difference")
            self._original_batch_size = len(batch)

            if "is_original" in gen_batch_output.batch and "is_edited" in gen_batch_output.batch:
                new_batch = TensorDict(
                    {
                        **batch.batch,
                        "is_original": batch.batch["is_original"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_original"],
                        "is_edited": batch.batch["is_edited"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_edited"]                                     
                    },
                    batch_size=batch.batch.batch_size
                )
                maybe_balanced_full_batch = copy.deepcopy(new_batch)
                batch.batch = new_batch

                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
                raw_prompts = batch.non_tensor_batch.get("raw_prompts", [None] * len(prompts_text))
                responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
                feedback_texts = batch.non_tensor_batch.get("feedback_texts", [None] * len(prompts_text)) if hasattr(self, 'enable_feedback') and self.enable_feedback else [None] * len(prompts_text)

                token_level_scores = reward_tensor
                response_mask = batch.batch["response_mask"]
                final_scores = (token_level_scores * response_mask).sum(dim=-1)
                final_scores_list = final_scores.cpu().tolist()

                # select original and corresponding edited answers
                original_indices = torch.where(batch.batch["is_original"])[0]
                edited_indices = torch.where(batch.batch["is_edited"])[0]
                
                if len(original_indices) != len(edited_indices):
                    raise ValueError("Original and edited batch sizes must match for improvement calculation.")
                    
                original_scores = final_scores[original_indices]
                edited_scores = final_scores[edited_indices]
                
                score_diffs = edited_scores - original_scores
                
                num_to_select = int(len(edited_indices) * self.edit_sample_prob)
                m = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                if m > 0:
                    num_to_select = (num_to_select // m) * m
                
                if num_to_select > 0:
                    _, top_indices = torch.topk(score_diffs, num_to_select)
                    selected_edited_indices = edited_indices[top_indices]
                else:
                    selected_edited_indices = torch.tensor([], dtype=torch.long, device=original_indices.device)
                
                # final selected indices = all original answers + selected edited answers
                selected_indices = torch.cat([original_indices, selected_edited_indices])
                
                print(f"original batch size: {len(original_indices)}. add {len(selected_edited_indices)} edited answers sorted by improvement. final batch size: {len(selected_indices)}.")
                
                # filter batch
                new_batch = TensorDict({}, batch_size=[len(selected_indices)])
                for key in batch.batch.keys():
                    if isinstance(batch.batch.get(key), torch.Tensor):
                        selected_data = batch.batch.get(key)[selected_indices]
                        new_batch.set(key, selected_data)
                    else:
                        new_batch.set(key, batch.batch.get(key))
                
                if batch.non_tensor_batch:
                    new_non_tensor_batch = {}
                    for key, value in batch.non_tensor_batch.items():
                        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == self._original_batch_size:
                            selected_data = value[selected_indices.cpu().numpy()]
                            new_non_tensor_batch[key] = selected_data
                        else:
                            new_non_tensor_batch[key] = value
                    batch.non_tensor_batch = new_non_tensor_batch
                
                batch.batch = new_batch
                self._selected_indices = selected_indices.cpu().numpy()

                is_originals = [maybe_balanced_full_batch["is_original"][i].item() for i in range(len(prompts_text))]
                is_editeds = [maybe_balanced_full_batch["is_edited"][i].item() for i in range(len(prompts_text))]
                is_selected = [i in self._selected_indices for i in range(len(prompts_text))]

                self._save_selection_data(
                    prompts_text, 
                    raw_prompts, 
                    responses_text, 
                    feedback_texts, 
                    final_scores_list,
                    is_originals,
                    is_editeds,
                    is_selected
                )

        elif self.select_fun == "group_rank_by_improvement":
            print("select the edited answer with the biggest improvement for each group of questions")
            self._original_batch_size = len(batch)
            if "is_original" in gen_batch_output.batch and "is_edited" in gen_batch_output.batch:
                new_batch = TensorDict(
                    {
                        **batch.batch,
                        "is_original": batch.batch["is_original"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_original"],
                        "is_edited": batch.batch["is_edited"] if self.config.trainer.balance_batch else gen_batch_output.batch["is_edited"]                                     
                    },
                    batch_size=batch.batch.batch_size
                )
                maybe_balanced_full_batch = copy.deepcopy(new_batch)
                batch.batch = new_batch

                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
                raw_prompts = batch.non_tensor_batch.get("raw_prompts", [None] * len(prompts_text))
                responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
                feedback_texts = batch.non_tensor_batch.get("feedback_texts", [None] * len(prompts_text)) if hasattr(self, 'enable_feedback') and self.enable_feedback else [None] * len(prompts_text)

                # 计算分数
                token_level_scores = reward_tensor
                response_mask = batch.batch["response_mask"]
                final_scores = (token_level_scores * response_mask).sum(dim=-1)
                final_scores_list = final_scores.cpu().tolist()

                # 使用question_uid分组，然后在组内用pair_uid配对计算improvement
                question_uids = batch.non_tensor_batch["question_uid"]
                pair_uids = batch.non_tensor_batch["pair_uid"]
                unique_question_uids = np.unique(question_uids)

                all_original_indices = torch.where(batch.batch["is_original"])[0]
                print(f'all_original_indices {len(all_original_indices)=}')
                selected_indices = all_original_indices.tolist() # Start with all original responses

                # 计算每个question_uid需要选择的编辑回答数量  
                rollout_per_query = self.config.actor_rollout_ref.rollout.n
                edits_per_uid = int(np.ceil(rollout_per_query * self.edit_sample_prob))

                best_improvement_candidates = []
                for question_uid in unique_question_uids:
                    # 找到该question_uid组内的所有样本
                    group_mask = (question_uids == question_uid)
                    is_original_mask = batch.batch["is_original"].cpu().numpy()
                    is_edited_mask = batch.batch["is_edited"].cpu().numpy()

                    original_indices_in_group = np.where(group_mask & is_original_mask)[0]
                    edited_indices_in_group = np.where(group_mask & is_edited_mask)[0]

                    if not (len(original_indices_in_group) > 0 and len(edited_indices_in_group) > 0):
                        assert False, 'should not happen'
                        print(f'should not happen! ####### len(edited_indices_in_group):{len(edited_indices_in_group)},len(edited_indices_in_group):{len(edited_indices_in_group)}\n')

                    else:

                        improvement_candidates_in_group = []
                        
                        for edited_idx in edited_indices_in_group:
                            edited_pair_uid = pair_uids[edited_idx]
                            
                            # 找到对应的original（相同pair_uid）
                            matching_original_indices = []
                            for orig_idx in original_indices_in_group:
                                if pair_uids[orig_idx] == edited_pair_uid:
                                    matching_original_indices.append(orig_idx)
                            
                            # if len(matching_original_indices) > 0:
                            assert len(matching_original_indices) == 1, f'{len(matching_original_indices)=}'
                            original_idx = matching_original_indices[0]  # 应该只有一个匹配的original
                            original_score = final_scores.cpu().numpy()[original_idx]
                            edited_score = final_scores.cpu().numpy()[edited_idx]
                            improvement = edited_score - original_score
                            improvement_candidates_in_group.append((edited_idx, improvement))
                        
                        # 在该组内按improvement排序，选择top-k
                        if improvement_candidates_in_group:
                            improvement_candidates_in_group.sort(key=lambda x: x[1], reverse=True)
                            num_to_select_in_group = min(edits_per_uid, len(improvement_candidates_in_group))
                            
                            for i in range(num_to_select_in_group):
                                best_improvement_candidates.append(improvement_candidates_in_group[i])

                print(f'best_improvement_candidates {len(best_improvement_candidates)=}')
                print(f'edits_per_uid {edits_per_uid=}')

                if best_improvement_candidates:
                    # edit_sample_prob * original_batch_size, align with other methods
                    num_to_select = int(len(all_original_indices) * self.edit_sample_prob)
                    m = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    if m > 0:
                        num_to_select = (num_to_select // m) * m

                    # if the number of candidates is greater than the number needed, select the top k by improvement globally
                    if len(best_improvement_candidates) > num_to_select:
                        best_improvement_candidates.sort(key=lambda x: x[1], reverse=True)
                        final_selected_edited_indices = [idx for idx, _ in best_improvement_candidates[:num_to_select]]
                    else:
                        final_selected_edited_indices = [idx for idx, _ in best_improvement_candidates]
                    
                    selected_indices.extend(final_selected_edited_indices)
                else:
                    assert False, 'should not happen'
                    # final_selected_edited_indices = []

                print(f'final_selected_edited_indices {len(final_selected_edited_indices)=}')

                # here deleted debug, hope it's not needed

                final_indices_tensor = torch.tensor(sorted(list(set(selected_indices))), dtype=torch.long)
                selected_indices = final_indices_tensor
                
                print(f"original batch size: {len(all_original_indices)}. add {len(final_selected_edited_indices)} edited answers sorted by improvement in each group. final batch size: {len(selected_indices)}.")

                new_batch = TensorDict({}, batch_size=[len(selected_indices)])
                for key in batch.batch.keys():
                    if isinstance(batch.batch.get(key), torch.Tensor):
                        selected_data = batch.batch.get(key)[selected_indices]
                        new_batch.set(key, selected_data)
                    else:
                        new_batch.set(key, batch.batch.get(key))
                
                if batch.non_tensor_batch:
                    new_non_tensor_batch = {}
                    for key, value in batch.non_tensor_batch.items():
                        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == self._original_batch_size:
                            selected_data = value[selected_indices.cpu().numpy()]
                            new_non_tensor_batch[key] = selected_data
                        else:
                            new_non_tensor_batch[key] = value
                    batch.non_tensor_batch = new_non_tensor_batch
                
                batch.batch = new_batch
                self._selected_indices = selected_indices.cpu().numpy()

                is_originals = [maybe_balanced_full_batch["is_original"][i].item() for i in range(len(prompts_text))]
                is_editeds = [maybe_balanced_full_batch["is_edited"][i].item() for i in range(len(prompts_text))]
                is_selected = [i in self._selected_indices for i in range(len(prompts_text))]

                # 保存选择数据
                self._save_selection_data(
                    prompts_text, 
                    raw_prompts, 
                    responses_text, 
                    feedback_texts, 
                    final_scores_list,
                    is_originals,
                    is_editeds,
                    is_selected
                )


        # if batch filtering is performed, filter reward_tensor and reward_extra_infos_dict  
        # if self.select_fun == "sample_edit_reward" and hasattr(self, '_selected_indices'):  
        if (self.select_fun == "sample_edit_reward" or 
            self.select_fun == "select_edited_if_better" or 
            self.select_fun == "rank_by_reward" or 
            self.select_fun == "rank_by_improvement" or
            self.select_fun == "group_rank_by_improvement") and hasattr(self, '_selected_indices'):
            # filter reward_tensor
            reward_tensor = reward_tensor[self._selected_indices]
            
            # filter reward_extra_infos_dict  
            if reward_extra_infos_dict:  
                filtered_reward_extra_infos_dict = {}  
                for key, value_list in reward_extra_infos_dict.items():  
                    if isinstance(value_list, list) and len(value_list) == self._original_batch_size:  
                        # 根据 selected_indices 过滤列表
                        filtered_value_list = [value_list[i] for i in self._selected_indices]  
                        filtered_reward_extra_infos_dict[key] = filtered_value_list  
                    else:  
                        filtered_reward_extra_infos_dict[key] = value_list  
                reward_extra_infos_dict = filtered_reward_extra_infos_dict
        return batch, reward_tensor,reward_extra_infos_dict
        


    def reset_actor_states_to_ref(self, synchronize_ref=True):
        """Reset the actor states to the reference model and clear the optimizer state.
        
        Args:
            synchronize_ref: Whether to synchronize the reference model with the actor model
        """
        print(f"[reset_actor_states_to_ref] Starting with synchronize_ref={synchronize_ref}")
        
        if synchronize_ref:
            assert False, 'deprecated'
            try:
                print("[reset_actor_states_to_ref] Saving actor model state to temporary file...")
                # Save actor state to temporary file
                actor_state_path = f"verl/reset_tmp/actor_state_mid_sft_rollout5_kl0_freq{self.global_steps}_2"
                self.actor_rollout_wg.save_checkpoint(actor_state_path)
                print(f"[reset_actor_states_to_ref] Actor state saved to: {actor_state_path}")
                
                print("[reset_actor_states_to_ref] Loading actor state into reference model...")
                # Load to reference model
                self.ref_policy_wg.load_checkpoint(actor_state_path, None, True)
                print("[reset_actor_states_to_ref] Reference model weights updated successfully")
                
            except Exception as e:
                print(f"[reset_actor_states_to_ref] Error during synchronization: {e}")
                print("[reset_actor_states_to_ref] Continuing with optimizer reset...")
        
        try:
            print("[reset_actor_states_to_ref] Resetting actor optimizer state...")
            # Reset optimizer state regardless of synchronization success
            self.actor_rollout_wg.reset_optimizer_state()
            print("[reset_actor_states_to_ref] Actor optimizer state reset successfully")
        except Exception as e:
            print(f"[reset_actor_states_to_ref] Error resetting optimizer: {e}")
        
        print("[reset_actor_states_to_ref] Complete")

    def _apply_selection_strategy_in_gen(self, batch, gen_batch_output):
        """
        apply different pipeline in generation
        according to 
        self.select_fun (when self.enable_edit)


        input:batch, gen_batch_output
        output:batch 
        """
        if not self.enable_edit:
            raise ValueError('should enable edit')       


        if self.select_fun in ["select_edited_if_better", "sample_edit_reward", "rank_by_improvement",
                                "only_edit", "random", "max_distance","select_best","rank_by_reward",
                                "group_rank_by_improvement"]:
            # batch = batch.repeat(repeat_times=2, interleave=False)
            batch = batch.repeat(repeat_times=2, interleave=True)
            batch = batch.union(gen_batch_output)

        elif self.select_fun == "sample_edit":
                            
            print("----------------self.global_steps---------------")
            print(self.global_steps)

            batch = batch.repeat(repeat_times=2, interleave=False) # 512

            
            # 检查gen_batch_output中的标记
            if hasattr(gen_batch_output, 'batch') and "is_original" in gen_batch_output.batch:
                is_orig = gen_batch_output.batch["is_original"]
                is_edit = gen_batch_output.batch["is_edited"]


            batch = batch.union(gen_batch_output)


            if "is_original" in gen_batch_output.batch and "is_edited" in gen_batch_output.batch:
                new_batch = TensorDict(
                    {
                        **batch.batch,
                        "is_original": gen_batch_output.batch["is_original"],
                        "is_edited": gen_batch_output.batch["is_edited"]
                    },
                    batch_size=gen_batch_output.batch.batch_size
                )
                batch.batch = new_batch

                original_indices = torch.where(batch.batch["is_original"])[0]
                edited_indices = torch.where(batch.batch["is_edited"])[0]

            # 保留所有原始回答
            selected_indices = original_indices.tolist()
            
            # 从编辑后的回答中选择一部分
            num_to_select = int(len(edited_indices) * self.edit_sample_prob)                            
            
            m = self.config.trainer.n_gpus_per_node
            n = self.config.trainer.nnodes
            m = m * n
            num_to_select = (num_to_select // m) * m


            # 随机选择编辑后的回答
            selected_edited = edited_indices[torch.randperm(len(edited_indices))[:num_to_select]]
            selected_indices.extend(selected_edited.tolist())

            new_batch = TensorDict({}, batch_size=[len(selected_indices)])
            
            for key in batch.batch.keys():
                if isinstance(batch.batch.get(key), torch.Tensor):
                    selected_data = batch.batch.get(key)[selected_indices]
                    new_batch.set(key, selected_data)
                else:
                    new_batch.set(key, batch.batch.get(key))
            
            if batch.non_tensor_batch:
                new_non_tensor_batch = {}
                for key, value in batch.non_tensor_batch.items():
                    if isinstance(value, np.ndarray):
                        # selected_data = value[selected_indices.cpu().numpy()]
                        selected_data = value[selected_indices]
                        if len(selected_data.shape) == 2:  # 对于二维数组
                            new_non_tensor_batch[key] = selected_data
                        else:  # 对于一维数组
                            new_non_tensor_batch[key] = selected_data
                    else:
                        new_non_tensor_batch[key] = value
                
                batch.non_tensor_batch = new_non_tensor_batch
            
            batch.batch = new_batch

        
        else:
            raise ValueError(f"Unknown selection function: {self.select_fun}")
        
        
        return batch
        

    def _save_selection_data(self, prompts_text, raw_prompts, responses_text, feedback_texts, final_scores_list, 
                        is_originals, is_editeds, is_selected=None):
        """save the selection situation to file
        
        Args:
            prompts_text: prompt text list
            raw_prompts: raw prompt text list
            responses_text: response text list
            feedback_texts: feedback text list
            final_scores_list: score list
            is_originals: boolean list indicating whether the answer is original
            is_editeds: boolean list indicating whether the answer is edited
            is_selected: boolean list indicating whether the answer is selected, default all selected
        """
        if not (self.save_data and self.save_path):
            return
        
        # 如果没有提供is_selected，则默认所有样本都被选中
        if is_selected is None:
            is_selected = [True] * len(prompts_text)
            
        save_data = []
        for i in range(len(prompts_text)):
            data_entry = {
                "prompt": prompts_text[i],
                "raw_prompt": raw_prompts[i] if i < len(raw_prompts) else None,
                "response": responses_text[i],
                "feedback": feedback_texts[i] if i < len(feedback_texts) else None,
                "score": final_scores_list[i],
                "is_original": is_originals[i],
                "is_edited": is_editeds[i],
                "is_selected": is_selected[i],
                "timestamp": datetime.now().isoformat()
            }
            save_data.append(data_entry)
                    
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, f"{self.select_fun}_responses_{self.save_counter}.json")
        self.save_counter += 1

        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        # print(f'saved to \n{save_file}')

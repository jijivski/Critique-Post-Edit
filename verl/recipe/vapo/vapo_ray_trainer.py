# not used in this release, but maybe someone could be interested

# RayPPOTrainer, copy from ppo, and like dapo, just import them


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

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
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
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          _timer, apply_kl_penalty,
                                          compute_response_mask)


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




# class AdvantageEstimator(str, Enum):
#     """
#     Using an enumeration class to avoid spelling errors in adv_estimator
#     """

#     GAE = "gae"
#     GRPO = "grpo"
#     REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
#     REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
#     REMAX = "remax"
#     RLOO = "rloo"
#     GRPO_PASSK = "grpo_passk"


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


# def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
#     responses = data.batch["responses"]
#     response_length = responses.size(1)
#     token_level_scores = data.batch["token_level_scores"]
#     batch_size = data.batch.batch_size[0]

#     if multi_turn:
#         loss_mask = data.batch["loss_mask"]
#         response_mask = loss_mask[:, -response_length:]
#     else:
#         attention_mask = data.batch["attention_mask"]
#         response_mask = attention_mask[:, -response_length:]

#     # compute kl between ref_policy and current policy
#     # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
#     kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
#     kld = kld * response_mask
#     beta = kl_ctrl.value

#     token_level_rewards = token_level_scores - beta * kld

#     current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
#     current_kl = torch.mean(current_kl, dim=0).item()

#     # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
#     kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
#     data.batch["token_level_rewards"] = token_level_rewards

#     metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

#     return data, metrics


# def compute_response_mask(data: DataProto):
#     responses = data.batch["responses"]
#     response_length = responses.size(1)
#     attention_mask = data.batch["attention_mask"]
#     return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        # this time lam is a dic
        # breakpoint()
        if "critic_lam_vapo" in lam and 'actor_lam_vapo' in lam:
            values = data.batch['values']
            # values
            # tensor([[3.4219, 4.0312, 1.8906,  ..., 0.0000, 0.0000, 0.0000],
            #         [4.2500, 0.9570, 1.3281,  ..., 0.0000, 0.0000, 0.0000],
            #         [3.4219, 4.0312, 1.8906,  ..., 0.0000, 0.0000, 0.0000],
            #         [4.2500, 5.9375, 5.5938,  ..., 0.0000, 0.0000, 0.0000]],
            #     dtype=torch.bfloat16)
            # values.shape
            # torch.Size([4, 4096])


            if lam["actor_lam_vapo"]=='adaptive':
                print("Using GAE with VAPO with adaptive lambda in actor",lam)
                advantages, _ = core_algos.compute_gae_advantage_return_adaptive_lambda_vapo(token_level_rewards=data.batch['token_level_rewards'],values=data.batch['values'],response_mask=data.batch['response_mask'],gamma=gamma,lam=lam['actor_lam_vapo'])
                            
                # advantages
                # tensor([[ 0.0126, -0.3281,  0.7891,  ...,  0.1377,  0.1377,  0.1377],
                #         [ 0.1348,  2.0781,  2.2656,  ...,  0.1377,  0.1377,  0.1377],
                #         [ 0.1030, -0.2402,  0.9023,  ...,  0.1377,  0.1377,  0.1377],
                #         [ 1.2031,  0.4082,  0.6562,  ...,  0.1377,  0.1377,  0.1377]],
                #     dtype=torch.bfloat16)
            else:
                print("Using GAE with VAPO",lam)
                advantages, _ = core_algos.compute_gae_advantage_return(
                    token_level_rewards=data.batch['token_level_rewards'],
                    values=data.batch['values'],
                    response_mask=data.batch['response_mask'],
                    gamma=gamma,
                    lam=lam['actor_lam_vapo'])
            
            
            
            data.batch['advantages'] = advantages
            values = data.batch['values']
            _, critic_reutrn_vapo = core_algos.compute_gae_advantage_return(token_level_rewards=data.batch['token_level_rewards'],values=data.batch['values'],response_mask=data.batch['response_mask'],gamma=gamma,lam=lam['critic_lam_vapo'])
            # critic_reutrn_vapo
            # tensor([[ 0.1562,  0.1562,  0.1562,  ...,  0.0000,  0.0000,  0.0000],
            #         [-4.0625, -4.0312, -4.0625,  ...,  0.0000,  0.0000,  0.0000],
            #         [ 0.6562,  0.6562,  0.6562,  ...,  0.0000,  0.0000,  0.0000],
            #         [-5.3125, -5.3125, -5.2812,  ...,  0.0000,  0.0000,  0.0000]],
            #     dtype=torch.bfloat16)
            # critic_reutrn_vapo.shape
            # torch.Size([4, 4096])
            # data.batch['critic_reutrn_vapo'] = critic_reutrn_vapo
            data.batch['returns'] = critic_reutrn_vapo

        else:# orig 
            values = data.batch['values']
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch['token_level_rewards'],
                values=data.batch['values'],
                response_mask=data.batch['response_mask'],
                gamma=gamma,
                lam=lam)
            data.batch['advantages'] = advantages
            data.batch['returns'] = returns


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


# @contextmanager
# def _timer(name: str, timing_raw: Dict[str, float]):
#     with Timer(name=name, logger=None) as timer:
#         yield
#     if name not in timing_raw:
#         timing_raw[name] = 0
#     timing_raw[name] += timer.last



                                        #   compute_advantage,
                                          


class RayVAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

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

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

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

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    self.enable_edit = self.config.actor_rollout_ref.rollout.get("edit_config", {}).get("enable_edit", False)
                    self.select_fun = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("select_fun", None)
                    self.save_data = self.config.get("save_data", False)
                    self.save_path = self.config.get("save_path", "")
                    self.edit_sample_prob = self.config.actor_rollout_ref.rollout.get("select_config", {}).get("edit_sample_prob", 0.2)
                    print("---------------enable_edit---------------")
                    print(self.enable_edit)
                    print("---------------select_fun---------------")
                    print(self.select_fun)
                    
                    if self.enable_edit:
                        if self.select_fun == "sample_edit":
                            print("------------gen_batch_output_length------------------")
                            print(len(gen_batch_output.batch["responses"]))
                            print("------------batch_length------------------")
                            print(len(batch.batch))                            
                            
                            # Get the first half as original responses
                            half_size = len(gen_batch_output.batch["responses"]) // 2
                            original_responses = gen_batch_output.batch["responses"][:half_size]
                            edited_responses = gen_batch_output.batch["responses"][half_size:]
                            
                            print("------------dimensions------------------")
                            print(f"original_responses shape: {original_responses.shape}")
                            print(f"edited_responses shape: {edited_responses.shape}")
                            
                            # 计算需要选择的编辑样本数量，确保是8的倍数
                            total_edited = edited_responses.size(0)
                            num_to_select = (total_edited // 8) * 8  # 向下取整到最近的8的倍数
                            
                            # 生成随机索引
                            # indices = torch.randperm(total_edited)[:num_to_select]
                            # selected_edited = edited_responses[indices]

                            indices = torch.randperm(total_edited)[:num_to_select] + half_size
                            selected_edited = edited_responses[indices - half_size]  

                            print(f"selected_edited shape: {selected_edited.shape}")
                            print(f"indices shape: {indices.shape}")
                            print(f"indices values: {indices}")
                            print(f"total_edited: {total_edited}")
                            print(f"num_to_select: {num_to_select}")

                            # Combine original and selected edited responses
                            final_responses = torch.cat([original_responses, selected_edited], dim=0)
                            print(f"final_responses shape: {final_responses.shape}")
                            print(f"gen_batch_output.batch['responses'] shape: {gen_batch_output.batch['responses'].shape}")
                            
                            # Create new tensors for each field
                            new_responses = final_responses
                            new_prompts = torch.cat([
                                gen_batch_output.batch["prompts"][:half_size],
                                # gen_batch_output.batch["prompts"][half_size:][indices]
                                gen_batch_output.batch["prompts"][half_size:][indices - half_size]
                            ], dim=0)
                            
                            # Handle input_ids
                            new_input_ids = torch.cat([new_prompts, new_responses], dim=-1)
                            
                            # Handle position_ids and attention_mask
                            response_length = new_responses.size(1)
                            delta_position_id = torch.arange(1, response_length + 1, device=gen_batch_output.batch["position_ids"].device)
                            delta_position_id = delta_position_id.unsqueeze(0).expand(new_responses.size(0), -1)
                            if gen_batch_output.batch["position_ids"].dim() == 3:  # qwen2vl mrope
                                delta_position_id = delta_position_id.view(new_responses.size(0), 1, -1).expand(new_responses.size(0), 3, -1)
                            
                            original_position_ids = torch.cat([
                                gen_batch_output.batch["position_ids"][:half_size],
                                # gen_batch_output.batch["position_ids"][half_size:][indices]
                                gen_batch_output.batch["position_ids"][half_size:][indices - half_size]
                            ], dim=0)
                            response_position_ids = original_position_ids[..., -1:] + delta_position_id
                            new_position_ids = torch.cat([original_position_ids, response_position_ids], dim=-1)
                            
                            original_attention_mask = torch.cat([
                                gen_batch_output.batch["attention_mask"][:half_size],
                                # gen_batch_output.batch["attention_mask"][half_size:][indices]
                                gen_batch_output.batch["attention_mask"][half_size:][indices - half_size]
                            ], dim=0)
                            print("-------------self.tokenizer.eos_token_id-------------")
                            print(self.tokenizer.eos_token_id)
                            response_attention_mask = get_response_mask(response_id=new_responses, eos_token=self.tokenizer.eos_token_id, dtype=original_attention_mask.dtype)
                            new_attention_mask = torch.cat((original_attention_mask, response_attention_mask), dim=-1)
                            
                            # Create a new TensorDict with the correct batch size
                            from tensordict import TensorDict
                            new_batch = TensorDict({
                                "responses": new_responses,
                                "prompts": new_prompts,
                                "input_ids": new_input_ids,
                                "position_ids": new_position_ids,
                                "attention_mask": new_attention_mask
                            }, batch_size=[new_responses.size(0)])
                            
                            # Update the batch
                            gen_batch_output.batch = new_batch
                            
                            # Update non_tensor_batch if it exists
                            if gen_batch_output.non_tensor_batch:
                                new_non_tensor_batch = {}
                                for key, value in gen_batch_output.non_tensor_batch.items():
                                    if isinstance(value, np.ndarray):
                                        # 获取原始值
                                        original_values = value[:half_size]
                                        # 获取编辑后的值
                                        # edited_values = value[half_size:][indices]
                                        edited_values = value[half_size:][indices - half_size]
                                        
                                        # 检查维度并确保正确连接
                                        if len(original_values.shape) == 2:  # 对于 raw_prompt 这样的二维数组
                                            new_non_tensor_batch[key] = np.concatenate([original_values, edited_values], axis=0)
                                        else:  # 对于 tools_kwargs 这样的一维数组
                                            new_non_tensor_batch[key] = np.concatenate([original_values, edited_values])
                                            
                                        # 添加调试信息
                                        print(f"=== Processing {key} ===")
                                        print(f"Original shape: {original_values.shape}")
                                        print(f"Edited shape: {edited_values.shape}")
                                        print(f"Final shape: {new_non_tensor_batch[key].shape}")
                                        print("---")
                                    else:
                                        new_non_tensor_batch[key] = value
                                gen_batch_output.non_tensor_batch = new_non_tensor_batch
                            
                            if self.save_data and self.save_path:
                                import datetime
                                import json
                                import os
                                
                                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
                                original_responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
                                edited_responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in edited_responses]
                                
                                save_data = []
                                for i in range(len(prompts_text)):
                                    data_entry = {
                                        "prompt": prompts_text[i],
                                        "original_response": original_responses_text[i],
                                        "edited_response": edited_responses_text[i],
                                        "selected": bool(i < len(selected_edited))
                                    }
                                    save_data.append(data_entry)
                                
                                os.makedirs(self.save_path, exist_ok=True)
                                if not hasattr(self, 'save_counter'):
                                    self.save_counter = 0
                                save_file = os.path.join(self.save_path, f"responses_{self.save_counter}.json")
                                self.save_counter += 1
                                with open(save_file, 'w', encoding='utf-8') as f:
                                    json.dump(save_data, f, ensure_ascii=False, indent=2)

                            # Ensure batch sizes match before union
                            print(f"batch.batch.batch_size: {batch.batch.batch_size}")
                            print(f"gen_batch_output.batch.batch_size: {gen_batch_output.batch.batch_size}")
                            
                            # If batch sizes don't match, adjust batch to match gen_batch_output
                            # if batch.batch.batch_size != gen_batch_output.batch.batch_size:
                            #     # Create a new batch with the same size as gen_batch_output
                            #     new_batch = TensorDict({}, batch_size=gen_batch_output.batch.batch_size)
                            #     for key in batch.batch:
                            #         if isinstance(batch.batch[key], torch.Tensor):
                            #             if batch.batch[key].size(0) > gen_batch_output.batch.batch_size[0]:
                            #                 new_batch[key] = batch.batch[key][:gen_batch_output.batch.batch_size[0]]
                            #             else:
                            #                 original_values = batch.batch[key][:half_size]
                            #                 # edited_values = batch.batch[key][half_size:][indices]
                            #                 edited_values = gen_batch_output.batch[key][half_size:][indices - half_size]
                            #                 new_batch[key] = torch.cat([original_values, edited_values], dim=0)
                            #         else:
                            #             new_batch[key] = batch.batch[key]
                            #     batch.batch = new_batch

                                # if batch.non_tensor_batch:
                                #     new_non_tensor_batch = {}
                                #     for key, value in batch.non_tensor_batch.items():
                                #         print("----------key----------")
                                #         print(key)
                                #         if isinstance(value, np.ndarray):
                                #             if self.select_fun == 'sample_edit':
                                #                 original_values = value[:half_size]
                                #                 # 检查key是否存在于gen_batch_output.non_tensor_batch中
                                #                 if key in gen_batch_output.non_tensor_batch:
                                #                     edited_values = gen_batch_output.non_tensor_batch[key][half_size:][indices - half_size]
                                #                     if len(original_values.shape) == 2:  # 对于 raw_prompt 这样的二维数组
                                #                         new_non_tensor_batch[key] = np.concatenate([original_values, edited_values], axis=0)
                                #                     else:  # 对于 tools_kwargs 这样的一维数组
                                #                         new_non_tensor_batch[key] = np.concatenate([original_values, edited_values])
                                #                 else:
                                #                     # 如果key不存在，则使用原始值
                                #                     new_non_tensor_batch[key] = value
                                                                                                        
                                #             else:
                                #                 new_non_tensor_batch[key] = value
                                #         else:
                                #             new_non_tensor_batch[key] = value

                            
                            if batch.batch.batch_size != gen_batch_output.batch.batch_size:
                                # Create a new batch with the same size as gen_batch_output
                                new_batch = TensorDict({}, batch_size=gen_batch_output.batch.batch_size)
                                for key in batch.batch:
                                    if isinstance(batch.batch[key], torch.Tensor):
                                        # 确保索引不会越界
                                        assert (indices - half_size).max() < batch.batch[key].size(0), f"indices - half_size max ({(indices - half_size).max()}) >= tensor size ({batch.batch[key].size(0)}) for key {key}"
                                        original_values = batch.batch[key]
                                        edited_values = batch.batch[key][indices - half_size]
                                        new_batch[key] = torch.cat([original_values, edited_values], dim=0)
                                    else:
                                        new_batch[key] = batch.batch[key]
                                batch.batch = new_batch

                                if batch.non_tensor_batch:
                                    new_non_tensor_batch = {}
                                    for key, value in batch.non_tensor_batch.items():
                                        print("----------key----------")
                                        print(key)
                                        if isinstance(value, np.ndarray):
                                            if self.select_fun == 'sample_edit':
                                                # 确保索引不会越界
                                                assert (indices - half_size).max() < len(value), f"indices - half_size max ({(indices - half_size).max()}) >= array length ({len(value)}) for key {key}"
                                                original_values = value
                                                edited_values = value[indices - half_size]

                                                if len(original_values.shape) == 2:  # 对于 raw_prompt 这样的二维数组
                                                    new_non_tensor_batch[key] = np.concatenate([original_values, edited_values], axis=0)
                                                else:  # 对于 tools_kwargs 这样的一维数组
                                                    new_non_tensor_batch[key] = np.concatenate([original_values, edited_values])                                                            
                                            else:
                                                new_non_tensor_batch[key] = value
                                        else:
                                            new_non_tensor_batch[key] = value                                
                                    batch.non_tensor_batch = new_non_tensor_batch

                                    print("After adjustment, non_tensor_batch sizes:")
                                    for key, value in batch.non_tensor_batch.items():
                                        if isinstance(value, np.ndarray):
                                            print(f"{key}: {value.shape}")

                            # 确保所有必要的字段都存在
                            required_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
                            for key in required_keys:
                                assert key in batch.batch or key in gen_batch_output.batch, f"Required key {key} not found in either batch.batch or gen_batch_output.batch"
                                if key not in batch.batch and key in gen_batch_output.batch:
                                    batch.batch[key] = gen_batch_output.batch[key]

                            batch = batch.union(gen_batch_output)
            
                        elif self.select_fun in ["only_edit", "random", "max_distance","select_best"]:
                            batch = batch.union(gen_batch_output)
                        else:
                            batch = batch.repeat(repeat_times=2, interleave=False)
                            batch = batch.union(gen_batch_output)
                    else:
                        batch = batch.union(gen_batch_output)



                    # else:
                    #     # Align the batch
                    #     traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                    #     batch = batch[:traj_bsz]
                    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n

                    print('verl/recipe/vapo/vapo_ray_trainer.py L701')

                    if len(batch)>traj_bsz and self.config.actor_rollout_ref.rollout.get("filter_overlen_ngram",False) :
                        # filter 
                        # batch_??(traj_bsz=traj_bsz,max_len=,n_gram_selection_dic,)
                        from verl.utils.select import \
                            filter_batch_by_ngram_maxlen
                        ngram_thresholds = self.config.actor_rollout_ref.rollout.get("ngram_thresholds", {2: 0.2, 3: 0.12})  
                        print(f'{ngram_thresholds=}')
                        # breakpoint()
                        batch = filter_batch_by_ngram_maxlen(batch, traj_bsz=traj_bsz, max_len=self.config.data.get("max_response_length", None), ngram_thresholds=ngram_thresholds)
                        if len(batch)!=traj_bsz:
                            print(f'{len(batch)=}')
                            breakpoint()
                        



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
                            # breakpoint()
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


                        actor_lam_vapo = self.config.algorithm.get('actor_lam_vapo',None)  
                        critic_lam_vapo = self.config.algorithm.get('critic_lam_vapo',None)  
                        # breakpoint()
                        if actor_lam_vapo is not None and critic_lam_vapo is not None:  
                            lam = {'actor_lam_vapo': actor_lam_vapo, 'critic_lam_vapo': critic_lam_vapo}  
                        else:  
                            lam = self.config.algorithm.lam

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=lam,
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
                            threshold = self.config.algorithm.get("NLL_threshold", 0.5)  
                            print(f"vapo NLL pos {threshold=}")
                            # positive_mask = (batch.batch["reward"][:,-1] > threshold).float()  # 对。。对吗
                            token_level_scores = batch.batch["token_level_scores"]  
                            # token_level_scores2 = batch.batch["token_level_rewards"]  
                            # (token_level_scores2 * response_mask).sum(dim=-1)  

                            response_mask = batch.batch["response_mask"]
                            final_scores = (token_level_scores * response_mask).sum(dim=-1)  
                            NLL_loss_mask = (final_scores > threshold).float()
                            batch.batch["NLL_loss_mask"] = NLL_loss_mask
                            # tensor([0., 1., 0., 0.])
                            
                            # breakploint()
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
                    
                    
                    # update reference model with actor
                    if (
                        self.use_reference_policy
                        and self.config.actor_rollout_ref.ref.sync_actor
                        and self.global_steps % self.config.actor_rollout_ref.ref.sync_steps == 0
                    ):
                        with _timer("update_ref", timing_raw):
                            self.ref_policy_wg.sync_with_actor(
                                self.actor_rollout_wg.export_actor_weights(),
                                self.config.actor_rollout_ref.ref.sync_actor_alpha,
                            )

                            
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

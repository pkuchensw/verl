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

import json
import logging
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging.version import InvalidVersion, Version
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_data_metrics,
    compute_moe_lb_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    get_metric_data_with_optional_routed_experts,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_spec_decode_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    Role,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.trainer.ppo.v1.precision_escalation import PrecisionEscalationController
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY, ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator, compute_advantage_for_multi_trajectories
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_type
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip import SkipManager
from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger
from verl.workers.config import CriticConfig, DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import response_from_nested, response_to_nested


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _tq_supports_checkpoint() -> bool:
    """Whether the installed TransferQueue can snapshot/restore its state for checkpoint consistency."""
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


class PPOTrainer(ABC):
    """Base class for PPO trainer.

    Args:
        config: DictConfig from yaml config file.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.trainer_mode = self.config.trainer.v1.trainer_mode
        self.parameter_sync_step = self.config.trainer.v1.get(self.trainer_mode, {}).get("parameter_sync_step", 1)
        self.replay_buffer = self._build_replay_buffer()
        self._rollout_moe_lb_metrics_accumulator = RolloutMoELoadBalanceMetricsAccumulator(
            model_config=self.config.actor_rollout_ref.model
        )
        # Algorithm control layer for mismatch-driven precision escalation. It only
        # tracks rollbacks and decides when a batch is worth regenerating on a
        # high-precision engine; the actual engine swap lives in the trainer
        # subclass (see engine_precision_switch.py).
        self.precision_escalation = PrecisionEscalationController(
            self.config.algorithm.get("mismatch_precision_escalation", None)
        )

        # Score Centering: propagate the algorithm-level switch onto the actor config (what
        # ppo_loss receives) and the top-k width onto the rollout config. Neither key is
        # declared in the component yaml schema, so they must be added under open_dict
        # (the Hydra config is in struct mode, which rejects new keys otherwise).
        sc_cfg = self.config.algorithm.get("score_centering", None)
        sc_enabled = sc_cfg is not None and sc_cfg.get("enable", False)
        if sc_enabled:
            if int(sc_cfg.get("topk", 0)) <= 0:
                raise ValueError("algorithm.score_centering.topk must be > 0 when score_centering is enabled")
            if self.config.actor_rollout_ref.model.get("use_fused_kernels", False):
                raise NotImplementedError(
                    "score_centering requires full logits; use_fused_kernels=True is not supported"
                )
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.score_centering = (
                OmegaConf.to_container(sc_cfg) if sc_cfg is not None else {}
            )
            self.config.actor_rollout_ref.rollout.score_centering_topk = int(
                sc_cfg.get("topk", 0) if sc_enabled else 0
            )

    def _build_replay_buffer(self) -> ReplayBuffer:
        """Instantiate the replay buffer (or a user-provided custom sampler).

        Set ``trainer.v1.sampler.custom_sampler.{path,name}`` to plug in a custom
        ``ReplayBuffer`` subclass; otherwise the built-in implementation is used.
        """
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        has_custom_sampler = bool(
            custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name")
        )
        if has_custom_sampler:
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
        else:
            sampler_cls = ReplayBuffer if self.trainer_mode == "sync" else ReplayBufferAsync

        replay_buffer_kwargs = dict(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._add_prompts_to_generate,
        )
        # Preserve the existing constructor contract for external samplers; custom implementations own
        # their filtering semantics and can consume algorithm.filter_groups through their own config.
        if not has_custom_sampler:
            filter_groups_metric = self._resolve_filter_groups_metric()
            sync_refill_failed_groups = bool(sampler_config.get("sync_refill_failed_groups", False))
            replay_buffer_kwargs.update(
                filter_groups_metric=filter_groups_metric,
                sync_refill_failed_groups=sync_refill_failed_groups,
            )
            # Both buffer variants need the batch geometry: ReplayBuffer for its
            # in-flight/Sync-DAPO bounds, ReplayBufferAsync for the starved-pool
            # self-heal refill size (dispatching a full gen-batch rebuilds surplus).
            train_batch_size = self.config.data.train_batch_size
            replay_buffer_kwargs.update(
                train_batch_size=train_batch_size,
                gen_batch_size=1
                if filter_groups_metric is not None or sync_refill_failed_groups
                else (self.config.data.get("gen_batch_size", None) or train_batch_size),
            )
            if sampler_cls is ReplayBuffer:
                filter_groups = self.config.algorithm.get("filter_groups", None)
                max_inflight_gen_batches = 1
                if filter_groups_metric is not None:
                    max_inflight_gen_batches = filter_groups.get("max_inflight_gen_batches", 1)
                replay_buffer_kwargs.update(
                    max_inflight_gen_batches=max_inflight_gen_batches,
                )
        return sampler_cls(**replay_buffer_kwargs)

    def _resolve_filter_groups_metric(self) -> str | None:
        """Resolve DAPO's group metric and verify that rollout computes it before sampling."""
        filter_groups = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        if not filter_enabled:
            return None

        filter_metric = filter_groups.get("metric", None)
        if not filter_metric:
            raise ValueError("algorithm.filter_groups.metric must be set when group filtering is enabled")

        reward_model = self.config.reward.reward_model
        streaming_reward_path = not reward_model.enable or reward_model.enable_resource_pool
        assert streaming_reward_path, (
            "algorithm.filter_groups requires the reward metric at sampling time: use rule-based reward or "
            "reward.reward_model.enable_resource_pool=True. A colocated reward model computes rewards only "
            "after replay-buffer sampling."
        )
        max_num_gen_batches = filter_groups.get("max_num_gen_batches", 0)
        if max_num_gen_batches > 0:
            logger.warning(
                "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
                "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
                max_num_gen_batches,
            )
        return str(filter_metric)

    def init(self):
        """Initialize all components of the trainer.

        1. WorkerGroup: actor, critic, reference with model engine: FSDP/Megatron/VeOmni/...
        2. LLMServerManager: launch and manage LLM server replicas for generation.
        3. CheckpointEngineManager: sync weights between worker group and LLM server replicas.
        4. RewardLoopManager: reward workers for rule-based reward, optional LLM server for model-based reward.
        5. [Optional] MultiTeacherModelManager: LLM teacher servers for on-policy distillation.
        """
        self._setup()
        self.on_init_end()

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu

            # Wire the critic profiler config via the hydra path (real dataclass tool_config), so the
            # standalone critic TrainingWorker gets a working DistProfiler instead of a silent no-op.
            critic_omega_profiler_config = self.config.critic.get("profiler", {})
            critic_profiler_config = (
                omega_conf_to_dataclass(critic_omega_profiler_config) if critic_omega_profiler_config else None
            )

            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
                profiler_config=critic_profiler_config,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initialize critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize teacher loop manager
        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # 9. initialize agent loop manager
        self.llm_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # 10. initialize checkpoint engine manager
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_engine_config.backend = "naive"
        self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()

        logger.info("all initialize finished, ready to fit")

    def get_llm_client(self) -> LLMServerClient:
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client()

    def get_teacher_client(self) -> Optional[dict[str, LLMServerClient]]:
        """Get the On-Policy Distillation teacher server clients.

        Returns:
            dict[str, LLMServerClient]: The teacher server clients.
        """
        return self.teacher_model_manager.get_client() if self.use_teacher_policy else None

    def get_reward_handles(self) -> list[ray.actor.ActorHandle]:
        """Get the handles of reward loop workers."""
        return self.reward_loop_manager.reward_loop_worker_handles

    def fit(self, agent_loop_manager: AgentLoopManager):
        """Fit the trainer with the agent loop manager.

        Args:
            agent_loop_manager: The agent loop manager to generate sequences.
        """
        self.agent_loop_manager = agent_loop_manager

        # initialize SkipManager for V1 rollout skip support
        SkipManager.init(self.config)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.dapo_filtered_reward_logger = DapoFilteredRewardTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            self.on_validate_begin()
            val_metrics = self._validate()
            self.on_validate_end()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        current_epoch = self.global_steps // self.steps_per_epoch
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        # SkipManager skips warmup batches in async trainers, so it doesn't conflict with reissue.
        SkipManager.set_step(self.global_steps)
        self._reissue_inflight_prompts()
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        self.on_train_begin()
        last_val_metrics = None
        while current_epoch < self.config.trainer.total_epochs and self.global_steps <= self.total_training_steps:
            is_last_step = self.global_steps >= self.total_training_steps
            metrics = {}
            self.timing_raw = {}

            # 1. perform rollout and actor/critic training
            with marked_timer("step", self.timing_raw):
                self.on_step_begin()

                self._start_profiling()
                batch = self.step(metrics, self.timing_raw)
                self._stop_profiling()

                # 2. save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", self.timing_raw, color="green"):
                        self._save_checkpoint()

                self.on_step_end()
                metrics.update(self._consume_sync_metrics())

            # 4. validate
            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", self.timing_raw, color="green"):
                    self.on_validate_begin()
                    val_metrics: dict = self._validate()
                    self.on_validate_end()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # 5. record metrics
            self._compute_metrics(batch, metrics, self.timing_raw, global_steps=self.global_steps, epoch=current_epoch)

            # 6. dump rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                self._log_rollout_data(batch, self.timing_raw, rollout_data_dir)

            # 7. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

            dapo_filtered_reward_counts = metrics.pop(DAPO_FILTERED_REWARD_COUNTS_KEY, None)
            self.logger.log(data=metrics, step=self.global_steps)
            if dapo_filtered_reward_counts:
                self.dapo_filtered_reward_logger.log(
                    self.config.trainer.logger, dapo_filtered_reward_counts, self.global_steps
                )
            progress_bar.update(1)
            self.global_steps += 1
            SkipManager.set_step(self.global_steps)
            current_epoch = (self.global_steps - 1) // self.steps_per_epoch
            if is_last_step:
                self._shutdown_dump_executor()
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

        self.on_train_end()
        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        train_batch_size = self.config.data.train_batch_size
        assert train_batch_size % self.parameter_sync_step == 0, (
            f"train_batch_size ({train_batch_size}) must be divisible by "
            f"parameter_sync_step ({self.parameter_sync_step})"
        )
        sample_batch_size = train_batch_size // self.parameter_sync_step

        if not hasattr(self, "_mem_log_interval"):
            self._mem_log_interval = 0
        self._mem_log_interval += 1
        if self._mem_log_interval % max(self.config.trainer.get("mem_log_interval", 5), 1) == 0:
            self._log_mem_usage()

        # Fixed oversample: sample a few extra prompt groups so a bounded number of
        # the most-mismatched groups (at most `oversample`) can be re-dispatched
        # without under-filling the training batch. _screen_and_redispatch_dirty caps
        # its redispatch at this same budget, so the trained batch stays at
        # sample_batch_size regardless of dirty load (128+oversample sampled - <=oversample
        # redispatched = >=128 trained). The oversample must stay small enough that the
        # pipeline can actually produce sample_batch_size + oversample terminal groups
        # per step; a large oversample makes replay_buffer.sample() wait forever for
        # groups the pipeline cannot produce (observed deadlock when oversample grew
        # past ~128).
        gate = self.config.algorithm.get("train_mismatch_gate", {})
        gate_enabled = gate.get("enable", False)
        oversample = gate.get("oversample", 0) if gate_enabled else 0

        self._add_batch_to_generate()

        metrics_aggregator = MetricsAggregator()
        combined_keys: list = []
        combined_tags: list = []
        combined_partition_id = "train"
        step_real_traj = 0
        for idx in range(self.parameter_sync_step):
            iter_metrics: dict = {}
            target = sample_batch_size + oversample
            batch = self._step_once(iter_metrics, timing_raw, target)
            sample_count = sum(not tag.get("is_padding", False) for tag in batch.tags)
            step_real_traj += sample_count
            metrics_aggregator.add_step_metrics(iter_metrics, sample_count=sample_count)
            combined_keys.extend(batch.keys)
            combined_tags.extend(batch.tags)
            combined_partition_id = batch.partition_id

        metrics.update(metrics_aggregator.get_aggregated_metrics())

        if gate_enabled:
            num_repeat = self.config.actor_rollout_ref.rollout.n
            clean_uids = step_real_traj // num_repeat
            logger.info(
                "[mismatch-gate-batch] step=%s oversample=%d trained_real_uids=%d/%d sample_target=%d",
                self.global_steps,
                oversample,
                clean_uids,
                sample_batch_size,
                target,
            )

        return KVBatchMeta(partition_id=combined_partition_id, keys=combined_keys, tags=combined_tags)

    def _log_mem_usage(self) -> None:
        """Log host-RAM footprint (process RSS + cgroup usage/limit) to diagnose OOM kills.

        Training crashes have been traced to driver host RAM near ~54GB. This periodic
        log leaves a trail right before any OOM so the peak and the effective cgroup
        limit are visible in the last lines of the training log.
        """
        try:
            with open("/proc/self/status") as f:
                rss_kb = None
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
            rss_gb = f"{rss_kb / 1048576:.1f}" if rss_kb is not None else "n/a"

            cgroup_path = None
            try:
                with open("/proc/self/cgroup") as f:
                    for line in f:
                        if "memory:" in line:
                            cgroup_path = line.split("memory:", 1)[1].strip()
                            break
            except OSError:
                pass

            limit_gb = cur_gb = "n/a"
            if cgroup_path:
                for p, name in (
                    (f"/sys/fs/cgroup{cgroup_path}/memory.max", "max"),
                    (f"/sys/fs/cgroup{cgroup_path}/memory/memory.limit_in_bytes", "limit_in_bytes"),
                ):
                    try:
                        with open(p) as f:
                            v = f.read().strip()
                        if v:
                            limit_gb = f"{int(v) / 1073741824:.1f}"
                            break
                    except OSError:
                        continue
                for p in (
                    f"/sys/fs/cgroup{cgroup_path}/memory.current",
                    f"/sys/fs/cgroup{cgroup_path}/memory/memory.usage_in_bytes",
                ):
                    try:
                        with open(p) as f:
                            v = f.read().strip()
                        if v:
                            cur_gb = f"{int(v) / 1073741824:.1f}"
                            break
                    except OSError:
                        continue

            logger.info(
                "[mem] step=%s rss=%sGB cgroup_current=%sGB cgroup_limit=%sGB",
                self.global_steps,
                rss_gb,
                cur_gb,
                limit_gb,
            )
        except Exception as e:  # pragma: no cover - diagnostic logging must never crash training
            logger.warning("[mem] failed to log memory usage: %s", e)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        """Run a single local update: sample one mini-batch and perform the full PPO pipeline once."""
        # 1. sample batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            # Precision-escalation "drop": groups the controller rejected are removed by
            # the sampler (cleared + refilled) so they never enter the batch, instead of
            # being rescued by an engine switch. Empty under the "switch" strategy.
            rejected = self.precision_escalation.pending_rejections()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
                exclude_uids=set(rejected) if rejected else None,
            )
            if rejected:
                self._reconcile_precision_rejections(rejected)
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        # 2. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        # 3. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 4. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 4.5 [OPTIONAL] batch-level mismatch screening: drop high-LLR uid groups from
        # this step's training batch and re-dispatch them for regeneration, so every
        # sample reaching the actor update has been screened at least once.
        with marked_timer("mismatch_screen", timing_raw, color="magenta"):
            batch = self._screen_and_redispatch_dirty(batch, metrics=metrics)

        # 5. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 7. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 8. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 9. update actor
        # The groups reaching the actor update are the ones actually trained on this
        # step -- groups the mismatch gate rolled back were already dropped from the
        # batch. Their trajectories are consumed, so drop their rollback counters:
        # this bounds the escalation controller's memory to the in-flight keys and
        # stops a uid's next generation from inheriting a count accrued by the
        # generation just trained. Captured before the actor update in case it
        # reshapes the batch.
        trained_keys = list(batch.keys)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        self.precision_escalation.forget(trained_keys)

        return batch

    def _reconcile_precision_rejections(self, rejected: list[str], partition_id: str = "train") -> None:
        """Release rejected uids whose group has already left the transfer queue.

        A rejected uid disappears from the queue exactly when the sampler drops it
        (it had become terminal). A uid still present was in flight when the batch was
        sampled, so it stays rejected and is retried on the next selection round.
        """
        items = tq.kv_list(partition_id).get(partition_id, {})
        self.precision_escalation.rejections_present([uid for uid in rejected if uid in items])

    # ------------------------------ abstract methods ------------------------------

    def on_init_end(self):
        """Called after the initialization ends."""
        return

    def on_train_begin(self):
        """Called before the training loop starts."""
        return

    def on_train_end(self):
        """Called after the training loop ends."""
        return

    def on_validate_begin(self):
        """Called before the validation loop starts."""
        return

    def on_validate_end(self):
        """Called after the validation loop ends."""
        return

    def on_step_begin(self):
        """Called at the beginning of each training step."""
        return

    @abstractmethod
    def on_step_end(self):
        """Called at the end of each training step."""
        return

    def _consume_sync_metrics(self) -> dict:
        """Weight-sync stats stashed by ``on_step_end`` (e.g. the delta engines'
        changed ratio / wire payload), merged into this step's logged metrics."""
        metrics = getattr(self, "_pending_sync_metrics", None) or {}
        self._pending_sync_metrics = {}
        mtrunc = getattr(self, "_pending_mismatch_trunc_metrics", None) or {}
        self._pending_mismatch_trunc_metrics = {}
        metrics.update(mtrunc)
        metrics.update(self.precision_escalation.metrics())
        return metrics

    def on_sample_begin(self):
        """Called at the beginning of sampling batch from replay buffer."""
        return

    @abstractmethod
    def on_sample_end(self):
        """Called after sampling a batch from replay buffer."""
        return

    # ------------------------------ common methods ------------------------------

    def _get_n_gpus_for_throughput(self) -> int:
        """Return the total number of GPUs used for throughput normalization.

        By default this is the trainer-side GPU count from the resource pool
        manager.  Modes that use additional dedicated GPUs (e.g. separate-async
        standalone rollout) should override this to include them.
        """
        return self.resource_pool_manager.get_n_gpus()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        # Exact refill counts require single-prompt dataloader fetches.
        filter_groups = self.config.algorithm.get("filter_groups", None)
        dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
        sync_refill_failed_groups = bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
        requires_exact_refill = self.trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups
        if requires_exact_refill:
            user_gen_batch_size = self.config.data.get("gen_batch_size", None)
            if user_gen_batch_size not in (None, 1):
                logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
            elif user_gen_batch_size is None:
                logger.info("data.gen_batch_size defaulted to 1.")
            with open_dict(self.config):
                self.config.data.gen_batch_size = 1

        # use gen_batch_size as the batch size for the dataloader if set, otherwise use train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.train_dataloader_it = None
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        self.steps_per_epoch = len(self.train_dataset) // self.config.data.train_batch_size

        # adjust total_training_steps
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        # The LR scheduler steps once per local update, and each global step performs
        # ``parameter_sync_step`` local updates (see ``PPOTrainer.step``). The optimizer's
        # schedule horizon must therefore count optimizer updates.
        optim_total_training_steps = total_training_steps * self.parameter_sync_step
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = optim_total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _init_resource_pool_mgr(self):
        config = self.config
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

        # Add actor rollout worker to mapping
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

        # Add critic worker to mapping.
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            self.mapping[Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # 5. restore TransferQueue state (async modes). Re-issuing the restored in-flight prompts is
        # deferred to fit() to use the agent_loop_manager.
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
            if os.path.exists(tq_ckpt_path):
                logger.info(f"Loading TransferQueue state from {tq_ckpt_path}")
                tq.load_checkpoint(tq_ckpt_path)

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Restart checkpointed pending/running prompt groups from their persisted prompt data."""
        if self.trainer_mode == "sync" or not _tq_supports_checkpoint():
            return 0
        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)

        # Treat this as a new dispatch attempt for the resumed training step.
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

        logger.info(
            f"Re-issued {len(inflight_uids)} in-flight prompts for step {self.global_steps}; "
            f"cleared {len(old_trajectory_keys)} old trajectories from partition {partition_id}"
        )
        return len(inflight_uids)

    def _save_checkpoint(self):
        """Save actor, critic, and dataloader checkpoints to local (and optionally remote) storage."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        logger.info(f"Saving checkpoint to {local_global_step_folder}")

        # resolve max checkpoints to keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # save actor
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)

        # save TransferQueue state for async modes so in-flight prompts (already fetched from the
        # dataloader but not yet trained into this checkpoint's weights) survive a restart:
        # finished trajectories are restored as-is, pending/running prompts are re-issued on resume.
        # Requires a TransferQueue release with checkpoint support (see _tq_supports_checkpoint).
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            tq.save_checkpoint(
                os.path.join(local_global_step_folder, "transfer_queue"),
                metadata={"global_steps": self.global_steps},
            )

        # write latest checkpointed iteration tracker for atomic resume
        actor_ckpt_cfg = self.config.actor_rollout_ref.actor.get("checkpoint", {})
        if actor_ckpt_cfg.get("async_save", False):
            logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _validate(self) -> dict[str, float]:
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            self.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

            uids = data.pop("uid").tolist()
            inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
            scores = data["rm_scores"].sum(dim=1).tolist()

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
            else:
                gts = [None] * len(uids)

            # Sort by uid key ({sample}_{rollout}_{output})
            sort_keys = []
            for key in batch.keys:
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
                else:
                    sort_keys.append((key, 0, 0))
            sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

            inputs = [inputs[i] for i in sorted_indices]
            outputs = [outputs[i] for i in sorted_indices]
            gts = [gts[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]

            reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=rollout_data_dir,
            )

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _fetch_one_gen_batch(self) -> TensorDict:
        """Fetch one ``gen_batch_size`` chunk from the dataloader."""
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int | None = None) -> TensorDict:
        """Fetch and coalesce the requested number of prompts.

        ``num_prompts`` need not be a multiple of ``gen_batch_size``: the surplus
        of the final dataloader chunk is sliced off. This lets replay-buffer refill
        dispatches (eviction or starved-pool self-heal) top up a batch by an exact
        small count instead of demanding whole ``gen_batch_size`` fetches.
        """
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        gen_batch_size = self.config.data.get("gen_batch_size", None) or train_batch_size
        if num_prompts <= 0:
            raise ValueError(f"num_prompts ({num_prompts}) must be positive")

        num_chunks = (num_prompts + gen_batch_size - 1) // gen_batch_size
        chunks = [self._fetch_one_gen_batch() for _ in range(num_chunks)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        if len(batch) > num_prompts:
            batch = batch[:num_prompts]
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Register prompts in TransferQueue and dispatch them for generation."""
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if self.trainer_mode != "sync":
            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id="train",
                tags=tags,
                # Persist prompt data for async checkpoint recovery.
                # TODO: maybe let workers do it?
                fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        """Add an exact number of prompts to the AgentLoopManager."""
        batch = self._next_train_batch(num_prompts)
        return self._submit_batch_to_rollout(batch)

    @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
    def _add_batch_to_generate(self):
        """Add one training batch to the AgentLoopManager."""
        batch = self._next_train_batch()
        self._submit_batch_to_rollout(batch)

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict | None = None) -> KVBatchMeta:
        """Compute the reward score with a colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"

        # 1. read the fields required by the reward model from TransferQueue.
        fields = ["prompts", "responses", "raw_prompt"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        prompt_lengths = data["prompts"].offsets().diff()
        response_lengths = data["responses"].offsets().diff()
        prompts = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        # 2. rebuild the attention mask aligned with the [prompts | responses] layout.
        prompt_mask = self._lengths_to_mask(prompt_lengths, prompts.size(1))
        response_mask = self._lengths_to_mask(response_lengths, responses.size(1))
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

        # `raw_prompt` is a non-tensor field; depending on the TransferQueue backend it
        # comes back as a tensordict LinkedList (a `list` subclass), a NonTensorStack or a
        # numpy array. `list(...)` normalizes all of them to a plain list where each element
        # is one sample's chat-message list (whereas `.tolist()` only exists on numpy/tensors).
        raw_prompts = list(data["raw_prompt"])
        raw_prompt_arr = np.empty(len(raw_prompts), dtype=object)
        raw_prompt_arr[:] = raw_prompts

        rm_input = DataProto(
            batch=TensorDict(
                {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
                batch_size=len(batch),
            ),
            non_tensor_batch={"raw_prompt": raw_prompt_arr},
        )

        # 3. run the reward model (wakes/sleeps the reward model internally).
        rm_output = self.reward_loop_manager.compute_rm_score(rm_input)

        # 4. write rm_scores (and reward extra info) back to TransferQueue.
        padded_rm_scores = rm_output.batch["rm_scores"]
        rm_scores = torch.nested.as_nested_tensor(
            [padded_rm_scores[i, : response_lengths[i]] for i in range(len(batch))],
            layout=torch.jagged,
        )
        write_back = {"rm_scores": rm_scores}
        for key in rm_output.meta_info.get("reward_extra_keys", []):
            write_back[key] = rm_output.non_tensor_batch[key]
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(write_back),
        )

        return batch

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        """Build a right-padded mask of shape (len(lengths), width) from per-row valid lengths."""
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).to(torch.int64)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # Upsampling the batch with padding sequences. Normally the batch is padded up
        # to the full mini-batch multiple so every mini-batch is full. With the gate
        # enabled, dirty redispatch leaves a ragged clean count; padding that to the
        # full multiple would waste a step training synthetic no-op samples. Real
        # trajectory counts are always uid * n (dp-divisible), so padding only to
        # dp_size suffices and the engine's ragged final mini-batch handles the rest.
        gate = self.config.algorithm.get("train_mismatch_gate", {})
        if gate.get("enable", False):
            batch_multiple = dp_size
        else:
            batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        # 1. compute log probs
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        batch = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _screen_and_redispatch_dirty(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Screen sampled trajectories by per-token mismatch and roll back dirty groups.

        Runs after ``_compute_old_log_prob`` and before ``_compute_advantage``. pi_k logprobs
        are reused from ``_compute_old_log_prob`` when it recomputed them (is_mode=tis/mis);
        in bypass mode (is_mode=none) they are recomputed here with ``compute_log_prob``. For
        every sampled trajectory we compute the per-token mismatch

            max_t |pi_k_logprob_t - rollout_logprob_t|

        aligned with ``partial_rollout_mismatch_truncation`` (both reject a sequence when
        some single token deviates from rollout by more than the threshold). A uid group is
        marked dirty when ANY of its trajectories is out of bounds; the whole group is
        dropped from this training step (GRPO advantage normalizes per uid) and re-dispatched
        to the rollout engine with per-session seeds: each session resumes from
        ``b = max(t*-rollback_window, 0)`` (t* = first mismatched token), keeping its prefix
        and only regenerating the tail -- the same local rollback truncation applies to
        in-flight prefixes at weight-sync time. Re-sampling is relative top-K, not a pure
        threshold filter: every sampled group is a candidate (ranked by per-token mismatch),
        and the K most-diverged groups above a soft floor (``min_llr``) are rolled back, so we
        actively pull the most off-policy groups back to pi_k instead of relying on a calibrated
        tau. Groups not re-sampled are trained as-is; every sample that reaches
        ``_compute_advantage`` has therefore been screened.

        Anti-starvation: a uid that is still dirty after ``max_force_accept_regens``
        rollbacks is force-accepted and trained anyway this step.
        """
        gate = self.config.algorithm.get("train_mismatch_gate", None)
        if gate is None or not gate.get("enable", False):
            return batch
        if self.agent_loop_manager is None:
            logger.warning("[train-mismatch-gate] screening skipped: no agent_loop_manager")
            return batch
        if not self.config.actor_rollout_ref.rollout.calculate_log_probs:
            logger.warning("[train-mismatch-gate] screening skipped: rollout.calculate_log_probs is False")
            return batch
        if not batch.keys:
            return batch

        threshold = gate.get("log_ratio_threshold", 20.0)
        max_force_accept = gate.get("max_force_accept_regens", 3)
        rollback_window = gate.get("rollback_window", 32)
        # Soft lower bound for top-K re-sampling: a group below this per-token mismatch is
        # not re-generated even if it ranks in the top-K (it is engine noise, not drift).
        min_llr = gate.get("min_llr", 1.5)
        # Per-session rollback point for the groups redispatch decides to roll back. Rebuilt
        # every screening call so it only ever carries this step's sampled keys; consumed by
        # _redispatch_dirty_uids immediately after. None = no mismatched token (session is
        # clean and is re-inserted verbatim when its group is rolled back).
        rollback_points: dict[str, int | None] = {}
        if not hasattr(self, "_mismatch_gate_regen_count"):
            self._mismatch_gate_regen_count = defaultdict(int)

        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["responses", "response_mask", "rollout_log_probs", "old_log_probs"],
        )

        # Recover uid from the trajectory key "{uid}_{session}_{index}". The uid comes from
        # uuid4().hex (no underscores), so the first segment is always the uid; this mirrors
        # _reissue_inflight_prompts and avoids fragile unwrapping of NonTensorData uids.
        uids = [key.split("_", 1)[0] for key in batch.keys]
        responses = data["responses"]
        response_mask = data["response_mask"]
        rollout_lp = data["rollout_log_probs"]

        # pi_k logprobs: in the decoupled path (is_mode=tis/mis) _compute_old_log_prob already
        # recomputed them into old_log_probs, so reuse (screening is a free read). In bypass
        # mode (is_mode=none) old_log_probs == rollout_log_probs, so recompute pi_k here.
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            batch.extra_info.update(
                {
                    "calculate_entropy": False,
                    "compute_loss": False,
                    "temperature": self.config.actor_rollout_ref.rollout.temperature,
                }
            )
            output = self.actor_rollout_wg.compute_log_prob(batch)
            assert len(output) == len(batch)
            lp_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["log_probs", "response_mask"],
            )
            pi_k_lp = response_from_nested(lp_data["log_probs"], lp_data["response_mask"])
        else:
            pi_k_lp = data["old_log_probs"]

        # Per-trajectory extreme per-token mismatch: max |log_ratio| over response
        # tokens, aligned with partial_rollout_mismatch_truncation (both reject a
        # sequence when some single token deviates from rollout by > threshold).
        # The old sum-based LLR was dominated by response length (L x per-token
        # bias), so the "most mismatched" groups were just the longest ones; a
        # per-token max keeps length out of the decision. Padding rows carry
        # response_mask == 0 -> |log_ratio| 0.
        llrs = []
        llr_means = []
        for i in range(len(batch)):
            # Fail-closed: a trajectory whose stored responses / response_mask /
            # rollout_log_probs lengths disagree (rare async truncation+resume
            # race) would crash ppo_loss at `log_prob - old_log_prob`. Mark it
            # dirty (NaN) so the redispatch regenerates it cleanly instead.
            resp_len = responses[i].shape[0]
            mask_len = response_mask[i].shape[0]
            lp_len = rollout_lp[i].shape[0]
            if not (resp_len == mask_len == lp_len):
                llrs.append(float("nan"))
                llr_means.append(float("nan"))
                # b = 0: regenerate the broken trajectory from scratch (its kept prefix would
                # itself be inconsistent, so there is nothing safe to keep).
                rollback_points[batch.keys[i]] = 0
                continue
            mask = response_mask[i].to(dtype=torch.float32)
            delta = (pi_k_lp[i] - rollout_lp[i]).abs() * mask
            llrs.append(float(delta.max().item()) if delta.numel() else 0.0)
            llr_means.append(float(delta.sum().item() / mask.sum().item()) if mask.sum() > 0 else 0.0)
            # First token exceeding the threshold marks the rollback point; on re-dispatch
            # the session resumes from b = max(t*-rollback_window, 0), keeping the prefix
            # and dropping the mismatched tail (same rule as weight-sync truncation).
            # None = clean session, re-inserted verbatim when its group is rolled back.
            hit = (delta > threshold).nonzero()
            if hit.numel():
                rollback_points[batch.keys[i]] = max(int(hit[0].item()) - rollback_window, 0)
            else:
                rollback_points[batch.keys[i]] = None
        llr_arr = np.abs(np.array(llrs))

        # A group is dirty if any of its trajectories is out of bounds (NaN/inf LLR too).
        uid_to_keys: dict[str, list[str]] = defaultdict(list)
        dirty_uids: set[str] = set()
        for key, uid, llr in zip(batch.keys, uids, llrs):
            uid_to_keys[uid].append(key)
            if not np.isfinite(llr) or abs(llr) > threshold:
                dirty_uids.add(uid)

        stats = {
            "train_mismatch_gate/screened": len(llrs),
            "train_mismatch_gate/dirty": len(dirty_uids),
            "train_mismatch_gate/dirty_rate": len(dirty_uids) / max(len(llrs), 1),
            "train_mismatch_gate/redispatch": 0,
            "train_mismatch_gate/force_accept": 0,
            "train_mismatch_gate/llr_mean": float(np.nanmean(llr_arr)),
            "train_mismatch_gate/llr_p95": float(np.nanpercentile(llr_arr, 95)),
            "train_mismatch_gate/llr_per_tok_mean": float(np.nanmean(llr_means)),
        }
        # Rollback budget: at most `max_redispatch` (= oversample) dirty groups are rolled
        # back per step so the trained batch never falls below sample_batch_size. Only
        # groups that are ACTUALLY dirty (some token exceeded the threshold) are selected,
        # most-mismatched first; clean steps roll back nothing and train the whole sampled
        # batch. Dirty groups beyond the budget, and groups rolled back too many times, are
        # force-accepted (trained) to avoid starvation. Unfinished high-mismatch prefixes
        # are additionally handled by partial_rollout_mismatch_truncation at weight-sync time.
        max_redispatch = gate.get("max_redispatch_per_step", 0) or gate.get("oversample", 0)
        if max_redispatch <= 0 or not uid_to_keys:
            metrics.update(stats)
            return batch

        uid_llr: dict[str, float] = {}
        for _key, _uid, llr in zip(batch.keys, uids, llrs):
            a = abs(llr) if np.isfinite(llr) else float("inf")
            uid_llr[_uid] = max(uid_llr.get(_uid, 0.0), a)

        force_accept_uids: set[str] = set()
        redispatch_uids: set[str] = set()
        # Anti-starvation: never roll back a group more than max_force_accept times.
        force_accept_uids |= {u for u in uid_llr if self._mismatch_gate_regen_count.get(u, 0) >= max_force_accept}
        # Top-K relative re-sampling: every sampled group (not only those above the
        # threshold) is a candidate, ranked by per-token mismatch, so we actively pull
        # the K most-diverged groups back to the current policy instead of relying on a
        # calibrated tau. A soft lower bound (min_llr) keeps us from re-generating
        # groups whose mismatch is just engine noise (per-token noise floor ~1.0).
        redispatch_candidates = sorted(
            (u for u in uid_llr if u not in force_accept_uids and uid_llr[u] > min_llr),
            key=lambda u: uid_llr[u],
            reverse=True,
        )
        redispatch_uids = set(redispatch_candidates[:max_redispatch])
        # Groups not re-sampled this step are trained as-is; their regen counters are
        # reset below (see _redispatch_dirty_uids caller), so a diverged-but-not-top-K
        # group stays eligible to be re-sampled once it rises back into the top-K.
        for uid in redispatch_uids:
            self._mismatch_gate_regen_count[uid] = self._mismatch_gate_regen_count.get(uid, 0) + 1
        for uid in force_accept_uids:
            self._mismatch_gate_regen_count.pop(uid, None)

        stats.update(
            {
                "train_mismatch_gate/redispatch": len(redispatch_uids),
                "train_mismatch_gate/force_accept": len(force_accept_uids),
                # Re-sampled groups the relative top-K pulled in that were actually below
                # the threshold (a pure threshold filter would have kept them). Non-zero
                # -> the threshold is too loose for the tail; large -> too tight for the rest.
                "train_mismatch_gate/redispatch_topk_extra": sum(
                    1 for u in redispatch_uids if uid_llr[u] <= threshold
                ),
            }
        )
        if not redispatch_uids:
            # everything dirty was force-accepted; train the whole batch this step
            metrics.update(stats)
            return batch

        # Reset regen counters for uids trained clean this step.
        for uid in uid_to_keys:
            if uid not in redispatch_uids and uid not in force_accept_uids:
                self._mismatch_gate_regen_count.pop(uid, None)

        # Drop the redispatch keys from this step's training batch. The re-dispatch clears
        # their trajectories from TransferQueue, so they must NOT reach _compute_advantage.
        redispatch_key_set = set(k for uid in redispatch_uids for k in uid_to_keys[uid])
        clean_keys = [k for k in batch.keys if k not in redispatch_key_set]
        clean_tags = [t for k, t in zip(batch.keys, batch.tags) if k not in redispatch_key_set]
        if not clean_keys:
            # Pathological: every sampled group is dirty and none are force-acceptable yet.
            # Do not clear trajectories (no re-dispatch) and train the full batch rather
            # than stalling training or double-handling keys that were already cleared.
            logger.warning(
                "[train-mismatch-gate] step=%s all %d sampled groups dirty; force-accepting all "
                "(set a tighter log_ratio_threshold or larger max_force_accept_regens)",
                self.global_steps,
                len(llrs),
            )
            for uid in redispatch_uids:
                self._mismatch_gate_regen_count.pop(uid, None)
            metrics.update(stats)
            return batch

        self._redispatch_dirty_uids(batch, redispatch_uids, uid_to_keys, rollback_points)

        # Re-balance the cleaned batch (idempotent: pads back to the required multiple).
        batch = KVBatchMeta(
            partition_id=batch.partition_id,
            keys=clean_keys,
            tags=clean_tags,
            fields=batch.fields,
            extra_info=batch.extra_info or {},
        )
        batch = self._balance_batch(batch, metrics=metrics, logging_prefix="global_seqlen")

        rb_vals = [v for k in redispatch_key_set if (v := rollback_points.get(k)) is not None]
        stats.update(
            {
                "train_mismatch_gate/rollback_sessions": len(rb_vals),
                "train_mismatch_gate/rollback_avg_b": (sum(rb_vals) / len(rb_vals)) if rb_vals else 0.0,
            }
        )
        metrics.update(stats)
        logger.info(
            "[train-mismatch-gate] step=%s screened=%d dirty=%d rate=%.3f redispatch=%d topk_extra=%d "
            "force_accept=%d rollback_sess=%d avg_b=%.1f |llr| max mean=%.3f p95=%.3f per_tok_mean=%.4f",
            self.global_steps,
            len(llrs),
            stats["train_mismatch_gate/dirty"],
            stats["train_mismatch_gate/dirty_rate"],
            len(redispatch_uids),
            stats["train_mismatch_gate/redispatch_topk_extra"],
            len(force_accept_uids),
            stats["train_mismatch_gate/rollback_sessions"],
            stats["train_mismatch_gate/rollback_avg_b"],
            float(np.nanmean(llr_arr)),
            float(np.nanpercentile(llr_arr, 95)),
            float(np.nanmean(llr_means)),
        )
        return batch

    # Fields generated by the agent-loop rollout (AgentLoopOutput.as_dict +
    # AgentLoopWorkerTQ._agent_loop_postprocess + trainer log-prob writebacks).
    # They live on trajectory keys alongside prompt kwargs but must NOT be forwarded
    # onto a re-dispatched prompt batch (they are per-trajectory, not per-prompt).
    _TRAJECTORY_ONLY_FIELDS = frozenset(
        {
            # per-request local-rollback control signals: per-session lists sized by the
            # batch being re-dispatched, never part of a persisted trajectory
            "seed_token_ids", "seed_log_probs", "seed_is_complete",
            # per-response prompt kwarg: session index 0..n-1, reassigned on redispatch
            # (forwarding it collides with _run_prompt's explicit session_id=i)
            "session_id",
            # AgentLoopOutput.as_dict() -> trajectory tensors
            "prompts", "responses", "response_mask", "rollout_log_probs", "rm_scores",
            "routed_experts", "multi_modal_data", "reward_score", "num_turns", "metrics",
            "extra_fields", "mm_processor_kwargs",
            # AgentLoopWorkerTQ._agent_loop_postprocess
            "loss_mask", "input_ids", "position_ids", "multi_modal_inputs",
            # Score Centering: per-trajectory sampler top-k (ids + q logprobs)
            "sc_topk_ids", "sc_topk_logprobs",
            # trainer log-prob writebacks (_compute_old_log_prob / gate screening)
            "old_log_probs", "log_probs", "entropy",
            # informational tags / reward extras (harmless to exclude if ever fields)
            "prompt_len", "response_len", "seq_len", "score", "reward_metrics",
        }
    )

    def _redispatch_dirty_uids(
        self,
        batch: KVBatchMeta,
        dirty_uids: set[str],
        uid_to_keys: dict[str, list[str]],
        rollback_points: dict[str, int | None],
    ) -> None:
        """Roll back a dirty uid's group instead of regenerating it from scratch.

        Clears the group's trajectories and re-submits the prompt with per-session seed
        prefixes: each session resumes from ``response[:b]`` (b = rollback point computed
        during screening, local rollback like weight-sync truncation) and only regenerates
        the tail. Clean sessions of the group carry their full response and are re-inserted
        verbatim (``seed_is_complete``), so their good content is never discarded just
        because a sibling session diverged.
        """
        uids = list(dirty_uids)
        # A dirty group reaching this path has already been rolled back; count it
        # for precision escalation. The truncation path counts per session, this
        # one per group, so a group is escalated once *any* signal repeats.
        self.precision_escalation.note_rollbacks(uids)
        rep_keys = [uid_to_keys[uid][0] for uid in dirty_uids]
        rep_data = tq.kv_batch_get(keys=rep_keys, partition_id=batch.partition_id)

        # Per-session responses / rollout logprobs, needed to slice the kept prefix.
        all_keys = [k for uid in dirty_uids for k in uid_to_keys[uid]]
        all_data = tq.kv_batch_get(
            keys=all_keys,
            partition_id=batch.partition_id,
            select_fields=["responses", "rollout_log_probs"],
        )
        responses = all_data["responses"]
        rollout_lp = all_data["rollout_log_probs"]
        resp_by_key = {key: responses[i] for i, key in enumerate(all_keys)}
        lp_by_key = {key: rollout_lp[i] for i, key in enumerate(all_keys)}

        n_sessions = self.config.actor_rollout_ref.rollout.n
        seed_token_ids, seed_log_probs, seed_is_complete = [], [], []
        for uid in dirty_uids:
            # Order seeds by session id so session j of the regenerated group picks seed[j].
            group_keys = sorted(uid_to_keys[uid], key=lambda k: int(k.split("_")[1]))
            group_seeds_tok, group_seeds_lp, group_seeds_complete = [], [], []
            for key in group_keys:
                resp_t, lp_t = resp_by_key[key], lp_by_key[key]
                b = rollback_points.get(key)
                if b is not None:
                    group_seeds_tok.append(resp_t[:b].tolist())
                    group_seeds_lp.append(lp_t[:b].tolist())
                    group_seeds_complete.append(False)
                else:
                    group_seeds_tok.append(resp_t.tolist())
                    group_seeds_lp.append(lp_t.tolist())
                    group_seeds_complete.append(True)
            # sample() returns complete groups; anything else falls back to from-scratch regen.
            if len(group_keys) != n_sessions:
                logger.warning(
                    "[train-mismatch-gate] uid=%s group has %d sessions (expected %d); regenerating from scratch",
                    uid,
                    len(group_keys),
                    n_sessions,
                )
                group_seeds_tok = [[] for _ in range(n_sessions)]
                group_seeds_lp = [[] for _ in range(n_sessions)]
                group_seeds_complete = [False] * n_sessions
            seed_token_ids.append(group_seeds_tok)
            seed_log_probs.append(group_seeds_lp)
            seed_is_complete.append(group_seeds_complete)

        prompt_fields: dict[str, Any] = {
            "uid": np.array(uids, dtype=object),
            "raw_prompt": list(rep_data["raw_prompt"]),
            "index": torch.tensor([int(x) for x in rep_data["index"]], dtype=torch.int64),
            "agent_name": list(rep_data["agent_name"]) if "agent_name" in rep_data else None,
            "seed_token_ids": seed_token_ids,
            "seed_log_probs": seed_log_probs,
            "seed_is_complete": seed_is_complete,
        }
        if prompt_fields["agent_name"] is None:
            prompt_fields.pop("agent_name")
        for key in ("priority", "__do_sample__"):
            if key in rep_data:
                prompt_fields[key] = list(rep_data[key])
        # Forward every other prompt-level metadata field carried by the persisted
        # trajectory dict. Trajectory keys ({uid}_{session}_{index}) hold all prompt
        # kwargs because AgentLoopWorkerTQ._agent_loop_postprocess does
        # field.update(kwargs); they additionally hold trajectory-generated fields,
        # excluded via _TRAJECTORY_ONLY_FIELDS below. The previous whitelist silently
        # dropped columns the reward manager requires (data_source, reward_model,
        # extra_info, ability, ...), so re-dispatched prompts crashed with
        # KeyError: 'data_source'.
        # kv_batch_get may return a plain dict or a TensorDict; cover both by taking
        # the union of top-level keys and (when present) the non-tensor namespace.
        # Any remaining torch.Tensor field is trajectory data not covered by the
        # denylist; skip it (a tensor inside get_tensordict's list branch asserts).
        field_names = set(rep_data.keys())
        if hasattr(rep_data, "non_tensor_batch"):
            field_names |= set(rep_data.non_tensor_batch.keys())
        for key in field_names:
            if key in prompt_fields or key in self._TRAJECTORY_ONLY_FIELDS or key == "global_steps":
                continue
            val = rep_data[key]
            if isinstance(val, torch.Tensor):
                continue
            if isinstance(val, NonTensorStack):
                # Normalize to raw per-sample values so get_tensordict does not wrap
                # NonTensorData entries a second time (which would leave downstream
                # reward/agent-loop code reading NonTensorData instead of the value).
                val = [v.data for v in val]
            prompt_fields[key] = list(val)
        prompt_batch = tu.get_tensordict(prompt_fields)
        tu.assign_non_tensor_data(prompt_batch, "global_steps", self.global_steps)

        # Remove the old (dirty) trajectories so they are not sampled again, then
        # re-dispatch through the standard prompt-submission path (writes pending
        # prompt keys + generate_sequences). Requests queue while replicas sleep and
        # are served on the next resume_generation_replicas.
        dirty_keys = [k for uid in dirty_uids for k in uid_to_keys[uid]]
        tq.kv_clear(keys=dirty_keys, partition_id=batch.partition_id)
        self._submit_batch_to_rollout(prompt_batch)

    def _get_actor_dp_size(self) -> int:
        """Return the actor mesh's data-parallel size (for dispatch-shard padding)."""
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            worker_group._dispatch_info[role] = worker_group._query_dispatch_info(role)
        return max(worker_group._dispatch_info[role]) + 1

    def _truncate_mismatched_partial_prefixes(self) -> None:
        """Roll back high-mismatch prefixes of unfinished partial-rollout requests.

        Called at weight-sync time (on_step_end), after the actor update has moved
        the training engine to pi_{k+1}. For each in-flight prefix we recompute its
        per-token logprobs under pi_{k+1} and compare with the rollout logprobs
        saved by the client (pi_old). The first token whose |log_ratio| exceeds the
        threshold marks the rollback point; the client drops everything after it on
        the next resume so the continuation re-prefills from the new model version.

        GRPO loss / reward / batch handling are untouched and finished sequences
        are never modified.
        """
        stats = {
            "mismatch_trunc/in_flight": 0,
            "mismatch_trunc/truncated": 0,
            "mismatch_trunc/ratio": 0.0,
            "mismatch_trunc/tokens_dropped": 0,
            "mismatch_trunc/token_drop_ratio": 0.0,
            "mismatch_trunc/avg_rollback": 0.0,
            "mismatch_trunc/max_rollback": 0,
        }
        config = self.config.algorithm.get("partial_rollout_mismatch_truncation", None)
        if config is None or not config.get("enable", False):
            self._pending_mismatch_trunc_metrics = stats
            return
        if self.agent_loop_manager is None:
            self._pending_mismatch_trunc_metrics = stats
            return

        prefixes = self.agent_loop_manager.get_inflight_prefixes()
        entries = [
            p
            for p in prefixes
            if len(p["response_prefix"]) >= 1 and len(p["rollout_log_probs"]) == len(p["response_prefix"])
        ]
        if not entries:
            self._pending_mismatch_trunc_metrics = stats
            return

        threshold = config.get("log_ratio_threshold", 5.0)
        rollback_window = config.get("rollback_window", 32)

        # Build a [prompt | response] batch. Duplicate the first entry so the batch
        # is divisible by the actor DP size (dispatch rejects empty shards); only
        # the first len(entries) output rows are read back.
        fields = []
        for p in entries:
            prompt_t = torch.tensor(p["prompt_ids"], dtype=torch.int64)
            resp_t = torch.tensor(p["response_prefix"], dtype=torch.int64)
            input_ids = torch.cat([prompt_t, resp_t], dim=0)
            fields.append(
                {
                    "prompts": prompt_t,
                    "responses": resp_t,
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(input_ids.size(0), dtype=torch.int64),
                    "position_ids": torch.arange(input_ids.size(0), dtype=torch.int64),
                    "loss_mask": torch.ones(resp_t.size(0), dtype=torch.int64),
                }
            )
        dp_size = self._get_actor_dp_size()
        while len(fields) % dp_size != 0:
            fields.append(fields[0])

        batch = tu.list_of_dict_to_tensordict(fields)
        # Batch-level non-tensor config, mirroring _compute_old_log_prob's
        # batch.extra_info update (a per-sample NonTensorStack would break the
        # engine's temperature unwrap in prepare_model_inputs).
        tu.assign_non_tensor(
            batch,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            calculate_entropy=False,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch)

        responses = batch["responses"]
        response_mask = torch.nested.as_nested_tensor(
            [torch.ones_like(r, dtype=torch.int64) for r in responses.unbind()], layout=torch.jagged
        )
        resp_log_probs = response_from_nested(output["log_probs"], response_mask)

        truncations_by_worker: dict[int, dict[str, int]] = defaultdict(dict)
        total_prefix_tokens = sum(len(p["response_prefix"]) for p in entries)
        truncated = 0
        total_dropped = 0
        rollbacks = []
        for i, p in enumerate(entries):
            log_ratio = [a - b for a, b in zip(resp_log_probs[i].tolist(), p["rollout_log_probs"])]
            t_star = next((j for j, r in enumerate(log_ratio) if abs(r) > threshold), None)
            if t_star is None:
                continue
            b = max(t_star - rollback_window, 0)
            truncations_by_worker[p["worker_idx"]][p["request_id"]] = b
            truncated += 1
            total_dropped += len(p["response_prefix"]) - b
            rollbacks.append(b)
            # Attribute this rollback to its prompt group so a session that keeps
            # tripping the fp8/bf16 boundary can be escalated to a high-precision
            # regeneration later (see PrecisionEscalationController).
            escalation_key = p.get("escalation_key")
            if escalation_key:
                self.precision_escalation.note_rollback(escalation_key, escalation_key.rsplit("_", 1)[0])

        if truncations_by_worker:
            self.agent_loop_manager.set_truncation_points(truncations_by_worker)
        stats.update(
            {
                "mismatch_trunc/in_flight": len(entries),
                "mismatch_trunc/truncated": truncated,
                "mismatch_trunc/ratio": truncated / len(entries),
                "mismatch_trunc/tokens_dropped": total_dropped,
                "mismatch_trunc/token_drop_ratio": total_dropped / total_prefix_tokens,
                "mismatch_trunc/avg_rollback": (sum(rollbacks) / len(rollbacks)) if rollbacks else 0.0,
                "mismatch_trunc/max_rollback": max(rollbacks) if rollbacks else 0,
            }
        )
        self._pending_mismatch_trunc_metrics = stats
        logger.info(
            "[mismatch-truncate] step=%s in_flight=%d truncated=%d ratio=%.3f "
            "dropped=%d/%d (%.3f) rollback_b avg=%.2f max=%d",
            self.global_steps,
            len(entries),
            truncated,
            stats["mismatch_trunc/ratio"],
            total_dropped,
            total_prefix_tokens,
            stats["mismatch_trunc/token_drop_ratio"],
            stats["mismatch_trunc/avg_rollback"],
            stats["mismatch_trunc/max_rollback"],
        )

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        batch.extra_info.update(
            {
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = [
            "uid",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
            "entropy",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 4. write nested advantages and returns back to TransferQueue
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        output = TensorDict(output, batch_size=len(batch))

        batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _update_critic(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the critic network."""
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        extra_info = {
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.critic.ppo_epochs,
            "seed": self.config.critic.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.critic.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: DataProtoFuture = self.critic_wg.train_mini_batch(batch)
        output: TensorDict = output.get()
        output = rename_dict(output["metrics"], "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_metrics = reduce_metrics(output)
        metrics.update(critic_metrics)

        return batch

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Update the actor network."""
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        ) or True  # always compute for logging even when entropy_coeff=0
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
        if is_distillation_enabled(self.config.get("distillation")):
            distillation_loss_cfg = self.distillation_config.distillation_loss
            distillation_only = (
                distillation_use_topk
                and not distillation_loss_cfg.use_task_rewards
                and not distillation_loss_cfg.use_policy_gradient
            )
        extra_info = {
            "calculate_entropy": calculate_entropy,
            "distillation_use_topk": distillation_use_topk,
            "distillation_only": distillation_only,
            "global_batch_size": ppo_mini_batch_size,
            "mini_batch_size": ppo_mini_batch_size,
            "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
            "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
            "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
        batch.extra_info.update(extra_info)

        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
        output = rename_dict(output["metrics"], "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        actor_metrics = reduce_metrics(output)
        metrics.update(actor_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        # 1. collect necessary fields from TransferQueue for computing metrics
        non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
        ]
        moe_lb_metrics_interval = self.config.actor_rollout_ref.rollout.get("moe_load_balance_metrics_interval", 0)
        data = get_metric_data_with_optional_routed_experts(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=fields,
            moe_lb_metrics_interval=moe_lb_metrics_interval,
            global_steps=global_steps,
            accumulator=self._rollout_moe_lb_metrics_accumulator,
            kv_batch_get=tq.kv_batch_get,
        )

        num_turns = np.array(data.pop("num_turns").tolist())
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        min_global_steps = np.array([tag["min_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]
        max_global_steps = np.array([tag["max_global_steps"] for tag in batch.tags], dtype=int)[non_padding_mask]

        # Only fetch speculative decoding stats when rollout writes them.
        spec_drafts = spec_accepts = spec_verifies = None
        mtp_config = getattr(self.config.actor_rollout_ref.model, "mtp", None)
        if mtp_config is not None and mtp_config.enable and mtp_config.enable_rollout:
            spec_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = spec_data.pop("extra_fields").tolist()
            # The rollout omits the spec_* stats when the backend does not report
            # per-request spec-decode stats; leave all three as None in that case.
            if extra_fields and all(
                isinstance(extra_field, dict) and "spec_num_draft_tokens" in extra_field for extra_field in extra_fields
            ):
                spec_drafts = [extra_field["spec_num_draft_tokens"] for extra_field in extra_fields]
                spec_accepts = [extra_field["spec_num_accepted_tokens"] for extra_field in extra_fields]
                spec_verifies = [extra_field["spec_num_verify_steps"] for extra_field in extra_fields]

        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(batch=data, meta_info={"global_token_num": global_token_num})
        metrics_batch = batch.select_idxs(non_padding_mask) if non_padding_mask.any() else batch

        # 2. compute metrics
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(
            compute_moe_lb_metrics(
                metrics_batch=metrics_batch,
                moe_lb_metrics_interval=moe_lb_metrics_interval,
                global_steps=global_steps,
                accumulator=self._rollout_moe_lb_metrics_accumulator,
            )
        )
        metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self._get_n_gpus_for_throughput()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=metrics_batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        if non_padding_mask.any():
            num_turns = num_turns[non_padding_mask]
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        # 4. per-request speculative-decoding aggregation (same metrics async PPO logs;
        # see compute_spec_decode_metrics in verl/trainer/ppo/ray_trainer.py).
        metrics.update(compute_spec_decode_metrics(spec_drafts, spec_accepts, spec_verifies, non_padding_mask))

        # 5. off-policy staleness metrics
        #   global_steps is the model weight version (one update_weights per global_step), and
        #   min/max_global_steps are the versions a trajectory was generated across, so all quantities
        #   below are already in model-version units.
        #   - trajectory_spans: how many distinct model versions a single trajectory was
        #     generated across (1 == fully generated on a single version). This captures the
        #     within-trajectory policy inconsistency caused by partial rollout / continuation.
        #   - trajectory_staleness: how many model versions the trajectory lags behind the
        #     *current* policy. A trajectory spans versions [min_global_steps, max_global_steps],
        #     so the lag is a range: the freshest weights used give the lower bound
        #     (global_steps - max_global_steps) and the oldest weights the worst case
        #     (global_steps - min_global_steps). We log the lower bound as the primary metric.
        trajectory_spans = max_global_steps - min_global_steps + 1
        trajectory_staleness = (global_steps - 1) - max_global_steps
        trajectory_staleness_worst = (global_steps - 1) - min_global_steps
        metrics.update(
            {
                "training/off_policy/trajectory_spans/mean": trajectory_spans.mean(),
                "training/off_policy/trajectory_spans/max": trajectory_spans.max(),
                "training/off_policy/trajectory_spans/min": trajectory_spans.min(),
                "training/off_policy/trajectory_staleness/mean": trajectory_staleness.mean(),
                "training/off_policy/trajectory_staleness/max": trajectory_staleness.max(),
                "training/off_policy/trajectory_staleness/min": trajectory_staleness.min(),
                "training/off_policy/trajectory_staleness_worst/mean": trajectory_staleness_worst.mean(),
                "training/off_policy/trajectory_staleness_worst/max": trajectory_staleness_worst.max(),
                "training/off_policy/trajectory_staleness_worst/min": trajectory_staleness_worst.min(),
            }
        )


TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}


def register_trainer(name: str):
    """Class decorator that registers a :class:`PPOTrainer` subclass under ``name``."""

    def decorator(cls: type[PPOTrainer]) -> type[PPOTrainer]:
        if not (isinstance(cls, type) and issubclass(cls, PPOTrainer)):
            raise TypeError(f"register_trainer expected a PPOTrainer subclass, got {cls!r}")
        existing = TRAINER_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Trainer name '{name}' is already registered to {existing.__name__}; "
                f"cannot re-register it to {cls.__name__}."
            )
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name: str) -> type[PPOTrainer]:
    """Return the :class:`PPOTrainer` subclass registered under ``name``."""
    try:
        return TRAINER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TRAINER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown trainer '{name}'. Available trainers: {available}.") from None

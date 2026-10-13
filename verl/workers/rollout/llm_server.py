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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Deterministic Routing**: When ``full_determinism=True``, tie-breaking
      among equally-loaded servers uses ``hash(request_id)`` so the same
      request always routes to the same server across runs.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
    ):
        # Allow empty initial servers: in dynamic-resource-scheduling mode all
        # replicas are hybrid and will be registered later via add_servers().

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism

    def acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        min_count = min(self._inflight_requests.values())
        candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
        if len(candidates) == 1:
            server_id = candidates[0]
        elif self._full_determinism:
            # Deterministic tie-breaking: same request_id → same server across runs
            server_id = candidates[hash(request_id) % len(candidates)]
        else:
            server_id = candidates[0]
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def clear_sticky_cache(self) -> dict:
        """Clear the sticky-session cache to force request redistribution.

        After clearing, all subsequent ``acquire_server()`` calls will select
        the least-loaded server (based on ``_inflight_requests``), which
        naturally balances load across all active replicas — including newly
        added ones with zero in-flight requests.

        Returns:
            A dict with ``cleared_entries`` (number of cache entries dropped)
            and ``server_loads`` (current per-server inflight counts for
            diagnostics).
        """
        cleared = len(self._request_id_to_server)
        self._request_id_to_server.clear()
        logger.info(
            f"[GlobalLoadBalancer] Sticky cache cleared: {cleared} entries dropped. "
            f"Server loads: {dict(self._inflight_requests)}"
        )
        return {
            "cleared_entries": cleared,
            "server_loads": dict(self._inflight_requests),
        }

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }

    def get_total_inflight(self) -> int:
        """Return the sum of in-flight requests across all currently registered servers."""
        return sum(self._inflight_requests.values())


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        server_id, handle = await self._load_balancer.acquire_server.remote(request_id=request_id)
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        self._load_balancer.release_server.remote(server_id=server_id)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        escalation_key: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            escalation_key (Optional[str]): ``uid_session`` label used by the
                mismatch-driven precision-escalation control layer to attribute
                rollbacks. Consumed by subclasses that track in-flight prefixes;
                never forwarded to the server.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        server_id, server = await self._acquire_server(request_id)
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(server_id)


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        only_hybrid: bool = False,
        **kwargs,
    ):
        """Initialize the FullyAsyncLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
            only_hybrid (bool): When ``True``, hybrid replicas are the *only* rollout
                resource.  If the load balancer is temporarily empty (e.g. during
                weight synchronisation) :meth:`_acquire_server` will keep retrying
                every 1 second instead of raising immediately.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._only_hybrid = only_hybrid
        # In-flight partial-rollout state, keyed by request_id. Each entry holds
        # the prompt and a reference to the generate()'s accumulating final_output
        # so the trainer can read the produced prefix (and its rollout logprobs)
        # at weight-sync time. _truncations holds rollback points dispatched by
        # the trainer; each is consumed once by the matching in-flight generate.
        self._inflight: dict[str, dict] = {}
        self._truncations: dict[str, int] = {}

    def _apply_truncation(self, request_id: str) -> None:
        """Consume this request's rollback point, if one was dispatched.

        The trainer computes rollback points during the weight-sync window, when
        replicas are asleep and the client is blocked in ``_acquire_server``, so
        applying here (right before each resubmission) guarantees the first
        request served after resume already carries the truncated prefix.
        """
        info = self._inflight.get(request_id)
        if info is None:
            return
        b = self._truncations.pop(request_id, None)
        if b is None:
            return
        final_output = info["final_output"]
        final_output.token_ids = final_output.token_ids[:b]
        # No falsy guard: truncate log_probs unconditionally so the prefix keeps
        # len(log_probs) == len(token_ids) after rollback (a log_probs list left
        # untouched here can desync the trajectory's rollout_log_probs length).
        final_output.log_probs = final_output.log_probs[:b]
        # Restore the full remaining budget for the resumed generation; the
        # stale decrement was computed against the pre-truncation length.
        limit_key = info["limit_key"]
        if limit_key is not None:
            info["sampling_params"][limit_key] = info["original_max_tokens"] - b

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # When only_hybrid is True, hybrid replicas are the sole rollout resource and
        # the LB may be temporarily empty during weight sync / scaling transitions.
        # In that case keep retrying every 1 s until a server becomes available.
        # Otherwise raise immediately so callers see the error right away.
        while True:
            # Apply any rollback point dispatched since the last attempt so the
            # resume (prompt + final_output.token_ids) starts from the truncated
            # prefix, not the stale one the last abort interrupted.
            self._apply_truncation(request_id)
            try:
                return await super()._acquire_server(request_id)
            except RuntimeError as e:
                if "No available servers in load balancer" in str(e) and self._only_hybrid:
                    await asyncio.sleep(1)
                else:
                    raise

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        seed_token_ids: Optional[list[int]] = None,
        seed_log_probs: Optional[list[float]] = None,
        seed_is_complete: bool = False,
        seed_global_steps: Optional[int] = None,
        escalation_key: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.
            escalation_key (Optional[str]): ``uid_session`` label surfaced through
                ``get_inflight_prefixes`` so the mismatch-driven precision
                escalation layer can attribute rollbacks to a prompt group.
            seed_token_ids (Optional[List[int]]): Already-generated response prefix to keep.
                The request resumes from this prefix instead of generating from scratch, so
                a rolled-back trajectory's continuation is attached after the kept tokens.
                Only used by the FullyAsync partial-rollout resume path.
            seed_log_probs (Optional[List[float]]): Rollout logprobs aligned with
                ``seed_token_ids`` (produced by the model version that generated them).
            seed_is_complete (bool): Return the seed verbatim without generating. Used by
                the trainer to re-insert unchanged (clean) sessions of a rolled-back group.
            seed_global_steps (Optional[int]): global_steps stamped on the returned output
                when ``seed_is_complete`` skips the server round-trip.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=list(seed_token_ids) if seed_token_ids else [],
            log_probs=list(seed_log_probs) if seed_log_probs else [],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        if seed_is_complete:
            # No-op reinsert of a kept-complete trajectory (clean session of a group whose
            # sibling sessions were rolled back). The seed is returned verbatim; the only
            # thing lost is a server round-trip, so generation load is unchanged.
            final_output.stop_reason = "completed"
            if seed_global_steps is not None:
                final_output.extra_fields["global_steps"] = seed_global_steps
                final_output.extra_fields["min_global_steps"] = seed_global_steps
                final_output.extra_fields["max_global_steps"] = seed_global_steps
            # The kept prefix was generated by an earlier request whose top-k was never
            # persisted; emit invalid slots so the trajectory still carries the field
            # (mixed presence across a batch breaks the TransferQueue field set).
            sc_topk = int(sampling_params.get("score_centering_topk", 0) or 0)
            if sc_topk > 0:
                n_kept = len(final_output.token_ids)
                final_output.extra_fields["sc_topk_ids"] = [[-1] * sc_topk for _ in range(n_kept)]
                final_output.extra_fields["sc_topk_logprobs"] = [[0.0] * sc_topk for _ in range(n_kept)]
            self._inflight.pop(request_id, None)
            self._truncations.pop(request_id, None)
            return final_output

        # Expose the in-flight prefix to the trainer (read-only via
        # get_inflight_prefixes) for mismatch-aware partial rollout. A seeded request
        # starts with the kept prefix already in final_output, so an abort retry (and a
        # concurrent weight-sync truncation screen) sees exactly the resumed prefix.
        self._inflight[request_id] = {
            "prompt_ids": prompt_ids,
            "final_output": final_output,
            "sampling_params": sampling_params,
            "limit_key": limit_key,
            "original_max_tokens": original_max_tokens,
            "escalation_key": escalation_key,
        }

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

            # 2. merge output into final_output
            # Score Centering: accumulate the sampler's per-position top-k (ids + q
            # logprobs) across continuation calls, in lockstep with token_ids. A
            # partial-rollout resume keeps a prefix that carries no top-k of its own, so
            # backfill those leading rows with invalid slots (-1 id / 0.0 logprob): the
            # loss masks them out and those positions fall back to plain PG.
            sc_ids_call = output.extra_fields.get("sc_topk_ids")
            sc_lps_call = output.extra_fields.get("sc_topk_logprobs")
            if sc_ids_call is not None and sc_lps_call is not None:
                acc_ids = final_output.extra_fields.get("sc_topk_ids")
                acc_lps = final_output.extra_fields.get("sc_topk_logprobs")
                if acc_ids is None:
                    width = len(sc_ids_call[0]) if sc_ids_call else 0
                    n_kept = len(final_output.token_ids)
                    acc_ids = [[-1] * width for _ in range(n_kept)]
                    acc_lps = [[0.0] * width for _ in range(n_kept)]
                acc_ids.extend(sc_ids_call)
                acc_lps.extend(sc_lps_call)
                final_output.extra_fields["sc_topk_ids"] = acc_ids
                final_output.extra_fields["sc_topk_logprobs"] = acc_lps
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # Keep log_probs aligned with token_ids. A per-call payload whose
            # logprobs outnumber its own tokens (server abort/continuation edge)
            # would otherwise leave final_output.log_probs longer than
            # final_output.token_ids and corrupt the trajectory's rollout_log_probs
            # length, crashing the trainer's loss shape check downstream.
            if len(final_output.log_probs) > len(final_output.token_ids):
                logger.warning(
                    "[client-merge-guard] request=%s clip log_probs %d -> %d "
                    "(token_ids=%d stop=%s global_steps=%s)",
                    request_id,
                    len(final_output.log_probs),
                    len(final_output.token_ids),
                    len(final_output.token_ids),
                    output.stop_reason,
                    output.extra_fields.get("global_steps"),
                )
                final_output.log_probs = final_output.log_probs[: len(final_output.token_ids)]
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            # If partial rollout not enable, aborted samples will be dropped.
            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
            should_retry = True
            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
                should_retry = False
            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)

        # A stale truncation point (request completed without an abort retry)
        # must not leak into a later generate reusing this request_id.
        self._truncations.pop(request_id, None)
        self._inflight.pop(request_id, None)
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps

        # Keep the accumulated top-k aligned with the final token length, mirroring the
        # log_probs length guard in the merge loop. Trim over-long rows; pad short ones
        # with invalid slots so the response axis stays aligned downstream.
        sc_ids = final_output.extra_fields.get("sc_topk_ids")
        sc_lps = final_output.extra_fields.get("sc_topk_logprobs")
        if sc_ids is not None and sc_lps is not None and len(sc_ids) != len(final_output.token_ids):
            n = len(final_output.token_ids)
            width = len(sc_ids[0]) if sc_ids else int(sampling_params.get("score_centering_topk", 0) or 0)
            sc_ids = sc_ids[:n]
            sc_lps = sc_lps[:n]
            pad = n - len(sc_ids)
            if pad > 0:
                sc_ids = sc_ids + [[-1] * width for _ in range(pad)]
                sc_lps = sc_lps + [[0.0] * width for _ in range(pad)]
            final_output.extra_fields["sc_topk_ids"] = sc_ids
            final_output.extra_fields["sc_topk_logprobs"] = sc_lps
        return final_output

    def get_inflight_prefixes(self) -> list[dict[str, Any]]:
        """Return partial-rollout prefixes of unfinished generate calls.

        Each entry is ``{"request_id", "prompt_ids", "response_prefix",
        "rollout_log_probs", "escalation_key"}`` for a generate that has produced
        at least one token and is still in flight. Read at weight-sync time so the
        trainer can compute the new model's prefix logprobs and roll back mismatch
        tokens.
        """
        prefixes = []
        for request_id, info in self._inflight.items():
            final_output = info["final_output"]
            if len(final_output.token_ids) < 1:
                continue
            prefixes.append(
                {
                    "request_id": request_id,
                    "prompt_ids": info["prompt_ids"],
                    "response_prefix": list(final_output.token_ids),
                    "rollout_log_probs": list(final_output.log_probs),
                    "escalation_key": info.get("escalation_key"),
                }
            )
        return prefixes

    def set_truncation_points(self, truncations: dict[str, int]) -> None:
        """Record per-request rollback points (response-token index) computed by
        the trainer at the last weight sync. Each is consumed once by the
        matching in-flight generate on its next retry."""
        self._truncations.update(truncations)


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus / rl-insight metrics with server addresses
        needs_metrics = self.rollout_config.prometheus.enable or RLInsightLogger.enabled()
        if self.rollout_config.disable_log_stats:
            if needs_metrics:
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not self.rollout_config.disable_log_stats:
            if self.rollout_config.prometheus.enable:
                update_prometheus_config(
                    self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
                )
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    self.rollout_config.name,
                    labels=[{"replica": server.replica_rank} for server in self.rollout_replicas],
                )

    async def _init_global_load_balancer(self) -> None:
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
        )

    @auto_await
    async def relaunch_replicas(self) -> None:
        """Recreate the server replicas from the current ``rollout_config``.

        Used by the precision switch: the caller tears the old servers down first
        (see ``RolloutReplica.teardown_servers``), mutates the precision fields of
        ``rollout_config``, then calls this to build servers with the new precision.
        ``self.server_addresses`` / ``self.server_handles`` are refreshed.
        """
        await self._initialize_llm_servers()

    @auto_await
    async def swap_servers(self, old_addresses: list[str]) -> None:
        """Replace ``old_addresses`` with the freshly launched servers in the load balancer.

        The sticky-session cache is cleared so subsequent requests are balanced
        over the new (least-loaded) servers instead of being pinned to a stale,
        now-dead one.
        """
        new_servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        await self.global_load_balancer.remove_servers.remote(old_addresses)
        await self.global_load_balancer.add_servers.remote(new_servers)
        await self.global_load_balancer.clear_sticky_cache.remote()

    def get_client(self, client_cls=LLMServerClient, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate (default: ``LLMServerClient``).
                Pass ``FullyAsyncLLMServerClient`` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])

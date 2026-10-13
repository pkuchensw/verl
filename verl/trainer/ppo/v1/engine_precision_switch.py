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
"""Restart the colocated rollout engine at a different precision.

The rollout engine's precision is fixed when the SGLang server is built: the
quantization kernels, the KV-cache dtype and the weight layout are all chosen at
load time, so a running fp8 server cannot serve bf16 and vice versa. Switching
precision therefore means tearing the server actors down and relaunching them
from a modified rollout config, then re-syncing the current training weights into
the fresh engine.

This module holds that mechanism and nothing else; the policy that decides
*when* to escalate lives in ``precision_escalation.py``. The switch is serialized
and blocking by design -- the whole engine changes precision, so no other request
may be generating while it is in flight. The caller must quiesce the rollout
pipeline before entering.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig

from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)

SUPPORTED_PRECISIONS = ("bf16", "fp8")


def _set_rollout_field(cfg: Any, key: str, value: Any) -> None:
    """Write ``key`` into a rollout config that may be a DictConfig or a dataclass.

    The manager's ``rollout_config`` is an OmegaConf ``DictConfig``; the worker's is
    a frozen ``BaseConfig`` dataclass. ``object.__setattr__`` is only correct for
    the latter: on a ``DictConfig`` it writes the instance ``__dict__`` while
    ``.get()`` / key lookup / serialization all read the internal ``_content``, so
    the write is invisible to the code that actually builds the engine --
    ``RolloutReplica`` converts the config with ``omega_conf_to_dataclass`` and the
    server reads it with ``.get(...)``. Item assignment is what a ``DictConfig``
    honors, so use it there and keep ``object.__setattr__`` for the frozen dataclass.
    """
    if isinstance(cfg, DictConfig):
        cfg[key] = value
    else:
        object.__setattr__(cfg, key, value)


class EnginePrecisionSwitcher:
    """Switch the colocated rollout engine between low and high precision.

    Args:
        trainer: the (colocate) PPO trainer. Must expose ``llm_server_manager``,
            ``checkpoint_manager``, ``actor_rollout_wg`` and ``global_steps``.
        initial_precision: precision the engine is already built at, so a switch to
            that same precision is a no-op instead of a needless teardown.
    """

    def __init__(self, trainer: Any, initial_precision: Optional[str] = None) -> None:
        self.trainer = trainer
        # Original (quantization, kv_cache_dtype) captured on the first switch so
        # switching back to "fp8" restores exactly what the run was launched with.
        self._saved: Optional[tuple[Any, Any]] = None
        self._current: Optional[str] = initial_precision

    @property
    def current_precision(self) -> Optional[str]:
        """Precision the engine is currently built at, or None before the first switch."""
        return self._current

    # -- config plumbing ------------------------------------------------------

    def _rollout_config(self):
        return self.trainer.llm_server_manager.rollout_config

    def _snapshot(self) -> None:
        if self._saved is not None:
            return
        cfg = self._rollout_config()
        kv_dtype = None
        sglang_kwargs = cfg.get("engine_kwargs", None)
        if sglang_kwargs is not None:
            sglang = sglang_kwargs.get("sglang", None)
            if sglang is not None:
                kv_dtype = sglang.get("kv_cache_dtype", None)
        self._saved = (cfg.get("quantization", None), kv_dtype)

    def _apply_precision(self, precision: str) -> None:
        """Point ``rollout_config`` at the target precision.

        Only ``quantization`` and the sglang KV-cache dtype differ between the two
        builds; everything else (parallelism, memory fraction, attention backend)
        is shared so the relaunched engine slots into the same replica layout.
        """
        self._snapshot()
        cfg = self._rollout_config()
        if precision == "bf16":
            quantization, kv_dtype = None, None
        elif precision == "fp8":
            quantization, kv_dtype = self._saved
            # Only the precision the run was launched with is reproducible: the
            # other builds need engine settings (quantization mode, KV-cache dtype)
            # that are not in the config. Refuse rather than silently relabeling a
            # bf16 engine as fp8.
            if quantization is None:
                raise ValueError(
                    "cannot build an fp8 rollout engine: the run was launched without "
                    "actor_rollout_ref.rollout.quantization=fp8"
                )
        else:
            raise ValueError(f"unsupported precision {precision!r}, expected one of {SUPPORTED_PRECISIONS}")

        # ``quantization`` is a frozen RolloutConfig field on the worker's dataclass
        # and a plain field on the manager's DictConfig; ``_set_rollout_field``
        # writes whichever the config actually is.
        _set_rollout_field(cfg, "quantization", quantization)
        sglang_kwargs = cfg.get("engine_kwargs", None)
        if sglang_kwargs is not None:
            sglang = sglang_kwargs.get("sglang", None)
            if sglang is not None:
                # kv_cache_dtype is only present for the fp8-KV builds; a bf16
                # build must drop it entirely rather than pass None to ServerArgs.
                if kv_dtype is None:
                    if "kv_cache_dtype" in sglang:
                        sglang.pop("kv_cache_dtype")
                else:
                    sglang["kv_cache_dtype"] = kv_dtype

    # -- switch ---------------------------------------------------------------

    @auto_await
    async def switch_to(self, precision: str) -> None:
        """Rebuild the rollout engine at ``precision`` and resync the current weights.

        Preconditions: the rollout pipeline is quiesced (no in-flight or queued
        requests) and the replicas are asleep. Both hold at the pause point inside
        ``on_step_end``, between the abort in ``on_sample_end`` and the resume.
        """
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported precision {precision!r}, expected one of {SUPPORTED_PRECISIONS}")
        if precision == self._current:
            return

        manager = self.trainer.llm_server_manager
        checkpoint_manager = self.trainer.checkpoint_manager
        replicas = manager.get_replicas()
        old_addresses = list(manager.server_addresses)

        logger.info("precision-switch: rebuilding rollout engine at %s (was %s)", precision, self._current)

        # 1. Sleep the engine so its weight + KV memory is released before the new
        #    build claims the same GPUs.
        await checkpoint_manager.sleep_replicas()

        # 2. Tear the old servers down. Blocks until the named actors disappear,
        #    because the relaunch reuses the same names.
        for replica in replicas:
            replica.teardown_servers()

        # 3. Repoint the rollout config and the worker-side adapters at the new
        #    precision. The adapters also drop their cached engine handle so they
        #    re-resolve the freshly created server actors.
        self._apply_precision(precision)
        self.trainer.actor_rollout_wg.set_rollout_precision(precision)

        # 4. Rebuild the servers under the same replica layout.
        await manager.relaunch_replicas()

        # 5. Move the load balancer over to the new servers and drop sticky
        #    sessions pinned to the dead ones.
        await manager.swap_servers(old_addresses)

        # 6. Keep the checkpoint manager pointing at the live replicas so the
        #    weight sync below (and the normal per-step sync) targets them.
        checkpoint_manager.replicas = manager.get_replicas()

        # 7. Put the fresh engine to sleep before syncing. ``update_weights`` is
        #    written against a *sleeping* engine -- with the naive backend it
        #    starts by resuming the weights that sleep released. A freshly
        #    launched engine has never slept, so that resume is invalid: sglang
        #    restores the static state stashed by ``release_memory_occupation``,
        #    which does not exist yet. Release first so the fresh engine is in
        #    the same state every other weight sync sees.
        await checkpoint_manager.sleep_replicas()

        # 8. Push the current training weights into the fresh engine.
        await checkpoint_manager.update_weights(self.trainer.global_steps)

        self._current = precision
        # Read the quantization back out of the live config rather than echoing the
        # requested precision: the two must agree, and a divergence here is exactly
        # the silent failure mode this switch had.
        logger.info(
            "precision-switch: rollout engine now running at %s (config quantization=%r)",
            precision,
            self._rollout_config().get("quantization", None),
        )

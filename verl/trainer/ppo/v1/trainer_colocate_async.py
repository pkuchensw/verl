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
import logging
import os
import time

import ray
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1.engine_precision_switch import EnginePrecisionSwitcher
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Seed fields a mismatch-gate re-dispatch attaches to a prompt row to *resume* a
# prefix. An escalated regeneration restarts from scratch, so they are stripped.
_SEED_FIELDS = ("seed_token_ids", "seed_log_probs", "seed_is_complete")


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        if self.config.skip.rollout_tq.enable:
            return
        num_warmup_batches = self.config.trainer.v1.colocate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_step_end(self):
        # Compute mismatch rollback points for in-flight partial-rollout prefixes
        # (using the post-update pi_{k+1}) and dispatch them to the clients before
        # syncing weights, so the resumed generation starts from the new model.
        self._truncate_mismatched_partial_prefixes()
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            self.checkpoint_manager.resume_generation_replicas()
        # Regenerate the groups that keep getting rolled back on a high-precision
        # engine, then hand the rollout engine back to the base precision. Runs
        # last so the engine swap starts from a synced, running base engine.
        self._maybe_run_precision_escalation()

    # ------------------------------ precision escalation ------------------------------

    def _get_precision_switcher(self) -> EnginePrecisionSwitcher:
        switcher = getattr(self, "_precision_switcher", None)
        if switcher is None:
            # Captured before the first switch mutates the rollout config, so it
            # records the precision the run was launched with.
            rollout_quantization = self.config.actor_rollout_ref.rollout.get("quantization", None)
            self._rollout_base_precision = "fp8" if rollout_quantization == "fp8" else "bf16"
            switcher = EnginePrecisionSwitcher(self, initial_precision=self._rollout_base_precision)
            self._precision_switcher = switcher
        return switcher

    def _maybe_run_precision_escalation(self) -> None:
        """Regenerate the buffered repeatedly-rolled-back groups at high precision.

        Runs after the normal weight sync + resume: the base engine is synced,
        running, and holds the current weights. The escalation first drains the
        rollout pipeline so the engine swap starts from an idle engine, then swaps
        the engine, regenerates the batch on it, and swaps back.
        """
        controller = self.precision_escalation
        if not controller.enable or not controller.ready():
            return
        if self.config.actor_rollout_ref.rollout.name != "sglang":
            logger.warning("[precision-escalation] only the sglang rollout supports the engine switch; skipping")
            return

        switcher = self._get_precision_switcher()
        target = controller.target_precision
        if target == switcher.current_precision:
            # Nothing to switch to, and a regeneration here would just reproduce
            # the same distribution, so drop the buffered groups. Released as well:
            # the uids are in-flight in the controller and would otherwise stay
            # blocked from ever escalating again.
            dropped = controller.take_batch()
            controller.release(dropped)
            logger.info(
                "[precision-escalation] engine already at %s; dropped %d buffered group(s) without regenerating",
                target,
                len(dropped),
            )
            return

        uids = controller.take_batch()
        if not uids:
            return
        present = self._present_escalation_uids(uids)
        # A uid evicted from the transfer queue can never be regenerated (its prompt
        # row is gone), so it is dropped rather than re-buffered -- re-buffering would
        # keep no-op batches cycling through ``take_batch`` forever. Released as well,
        # so a uid is never left permanently blocked in the controller.
        release_absent = [uid for uid in uids if uid not in set(present)]
        if release_absent:
            controller.release(release_absent)
        if not present:
            logger.warning("[precision-escalation] dropped %d buffered group(s): none is still registered", len(uids))
            return

        logger.info(
            "[precision-escalation] step=%s escalating %d/%d group(s) to %s",
            self.global_steps,
            len(present),
            len(uids),
            target,
        )
        # The escalated engine must be handed back even if the regeneration fails
        # midway: leaving the run on the high-precision build would silently change
        # the experiment (every later step would roll out at bf16, not at the
        # launched precision), and nothing downstream would flag it.
        switched = False
        try:
            with marked_timer("precision_escalation", self.timing_raw, color="magenta"):
                self._drain_rollout_pipeline()
                switcher.switch_to(target)
                switched = True
                # The groups' prior inference is void: drop it and regenerate from scratch.
                self._clear_escalated_trajectories(present)
                self._submit_batch_to_rollout(self._build_escalated_prompt_batch(present))
                self._wait_for_escalated_batch(present)
            # The discarded inference's rollbacks must not weigh down the fresh
            # generation: reset the counters so these groups need another full
            # ``num_rollbacks_trigger`` trip before they escalate again.
            controller.release(present)
        except Exception:
            controller.return_batch(present)
            raise
        finally:
            if switched:
                switcher.switch_to(self._rollout_base_precision)

    def _drain_rollout_pipeline(self, timeout_s: float = 3600.0, idle_checks: int = 3) -> None:
        """Block until no rollout client is mid-request, so the engine can be rebuilt.

        Killing the server actors is only safe once nothing is awaiting them: a
        request still queued on the old engine fails when its actor dies. Nothing
        new is dispatched while the trainer is inside ``on_step_end``, so the
        in-flight requests finish on their own and the engine goes idle.

        The load balancer's in-flight counter brackets each client's server call
        (incremented on acquire, decremented after the response), so a sustained
        zero means no request is outstanding.
        """
        load_balancer = self.llm_server_manager.global_load_balancer
        deadline = time.time() + timeout_s
        idle = 0
        while idle < idle_checks:
            inflight = ray.get(load_balancer.get_total_inflight.remote())
            idle = idle + 1 if inflight == 0 else 0
            if idle >= idle_checks:
                break
            if time.time() > deadline:
                logger.warning(
                    "[precision-escalation] drain timed out with %d request(s) still in flight; rebuilding anyway",
                    inflight,
                )
                return
            time.sleep(1.0)
        logger.info("[precision-escalation] rollout pipeline drained")

    def _present_escalation_uids(self, uids: list[str], partition_id: str = "train") -> list[str]:
        """Keep only uids whose prompt row is still registered in the transfer queue.

        The escalated batch is rebuilt from those rows, so a uid that has already
        been evicted from the queue cannot be regenerated.
        """
        items = tq.kv_list(partition_id).get(partition_id, {})
        present = [uid for uid in uids if uid in items]
        missing = len(uids) - len(present)
        if missing:
            logger.warning("[precision-escalation] %d uid(s) are no longer registered; skipping", missing)
        return present

    def _build_escalated_prompt_batch(self, uids: list[str], partition_id: str = "train") -> TensorDict:
        """Read the escalated prompt rows back into a from-scratch prompt batch.

        ``_submit_batch_to_rollout`` persists every prompt row under its uid, so
        the batch can be rebuilt verbatim -- including the fields the agent loop
        and reward manager need -- without caching payloads at dispatch time.
        """
        prompt_batch = tq.kv_batch_get(keys=list(uids), partition_id=partition_id)
        # Re-assert the key column so the re-dispatch registers exactly the uids we
        # read, whatever order the queue returns them in.
        tu.assign_non_tensor_stack(prompt_batch, "uid", list(uids))
        for key in _SEED_FIELDS:
            tu.pop(prompt_batch, key, None)
        tu.assign_non_tensor_data(prompt_batch, "global_steps", self.global_steps)
        return prompt_batch

    def _clear_escalated_trajectories(self, uids: list[str], partition_id: str = "train") -> None:
        """Drop trajectories already recorded for the escalated groups.

        Their prior inference is discarded by design, so any trajectory that
        landed before the regeneration must not survive to be trained alongside it.
        """
        uid_set = set(uids)
        items = tq.kv_list(partition_id).get(partition_id, {})
        stale = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in uid_set
        ]
        if stale:
            tq.kv_clear(keys=stale, partition_id=partition_id)
            logger.info("[precision-escalation] cleared %d stale trajectory key(s)", len(stale))

    def _wait_for_escalated_batch(
        self, uids: list[str], partition_id: str = "train", timeout_s: float = 3600.0
    ) -> None:
        """Block until every escalated uid has produced its full trajectory group.

        The pipeline was drained and nothing else is dispatched during the
        escalation, so the only trajectories that can appear are this batch's.
        """
        n_sessions = self.config.actor_rollout_ref.rollout.n
        pending = set(uids)
        deadline = time.time() + timeout_s
        while pending:
            items = tq.kv_list(partition_id).get(partition_id, {})
            counts: dict[str, int] = {}
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    continue
                uid = key.split("_", 1)[0]
                if uid in pending:
                    counts[uid] = counts.get(uid, 0) + 1
            pending = {uid for uid in pending if counts.get(uid, 0) < n_sessions}
            if not pending:
                break
            if time.time() > deadline:
                logger.warning(
                    "[precision-escalation] timed out waiting for %d escalated uid(s): %s",
                    len(pending),
                    sorted(pending)[:8],
                )
                break
            time.sleep(2.0)

    def on_sample_end(self):
        # abort all unfinished requests and pause generation
        self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()

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
"""Algorithm-side control layer for mismatch-driven precision escalation.

Motivation
----------
In the async partial-rollout pipeline the low-precision (fp8) rollout engine and
the bf16 training engine disagree about the distribution of the tokens the rollout
produces. Two existing mechanisms fight that disagreement *in place*:

* ``partial_rollout_mismatch_truncation`` rolls an in-flight prefix back to
  ``b = max(t* - rollback_window, 0)`` at weight-sync time so the continuation
  re-prefills under the new model version, and
* ``train_mismatch_gate`` re-dispatches whole prompt groups whose per-token
  mismatch is large enough to be distinguishable from engine noise.

Both keep the *same* low-precision engine. A prompt that trips either mechanism
repeatedly is therefore evidence that the engine precision itself -- not just the
model version -- is the source of the mismatch: re-running it on the same engine
will keep tripping.

This controller tracks those repeated rollbacks. Once a uid has been rolled back
``num_rollbacks_trigger`` times it is marked for *escalation*: its entire prior
inference is discarded and it is parked in a buffer. When the buffer reaches
``batch_size`` the trainer switches the rollout engine to a high-precision build,
regenerates exactly that batch on it, and switches back. The high-precision
engine removes the engine-precision component of the mismatch -- precisely the
part the in-place mechanisms cannot touch.

The controller is deliberately engine-agnostic: it only decides *when* to
escalate *what*. The engine swap lives in ``engine_precision_switch.py`` so the
policy and the mechanism can be tuned independently.

Strategies
----------
``strategy`` selects what "escalation" means for a stuck group:

* ``"switch"`` (default): buffer the group and, once enough accumulate, switch the
  rollout engine to a high-precision build and regenerate them (the mechanism above).
* ``"drop"``: do **not** touch the engine. A group that trips the threshold is
  rejected outright -- the sampler clears its prompt + trajectories from the transfer
  queue before selecting the next training batch, so it is never trained on and the
  refill mechanism replaces it with fresh prompts. This trades the (costly, blocking)
  engine switch for losing the group's data: cheaper, but it discards rather than
  rescues. A rejected uid is only cleared once it is *terminal*; an in-flight
  rejection keeps generating until it settles, because its running sessions still
  write to the queue.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PrecisionEscalationController:
    """Track repeated rollout rollbacks and decide when to escalate a batch.

    A uid is a GRPO prompt group id (``uid`` in the transfer-queue batch); every
    session of that group shares it. The controller counts rollbacks per uid and
    promotes a uid to the escalation buffer once the count reaches
    ``num_rollbacks_trigger``.

    Only uids are buffered, never prompt payloads: the trainer persists each
    prompt row in the transfer queue under its uid, so the escalated batch is
    rebuilt by reading those rows back.

    A uid is counted only while it is not already awaiting (or undergoing) a
    regeneration, and its counters are wiped once that regeneration completes.
    ``num_rollbacks_trigger`` therefore reads as "rolled back this many times
    *since its last regeneration*", not "this many times ever" -- which is what
    makes escalation rare. Without both guards a single stale session, which
    keeps tripping until the engine swap finally kills it, re-queues its group on
    every step and the engine switches almost continuously.

    Escalation is a one-shot test per group: a uid that has been regenerated on
    the high-precision engine is never escalated again. The switch asks "is this
    group blocked by the rollout engine's precision?". If the group still trips
    after a bf16 regeneration the answer is no -- it is simply a hard prompt --
    and re-running the swap on it would cycle the engine forever at the
    ``num_rollbacks_trigger`` period. One shot per group also bounds the total
    number of switches by the number of distinct stuck groups.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.enable: bool = bool(cfg.get("enable", False))
        self.num_rollbacks_trigger: int = int(cfg.get("num_rollbacks_trigger", 2))
        self.batch_size: int = int(cfg.get("batch_size", 8))
        self.target_precision: str = str(cfg.get("target_precision", "bf16"))
        self.strategy: str = str(cfg.get("strategy", "switch"))
        if self.strategy not in ("switch", "drop"):
            raise ValueError(f"unsupported precision-escalation strategy {self.strategy!r}, expected 'switch' or 'drop'")
        # Soft cap on the escalation buffer: without it a workload where every
        # prompt gets stuck would grow the buffer without bound. Overflow uids are
        # dropped (and counted) rather than kept forever.
        max_buffer = int(cfg.get("max_buffer", 0))
        self.max_buffer: int = max_buffer if max_buffer > 0 else max(self.batch_size * 4, 64)
        # Hard cap on tracked keys. Trained groups are pruned every step, so this
        # only bounds the residue (keys whose group never reached training), and
        # prevents a pathological workload from growing the dict without bound.
        # Oldest keys are evicted first.
        max_tracked = int(cfg.get("max_tracked", 0))
        self.max_tracked: int = max_tracked if max_tracked > 0 else max(self.batch_size * 8, 4096)

        # escalation key (uid or uid_session) -> rollbacks since its last reset.
        self._rollbacks: dict[str, int] = {}
        # uid -> None in escalation order. OrderedDict so take_batch is FIFO.
        self._escalated: OrderedDict[str, None] = OrderedDict()
        # uid -> None for groups rejected under the "drop" strategy, in rejection
        # order. The sampler clears the ones that are terminal and refills; the
        # trainer prunes the rest once their groups leave the transfer queue.
        self._rejected: OrderedDict[str, None] = OrderedDict()
        # uids queued for, or currently being, regenerated. Their inference is
        # already void, so rollbacks arriving from those stale sessions are
        # ignored instead of being charged to the next generation.
        self._pending: set[str] = set()
        # uids that have ever been escalated. Kept for the whole run: the bf16
        # regeneration is a one-shot test of whether engine precision is this
        # group's problem, so a group that already had its shot is never queued
        # again. Reset by clear() (checkpoint resume / restart).
        self._attempted: set[str] = set()

        # Cumulative counters for metrics. Both rollback paths feed one count:
        # truncation and the mismatch gate are two symptoms of the same thing, and
        # the decision to escalate looks only at whether a session keeps tripping.
        self._num_rollbacks = 0
        self._num_escalated = 0
        self._num_rejected = 0
        self._num_batches = 0
        self._num_regens = 0
        self._num_overflow = 0
        self._num_evicted = 0

    # ------------------------------------------------------------------ records

    def note_rollback(self, key: str, uid: Optional[str] = None) -> bool:
        """Record one rollback; return True if ``uid`` just got escalated.

        ``key`` identifies *what* was rolled back -- a single trajectory
        (``uid_session``) for the truncation path, or the whole group (``uid``)
        for the mismatch gate. ``uid`` identifies the group that would be
        escalated (defaults to ``key``). Counting per trajectory while escalating
        per group matches the intent: a group is escalated once *any* of its
        sessions has been rolled back repeatedly.
        """
        if not self.enable:
            return False
        if uid is None:
            uid = key
        # Already had its high-precision regeneration: whatever keeps tripping it
        # is not engine precision, so do not count it toward another swap.
        if uid in self._attempted:
            return False
        self._num_rollbacks += 1
        if uid in self._pending:
            return False
        count = self._rollbacks.get(key, 0) + 1
        self._rollbacks[key] = count
        self._evict_oldest()
        if count < self.num_rollbacks_trigger:
            return False
        if self.strategy == "drop":
            # Reject outright: the sampler clears the group before it can enter a
            # training batch, so no engine switch is needed and the data is lost
            # rather than rescued. ``_pending`` blocks re-counting until the group
            # settles and the trainer releases it.
            self._rejected[uid] = None
            self._pending.add(uid)
            self._attempted.add(uid)
            self._num_rejected += 1
            logger.info("precision-escalation: uid=%s rejected after %d rollback(s) of %s", uid, count, key)
            return True
        if len(self._escalated) >= self.max_buffer:
            self._num_overflow += 1
            return False
        self._escalated[uid] = None
        self._pending.add(uid)
        self._attempted.add(uid)
        self._num_escalated += 1
        logger.info("precision-escalation: uid=%s escalated after %d rollback(s) of %s", uid, count, key)
        return True

    def note_rollbacks(self, keys: Any, uids: Optional[Any] = None) -> int:
        """Record one rollback per key; return the number of groups escalated.

        ``uids``, when given, is a parallel iterable of group ids (one per key).
        """
        if uids is None:
            return sum(1 for key in keys if self.note_rollback(str(key)))
        return sum(1 for key, uid in zip(keys, uids, strict=True) if self.note_rollback(str(key), str(uid)))

    def _evict_oldest(self) -> None:
        """Keep ``_rollbacks`` within ``max_tracked`` by dropping the oldest keys."""
        while len(self._rollbacks) > self.max_tracked:
            self._rollbacks.pop(next(iter(self._rollbacks)))
            self._num_evicted += 1

    def rollback_count(self, key: str) -> int:
        return self._rollbacks.get(key, 0)

    # ------------------------------------------------------------------- buffer

    def ready(self) -> bool:
        """True when enough uids have been escalated to form a batch."""
        return self.enable and len(self._escalated) >= self.batch_size

    def buffer_size(self) -> int:
        return len(self._escalated)

    def take_batch(self) -> list[str]:
        """Pop up to ``batch_size`` escalated uids in FIFO order."""
        uids: list[str] = []
        while self._escalated and len(uids) < self.batch_size:
            uid, _ = self._escalated.popitem(last=False)
            uids.append(uid)
        if uids:
            self._num_batches += 1
            self._num_regens += len(uids)
        return uids

    def return_batch(self, uids: list[str]) -> None:
        """Put an un-consumed batch back at the front of the buffer.

        Used when the engine switch fails before generation; the uids keep their
        escalation so the next opportunity retries them.
        """
        for uid in reversed(uids):
            self._escalated[uid] = None
            self._escalated.move_to_end(uid, last=False)
        if uids:
            self._num_batches = max(0, self._num_batches - 1)
            self._num_regens = max(0, self._num_regens - len(uids))

    @staticmethod
    def _as_uid(value: Any) -> str:
        """Normalize a uid or a trajectory key (``uid_session_index``) to its uid.

        Both forms are accepted because callers pass either: the trainer prunes
        with ``batch.keys`` (trajectory keys) while the buffer holds bare uids.
        uids come from ``uuid4().hex`` and therefore contain no underscore, so the
        first segment is always the uid for both forms.
        """
        return str(value).split("_", 1)[0]

    def forget(self, keys: Any) -> None:
        """Drop the rollback counters belonging to ``keys`` (uids or trajectory keys).

        Called every step for the groups whose data was just trained on: their
        trajectories are consumed, so any rollback history they carry is dead and
        must not outlive them -- keeping it would grow the dict without bound and
        -- worse -- let a *new* generation of the same uid inherit a stale count.
        """
        uid_set = {self._as_uid(k) for k in keys}
        if not uid_set:
            return
        for key in [k for k in self._rollbacks if k.split("_", 1)[0] in uid_set]:
            del self._rollbacks[key]

    def release(self, uids: list[str]) -> None:
        """Conclude the escalation of ``uids``: wipe their counters and unblock them.

        Called after a successful regeneration -- the new generation starts from a
        clean slate, so it must trip ``num_rollbacks_trigger`` fresh times before
        escalating again -- and also when a batch is dropped without being
        regenerated, so neither case leaves a group blocked or carrying counts
        from inference that was discarded.
        """
        self.forget(uids)
        uid_set = {self._as_uid(uid) for uid in uids}
        self._pending.difference_update(uid_set)
        for uid in list(self._rejected):
            if uid in uid_set:
                del self._rejected[uid]

    def clear(self) -> None:
        """Drop all tracking state (used on checkpoint resume / shutdown)."""
        self._rollbacks.clear()
        self._escalated.clear()
        self._rejected.clear()
        self._pending.clear()
        self._attempted.clear()

    # ------------------------------------------------------------- rejections

    def pending_rejections(self) -> list[str]:
        """Uids currently marked for dropping (``strategy == "drop"``).

        The trainer hands these to the sampler, which clears the ones that are
        terminal and refills their slots. Uids still in flight stay in the set so
        they are dropped as soon as they settle.
        """
        return list(self._rejected)

    def rejections_present(self, present_uids: Any) -> None:
        """Release rejections whose group has left the transfer queue.

        A rejected uid disappears from the queue precisely when the sampler clears
        it (terminal) -- that group is fully handled, so its tracking is dropped.
        A uid still present was not terminal yet and remains queued for the next
        selection round.
        """
        if not self._rejected:
            return
        present = {str(uid) for uid in present_uids}
        settled = [uid for uid in self._rejected if uid not in present]
        if settled:
            self.release(settled)

    # ------------------------------------------------------------------ metrics

    def metrics(self) -> dict[str, float]:
        return {
            "precision_escalation/rollbacks": float(self._num_rollbacks),
            "precision_escalation/tracked_keys": float(len(self._rollbacks)),
            "precision_escalation/buffer": float(len(self._escalated)),
            "precision_escalation/rejected": float(len(self._rejected)),
            "precision_escalation/pending": float(len(self._pending)),
            "precision_escalation/escalated": float(self._num_escalated),
            "precision_escalation/dropped": float(self._num_rejected),
            "precision_escalation/attempted_groups": float(len(self._attempted)),
            "precision_escalation/batches": float(self._num_batches),
            "precision_escalation/regens": float(self._num_regens),
            "precision_escalation/overflow": float(self._num_overflow),
            "precision_escalation/evicted": float(self._num_evicted),
        }

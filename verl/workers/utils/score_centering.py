# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Score Centering (SC) for off-policy RL.

Reference: "Score Centering Stabilizes Off-policy Reinforcement Learning"
(arXiv 2609.20807v1), Sections 4 + Appendix A.

Score Centering replaces the vanilla REINFORCE score ``log p(a)`` of the *sampled*
token ``a`` with a *centered* score

    log p(a) - E_{v ~ q}[log p(v)]

where ``p`` is the training-time next-token distribution and ``q`` the sampling-time
distribution of the rollout engine. Centering removes the ``E_q[grad log p]`` component
of the policy gradient, which is the part that blows up when ``p != q`` (off-policy /
stale-rollout / low-precision engine mismatch): under any distribution aligned with the
policy, the expected centered gradient is zero, so a mis-specified ``q`` no longer injects
systematic drift.

Because ``q`` is over the full vocabulary and only a finite set of candidates can be
scored with the training engine, we implement the **top-k** estimator. The rollout engine
records, for every generated position:

  * the sampled token id (already available as ``responses``),
  * the sampler top-k token ids ``H``,
  * their logprob under the sampling distribution ``q``.

The training engine scores the *same* token ids ``H`` under ``p`` (full-vocabulary
normalization -- never a softmax restricted to the top-k logits). The unobserved tail
``v not in H`` is approximated by assuming the tail of ``q`` is proportional to the tail
of ``p``:

    q_tail / p_tail = rho,   q(v) ~ rho * p(v)  for v in tail.

which yields the implemented correction

    q_head = exp(sampler_topk_logprob)          # mass of H under q
    p_head = exp(train_topk_logprob)            # mass of H under p (grad flows)
    q_tail = 1 - sum(q_head)
    p_tail = 1 - sum(p_head)
    rho    = q_tail / p_tail                    # detached
    coeff  = detach(q_head - rho * p_head)      # detached, incl. p_head and rho
    correction = sum(coeff * train_topk_logprob)          # NOT detached
    token_loss = -detach(A) * (train_sampled_logprob - correction)

All probability / tail / coefficient arithmetic is done in FP32 with a floor on the tail
mass; the number of positions where the floor triggers is counted and reported.

Combined mode (SC + token-level IS/TIS/MIS)
-------------------------------------------
When per-token importance weights ``w(v) = f(p(v)/q(v))`` are available (the same ``f``
the existing rollout-correction machinery uses), the centered score generalizes to

    token_loss = -detach(A) * [ detach(w(a)) * log p(a)
                                - sum_v detach(q(v) * w(v)) * log p(v) ]

with the top-k tail handled as

    head_weights = f(p_head / q_head)
    alpha        = rho * f(1 / rho)
    coeff        = detach(q_head * head_weights - alpha * p_head)
    correction   = sum(coeff * train_topk_logprob)
    token_loss   = -detach(A) * [ detach(w(a)) * log p(a) - correction ]

``w(a)`` is taken from the *actually sampled* token's true sampling logprob even when that
token is not in the top-k (no tail approximation for the sampled term). MIS zeroes the
sampled-term weight for out-of-band tokens, but the correction term may still be non-zero
-- the IS mask is therefore applied only to the sampled term, never across the whole SC
loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Floor applied to the tail mass (1 - sum of observed head mass) before dividing. Keeps
# ``rho`` finite when the training model (or the sampler) concentrates ~all mass in the
# top-k. 1e-6 on a probability is far below any numerically meaningful mass.
TAIL_EPS = 1e-6

# Floor applied to a head probability before it is used as a denominator in ``p/q``.
HEAD_EPS = 1e-12

# Upper clamp on ``rho = q_tail / p_tail``. Guards the case where the training model
# concentrates essentially all mass on the top-k (p_tail -> 0) while the sampler retains a
# real tail: without the clamp rho diverges and would dominate the coefficient.
RHO_MAX = 1e4


@dataclass
class ScoreCenteringRuntimeConfig:
    """The subset of ``algorithm.score_centering`` the loss math needs.

    Kept separate from the trainer config so this module stays importable without the
    full config stack (and unit-testable on CPU).
    """

    is_mode: str = "none"
    is_low: float = 0.5
    is_high: float = 5.0
    coefficient_mode: str = "centered"

    @classmethod
    def from_dict(cls, d) -> "ScoreCenteringRuntimeConfig":
        return cls(
            is_mode=str(d.get("is_mode", "none")),
            is_low=float(d.get("is_low", 0.5)),
            is_high=float(d.get("is_high", 5.0)),
            coefficient_mode=str(d.get("coefficient_mode", "centered")),
        )


@dataclass
class ScoreCenteringStats:
    """Accumulators for SC diagnostics, filled in-place during the loss computation."""

    tail_floor_triggers: int = 0
    head_floor_triggers: int = 0
    num_positions: int = 0
    q_head_mass_sum: float = 0.0
    p_head_mass_sum: float = 0.0

    def as_metrics(self) -> dict[str, "Metric"]:
        from verl.utils.metric import AggregationType, Metric

        n = max(self.num_positions, 1)
        return {
            "sc/tail_floor_frac": Metric(
                value=torch.tensor(self.tail_floor_triggers / n, dtype=torch.float32),
                aggregation=AggregationType.MEAN,
            ),
            "sc/head_floor_frac": Metric(
                value=torch.tensor(self.head_floor_triggers / n, dtype=torch.float32),
                aggregation=AggregationType.MEAN,
            ),
            "sc/q_topk_coverage": Metric(
                value=torch.tensor(self.q_head_mass_sum / n, dtype=torch.float32),
                aggregation=AggregationType.MEAN,
            ),
            "sc/p_topk_coverage": Metric(
                value=torch.tensor(self.p_head_mass_sum / n, dtype=torch.float32),
                aggregation=AggregationType.MEAN,
            ),
        }


def _is_weight(ratio: torch.Tensor, mode: str, lo: float, hi: float) -> torch.Tensor:
    """Token IS weight ``f(p/q)`` matching the rollout-correction conventions.

    ``mode``:
      * ``"none"``  -> identity (pure SC, no IS)
      * ``"tis"``   -> truncated IS: ``clamp(ratio, max=hi)`` (upper clip only)
      * ``"mis"``   -> masked IS: ``ratio`` inside ``[lo, hi]``, else ``0``
    """
    if mode == "none":
        return ratio
    if mode == "tis":
        return ratio.clamp(max=hi)
    if mode == "mis":
        in_band = (ratio >= lo) & (ratio <= hi)
        return torch.where(in_band, ratio, torch.zeros_like(ratio))
    raise ValueError(f"unknown score_centering.is_mode={mode!r} (expected none/tis/mis)")


def score_centering_token_loss(
    *,
    train_sampled_logprob: torch.Tensor,
    train_topk_logprob: torch.Tensor,
    sampler_topk_logprob: torch.Tensor,
    topk_mask: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sampled_weight: Optional[torch.Tensor] = None,
    is_mode: str = "none",
    is_low: float = 0.5,
    is_high: float = 5.0,
    coefficient_mode: str = "centered",
    stats: Optional[ScoreCenteringStats] = None,
) -> torch.Tensor:
    """Per-token Score Centering loss matrix, shape ``[batch, seq]``.

    Args:
        train_sampled_logprob: ``log p(a)`` for the sampled token ``a`` -- ``[B, S]``,
            gradient flows.
        train_topk_logprob: ``log p(v)`` for the sampler top-k ids ``v in H``, normalized
            over the **full** vocabulary -- ``[B, S, K]``, gradient flows.
        sampler_topk_logprob: ``log q(v)`` for ``v in H`` from the rollout engine --
            ``[B, S, K]``, used detached only.
        topk_mask: boolean/0-1 validity mask for the K axis -- ``[B, S, K]``. Rows that are
            padding / absent slots must be 0.
        advantages: ``[B, S]`` (detached inside).
        response_mask: ``[B, S]`` boolean, 1 for generated tokens carrying loss.
        sampled_weight: optional ``detach(w(a))`` -- ``[B, S]``. Only used in combined mode;
            when ``None`` the sampled term carries weight 1.

    Returns:
        ``token_loss`` of shape ``[B, S]``, already zero on masked-out positions.
    """
    if train_topk_logprob.shape != sampler_topk_logprob.shape:
        raise ValueError(
            f"train/sampler top-k shape mismatch: {tuple(train_topk_logprob.shape)} "
            f"vs {tuple(sampler_topk_logprob.shape)}"
        )

    # Everything probability-shaped is FP32 regardless of the model / rollout dtype.
    t_sampled = train_sampled_logprob.float()
    t_topk = train_topk_logprob.float()
    s_topk = sampler_topk_logprob.float()
    mask = topk_mask.to(t_sampled.dtype)
    adv = advantages.detach().float()

    # ---- head mass (keep original top-k mass, do NOT renormalize to sum 1) ----
    q_head = torch.exp(s_topk) * mask  # [B,S,K]
    p_head = torch.exp(t_topk) * mask  # [B,S,K]  (grad flows through t_topk)
    q_head_mass = q_head.sum(dim=-1)  # [B,S]
    p_head_mass = p_head.sum(dim=-1)  # [B,S]

    # ---- tail mass with numerical floor ----
    q_tail_raw = 1.0 - q_head_mass
    p_tail_raw = 1.0 - p_head_mass
    if stats is not None:
        stats.tail_floor_triggers += int(
            ((q_tail_raw < TAIL_EPS) | (p_tail_raw < TAIL_EPS)).sum().item()
        )
    q_tail = q_tail_raw.clamp(min=TAIL_EPS)
    p_tail = p_tail_raw.clamp(min=TAIL_EPS)

    # ---- coefficient (fully detached: q_head, p_head and rho) ----
    rho = (q_tail / p_tail).clamp(max=RHO_MAX).detach()  # [B,S]

    if is_mode == "none":
        coeff = (q_head - rho.unsqueeze(-1) * p_head).detach()
        if coefficient_mode == "centered":
            pass
        elif coefficient_mode == "sampled_only":
            # Ablation: drop the tail correction, keep only centering over H.
            coeff = q_head.detach()
        else:
            raise ValueError(f"unknown score_centering.coefficient_mode={coefficient_mode!r}")
    else:
        # Combined SC + IS. head_weights uses per-token p/q; the tail uses the scalar
        # rho so that q_tail * f-ish scaling collapses to alpha = rho * f(1/rho).
        head_weights = _is_weight(
            p_head.detach() / q_head.detach().clamp(min=HEAD_EPS), is_mode, is_low, is_high
        )
        alpha = rho * _is_weight((1.0 / rho).unsqueeze(-1), is_mode, is_low, is_high).squeeze(-1)
        coeff = (q_head * head_weights - alpha.unsqueeze(-1) * p_head).detach()
        if stats is not None:
            stats.head_floor_triggers += int((q_head.detach() < HEAD_EPS).sum().item())

    # ---- correction: NOT detached; gradient flows through train_topk_logprob ----
    correction = (coeff * t_topk).sum(dim=-1)  # [B,S]

    # ---- sampled term ----
    if sampled_weight is not None:
        sampled_term = sampled_weight.detach().float() * t_sampled
    else:
        sampled_term = t_sampled

    token_loss = -adv * (sampled_term - correction)

    if stats is not None:
        valid = response_mask.to(t_sampled.dtype)
        stats.num_positions += int(valid.sum().item())
        stats.q_head_mass_sum += float((q_head_mass * valid).sum().item())
        stats.p_head_mass_sum += float((p_head_mass * valid).sum().item())

    return token_loss * response_mask.to(token_loss.dtype)


def score_centering_loss(
    *,
    train_sampled_logprob: torch.Tensor,
    train_topk_logprob: torch.Tensor,
    sampler_topk_logprob: torch.Tensor,
    topk_mask: torch.Tensor,
    old_logprob: torch.Tensor,
    sampler_sampled_logprob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config,
    global_batch_info: Optional[dict] = None,
    stats: Optional[ScoreCenteringStats] = None,
):
    """Aggregate the SC token loss and build the metric dict.

    ``old_logprob`` is the training-engine logprob saved before the update (``pi_k``) and
    ``sampler_sampled_logprob`` the rollout logprob of the sampled token (``q(a)``); their
    ratio is the sampled-token IS weight ``w(a)`` used in combined mode.

    Returns ``(loss_scalar, metrics_dict)``.
    """
    sampled_weight = None
    if config.is_mode != "none":
        ratio_a = (old_logprob.float() - sampler_sampled_logprob.float()).exp()
        # The sampled token may fall outside the top-k; w(a) then comes from its true
        # sampling logprob directly -- no tail approximation.
        sampled_weight = _is_weight(ratio_a, config.is_mode, config.is_low, config.is_high)

    if stats is None:
        stats = ScoreCenteringStats()
    token_loss = score_centering_token_loss(
        train_sampled_logprob=train_sampled_logprob,
        train_topk_logprob=train_topk_logprob,
        sampler_topk_logprob=sampler_topk_logprob,
        topk_mask=topk_mask,
        advantages=advantages,
        response_mask=response_mask,
        sampled_weight=sampled_weight,
        is_mode=config.is_mode,
        is_low=config.is_low,
        is_high=config.is_high,
        coefficient_mode=getattr(config, "coefficient_mode", "centered"),
        stats=stats,
    )

    gbi = global_batch_info or {}
    loss = agg_token_loss(token_loss, response_mask, loss_agg_mode, **gbi)

    metrics = stats.as_metrics() if stats is not None else {}
    return loss, metrics


def agg_token_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """Thin wrapper over ``agg_loss`` kept import-light so this module can be unit-tested
    without importing the full trainer."""
    from verl.trainer.ppo.core_algos import agg_loss

    return agg_loss(
        loss_mat=loss_mat,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
    )


__all__ = [
    "ScoreCenteringStats",
    "score_centering_loss",
    "score_centering_token_loss",
]

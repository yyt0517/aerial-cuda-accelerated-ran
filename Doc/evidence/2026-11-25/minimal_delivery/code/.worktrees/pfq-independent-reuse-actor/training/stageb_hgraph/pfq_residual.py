from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .model import StageBHGraphPolicy


def initialize_pfq_residual_head(actor: StageBHGraphPolicy) -> None:
    pair = actor.action_head.candidate_pair_score
    blank = actor.action_head.candidate_null_head
    if (
        pair is None
        or blank is None
        or not isinstance(pair[-1], nn.Linear)
        or not isinstance(blank[-1], nn.Linear)
    ):
        raise ValueError("candidate residual head is unavailable")
    nn.init.zeros_(pair[-1].weight)
    nn.init.zeros_(pair[-1].bias)
    nn.init.zeros_(blank[-1].weight)
    nn.init.zeros_(blank[-1].bias)


def should_initialize_pfq_residual(
    checkpoint: Mapping[str, object] | None,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if checkpoint is None:
        return True
    initialization = checkpoint.get("initialization")
    if not isinstance(initialization, Mapping):
        return True
    return initialization.get("mode") != "pfq_residual"

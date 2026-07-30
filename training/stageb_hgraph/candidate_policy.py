from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .graph_ops import PackedPrgCandidates, _estimate_ue_prg_demand_cap
from .types import SparseEntityGraph


@dataclass(frozen=True)
class CandidatePolicyConfig:
    mode: Literal["sample", "greedy"] = "sample"
    demand_cap_slack_bytes: float = 0.0
    hard_cap: bool = True
    usage_penalty: float = 0.4
    overcap_penalty: float = 8.0
    first_service_bonus: float = 0.01


@dataclass(frozen=True)
class CandidatePolicyTrace:
    choice: torch.Tensor
    order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor
    forced_step_count: int


def _step_logits(
    logits_row: torch.Tensor,
    candidate_ue_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    used: torch.Tensor,
    cap: torch.Tensor,
    cfg: CandidatePolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = cap.gather(0, candidate_ue_ids) - used.gather(0, candidate_ue_ids)
    main_valid = candidate_valid & (remaining > 0)
    main = logits_row[:-1] - cfg.usage_penalty * used.gather(0, candidate_ue_ids).float()
    main = main + cfg.first_service_bonus * (
        (used.gather(0, candidate_ue_ids) == 0) & (cap.gather(0, candidate_ue_ids) > 0)
    ).to(main.dtype)
    if cfg.hard_cap:
        main = main.masked_fill(~main_valid, -1.0e9)
    else:
        main = main - (~main_valid).to(main.dtype) * cfg.overcap_penalty
        main = main.masked_fill(~candidate_valid, -1.0e9)
    adjusted = torch.cat((main, logits_row[-1:]), dim=0)
    valid = torch.cat((main > -1.0e8, torch.ones(1, dtype=torch.bool, device=main.device)))
    return adjusted, valid


def _validate_inputs(
    packed: PackedPrgCandidates,
    candidate_logits: torch.Tensor,
    cfg: CandidatePolicyConfig,
    candidate_valid: torch.Tensor | None = None,
) -> None:
    if cfg.mode not in ("sample", "greedy"):
        raise ValueError(f"candidate policy mode must be sample or greedy, got {cfg.mode!r}")
    expected_logits = (packed.num_prg_tokens, packed.max_candidates + 1)
    if tuple(candidate_logits.shape) != expected_logits:
        raise ValueError(
            f"candidate_logits mismatch: got={tuple(candidate_logits.shape)} expected={expected_logits}"
        )
    if candidate_valid is not None:
        expected_valid = (packed.num_prg_tokens, packed.max_candidates)
        if tuple(candidate_valid.shape) != expected_valid:
            raise ValueError(
                f"candidate_valid mismatch: got={tuple(candidate_valid.shape)} expected={expected_valid}"
            )


def _run_policy_kernel(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    candidate_logits: torch.Tensor,
    cfg: CandidatePolicyConfig,
    order: torch.Tensor,
    *,
    stored_choice: torch.Tensor | None = None,
    stored_valid_at_step: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    device = candidate_logits.device
    valid_main = candidate_valid.to(device=device, dtype=torch.bool)
    safe_ue_ids = packed.ue_ids.to(device=device, dtype=torch.long)
    if safe_ue_ids.numel() > 0:
        safe_ue_ids = safe_ue_ids.clamp(
            min=0,
            max=max(0, int(graph.snapshot.static_meta.n_ue) - 1),
        )
    cap = _estimate_ue_prg_demand_cap(
        graph,
        slack_bytes=float(cfg.demand_cap_slack_bytes),
    ).to(device=device, dtype=torch.long)
    used = torch.zeros_like(cap)
    choices = torch.empty((packed.num_prg_tokens,), dtype=torch.long, device=device)
    step_valid_rows: list[torch.Tensor] = []
    chosen_logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    forced_step_count = 0

    for step_idx in range(packed.num_prg_tokens):
        token_idx = int(order[step_idx].item())
        adjusted, computed_valid = _step_logits(
            candidate_logits[token_idx],
            safe_ue_ids[token_idx],
            valid_main[token_idx],
            used,
            cap,
            cfg,
        )
        step_valid = (
            computed_valid
            if stored_valid_at_step is None
            else stored_valid_at_step[step_idx].to(device=device, dtype=torch.bool)
        )
        adjusted = adjusted.masked_fill(~step_valid, -1.0e9)
        logp_all = F.log_softmax(adjusted, dim=-1)
        probs = logp_all.exp()

        if stored_choice is not None:
            selected = stored_choice[token_idx].to(device=device, dtype=torch.long)
        elif cfg.mode == "sample":
            selected = torch.multinomial(probs, 1, generator=generator).squeeze(0)
        else:
            selected = torch.argmax(adjusted, dim=-1)

        choices[token_idx] = selected
        step_valid_rows.append(step_valid)
        chosen_logps.append(logp_all[selected])
        entropies.append(-(probs * logp_all).sum())
        forced_step_count += int(step_valid.sum().item() == 1)

        if int(selected.item()) < packed.blank_class_index:
            ue_idx = int(safe_ue_ids[token_idx, selected].item())
            used[ue_idx] += 1

    if packed.num_prg_tokens == 0:
        zero = candidate_logits.sum() * 0.0
        valid_at_step = torch.empty(
            (0, packed.max_candidates + 1),
            dtype=torch.bool,
            device=device,
        )
        return choices, valid_at_step, zero, 0, zero

    valid_at_step = torch.stack(step_valid_rows)
    mean_logp = torch.stack(chosen_logps).mean()
    mean_entropy = torch.stack(entropies).mean()
    return choices, valid_at_step, mean_logp, forced_step_count, mean_entropy


def select_candidate_policy_action(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_valid: torch.Tensor,
    candidate_logits: torch.Tensor,
    cfg: CandidatePolicyConfig,
    generator: torch.Generator | None = None,
) -> CandidatePolicyTrace:
    _validate_inputs(packed, candidate_logits, cfg, candidate_valid)
    valid_main = candidate_valid.to(device=candidate_logits.device, dtype=torch.bool)
    if packed.max_candidates > 0:
        initial_main = candidate_logits[:, :-1].detach().masked_fill(~valid_main, -1.0e9)
        best_initial_main = initial_main.max(dim=-1).values
        order = torch.argsort(best_initial_main, descending=True, stable=True)
    else:
        order = torch.arange(packed.num_prg_tokens, device=candidate_logits.device)
    choice, valid_at_step, logp, forced_step_count, entropy = _run_policy_kernel(
        graph,
        packed,
        candidate_valid,
        candidate_logits,
        cfg,
        order,
        generator=generator,
    )
    return CandidatePolicyTrace(
        choice=choice,
        order=order.detach(),
        valid_at_step=valid_at_step.detach(),
        logp=logp,
        entropy=entropy,
        forced_step_count=forced_step_count,
    )


def replay_candidate_policy_trace(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    candidate_logits: torch.Tensor,
    trace: CandidatePolicyTrace,
    cfg: CandidatePolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(packed, candidate_logits, cfg)
    _, _, logp, _, entropy = _run_policy_kernel(
        graph,
        packed,
        packed.mask,
        candidate_logits,
        cfg,
        trace.order,
        stored_choice=trace.choice,
        stored_valid_at_step=trace.valid_at_step,
    )
    return logp, entropy

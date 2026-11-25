from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from training.gnnrl.slot_layout import build_type0_slot_layout

from .graph_ops import (
    DecodedType0Action,
    PackedPrgCandidates,
    pack_prg_candidates,
    type0_action_preflight_reject_counts,
)
from .independent_actor import PFQIndependentReuseActor
from .independent_blank_diagnostics import (
    IndependentPolicyDiagnostics,
    concatenate_independent_policy_diagnostics,
    compute_independent_policy_diagnostics,
)
from .independent_contract import (
    W0_V2_CHOSEN_LOGPROB_ATOL,
    W0_V2_VALID_LOGITS_ATOL,
)
from .ctbs_headroom import (
    CtbsDecisionDiagnostics,
    CtbsHeadroomInputs,
    CtbsHeadroomState,
    concatenate_ctbs_diagnostics,
)
from .ctbs_non_starvation_guard import (
    CtbsNonStarvationGuardConfig,
    GuardStepDiagnostics,
    apply_ctbs_non_starvation_guard_step,
    concatenate_guard_diagnostics,
)
from .rescue_cohort_diagnostics import (
    RescueDecisionDiagnostics,
    build_rescue_decision_diagnostics,
    concatenate_rescue_decision_diagnostics,
)
from .service_accounting_audit import (
    ServiceAccountingDecisionDiagnostics,
    build_service_accounting_decision_diagnostics,
    concatenate_service_accounting_decision_diagnostics,
)
from .blank_expiry_attribution_audit import (
    BlankExpiryDecisionDiagnostics,
    build_blank_expiry_decision_diagnostics,
    concatenate_blank_expiry_decision_diagnostics,
)
from .types import SparseEntityGraph


@dataclass(frozen=True)
class IndependentPolicyTrace:
    actor_choice: torch.Tensor
    choice: torch.Tensor
    guard_override: torch.Tensor
    local_prg_order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor
    guard_diagnostics: GuardStepDiagnostics
    diagnostics: IndependentPolicyDiagnostics | None = None
    rescue_diagnostics: RescueDecisionDiagnostics | None = None
    service_accounting_diagnostics: ServiceAccountingDecisionDiagnostics | None = None
    raw_logits_at_step: torch.Tensor | None = None
    chosen_logprob_at_step: torch.Tensor | None = None
    ctbs_headroom_inputs: CtbsHeadroomInputs | None = None
    ctbs_diagnostics: CtbsDecisionDiagnostics | None = None
    blank_expiry_diagnostics: BlankExpiryDecisionDiagnostics | None = None


def resolve_full_candidate_width(graph: SparseEntityGraph) -> int:
    association = graph.snapshot.static_meta.action_mask_cell_ue.to(dtype=torch.int64)
    return int(association.sum(dim=1).max().item()) if association.numel() else 0


def estimate_candidate_service_bytes(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    if packed.num_prg_tokens != graph.snapshot.n_prg_tokens:
        raise ValueError("packed PRG token count does not match graph")
    if meta.post_eq_sinr is None:
        raise ValueError("independent service estimation requires post_eq_sinr")
    post_eq = meta.post_eq_sinr.to(device=packed.ue_ids.device, dtype=torch.float32)
    if post_eq.dim() == 2:
        post_eq = post_eq.unsqueeze(-1)
    if post_eq.dim() != 3 or tuple(post_eq.shape[:2]) != (meta.n_ue, meta.n_prg):
        raise ValueError(
            "post_eq_sinr must be [U,P] or [U,P,L], "
            f"got={tuple(post_eq.shape)}"
        )
    if packed.max_candidates == 0:
        return torch.empty(
            (packed.num_prg_tokens, 0),
            dtype=torch.float32,
            device=packed.ue_ids.device,
        )
    safe_ue = packed.ue_ids.clamp(min=0, max=max(0, meta.n_ue - 1))
    local_prg = meta.prg_local_index.to(device=safe_ue.device, dtype=torch.long)
    quality = post_eq[safe_ue, local_prg.unsqueeze(-1)]
    spectral_efficiency = torch.log2(1.0 + torch.clamp_min(quality, 0.0)).sum(dim=-1)
    rate_bps = spectral_efficiency * (16.0 * 12.0 * 30_000.0) * (11.0 / 14.0)
    service_bytes = torch.clamp_min(rate_bps * 0.0005 / 8.0, 1.0)
    return torch.where(packed.mask.to(service_bytes.device), service_bytes, torch.ones_like(service_bytes))


def _initial_demand_bytes(graph: SparseEntityGraph) -> torch.Tensor:
    features = graph.snapshot.ue_features.values
    demand = torch.clamp_min(features[:, 0], 0.0)
    new_data = features[:, 6] > 0.5 if features.shape[1] > 6 else torch.zeros_like(demand, dtype=torch.bool)
    return torch.where(new_data & (demand <= 0.0), torch.ones_like(demand), demand)


def _candidate_state(
    packed: PackedPrgCandidates,
    token_indices: torch.Tensor,
    initial_demand: torch.Tensor,
    remaining_demand: torch.Tensor,
    allocated_prgs: torch.Tensor,
    predicted_service: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_ids = packed.ue_ids.to(token_indices.device).index_select(0, token_indices)
    candidate_mask = packed.mask.to(token_indices.device).index_select(0, token_indices)
    safe_ids = candidate_ids.clamp(min=0, max=max(0, int(initial_demand.numel()) - 1))
    denominator = initial_demand.gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids).clamp_min(1.0)
    remaining = remaining_demand.gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids)
    allocated = allocated_prgs.gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids)
    predicted = predicted_service.gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids)
    features = torch.stack(
        (
            torch.clamp(remaining / denominator, 0.0, 1.0),
            torch.clamp(allocated / 17.0, 0.0, 1.0),
            torch.clamp(predicted / denominator, 0.0, 1.0),
        ),
        dim=-1,
    )
    valid_main = candidate_mask & (remaining > 0.0)
    return features, valid_main


def _validate_policy_inputs(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
) -> None:
    meta = graph.snapshot.static_meta
    if packed.num_prg_tokens != meta.n_cell * meta.n_prg:
        raise ValueError("packed PRG token count mismatch")
    expected_width = resolve_full_candidate_width(graph)
    if packed.max_candidates < expected_width:
        raise ValueError(
            "candidate width would truncate topology candidates: "
            f"got={packed.max_candidates} expected_at_least={expected_width}"
        )


def _select_hierarchical_greedy(
    adjusted_logits: torch.Tensor,
    valid_main: torch.Tensor,
    blank_class_index: int,
) -> torch.Tensor:
    """Choose aggregate Serve strictly over Blank, then the best valid UE."""
    if adjusted_logits.dim() != 2:
        raise ValueError("hierarchical logits must be rank-2")
    if (
        blank_class_index <= 0
        or blank_class_index != adjusted_logits.shape[1] - 1
        or valid_main.shape != (adjusted_logits.shape[0], blank_class_index)
        or valid_main.dtype is not torch.bool
        or valid_main.device != adjusted_logits.device
    ):
        raise ValueError("hierarchical logits, mask, or blank index mismatch")
    if not bool(torch.isfinite(adjusted_logits[:, blank_class_index]).all().item()):
        raise ValueError("hierarchical blank logits must be finite")
    if not bool(torch.isfinite(adjusted_logits[:, :blank_class_index][valid_main]).all().item()):
        raise ValueError("hierarchical valid UE logits must be finite")

    service_logits = adjusted_logits[:, :blank_class_index].masked_fill(~valid_main, -torch.inf)
    serve_logit = torch.logsumexp(service_logits, dim=-1)
    best_ue = service_logits.argmax(dim=-1)
    has_valid = valid_main.any(dim=-1)
    return torch.where(
        has_valid & (serve_logit > adjusted_logits[:, blank_class_index]),
        best_ue,
        torch.full_like(best_ue, blank_class_index),
    )


def _run_policy(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    *,
    mode: Literal["sample", "greedy", "hierarchical_greedy"],
    generator: torch.Generator | None = None,
    collect_diagnostics: bool = False,
    collect_numerical_trace: bool = False,
    collect_blank_expiry_attribution: bool = False,
    stored_trace: IndependentPolicyTrace | None = None,
    teacher_actions: tuple[torch.Tensor, torch.Tensor] | None = None,
    ctbs_headroom_inputs: CtbsHeadroomInputs | None = None,
    non_starvation_guard: CtbsNonStarvationGuardConfig = CtbsNonStarvationGuardConfig(),
) -> IndependentPolicyTrace:
    if mode not in ("sample", "greedy", "hierarchical_greedy"):
        raise ValueError(
            "independent policy mode must be sample, greedy, or hierarchical_greedy, "
            f"got {mode!r}"
        )
    if non_starvation_guard.enabled and mode != "greedy":
        raise ValueError("enabled non-starvation guard requires greedy mode")
    if non_starvation_guard.enabled and teacher_actions is not None:
        raise ValueError("enabled non-starvation guard does not support teacher actions")
    if non_starvation_guard.enabled and ctbs_headroom_inputs is None:
        raise ValueError("enabled non-starvation guard requires CTBS inputs")
    _validate_policy_inputs(actor, graph, packed)
    meta = graph.snapshot.static_meta
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype
    encoded = actor.encode_graph(graph)
    raw_service = estimate_candidate_service_bytes(graph, packed).to(device=device, dtype=dtype)
    effective_service = raw_service
    initial = _initial_demand_bytes(graph).to(device=device, dtype=dtype)
    remaining = initial.clone()
    allocated = torch.zeros_like(initial)
    predicted = torch.zeros_like(initial)
    local_order = torch.arange(meta.n_prg, device=device, dtype=torch.long)
    if stored_trace is not None:
        stored_order = torch.as_tensor(stored_trace.local_prg_order, device=device, dtype=torch.long)
        if not torch.equal(stored_order, local_order):
            raise ValueError("stored local PRG order does not match fixed order")
    if stored_trace is not None and teacher_actions is not None:
        raise ValueError("stored trace and teacher actions are mutually exclusive")
    if stored_trace is not None:
        if ctbs_headroom_inputs is not None:
            raise ValueError("replay must use the CTBS tuple frozen in the stored trace")
        ctbs_headroom_inputs = stored_trace.ctbs_headroom_inputs
    stored_actor_choices: torch.Tensor | None = None
    stored_executed_choices: torch.Tensor | None = None
    stored_guard_overrides: torch.Tensor | None = None
    if stored_trace is not None:
        expected_shape = (meta.n_cell * meta.n_prg,)
        stored_actor_choices = torch.as_tensor(
            stored_trace.actor_choice, device=device, dtype=torch.long
        ).reshape(-1)
        stored_executed_choices = torch.as_tensor(
            stored_trace.choice, device=device, dtype=torch.long
        ).reshape(-1)
        stored_guard_overrides = torch.as_tensor(
            stored_trace.guard_override, device=device, dtype=torch.bool
        ).reshape(-1)
        if (
            stored_actor_choices.shape != expected_shape
            or stored_executed_choices.shape != expected_shape
            or stored_guard_overrides.shape != expected_shape
        ):
            raise ValueError("stored raw, executed, or guard override shape mismatch")
        expected_override = stored_actor_choices != stored_executed_choices
        if not torch.equal(expected_override, stored_guard_overrides):
            raise ValueError("stored guard override disagrees with raw/executed choices")
        if bool(
            (
                stored_guard_overrides
                & (
                    (stored_actor_choices != packed.blank_class_index)
                    | (stored_executed_choices == packed.blank_class_index)
                )
            ).any().item()
        ):
            raise ValueError("stored guard override must replace Blank with nonblank")

        local_major_actor = stored_actor_choices.reshape(
            meta.n_cell, meta.n_prg
        ).transpose(0, 1).reshape(-1)
        local_major_executed = stored_executed_choices.reshape(
            meta.n_cell, meta.n_prg
        ).transpose(0, 1).reshape(-1)
        local_major_override = stored_guard_overrides.reshape(
            meta.n_cell, meta.n_prg
        ).transpose(0, 1).reshape(-1)
        guard_diagnostics = stored_trace.guard_diagnostics
        diagnostic_actor = torch.as_tensor(
            guard_diagnostics.actor_choice, device=device, dtype=torch.long
        ).reshape(-1)
        diagnostic_executed = torch.as_tensor(
            guard_diagnostics.executed_choice, device=device, dtype=torch.long
        ).reshape(-1)
        diagnostic_override = torch.as_tensor(
            guard_diagnostics.guard_override, device=device, dtype=torch.bool
        ).reshape(-1)
        if not (
            torch.equal(diagnostic_actor, local_major_actor)
            and torch.equal(diagnostic_executed, local_major_executed)
            and torch.equal(diagnostic_override, local_major_override)
        ):
            raise ValueError(
                "stored guard diagnostics disagree with raw/executed choices"
            )
    ctbs_states: list[CtbsHeadroomState] | None = None
    if ctbs_headroom_inputs is not None:
        if len(ctbs_headroom_inputs.ue_ids) != meta.n_ue:
            raise ValueError("CTBS UE count does not match graph")
        if any(len(row) != meta.n_prg for row in ctbs_headroom_inputs.prg_layer_cap):
            raise ValueError("CTBS PRG count does not match graph")
        association = meta.action_mask_cell_ue.to(dtype=torch.bool).detach().cpu()
        serving = tuple(
            int(torch.nonzero(association[:, ue], as_tuple=False).flatten().item())
            for ue in range(meta.n_ue)
        )
        if serving != ctbs_headroom_inputs.serving_cell:
            raise ValueError("CTBS serving-cell identities do not match graph association")
        ctbs_states = [
            CtbsHeadroomState(
                initial_demand_bytes=float(initial[ue].item()),
                reconstructed_mcs=ctbs_headroom_inputs.reconstructed_mcs_for_ue(ue),
                config=ctbs_headroom_inputs.config,
            )
            for ue in range(meta.n_ue)
        ]
    teacher_ue = None
    teacher_prg = None
    if teacher_actions is not None:
        teacher_ue = torch.as_tensor(
            teacher_actions[0], device=device, dtype=torch.long
        ).reshape(meta.n_sched_ue)
        teacher_prg = torch.as_tensor(
            teacher_actions[1], device=device, dtype=torch.long
        ).reshape(meta.n_cell * meta.n_prg)

    actor_choices = torch.full(
        (meta.n_cell * meta.n_prg,),
        packed.blank_class_index,
        dtype=torch.long,
        device=device,
    )
    choices = torch.full(
        (meta.n_cell * meta.n_prg,),
        packed.blank_class_index,
        dtype=torch.long,
        device=device,
    )
    guard_overrides = torch.zeros(
        (meta.n_cell * meta.n_prg,), dtype=torch.bool, device=device
    )
    override_used_by_cell = torch.zeros(
        meta.n_cell, dtype=torch.bool, device=device
    )
    masks: list[torch.Tensor] = []
    logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    diagnostic_rows: list[IndependentPolicyDiagnostics] = []
    rescue_diagnostic_rows: list[RescueDecisionDiagnostics] = []
    service_accounting_rows: list[ServiceAccountingDecisionDiagnostics] = []
    numerical_logits: list[torch.Tensor] = []
    numerical_chosen_logps: list[torch.Tensor] = []
    ctbs_diagnostic_rows: list[CtbsDecisionDiagnostics] = []
    blank_expiry_rows: list[BlankExpiryDecisionDiagnostics] = []
    guard_diagnostic_rows: list[GuardStepDiagnostics] = []
    for local_prg in range(meta.n_prg):
        tokens = torch.arange(meta.n_cell, device=device, dtype=torch.long) * meta.n_prg + local_prg
        sequence_features, valid_main = _candidate_state(
            packed,
            tokens,
            initial,
            remaining,
            allocated,
            predicted,
        )
        logits = actor.score_reuse_group(
            encoded,
            graph,
            packed,
            local_prg,
            sequence_features,
        )
        valid = torch.cat(
            (
                valid_main,
                torch.ones((meta.n_cell, 1), dtype=torch.bool, device=device),
            ),
            dim=-1,
        )
        if stored_trace is not None:
            stored_valid = torch.as_tensor(
                stored_trace.valid_at_step[local_prg],
                device=device,
                dtype=torch.bool,
            )
            if not torch.equal(stored_valid, valid):
                raise ValueError(f"stored mask mismatch at local PRG {local_prg}")
        adjusted = logits.masked_fill(~valid, -1.0e9)
        logp_all = F.log_softmax(adjusted, dim=-1)
        probabilities = logp_all.exp()
        if stored_trace is not None:
            assert stored_actor_choices is not None
            assert stored_executed_choices is not None
            actor_selected = stored_actor_choices.index_select(0, tokens)
            executed_selected = stored_executed_choices.index_select(0, tokens)
            for layer_name, layer_choice in (
                ("raw actor", actor_selected),
                ("executed", executed_selected),
            ):
                if bool(
                    (
                        (layer_choice < 0)
                        | (layer_choice > packed.blank_class_index)
                    ).any().item()
                ):
                    raise ValueError(
                        f"stored {layer_name} choice out of range at local PRG {local_prg}"
                    )
                if not bool(
                    valid.gather(1, layer_choice.unsqueeze(-1)).all().item()
                ):
                    raise ValueError(
                        f"stored {layer_name} choice is invalid at local PRG {local_prg}"
                    )
        elif teacher_ue is not None and teacher_prg is not None:
            selected_rows: list[int] = []
            for cell_idx in range(meta.n_cell):
                token_idx = int(tokens[cell_idx].item())
                native_idx = local_prg * meta.n_cell + cell_idx
                slot_idx = int(teacher_prg[native_idx].item())
                if slot_idx < 0:
                    target = packed.blank_class_index
                else:
                    if slot_idx >= meta.n_sched_ue:
                        raise ValueError(f"teacher slot out of range at token {token_idx}")
                    ue_idx = int(teacher_ue[slot_idx].item())
                    if ue_idx < 0 or ue_idx >= meta.n_ue:
                        raise ValueError(f"teacher slot is empty at token {token_idx}")
                    if not bool(meta.action_mask_cell_ue[cell_idx, ue_idx].item()):
                        raise ValueError(f"teacher UE is assigned to wrong cell at token {token_idx}")
                    matches = torch.nonzero(
                        packed.mask[token_idx].to(device=device)
                        & (packed.ue_ids[token_idx].to(device=device) == ue_idx),
                        as_tuple=False,
                    ).flatten()
                    if matches.numel() != 1:
                        raise ValueError(f"teacher UE missing or duplicated at token {token_idx}")
                    target = int(matches[0].item())
                if not bool(valid[cell_idx, target].item()):
                    if target < packed.blank_class_index:
                        ue_idx = int(
                            packed.ue_ids[token_idx, target].to(device=device).item()
                        )
                        raise ValueError(
                            "teacher target is invalid at token "
                            f"{token_idx}: cell={cell_idx} local_prg={local_prg} "
                            f"ue={ue_idx} initial_demand={float(initial[ue_idx].item()):.6f} "
                            f"remaining_demand={float(remaining[ue_idx].item()):.6f} "
                            f"allocated_prgs={float(allocated[ue_idx].item()):.0f} "
                            f"predicted_service={float(predicted[ue_idx].item()):.6f} "
                            f"next_service={float(raw_service[token_idx, target].item()):.6f}"
                        )
                    raise ValueError(f"teacher target is invalid at token {token_idx}")
                selected_rows.append(target)
            actor_selected = torch.tensor(
                selected_rows, dtype=torch.long, device=device
            )
        elif mode == "sample":
            actor_selected = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(-1)
        elif mode == "hierarchical_greedy":
            actor_selected = _select_hierarchical_greedy(
                adjusted,
                valid_main,
                packed.blank_class_index,
            )
        else:
            actor_selected = adjusted.argmax(dim=-1)

        if collect_blank_expiry_attribution:
            candidate_ids = packed.ue_ids.to(device=device).index_select(0, tokens)
            candidate_present = packed.mask.to(device=device).index_select(0, tokens)
            safe_ids = candidate_ids.clamp(min=0, max=max(0, meta.n_ue - 1))

            def gather_ue(values: torch.Tensor, *, absent_value: float = 0.0) -> torch.Tensor:
                gathered = values.to(device=device).gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids)
                return torch.where(
                    candidate_present,
                    gathered,
                    torch.full_like(gathered, float(absent_value)),
                )

            ue_features = graph.snapshot.ue_features.values.to(device=device, dtype=dtype)
            blank_expiry_rows.append(
                build_blank_expiry_decision_diagnostics(
                    cell_index=meta.prg_owner_cell.to(device=device, dtype=torch.long).index_select(0, tokens),
                    local_prg=local_prg,
                    candidate_ue_ids=candidate_ids,
                    candidate_present=candidate_present,
                    candidate_valid=valid_main,
                    logits=logits,
                    selected_class=actor_selected,
                    initial_backlog_bytes=gather_ue(initial),
                    remaining_demand_before_bytes=gather_ue(remaining),
                    cumulative_capacity_before_bytes=gather_ue(predicted),
                    hol_delay_ms=gather_ue(ue_features[:, 8] if ue_features.shape[1] > 8 else torch.zeros_like(initial)),
                    ttl_slack_ms=gather_ue(
                        ue_features[:, 9] if ue_features.shape[1] > 9 else torch.full_like(initial, -1.0),
                        absent_value=-1.0,
                    ),
                )
            )

        diagnostic_row: IndependentPolicyDiagnostics | None = None
        if collect_diagnostics:
            diagnostic_row = compute_independent_policy_diagnostics(
                adjusted_logits=adjusted,
                probabilities=probabilities,
                valid_main=valid_main,
                selected=actor_selected,
                blank_class_index=packed.blank_class_index,
            )
            diagnostic_rows.append(diagnostic_row)

        if stored_trace is None:
            guard_result = apply_ctbs_non_starvation_guard_step(
                config=non_starvation_guard,
                blank_class_index=packed.blank_class_index,
                local_prg=local_prg,
                actor_choice=actor_selected,
                candidate_ue_ids=packed.ue_ids.to(device=device).index_select(
                    0, tokens
                ),
                valid_main=valid_main,
                remaining_demand_bytes=remaining,
                ue_features=graph.snapshot.ue_features.values.to(
                    device=device, dtype=dtype
                ),
                ue_feature_names=graph.snapshot.ue_features.names,
                ctbs_states=ctbs_states,
                prg_layer_cap=(
                    ctbs_headroom_inputs.prg_layer_cap
                    if ctbs_headroom_inputs is not None
                    else None
                ),
                override_used_by_cell=override_used_by_cell,
            )
            executed_selected = guard_result.executed_choice
            override_used_by_cell = guard_result.override_used_by_cell
            step_guard_override = guard_result.diagnostics.guard_override
            guard_diagnostic_rows.append(guard_result.diagnostics)
        else:
            assert stored_guard_overrides is not None
            step_guard_override = stored_guard_overrides.index_select(0, tokens)

        if collect_diagnostics:
            assert diagnostic_row is not None
            rescue_diagnostic_rows.append(
                build_rescue_decision_diagnostics(
                    graph=graph,
                    packed=packed,
                    token_indices=tokens,
                    local_prg=local_prg,
                    selected=executed_selected,
                    service_bytes=raw_service,
                    sequential_features=sequence_features,
                    policy_diagnostics=diagnostic_row,
                )
            )
            service_accounting_rows.append(
                build_service_accounting_decision_diagnostics(
                    packed=packed,
                    token_indices=tokens,
                    selected=executed_selected,
                    raw_service_bytes=raw_service,
                    effective_service_bytes=effective_service,
                    initial_demand_bytes=initial,
                    remaining_demand_before_bytes=remaining,
                )
            )

        actor_choices[tokens] = actor_selected
        choices[tokens] = executed_selected
        guard_overrides[tokens] = step_guard_override
        masks.append(valid)
        chosen_logp = logp_all.gather(
            1, actor_selected.unsqueeze(-1)
        ).squeeze(-1)
        logps.append(chosen_logp)
        entropies.append(-(probabilities * logp_all).sum(dim=-1))
        if collect_numerical_trace:
            numerical_logits.append(logits.detach())
            numerical_chosen_logps.append(chosen_logp.detach())

        # The three cells above were all scored from one pre-update state. Only
        # after the reuse group is complete do selected nonblank UEs update.
        selected_main = executed_selected < packed.blank_class_index
        ctbs_selected_ue = torch.full(
            (meta.n_cell,), -1, dtype=torch.long, device=device
        )
        ctbs_raw_service = torch.zeros((meta.n_cell,), dtype=dtype, device=device)
        ctbs_capacity_before = torch.zeros_like(ctbs_raw_service)
        ctbs_capacity_after = torch.zeros_like(ctbs_raw_service)
        ctbs_remaining_before = torch.zeros_like(ctbs_raw_service)
        ctbs_remaining_after = torch.zeros_like(ctbs_raw_service)
        ctbs_layer_before = torch.zeros((meta.n_cell,), dtype=torch.long, device=device)
        ctbs_layer_after = torch.zeros((meta.n_cell,), dtype=torch.long, device=device)
        if bool(selected_main.any().item()):
            selected_class = executed_selected.clamp(
                max=max(0, packed.max_candidates - 1)
            )
            candidate_ids = packed.ue_ids.to(device=device).index_select(0, tokens)
            chosen_ues = candidate_ids.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)
            selected_service = effective_service.index_select(0, tokens).gather(
                1,
                selected_class.unsqueeze(-1),
            ).squeeze(-1)
            for cell_idx in torch.nonzero(selected_main, as_tuple=False).flatten().tolist():
                ue_idx = int(chosen_ues[cell_idx].item())
                ctbs_selected_ue[cell_idx] = ue_idx
                ctbs_raw_service[cell_idx] = selected_service[cell_idx]
                if ctbs_states is None or ctbs_headroom_inputs is None:
                    service_bytes = selected_service[cell_idx]
                    remaining[ue_idx] = torch.clamp_min(remaining[ue_idx] - service_bytes, 0.0)
                    allocated[ue_idx] += 1.0
                    predicted[ue_idx] += service_bytes
                else:
                    previous_state = ctbs_states[ue_idx]
                    ctbs_capacity_before[cell_idx] = float(
                        previous_state.cumulative_tbs_bytes
                    )
                    ctbs_remaining_before[cell_idx] = float(
                        previous_state.remaining_demand_bytes
                    )
                    ctbs_layer_before[cell_idx] = (
                        min(previous_state.selected_layer_caps)
                        if previous_state.selected_layer_caps
                        else 0
                    )
                    state = ctbs_states[ue_idx].with_selected_prg(
                        local_prg,
                        layer_cap=ctbs_headroom_inputs.prg_layer_cap[ue_idx][local_prg],
                    )
                    ctbs_states[ue_idx] = state
                    remaining[ue_idx] = state.remaining_demand_bytes
                    allocated[ue_idx] = float(len(state.selected_local_prgs))
                    predicted[ue_idx] = float(state.cumulative_tbs_bytes)
                    ctbs_capacity_after[cell_idx] = float(
                        state.cumulative_tbs_bytes
                    )
                    ctbs_remaining_after[cell_idx] = float(
                        state.remaining_demand_bytes
                    )
                    ctbs_layer_after[cell_idx] = min(state.selected_layer_caps)
        if collect_diagnostics and ctbs_states is not None:
            ctbs_diagnostic_rows.append(
                CtbsDecisionDiagnostics(
                    selected_ue=ctbs_selected_ue.detach(),
                    cell_index=torch.arange(
                        meta.n_cell, device=device, dtype=torch.long
                    ),
                    local_prg_index=torch.full(
                        (meta.n_cell,), local_prg, device=device, dtype=torch.long
                    ),
                    selected_nonblank=selected_main.detach(),
                    raw_service_bytes=ctbs_raw_service.detach(),
                    cumulative_tbs_before_bytes=ctbs_capacity_before.detach(),
                    cumulative_tbs_after_bytes=ctbs_capacity_after.detach(),
                    remaining_before_bytes=ctbs_remaining_before.detach(),
                    remaining_after_bytes=ctbs_remaining_after.detach(),
                    layer_before=ctbs_layer_before.detach(),
                    layer_after=ctbs_layer_after.detach(),
                )
            )

    return IndependentPolicyTrace(
        actor_choice=actor_choices,
        choice=choices,
        guard_override=guard_overrides,
        local_prg_order=local_order.detach(),
        valid_at_step=torch.stack(masks).detach(),
        logp=torch.cat(logps).mean(),
        entropy=torch.cat(entropies).mean(),
        guard_diagnostics=(
            stored_trace.guard_diagnostics
            if stored_trace is not None
            else concatenate_guard_diagnostics(guard_diagnostic_rows)
        ),
        diagnostics=(
            concatenate_independent_policy_diagnostics(diagnostic_rows)
            if collect_diagnostics
            else None
        ),
        rescue_diagnostics=(
            concatenate_rescue_decision_diagnostics(rescue_diagnostic_rows)
            if collect_diagnostics
            else None
        ),
        service_accounting_diagnostics=(
            concatenate_service_accounting_decision_diagnostics(
                service_accounting_rows
            )
            if collect_diagnostics
            else None
        ),
        raw_logits_at_step=(
            torch.stack(numerical_logits) if collect_numerical_trace else None
        ),
        chosen_logprob_at_step=(
            torch.stack(numerical_chosen_logps) if collect_numerical_trace else None
        ),
        ctbs_headroom_inputs=ctbs_headroom_inputs,
        ctbs_diagnostics=(
            concatenate_ctbs_diagnostics(ctbs_diagnostic_rows)
            if ctbs_diagnostic_rows
            else None
        ),
        blank_expiry_diagnostics=(
            concatenate_blank_expiry_decision_diagnostics(blank_expiry_rows)
            if blank_expiry_rows
            else None
        ),
    )


def select_independent_action(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    mode: Literal["sample", "greedy", "hierarchical_greedy"],
    generator: torch.Generator | None = None,
    collect_diagnostics: bool = False,
    collect_numerical_trace: bool = False,
    collect_blank_expiry_attribution: bool = False,
    ctbs_headroom_inputs: CtbsHeadroomInputs | None = None,
    non_starvation_guard: CtbsNonStarvationGuardConfig = CtbsNonStarvationGuardConfig(),
) -> tuple[IndependentPolicyTrace, DecodedType0Action]:
    trace = _run_policy(
        actor,
        graph,
        packed,
        mode=mode,
        generator=generator,
        collect_diagnostics=collect_diagnostics,
        collect_numerical_trace=collect_numerical_trace,
        collect_blank_expiry_attribution=collect_blank_expiry_attribution,
        ctbs_headroom_inputs=ctbs_headroom_inputs,
        non_starvation_guard=non_starvation_guard,
    )
    decoded = materialize_independent_choices_to_type0(graph, packed, trace.choice)
    return trace, decoded


def replay_independent_trace(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    trace: IndependentPolicyTrace,
) -> tuple[torch.Tensor, torch.Tensor, DecodedType0Action]:
    replayed = _run_policy(
        actor,
        graph,
        packed,
        mode="greedy",
        stored_trace=trace,
    )
    if not torch.equal(replayed.choice, torch.as_tensor(trace.choice, device=replayed.choice.device)):
        raise ValueError("replayed choices differ from stored choices")
    decoded = materialize_independent_choices_to_type0(graph, packed, replayed.choice)
    return replayed.logp, replayed.entropy, decoded


def compare_candidate_padding_semantics(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    *,
    reference_width: int,
    padded_width: int,
    mode: Literal["greedy", "hierarchical_greedy"],
) -> dict[str, bool]:
    reference = pack_prg_candidates(
        graph,
        max_candidates=int(reference_width),
        fixed_width=True,
    )
    padded = pack_prg_candidates(
        graph,
        max_candidates=int(padded_width),
        fixed_width=True,
    )
    if reference.max_candidates >= padded.max_candidates:
        raise ValueError("padded candidate width must exceed reference width")
    reference_trace, reference_action = select_independent_action(
        actor,
        graph,
        reference,
        mode=mode,
    )
    padded_trace, padded_action = select_independent_action(
        actor,
        graph,
        padded,
        mode=mode,
    )
    reference_logp, _reference_entropy, reference_replay = replay_independent_trace(
        actor,
        graph,
        reference,
        reference_trace,
    )
    padded_logp, _padded_entropy, padded_replay = replay_independent_trace(
        actor,
        graph,
        padded,
        padded_trace,
    )

    normalized_reference_mask = torch.zeros_like(padded_trace.valid_at_step)
    normalized_reference_mask[..., : reference.max_candidates] = (
        reference_trace.valid_at_step[..., : reference.max_candidates]
    )
    normalized_reference_mask[..., padded.blank_class_index] = (
        reference_trace.valid_at_step[..., reference.blank_class_index]
    )
    extra_slice = slice(reference.max_candidates, padded.max_candidates)
    extra_padding_invalid = bool(
        not padded.mask[:, extra_slice].any().item()
        and not padded_trace.valid_at_step[..., extra_slice].any().item()
    )
    replay_equal = bool(
        torch.equal(reference_logp, reference_trace.logp)
        and torch.equal(padded_logp, padded_trace.logp)
        and torch.equal(reference_action.action_ue_select, reference_replay.action_ue_select)
        and torch.equal(reference_action.action_prg_alloc, reference_replay.action_prg_alloc)
        and torch.equal(padded_action.action_ue_select, padded_replay.action_ue_select)
        and torch.equal(padded_action.action_prg_alloc, padded_replay.action_prg_alloc)
    )
    return {
        "candidate_choice_equal": bool(
            torch.equal(
                reference_action.chosen_ue_per_prg,
                padded_action.chosen_ue_per_prg,
            )
        ),
        "mask_equal": bool(
            torch.equal(normalized_reference_mask, padded_trace.valid_at_step)
        ),
        "logprob_equal": bool(torch.equal(reference_trace.logp, padded_trace.logp)),
        "replay_equal": replay_equal,
        "type0_action_equal": bool(
            torch.equal(reference_action.action_ue_select, padded_action.action_ue_select)
            and torch.equal(reference_action.action_prg_alloc, padded_action.action_prg_alloc)
        ),
        "extra_padding_invalid": extra_padding_invalid,
    }


def compare_candidate_padding_fp32_equivalence(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    *,
    reference_width: int,
    padded_width: int,
    mode: Literal["greedy", "hierarchical_greedy"],
    valid_logits_atol: float = W0_V2_VALID_LOGITS_ATOL,
    chosen_logprob_atol: float = W0_V2_CHOSEN_LOGPROB_ATOL,
) -> dict[str, bool | float | int]:
    """Compare discrete behavior exactly while bounding cross-width FP32 drift."""
    if valid_logits_atol < 0.0 or chosen_logprob_atol < 0.0:
        raise ValueError("W0-v2 numerical tolerances must be non-negative")
    reference = pack_prg_candidates(
        graph,
        max_candidates=int(reference_width),
        fixed_width=True,
    )
    padded = pack_prg_candidates(
        graph,
        max_candidates=int(padded_width),
        fixed_width=True,
    )
    if reference.max_candidates >= padded.max_candidates:
        raise ValueError("padded candidate width must exceed reference width")
    reference_trace, reference_action = select_independent_action(
        actor,
        graph,
        reference,
        mode=mode,
        collect_numerical_trace=True,
    )
    padded_trace, padded_action = select_independent_action(
        actor,
        graph,
        padded,
        mode=mode,
        collect_numerical_trace=True,
    )
    if (
        reference_trace.raw_logits_at_step is None
        or reference_trace.chosen_logprob_at_step is None
        or padded_trace.raw_logits_at_step is None
        or padded_trace.chosen_logprob_at_step is None
    ):
        raise RuntimeError("W0-v2 numerical traces were not collected")
    reference_logp, _reference_entropy, reference_replay = replay_independent_trace(
        actor,
        graph,
        reference,
        reference_trace,
    )
    padded_logp, _padded_entropy, padded_replay = replay_independent_trace(
        actor,
        graph,
        padded,
        padded_trace,
    )

    normalized_reference_mask = torch.zeros_like(padded_trace.valid_at_step)
    normalized_reference_mask[..., : reference.max_candidates] = (
        reference_trace.valid_at_step[..., : reference.max_candidates]
    )
    normalized_reference_mask[..., padded.blank_class_index] = (
        reference_trace.valid_at_step[..., reference.blank_class_index]
    )
    normalized_reference_logits = torch.zeros_like(padded_trace.raw_logits_at_step)
    normalized_reference_logits[..., : reference.max_candidates] = (
        reference_trace.raw_logits_at_step[..., : reference.max_candidates]
    )
    normalized_reference_logits[..., padded.blank_class_index] = (
        reference_trace.raw_logits_at_step[..., reference.blank_class_index]
    )
    extra_slice = slice(reference.max_candidates, padded.max_candidates)
    extra_padding_invalid = bool(
        not padded.mask[:, extra_slice].any().item()
        and not padded_trace.valid_at_step[..., extra_slice].any().item()
    )
    mask_equal = bool(
        torch.equal(normalized_reference_mask, padded_trace.valid_at_step)
    )

    paired_valid = normalized_reference_mask & padded_trace.valid_at_step
    reference_valid = normalized_reference_logits[normalized_reference_mask]
    padded_valid = padded_trace.raw_logits_at_step[padded_trace.valid_at_step]
    reference_chosen_logp = reference_trace.chosen_logprob_at_step
    padded_chosen_logp = padded_trace.chosen_logprob_at_step
    nan_or_inf_count = int(
        (~torch.isfinite(reference_valid)).sum().item()
        + (~torch.isfinite(padded_valid)).sum().item()
        + (~torch.isfinite(reference_chosen_logp)).sum().item()
        + (~torch.isfinite(padded_chosen_logp)).sum().item()
    )
    finite_logit_pairs = (
        paired_valid
        & torch.isfinite(normalized_reference_logits)
        & torch.isfinite(padded_trace.raw_logits_at_step)
    )
    valid_logit_differences = torch.abs(
        normalized_reference_logits[finite_logit_pairs]
        - padded_trace.raw_logits_at_step[finite_logit_pairs]
    )
    finite_chosen_pairs = torch.isfinite(reference_chosen_logp) & torch.isfinite(
        padded_chosen_logp
    )
    chosen_logprob_differences = torch.abs(
        reference_chosen_logp[finite_chosen_pairs]
        - padded_chosen_logp[finite_chosen_pairs]
    )
    valid_logits_max_abs_diff = (
        float(valid_logit_differences.max().item())
        if valid_logit_differences.numel()
        else 0.0
    )
    chosen_logprob_max_abs_diff = (
        float(chosen_logprob_differences.max().item())
        if chosen_logprob_differences.numel()
        else 0.0
    )

    reference_replay_equal = bool(
        torch.equal(reference_logp, reference_trace.logp)
        and torch.equal(reference_action.action_ue_select, reference_replay.action_ue_select)
        and torch.equal(reference_action.action_prg_alloc, reference_replay.action_prg_alloc)
    )
    runtime_replay_equal = bool(
        torch.equal(padded_logp, padded_trace.logp)
        and torch.equal(padded_action.action_ue_select, padded_replay.action_ue_select)
        and torch.equal(padded_action.action_prg_alloc, padded_replay.action_prg_alloc)
    )
    candidate_choice_equal = bool(
        torch.equal(reference_action.chosen_ue_per_prg, padded_action.chosen_ue_per_prg)
    )
    type0_action_equal = bool(
        torch.equal(reference_action.action_ue_select, padded_action.action_ue_select)
        and torch.equal(reference_action.action_prg_alloc, padded_action.action_prg_alloc)
    )
    valid_logits_within_tolerance = bool(
        valid_logits_max_abs_diff <= float(valid_logits_atol)
    )
    chosen_logprob_within_tolerance = bool(
        chosen_logprob_max_abs_diff <= float(chosen_logprob_atol)
    )
    passed = bool(
        candidate_choice_equal
        and mask_equal
        and type0_action_equal
        and extra_padding_invalid
        and reference_replay_equal
        and runtime_replay_equal
        and nan_or_inf_count == 0
        and valid_logits_within_tolerance
        and chosen_logprob_within_tolerance
    )
    return {
        "candidate_choice_equal": candidate_choice_equal,
        "mask_equal": mask_equal,
        "type0_action_equal": type0_action_equal,
        "extra_padding_invalid": extra_padding_invalid,
        "reference_replay_mismatch_count": int(not reference_replay_equal),
        "runtime_replay_mismatch_count": int(not runtime_replay_equal),
        "bitwise_mean_logprob_equal": bool(
            torch.equal(reference_trace.logp, padded_trace.logp)
        ),
        "valid_logits_atol": float(valid_logits_atol),
        "chosen_logprob_atol": float(chosen_logprob_atol),
        "valid_logits_max_abs_diff": valid_logits_max_abs_diff,
        "chosen_logprob_max_abs_diff": chosen_logprob_max_abs_diff,
        "nan_or_inf_count": nan_or_inf_count,
        "valid_logits_within_tolerance": valid_logits_within_tolerance,
        "chosen_logprob_within_tolerance": chosen_logprob_within_tolerance,
        "passed": passed,
    }


def independent_teacher_trace_from_type0(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
) -> IndependentPolicyTrace:
    return _run_policy(
        actor,
        graph,
        packed,
        mode="greedy",
        teacher_actions=(action_ue_select, action_prg_alloc),
    )


def _validate_candidate_rows(graph: SparseEntityGraph, packed: PackedPrgCandidates) -> None:
    meta = graph.snapshot.static_meta
    for token_idx in range(packed.num_prg_tokens):
        valid_ids = packed.ue_ids[token_idx][packed.mask[token_idx]].tolist()
        if len(valid_ids) != len(set(int(value) for value in valid_ids)):
            raise ValueError(f"duplicate candidate row at token {token_idx}")
        cell_idx = int(meta.prg_owner_cell[token_idx].item())
        for ue_idx in valid_ids:
            ue = int(ue_idx)
            if ue < 0 or ue >= meta.n_ue or not bool(meta.action_mask_cell_ue[cell_idx, ue].item()):
                raise ValueError(f"wrong-cell candidate at token {token_idx}: ue={ue} cell={cell_idx}")


def materialize_independent_choices_to_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    choice: torch.Tensor,
) -> DecodedType0Action:
    meta = graph.snapshot.static_meta
    if packed.num_prg_tokens != meta.n_cell * meta.n_prg:
        raise ValueError("packed PRG token count mismatch")
    _validate_candidate_rows(graph, packed)
    device = packed.ue_ids.device
    selected_class = torch.as_tensor(choice, dtype=torch.long, device=device).reshape(-1)
    if selected_class.shape != (packed.num_prg_tokens,):
        raise ValueError("choice shape mismatch")
    if bool(((selected_class < 0) | (selected_class > packed.blank_class_index)).any().item()):
        raise ValueError("choice class is out of range")
    blank = selected_class == packed.blank_class_index
    chosen_ue = torch.full(
        (packed.num_prg_tokens,),
        -1,
        dtype=torch.long,
        device=device,
    )
    main = ~blank
    if bool(main.any().item()):
        probe = selected_class.clamp(max=max(0, packed.max_candidates - 1))
        chosen_valid = packed.mask.to(device).gather(1, probe.unsqueeze(-1)).squeeze(-1)
        if not bool(chosen_valid[main].all().item()):
            raise ValueError("choice selects an absent candidate")
        chosen_ue[main] = packed.ue_ids.to(device).gather(1, probe.unsqueeze(-1)).squeeze(-1)[main]

    action_mask_ue = (
        torch.ones((meta.n_ue,), dtype=torch.bool, device=device)
        if meta.action_mask_ue is None
        else meta.action_mask_ue.to(device=device, dtype=torch.bool)
    )
    owner = meta.prg_owner_cell.to(device=device, dtype=torch.long)
    local = meta.prg_local_index.to(device=device, dtype=torch.long)
    for token_idx in torch.nonzero(main, as_tuple=False).flatten().tolist():
        cell_idx = int(owner[token_idx].item())
        ue_idx = int(chosen_ue[token_idx].item())
        prg_idx = int(local[token_idx].item())
        if not bool(action_mask_ue[ue_idx].item()) or not bool(
            meta.action_mask_cell_ue[cell_idx, ue_idx].item()
        ):
            raise ValueError(f"wrong-cell or masked UE choice at token {token_idx}")
        if not bool(meta.action_mask_prg_cell[cell_idx, prg_idx].item()):
            raise ValueError(f"invalid PRG mask choice at token {token_idx}")

    layout = build_type0_slot_layout(
        meta.action_mask_cell_ue.to(device=device, dtype=torch.bool).unsqueeze(0),
        n_sched_ue=int(meta.n_sched_ue),
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    slot_to_cell = layout.slot_to_cell.squeeze(0)
    slot_valid = layout.slot_valid_mask.squeeze(0)
    action_ue = torch.full((meta.n_sched_ue,), -1, dtype=torch.int32, device=device)
    action_prg = torch.full((meta.n_cell * meta.n_prg,), -1, dtype=torch.int16, device=device)
    for cell_idx in range(meta.n_cell):
        cell_tokens = torch.nonzero(main & (owner == cell_idx), as_tuple=False).flatten()
        selected_ues = sorted(set(int(chosen_ue[token].item()) for token in cell_tokens.tolist()))
        slots = torch.nonzero(
            slot_valid & (slot_to_cell == cell_idx),
            as_tuple=False,
        ).flatten().tolist()
        if len(selected_ues) > len(slots):
            raise ValueError(
                f"selected UE slot overflow for cell {cell_idx}: selected={len(selected_ues)} slots={len(slots)}"
            )
        ue_to_slot = {ue_idx: int(slots[pos]) for pos, ue_idx in enumerate(selected_ues)}
        for ue_idx, slot_idx in ue_to_slot.items():
            action_ue[slot_idx] = ue_idx
        for token_idx in cell_tokens.tolist():
            native_idx = int(local[token_idx].item()) * meta.n_cell + cell_idx
            action_prg[native_idx] = ue_to_slot[int(chosen_ue[token_idx].item())]

    reject_counts = type0_action_preflight_reject_counts(graph, action_ue, action_prg)
    reject_total = sum(int(value.sum().item()) for value in reject_counts.values())
    if reject_total:
        raise ValueError(f"identity materialization failed Type-0 preflight: {reject_counts}")
    decoded = DecodedType0Action(
        action_ue_select=action_ue,
        action_prg_alloc=action_prg,
        chosen_ue_per_prg=chosen_ue,
        prg_chosen_mask=main,
        final_blank_count=int(blank.sum().item()),
    )
    return decoded


def independent_teacher_targets_from_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
    valid_at_step: torch.Tensor,
) -> torch.Tensor:
    meta = graph.snapshot.static_meta
    ue_action = torch.as_tensor(action_ue_select, dtype=torch.long).reshape(meta.n_sched_ue)
    prg_action = torch.as_tensor(action_prg_alloc, dtype=torch.long).reshape(meta.n_cell * meta.n_prg)
    valid = torch.as_tensor(valid_at_step, dtype=torch.bool)
    expected_valid = (meta.n_prg, meta.n_cell, packed.max_candidates + 1)
    if tuple(valid.shape) != expected_valid:
        raise ValueError(f"valid_at_step mismatch: got={tuple(valid.shape)} expected={expected_valid}")
    targets = torch.full(
        (packed.num_prg_tokens,),
        packed.blank_class_index,
        dtype=torch.long,
    )
    for cell_idx in range(meta.n_cell):
        for local_prg in range(meta.n_prg):
            native_idx = local_prg * meta.n_cell + cell_idx
            token_idx = cell_idx * meta.n_prg + local_prg
            slot_idx = int(prg_action[native_idx].item())
            if slot_idx < 0:
                target = packed.blank_class_index
            else:
                if slot_idx >= meta.n_sched_ue:
                    raise ValueError(f"teacher slot out of range at token {token_idx}")
                ue_idx = int(ue_action[slot_idx].item())
                if ue_idx < 0 or ue_idx >= meta.n_ue:
                    raise ValueError(f"teacher slot is empty at token {token_idx}")
                if not bool(meta.action_mask_cell_ue[cell_idx, ue_idx].item()):
                    raise ValueError(f"teacher UE is assigned to wrong cell at token {token_idx}")
                matches = torch.nonzero(
                    packed.mask[token_idx] & (packed.ue_ids[token_idx] == ue_idx),
                    as_tuple=False,
                ).flatten()
                if matches.numel() != 1:
                    raise ValueError(f"teacher UE missing or duplicated at token {token_idx}")
                target = int(matches[0].item())
            if not bool(valid[local_prg, cell_idx, target].item()):
                raise ValueError(f"teacher target is headroom-invalid at token {token_idx}")
            targets[token_idx] = target
    return targets


__all__ = [
    "IndependentPolicyTrace",
    "W0_V2_CHOSEN_LOGPROB_ATOL",
    "W0_V2_VALID_LOGITS_ATOL",
    "compare_candidate_padding_fp32_equivalence",
    "compare_candidate_padding_semantics",
    "estimate_candidate_service_bytes",
    "independent_teacher_trace_from_type0",
    "independent_teacher_targets_from_type0",
    "materialize_independent_choices_to_type0",
    "replay_independent_trace",
    "resolve_full_candidate_width",
    "select_independent_action",
]

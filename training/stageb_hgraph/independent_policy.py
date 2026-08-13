from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from training.gnnrl.slot_layout import build_type0_slot_layout

from .graph_ops import (
    DecodedType0Action,
    PackedPrgCandidates,
    type0_action_preflight_reject_counts,
)
from .independent_actor import PFQIndependentReuseActor
from .types import SparseEntityGraph


@dataclass(frozen=True)
class IndependentPolicyTrace:
    choice: torch.Tensor
    local_prg_order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor


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
    if packed.max_candidates != expected_width:
        raise ValueError(
            f"candidate width mismatch: got={packed.max_candidates} expected={expected_width}"
        )
    configured_width = int(actor.cfg.max_candidates)
    if configured_width != expected_width:
        raise ValueError(
            f"actor max_candidates mismatch: got={configured_width} expected={expected_width}"
        )


def _run_policy(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    *,
    mode: Literal["sample", "greedy"],
    generator: torch.Generator | None = None,
    stored_trace: IndependentPolicyTrace | None = None,
    teacher_actions: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> IndependentPolicyTrace:
    if mode not in ("sample", "greedy"):
        raise ValueError(f"independent policy mode must be sample or greedy, got {mode!r}")
    _validate_policy_inputs(actor, graph, packed)
    meta = graph.snapshot.static_meta
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype
    encoded = actor.encode_graph(graph)
    service = estimate_candidate_service_bytes(graph, packed).to(device=device, dtype=dtype)
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
    teacher_ue = None
    teacher_prg = None
    if teacher_actions is not None:
        teacher_ue = torch.as_tensor(
            teacher_actions[0], device=device, dtype=torch.long
        ).reshape(meta.n_sched_ue)
        teacher_prg = torch.as_tensor(
            teacher_actions[1], device=device, dtype=torch.long
        ).reshape(meta.n_cell * meta.n_prg)

    choices = torch.full(
        (meta.n_cell * meta.n_prg,),
        packed.blank_class_index,
        dtype=torch.long,
        device=device,
    )
    masks: list[torch.Tensor] = []
    logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
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
            selected = torch.as_tensor(
                stored_trace.choice,
                device=device,
                dtype=torch.long,
            ).index_select(0, tokens)
            if bool(((selected < 0) | (selected > packed.blank_class_index)).any().item()):
                raise ValueError(f"stored choice out of range at local PRG {local_prg}")
            if not bool(valid.gather(1, selected.unsqueeze(-1)).all().item()):
                raise ValueError(f"stored choice is invalid at local PRG {local_prg}")
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
                            f"next_service={float(service[token_idx, target].item()):.6f}"
                        )
                    raise ValueError(f"teacher target is invalid at token {token_idx}")
                selected_rows.append(target)
            selected = torch.tensor(selected_rows, dtype=torch.long, device=device)
        elif mode == "sample":
            selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        else:
            selected = adjusted.argmax(dim=-1)

        choices[tokens] = selected
        masks.append(valid)
        logps.append(logp_all.gather(1, selected.unsqueeze(-1)).squeeze(-1))
        entropies.append(-(probabilities * logp_all).sum(dim=-1))

        # The three cells above were all scored from one pre-update state. Only
        # after the reuse group is complete do selected nonblank UEs update.
        selected_main = selected < packed.blank_class_index
        if bool(selected_main.any().item()):
            selected_class = selected.clamp(max=max(0, packed.max_candidates - 1))
            candidate_ids = packed.ue_ids.to(device=device).index_select(0, tokens)
            chosen_ues = candidate_ids.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)
            selected_service = service.index_select(0, tokens).gather(
                1,
                selected_class.unsqueeze(-1),
            ).squeeze(-1)
            for cell_idx in torch.nonzero(selected_main, as_tuple=False).flatten().tolist():
                ue_idx = int(chosen_ues[cell_idx].item())
                service_bytes = selected_service[cell_idx]
                remaining[ue_idx] = torch.clamp_min(remaining[ue_idx] - service_bytes, 0.0)
                allocated[ue_idx] += 1.0
                predicted[ue_idx] += service_bytes

    return IndependentPolicyTrace(
        choice=choices,
        local_prg_order=local_order.detach(),
        valid_at_step=torch.stack(masks).detach(),
        logp=torch.cat(logps).mean(),
        entropy=torch.cat(entropies).mean(),
    )


def select_independent_action(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    mode: Literal["sample", "greedy"],
    generator: torch.Generator | None = None,
) -> tuple[IndependentPolicyTrace, DecodedType0Action]:
    trace = _run_policy(
        actor,
        graph,
        packed,
        mode=mode,
        generator=generator,
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
    "estimate_candidate_service_bytes",
    "independent_teacher_trace_from_type0",
    "independent_teacher_targets_from_type0",
    "materialize_independent_choices_to_type0",
    "replay_independent_trace",
    "resolve_full_candidate_width",
    "select_independent_action",
]

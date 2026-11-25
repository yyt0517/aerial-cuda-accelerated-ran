"""Detached selected-candidate evidence for independent-policy rescue decisions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import torch

from .graph_ops import PackedPrgCandidates
from .independent_blank_diagnostics import IndependentPolicyDiagnostics
from .types import SparseEntityGraph


@dataclass(frozen=True)
class RescueDecisionDiagnostics:
    selected_ue: torch.Tensor
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    selected_nonblank: torch.Tensor
    fragmentation_rescued: torch.Tensor
    selected_forced_blank: torch.Tensor
    selected_conservative_blank: torch.Tensor
    estimated_service_bytes: torch.Tensor
    quality_db: torch.Tensor
    quality_gap_db: torch.Tensor
    prg_risk: torch.Tensor
    ue_tb_err_last: torch.Tensor
    remaining_demand_ratio: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    success_proxy: torch.Tensor


@dataclass(frozen=True)
class RescueUeTtiRow:
    tti: int
    ue: int
    rescue_exposed: bool
    planned_prg_count: int
    rescued_prg_count: int
    accepted_prg_count: int
    actual_prg_count: int
    predicted_service_bytes: float
    scheduled_bytes: float
    goodput_bytes: float
    tb_tx_count: int
    tb_err_count: int
    quality_db_sum: float
    quality_gap_db_sum: float
    prg_risk_sum: float
    ue_tb_err_last_sum: float
    remaining_demand_ratio_sum: float
    success_proxy_sum: float
    p_blank_sum: float
    p_service_sum: float
    p_best_ue_sum: float
    early_rescue_count: int
    middle_rescue_count: int
    late_rescue_count: int


@dataclass(frozen=True)
class RescueCellTtiRow:
    tti: int
    cell: int
    rescue_count: int
    first_rescue_local_prg: int
    decisions_before_or_without_prior_rescue: int
    decisions_after_prior_rescue: int
    forced_before_or_without_prior_rescue: int
    forced_after_prior_rescue: int
    conservative_before_or_without_prior_rescue: int
    conservative_after_prior_rescue: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _detached_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).cpu()


def _validate_policy_row(policy_diagnostics: IndependentPolicyDiagnostics, count: int) -> None:
    for field in fields(IndependentPolicyDiagnostics):
        value = getattr(policy_diagnostics, field.name)
        _require(value.shape == (count,), f"policy diagnostics {field.name} shape mismatch")
        if value.dtype is not torch.bool:
            _require(torch.isfinite(value).all().item(), f"policy diagnostics {field.name} must be finite")


def build_rescue_decision_diagnostics(
    *,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    token_indices: torch.Tensor,
    local_prg: int,
    selected: torch.Tensor,
    service_bytes: torch.Tensor,
    sequential_features: torch.Tensor,
    policy_diagnostics: IndependentPolicyDiagnostics,
) -> RescueDecisionDiagnostics:
    """Gather selected-class evidence for one reuse group without affecting actions."""
    meta = graph.snapshot.static_meta
    tokens = torch.as_tensor(token_indices, device=packed.ue_ids.device, dtype=torch.long).reshape(-1)
    selected = torch.as_tensor(selected, device=packed.ue_ids.device, dtype=torch.long).reshape(-1)
    count = int(tokens.numel())
    _require(count == meta.n_cell, "token_indices must contain one token per cell")
    _require(0 <= local_prg < meta.n_prg, "local_prg is out of range")
    _require(selected.shape == (count,), "selected shape mismatch")
    _require(((tokens >= 0) & (tokens < packed.num_prg_tokens)).all().item(), "token index is out of range")
    _require(
        torch.equal(
            meta.prg_local_index.to(device=tokens.device, dtype=torch.long).index_select(0, tokens),
            torch.full_like(tokens, local_prg),
        ),
        "token local PRG mismatch",
    )
    _require(packed.edge_attr is not None and packed.edge_attr.dim() == 3, "packed base edge attributes are required")
    _require(packed.edge_attr.shape[2] >= 5, "packed base edge width must be at least five")
    _require(graph.snapshot.ue_features.values.shape[1] > 5, "UE features require index five")
    _require(service_bytes.shape == packed.ue_ids.shape, "service_bytes shape mismatch")
    _require(
        sequential_features.shape[:2] == (count, packed.max_candidates)
        and sequential_features.dim() == 3
        and sequential_features.shape[2] > 0,
        "sequential_features shape mismatch",
    )
    _require(((selected >= 0) & (selected <= packed.blank_class_index)).all().item(), "selected class is out of range")
    _validate_policy_row(policy_diagnostics, count)

    selected_nonblank = selected < packed.blank_class_index
    selected_ue = torch.full((count,), -1, dtype=torch.long, device=tokens.device)
    zeros = torch.zeros((count,), dtype=service_bytes.dtype, device=tokens.device)
    if packed.max_candidates > 0:
        selected_class = selected.clamp(max=packed.max_candidates - 1)
        candidate_mask = packed.mask.to(device=tokens.device).index_select(0, tokens)
        _require(
            bool(candidate_mask.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)[selected_nonblank].all().item()),
            "selected nonblank class is not a candidate",
        )
        candidate_ids = packed.ue_ids.to(device=tokens.device).index_select(0, tokens)
        gathered_ue = candidate_ids.gather(1, selected_class.unsqueeze(-1)).squeeze(-1)
        _require(
            bool(((gathered_ue[selected_nonblank] >= 0) & (gathered_ue[selected_nonblank] < meta.n_ue)).all().item()),
            "selected UE is out of range",
        )
        selected_ue[selected_nonblank] = gathered_ue[selected_nonblank]
        edge_values = packed.edge_attr.to(device=tokens.device).index_select(0, tokens).gather(
            1,
            selected_class[:, None, None].expand(-1, 1, packed.edge_attr.shape[2]),
        ).squeeze(1)
        gathered_service = service_bytes.to(device=tokens.device).index_select(0, tokens).gather(
            1, selected_class.unsqueeze(-1)
        ).squeeze(-1)
        gathered_remaining = sequential_features.to(device=tokens.device).gather(
            1, selected_class[:, None, None].expand(-1, 1, sequential_features.shape[2])
        ).squeeze(1)[:, 0]
        safe_ue = gathered_ue.clamp(min=0, max=max(0, meta.n_ue - 1))
        gathered_tb_err = graph.snapshot.ue_features.values.to(device=tokens.device)[safe_ue, 5]
        _require(
            all(
                torch.isfinite(value[selected_nonblank]).all().item()
                for value in (
                    edge_values[:, 0], edge_values[:, 1], edge_values[:, 4], gathered_service,
                    gathered_remaining, gathered_tb_err,
                )
            ),
            "gathered selected-candidate values must be finite",
        )
        estimated_service_bytes = torch.where(selected_nonblank, gathered_service, zeros)
        quality_db = torch.where(selected_nonblank, edge_values[:, 0], zeros)
        quality_gap_db = torch.where(selected_nonblank, edge_values[:, 1], zeros)
        prg_risk = torch.where(selected_nonblank, edge_values[:, 4], zeros)
        ue_tb_err_last = torch.where(selected_nonblank, gathered_tb_err, zeros)
        remaining_demand_ratio = torch.where(selected_nonblank, gathered_remaining, zeros)
    else:
        _require(not bool(selected_nonblank.any().item()), "nonblank selection requires candidates")
        estimated_service_bytes = zeros
        quality_db = zeros
        quality_gap_db = zeros
        prg_risk = zeros
        ue_tb_err_last = zeros
        remaining_demand_ratio = zeros

    success_proxy = torch.where(
        selected_nonblank,
        torch.sigmoid((quality_db - 0.35 * quality_gap_db - 7.5 * prg_risk - 6.0 * ue_tb_err_last) / 4.0),
        torch.zeros_like(quality_db),
    )
    values = {
        "selected_ue": selected_ue,
        "cell_index": meta.prg_owner_cell.to(device=tokens.device, dtype=torch.long).index_select(0, tokens),
        "local_prg_index": torch.full_like(tokens, local_prg),
        "selected_nonblank": selected_nonblank,
        "fragmentation_rescued": policy_diagnostics.fragmentation_rescued.to(tokens.device),
        "selected_forced_blank": policy_diagnostics.selected_forced_blank.to(tokens.device),
        "selected_conservative_blank": policy_diagnostics.selected_conservative_blank.to(tokens.device),
        "estimated_service_bytes": estimated_service_bytes,
        "quality_db": quality_db,
        "quality_gap_db": quality_gap_db,
        "prg_risk": prg_risk,
        "ue_tb_err_last": ue_tb_err_last,
        "remaining_demand_ratio": remaining_demand_ratio,
        "p_blank": policy_diagnostics.p_blank.to(tokens.device),
        "p_service": policy_diagnostics.p_service.to(tokens.device),
        "p_best_ue": policy_diagnostics.p_best_ue.to(tokens.device),
        "success_proxy": success_proxy,
    }
    return RescueDecisionDiagnostics(**{name: _detached_vector(value) for name, value in values.items()})


def concatenate_rescue_decision_diagnostics(
    rows: Sequence[RescueDecisionDiagnostics],
) -> RescueDecisionDiagnostics:
    """Concatenate fixed-schema detached decision rows in policy execution order."""
    _require(bool(rows), "at least one rescue diagnostics row is required")
    values: dict[str, torch.Tensor] = {}
    for field in fields(RescueDecisionDiagnostics):
        tensors = [getattr(row, field.name) for row in rows]
        _require(all(tensor.dim() == 1 for tensor in tensors), f"{field.name} must be one-dimensional")
        values[field.name] = _detached_vector(torch.cat(tensors))
    lengths = {tensor.numel() for tensor in values.values()}
    _require(len(lengths) == 1, "rescue diagnostics fields must have equal lengths")
    return RescueDecisionDiagnostics(**values)


def summarize_local_prg_phy_interference(
    decisions: RescueDecisionDiagnostics,
    *,
    n_cell: int,
    n_prg: int,
) -> dict[str, object]:
    """Summarize detached selected-PHY and reuse evidence by local PRG."""
    _require(n_cell > 0 and n_prg > 0, "local-PRG diagnostic dimensions must be positive")
    values = {
        field.name: _detached_vector(getattr(decisions, field.name))
        for field in fields(decisions)
    }
    lengths = {value.numel() for value in values.values()}
    _require(len(lengths) == 1, "local-PRG diagnostic fields must have equal lengths")
    decision_count = next(iter(lengths))
    decisions_per_tti = n_cell * n_prg
    _require(
        decision_count > 0 and decision_count % decisions_per_tti == 0,
        "local-PRG diagnostic decision count does not cover whole TTIs",
    )
    tti_count = decision_count // decisions_per_tti
    expected_local = torch.arange(n_prg).repeat_interleave(n_cell)
    expected_cell = torch.arange(n_cell).repeat(n_prg)
    local_by_tti = values["local_prg_index"].to(dtype=torch.long).reshape(
        tti_count, decisions_per_tti
    )
    cell_by_tti = values["cell_index"].to(dtype=torch.long).reshape(
        tti_count, decisions_per_tti
    )
    _require(
        torch.equal(local_by_tti, expected_local.unsqueeze(0).expand_as(local_by_tti)),
        "local-PRG diagnostic ordering mismatch",
    )
    _require(
        torch.equal(cell_by_tti, expected_cell.unsqueeze(0).expand_as(cell_by_tti)),
        "local-PRG diagnostic cell ordering mismatch",
    )
    selected_by_tti = values["selected_nonblank"].to(dtype=torch.bool).reshape(
        tti_count, n_prg, n_cell
    )

    def selected_mean(name: str, mask: torch.Tensor) -> float | None:
        selected_values = values[name].to(dtype=torch.float64)[mask]
        return None if selected_values.numel() == 0 else float(selected_values.mean().item())

    rows: list[dict[str, object]] = []
    local_flat = values["local_prg_index"].to(dtype=torch.long)
    selected_flat = values["selected_nonblank"].to(dtype=torch.bool)
    for local_prg in range(n_prg):
        local_mask = local_flat == local_prg
        selected_mask = local_mask & selected_flat
        selected_count = int(selected_mask.sum().item())
        local_decisions = int(local_mask.sum().item())
        reuse = selected_by_tti[:, local_prg, :].sum(dim=1)
        rows.append(
            {
                "local_prg": local_prg,
                "decision_count": local_decisions,
                "selected_count": selected_count,
                "forced_blank_count": int(
                    values["selected_forced_blank"]
                    .to(dtype=torch.bool)[local_mask]
                    .sum()
                    .item()
                ),
                "conservative_blank_count": int(
                    values["selected_conservative_blank"]
                    .to(dtype=torch.bool)[local_mask]
                    .sum()
                    .item()
                ),
                "blank_ratio": float(1.0 - selected_count / local_decisions),
                "reuse_count_mean": float(reuse.to(dtype=torch.float64).mean().item()),
                "reuse_cardinality_tti_count": {
                    str(cardinality): int((reuse == cardinality).sum().item())
                    for cardinality in range(n_cell + 1)
                },
                "selected_quality_db_mean": selected_mean("quality_db", selected_mask),
                "selected_quality_gap_db_mean": selected_mean(
                    "quality_gap_db", selected_mask
                ),
                "selected_prg_risk_mean": selected_mean("prg_risk", selected_mask),
                "selected_predicted_service_bytes_mean": selected_mean(
                    "estimated_service_bytes", selected_mask
                ),
                "selected_success_proxy_mean": selected_mean(
                    "success_proxy", selected_mask
                ),
                "p_blank_mean": float(
                    values["p_blank"].to(dtype=torch.float64)[local_mask].mean().item()
                ),
                "p_service_mean": float(
                    values["p_service"].to(dtype=torch.float64)[local_mask].mean().item()
                ),
                "p_best_ue_mean": float(
                    values["p_best_ue"].to(dtype=torch.float64)[local_mask].mean().item()
                ),
            }
        )
    return {
        "schema_version": 1,
        "unit": "local_prg_decision",
        "tti_count": tti_count,
        "n_cell": n_cell,
        "n_prg": n_prg,
        "local_prgs": rows,
    }


def _outcome_vector(info: Mapping[str, object], name: str, n_ue: int) -> torch.Tensor:
    _require(name in info, f"missing required rescue outcome {name}")
    value = torch.as_tensor(info[name]).detach().cpu()
    _require(value.shape == (n_ue,), f"rescue outcome {name} shape mismatch")
    _require(value.dtype is not torch.bool, f"rescue outcome {name} must be numeric")
    value = value.to(dtype=torch.float64)
    _require(torch.isfinite(value).all().item(), f"rescue outcome {name} must be finite")
    _require((value >= 0).all().item(), f"rescue outcome {name} must be non-negative")
    return value


def _count_vector(info: Mapping[str, object], name: str, n_ue: int) -> torch.Tensor:
    value = _outcome_vector(info, name, n_ue)
    _require(torch.equal(value, value.round()), f"rescue outcome {name} must be integral")
    return value.to(dtype=torch.long)


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _validate_join_decisions(
    decisions: RescueDecisionDiagnostics,
    chosen_ue_per_prg: torch.Tensor,
    *,
    n_ue: int,
    n_cell: int,
    n_prg: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected = n_cell * n_prg
    _require(n_ue >= 0 and n_cell > 0 and n_prg > 0, "invalid rescue join dimensions")
    values = {field.name: _detached_vector(getattr(decisions, field.name)) for field in fields(decisions)}
    _require(all(value.shape == (expected,) for value in values.values()), "rescue decision shape mismatch")
    chosen = torch.as_tensor(chosen_ue_per_prg).detach().cpu()
    _require(chosen.shape == (expected,), "chosen_ue_per_prg shape mismatch")
    _require(chosen.dtype is not torch.bool, "chosen_ue_per_prg must be numeric")
    chosen = chosen.to(dtype=torch.float64)
    _require(torch.isfinite(chosen).all().item(), "chosen_ue_per_prg must be finite")
    _require(torch.equal(chosen, chosen.round()), "chosen_ue_per_prg must be integral")
    chosen = chosen.to(dtype=torch.long)
    _require(((chosen >= -1) & (chosen < n_ue)).all().item(), "chosen_ue_per_prg UE is out of range")

    selected_nonblank = values["selected_nonblank"].to(dtype=torch.bool)
    selected_ue = values["selected_ue"].to(dtype=torch.long)
    cell_index = values["cell_index"].to(dtype=torch.long)
    local_prg_index = values["local_prg_index"].to(dtype=torch.long)
    _require(((cell_index >= 0) & (cell_index < n_cell)).all().item(), "rescue decision cell is out of range")
    _require(((local_prg_index >= 0) & (local_prg_index < n_prg)).all().item(), "rescue decision local PRG is out of range")
    _require(((selected_ue[selected_nonblank] >= 0) & (selected_ue[selected_nonblank] < n_ue)).all().item(), "rescue decision UE is out of range")
    _require((selected_ue[~selected_nonblank] == -1).all().item(), "Blank rescue decision must have UE -1")
    tokens = cell_index * n_prg + local_prg_index
    _require(torch.unique(tokens).numel() == expected, "rescue decision tokens must be unique")
    _require(torch.equal(torch.sort(tokens).values, torch.arange(expected)), "rescue decision tokens must cover action")
    expected_chosen = torch.full_like(chosen, -1)
    expected_chosen[tokens[selected_nonblank]] = selected_ue[selected_nonblank]
    _require(torch.equal(chosen, expected_chosen), "rescue decisions do not reconstruct chosen UE action")
    _require((~values["fragmentation_rescued"].to(dtype=torch.bool) | selected_nonblank).all().item(), "rescued decision must be nonblank")
    for name in (
        "estimated_service_bytes", "quality_db", "quality_gap_db", "prg_risk", "ue_tb_err_last",
        "remaining_demand_ratio", "p_blank", "p_service", "p_best_ue", "success_proxy",
    ):
        _require(torch.isfinite(values[name].to(dtype=torch.float64)).all().item(), f"rescue decision {name} must be finite")
    _require((values["estimated_service_bytes"].to(dtype=torch.float64)[selected_nonblank] >= 0).all().item(), "estimated service bytes must be non-negative")
    return chosen, values


def join_rescue_cohort_tti(
    decisions: RescueDecisionDiagnostics,
    *,
    chosen_ue_per_prg: torch.Tensor,
    info: Mapping[str, object],
    n_ue: int,
    n_cell: int,
    n_prg: int,
    tti: int,
) -> tuple[list[RescueUeTtiRow], list[RescueCellTtiRow]]:
    """Join one action's planned decisions to the simulator's UE-level outcome."""
    _chosen, values = _validate_join_decisions(
        decisions, chosen_ue_per_prg, n_ue=n_ue, n_cell=n_cell, n_prg=n_prg
    )
    accepted = _count_vector(info, "accepted_pre_pdsch_ue_prg_count", n_ue)
    actual = _count_vector(info, "actual_ue_prg_count", n_ue)
    scheduled = _outcome_vector(info, "actual_ue_served_bytes", n_ue)
    goodput = _outcome_vector(info, "actual_ue_goodput_bytes", n_ue)
    tb_tx = _count_vector(info, "actual_ue_tb_tx_count", n_ue)
    tb_err = _count_vector(info, "actual_ue_tb_err_count", n_ue)
    _require((goodput <= scheduled).all().item(), "goodput bytes exceed scheduled bytes")
    _require((tb_err <= tb_tx).all().item(), "tb_err exceeds tb_tx")

    selected = values["selected_nonblank"].to(dtype=torch.bool)
    selected_ue = values["selected_ue"].to(dtype=torch.long)
    rescued = values["fragmentation_rescued"].to(dtype=torch.bool)
    planned_by_ue = torch.bincount(selected_ue[selected], minlength=n_ue)
    evidence = accepted + actual + scheduled.ne(0).to(dtype=torch.long) + goodput.ne(0).to(dtype=torch.long) + tb_tx + tb_err
    _require(((planned_by_ue > 0) | (evidence == 0)).all().item(), "unplanned UE has nonzero actual evidence")
    _require((accepted <= planned_by_ue).all().item(), "accepted PRGs exceed planned PRGs")
    _require((actual <= accepted).all().item(), "actual PRGs exceed accepted PRGs")

    ue_rows: list[RescueUeTtiRow] = []
    scalar_fields = (
        "estimated_service_bytes", "quality_db", "quality_gap_db", "prg_risk", "ue_tb_err_last",
        "remaining_demand_ratio", "success_proxy", "p_blank", "p_service", "p_best_ue",
    )
    for ue in torch.nonzero(planned_by_ue > 0, as_tuple=False).flatten().tolist():
        mask = selected & (selected_ue == ue)
        rescue_mask = mask & rescued
        local = values["local_prg_index"].to(dtype=torch.long)[rescue_mask]
        sums = {name: float(values[name].to(dtype=torch.float64)[mask].sum().item()) for name in scalar_fields}
        ue_rows.append(RescueUeTtiRow(
            tti=int(tti), ue=int(ue), rescue_exposed=bool(rescue_mask.any().item()),
            planned_prg_count=int(mask.sum().item()), rescued_prg_count=int(rescue_mask.sum().item()),
            accepted_prg_count=int(accepted[ue].item()), actual_prg_count=int(actual[ue].item()),
            predicted_service_bytes=sums["estimated_service_bytes"], scheduled_bytes=float(scheduled[ue].item()),
            goodput_bytes=float(goodput[ue].item()), tb_tx_count=int(tb_tx[ue].item()), tb_err_count=int(tb_err[ue].item()),
            quality_db_sum=sums["quality_db"], quality_gap_db_sum=sums["quality_gap_db"],
            prg_risk_sum=sums["prg_risk"], ue_tb_err_last_sum=sums["ue_tb_err_last"],
            remaining_demand_ratio_sum=sums["remaining_demand_ratio"], success_proxy_sum=sums["success_proxy"],
            p_blank_sum=sums["p_blank"], p_service_sum=sums["p_service"], p_best_ue_sum=sums["p_best_ue"],
            early_rescue_count=int((local <= 5).sum().item()),
            middle_rescue_count=int(((local >= 6) & (local <= 11)).sum().item()),
            late_rescue_count=int(((local >= 12) & (local <= 16)).sum().item()),
        ))

    cell_rows: list[RescueCellTtiRow] = []
    cell_index = values["cell_index"].to(dtype=torch.long)
    local_index = values["local_prg_index"].to(dtype=torch.long)
    forced = values["selected_forced_blank"].to(dtype=torch.bool)
    conservative = values["selected_conservative_blank"].to(dtype=torch.bool)
    for cell in range(n_cell):
        cell_mask = cell_index == cell
        cell_rescue_local = local_index[cell_mask & rescued]
        first_rescue = int(cell_rescue_local.min().item()) if cell_rescue_local.numel() else -1
        before = after = forced_before = forced_after = conservative_before = conservative_after = 0
        for local_prg in range(n_prg):
            index = torch.nonzero(cell_mask & (local_index == local_prg), as_tuple=False).flatten()
            _require(index.numel() == 1, "rescue decision cell-local PRG is missing")
            decision = int(index.item())
            has_prior_rescue = first_rescue >= 0 and first_rescue < local_prg
            if has_prior_rescue:
                after += 1
                forced_after += int(forced[decision].item())
                conservative_after += int(conservative[decision].item())
            else:
                before += 1
                forced_before += int(forced[decision].item())
                conservative_before += int(conservative[decision].item())
        cell_rows.append(RescueCellTtiRow(
            tti=int(tti), cell=cell, rescue_count=int((cell_mask & rescued).sum().item()),
            first_rescue_local_prg=first_rescue,
            decisions_before_or_without_prior_rescue=before, decisions_after_prior_rescue=after,
            forced_before_or_without_prior_rescue=forced_before, forced_after_prior_rescue=forced_after,
            conservative_before_or_without_prior_rescue=conservative_before,
            conservative_after_prior_rescue=conservative_after,
        ))
    return ue_rows, cell_rows


def _summarize_rows(rows: Sequence[RescueUeTtiRow]) -> dict[str, float | int | None]:
    planned = sum(row.planned_prg_count for row in rows)
    accepted = sum(row.accepted_prg_count for row in rows)
    actual = sum(row.actual_prg_count for row in rows)
    scheduled = sum(row.scheduled_bytes for row in rows)
    goodput = sum(row.goodput_bytes for row in rows)
    tb_tx = sum(row.tb_tx_count for row in rows)
    sums = {
        "ue_tti_count": len(rows), "tti_count": len({row.tti for row in rows}),
        "planned_prg_count": planned, "rescued_prg_count": sum(row.rescued_prg_count for row in rows),
        "accepted_prg_count": accepted, "actual_prg_count": actual,
        "predicted_service_bytes": float(sum(row.predicted_service_bytes for row in rows)),
        "scheduled_bytes": float(scheduled), "goodput_bytes": float(goodput),
        "failed_bytes": float(scheduled - goodput), "tb_tx_count": tb_tx,
        "tb_err_count": sum(row.tb_err_count for row in rows),
        "early_rescue_count": sum(row.early_rescue_count for row in rows),
        "middle_rescue_count": sum(row.middle_rescue_count for row in rows),
        "late_rescue_count": sum(row.late_rescue_count for row in rows),
    }
    result: dict[str, float | int | None] = dict(sums)
    result.update({
        "predicted_service_bytes_per_planned_prg": _ratio(sums["predicted_service_bytes"], planned),
        "scheduled_bytes_per_actual_prg": _ratio(scheduled, actual),
        "goodput_bytes_per_actual_prg": _ratio(goodput, actual),
        "failed_byte_ratio": _ratio(scheduled - goodput, scheduled),
        "tb_error_ratio": _ratio(sums["tb_err_count"], tb_tx),
        "planned_to_accepted_ratio": _ratio(accepted, planned),
        "accepted_to_actual_ratio": _ratio(actual, accepted),
        "quality_db_mean": _ratio(sum(row.quality_db_sum for row in rows), planned),
        "quality_gap_db_mean": _ratio(sum(row.quality_gap_db_sum for row in rows), planned),
        "prg_risk_mean": _ratio(sum(row.prg_risk_sum for row in rows), planned),
        "ue_tb_err_last_mean": _ratio(sum(row.ue_tb_err_last_sum for row in rows), planned),
        "remaining_demand_ratio_mean": _ratio(sum(row.remaining_demand_ratio_sum for row in rows), planned),
        "success_proxy_mean": _ratio(sum(row.success_proxy_sum for row in rows), planned),
        "p_blank_mean": _ratio(sum(row.p_blank_sum for row in rows), planned),
        "p_service_mean": _ratio(sum(row.p_service_sum for row in rows), planned),
        "p_best_ue_mean": _ratio(sum(row.p_best_ue_sum for row in rows), planned),
    })
    return result


def _per_tti_comparisons(ue_rows: Sequence[RescueUeTtiRow]) -> dict[str, dict[str, float | int | None]]:
    metric_names = (
        "predicted_service_bytes_per_planned_prg", "scheduled_bytes_per_actual_prg",
        "goodput_bytes_per_actual_prg", "failed_byte_ratio", "tb_error_ratio",
        "planned_to_accepted_ratio", "accepted_to_actual_ratio",
    )
    by_tti = sorted({row.tti for row in ue_rows})
    result: dict[str, dict[str, float | int | None]] = {}
    for metric in metric_names:
        deltas: list[float] = []
        for tti in by_tti:
            rescue = _summarize_rows([row for row in ue_rows if row.tti == tti and row.rescue_exposed])[metric]
            ordinary = _summarize_rows([row for row in ue_rows if row.tti == tti and not row.rescue_exposed])[metric]
            if rescue is not None and ordinary is not None:
                deltas.append(float(rescue) - float(ordinary))
        mean_delta = None if not deltas else float(sum(deltas) / len(deltas))
        if mean_delta is None:
            agreement = None
        elif mean_delta == 0.0:
            agreement = float(sum(delta == 0.0 for delta in deltas) / len(deltas))
        else:
            agreement = float(sum((delta > 0.0) == (mean_delta > 0.0) for delta in deltas) / len(deltas))
        result[metric] = {
            "mean_delta_rescue_minus_ordinary": mean_delta,
            "paired_tti_count": len(deltas),
            "direction_agreement_fraction": agreement,
        }
    return result


def summarize_rescue_cohorts(
    ue_rows: Sequence[RescueUeTtiRow],
    cell_rows: Sequence[RescueCellTtiRow],
) -> dict[str, object]:
    """Summarize measured-window cohorts using sums and explicit denominators."""
    cohorts = {
        "rescue_exposed": _summarize_rows([row for row in ue_rows if row.rescue_exposed]),
        "ordinary_service": _summarize_rows([row for row in ue_rows if not row.rescue_exposed]),
    }
    strata = {
        "zero": lambda count: count == 0,
        "1_2": lambda count: 1 <= count <= 2,
        "3_5": lambda count: 3 <= count <= 5,
        "ge_6": lambda count: count >= 6,
    }
    actual_prg_strata = {
        name: {
            "rescue_exposed": _summarize_rows([row for row in ue_rows if row.rescue_exposed and predicate(row.actual_prg_count)]),
            "ordinary_service": _summarize_rows([row for row in ue_rows if not row.rescue_exposed and predicate(row.actual_prg_count)]),
        }
        for name, predicate in strata.items()
    }
    return {
        "schema_version": 1,
        "unit": "ue_tti",
        "cohorts": cohorts,
        "actual_prg_strata": actual_prg_strata,
        "per_tti_comparisons": _per_tti_comparisons(ue_rows),
        "depletion": {
            "cell_tti_count": len(cell_rows),
            "rescue_count": sum(row.rescue_count for row in cell_rows),
            "decisions_before_or_without_prior_rescue": sum(row.decisions_before_or_without_prior_rescue for row in cell_rows),
            "decisions_after_prior_rescue": sum(row.decisions_after_prior_rescue for row in cell_rows),
            "forced_before_or_without_prior_rescue": sum(row.forced_before_or_without_prior_rescue for row in cell_rows),
            "forced_after_prior_rescue": sum(row.forced_after_prior_rescue for row in cell_rows),
            "conservative_before_or_without_prior_rescue": sum(row.conservative_before_or_without_prior_rescue for row in cell_rows),
            "conservative_after_prior_rescue": sum(row.conservative_after_prior_rescue for row in cell_rows),
        },
    }

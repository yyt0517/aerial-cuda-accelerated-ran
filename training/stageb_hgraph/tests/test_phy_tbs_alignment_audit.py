from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from training.stageb_hgraph.phy_tbs_alignment_audit import (
    PHY_TBS_AUDIT_V1_PROTOCOL_ID,
    PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    PhyTbsAuditUeTtiRow,
    PhyTbsConfig,
    compare_reference_labels,
    determine_tbs_bytes,
    join_phy_tbs_alignment_tti,
    phy_tbs_config_sha256,
    reconstruct_from_audit_record,
    summarize_phy_tbs_alignment_audit,
    validate_phy_tbs_audit_provenance,
)
from training.stageb_hgraph.online_protocol import (
    PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
    PHY_TBS_OLLA_SOURCE_CPU,
)
from training.stageb_hgraph.service_accounting_audit import (
    ServiceAccountingDecisionDiagnostics,
)


def _config() -> PhyTbsConfig:
    return PhyTbsConfig(
        mcs_lut_type=0,
        mcs_sinr_cap_db=25.99,
        layer_sinval_threshold=0.1,
        prbs_per_prg=16,
        pdsch_data_symbols=11,
        effective_layer_cap=4,
    )


def _record(*, prgs: tuple[int, ...], layer_caps: tuple[int, ...]) -> dict[str, object]:
    caps = [1] * 17
    for prg, cap in zip(prgs, layer_caps, strict=True):
        caps[prg] = cap
    return {
        "ue": 5,
        "cell": 1,
        "actual_prg_ids": list(prgs),
        "decision_pre_wb_sinr_first_layer_db": -4.0,
        "decision_pre_olla_delta": 0.0,
        "decision_pre_olla_delta_up": 1.0,
        "decision_pre_olla_delta_down": 0.1,
        "decision_pre_tb_err": -1,
        "decision_pre_prg_layer_cap": caps,
        "phy_tbs_config": _config().as_vector(),
    }


def test_determine_tbs_bytes_matches_hand_derived_small_and_large_cases() -> None:
    # Independent bit-table examples, converted once at the byte boundary.
    assert determine_tbs_bytes(16, 11, 1, 120.0 / 1024.0, 2) == 63
    assert determine_tbs_bytes(16, 11, 3, 120.0 / 1024.0, 2) == 185
    assert determine_tbs_bytes(32, 11, 4, 120.0 / 1024.0, 2) == 497
    assert determine_tbs_bytes(16, 11, 4, 602.0 / 1024.0, 2) == 1249


def test_prefix_uses_first_awarded_prgs_in_actor_order_and_exposes_negative_marginal() -> None:
    reconstructed = reconstruct_from_audit_record(
        _record(prgs=(2, 7, 12), layer_caps=(4, 4, 1))
    )

    assert reconstructed.reconstructed_mcs == 0
    assert reconstructed.prefix_local_prgs == ((2,), (2, 7), (2, 7, 12))
    assert reconstructed.prefix_layers == (4, 4, 1)
    assert reconstructed.prefix_tbs_bytes == (253, 497, 185)
    assert reconstructed.marginal_tbs_bytes == (253, 244, -312)


def test_zero_marginal_is_preserved_and_counted_as_zero() -> None:
    reconstructed = reconstruct_from_audit_record(
        _record(prgs=(2, 7), layer_caps=(2, 1))
    )

    assert reconstructed.prefix_tbs_bytes == (123, 123)
    assert reconstructed.marginal_tbs_bytes == (123, 0)
    assert reconstructed.zero_marginal_count == 1
    assert reconstructed.negative_marginal_count == 0


def test_reference_labels_are_not_reconstruction_inputs() -> None:
    record = _record(prgs=(2, 7), layer_caps=(2, 1))
    record.update(
        {
            "actual_ue_mcs": 0,
            "actual_ue_layer": 1,
            "actual_ue_tbs_capacity_bytes": 123,
        }
    )

    with_labels = reconstruct_from_audit_record(record)
    without_labels_record = deepcopy(record)
    del without_labels_record["actual_ue_mcs"]
    del without_labels_record["actual_ue_layer"]
    del without_labels_record["actual_ue_tbs_capacity_bytes"]
    without_labels = reconstruct_from_audit_record(without_labels_record)

    assert with_labels == without_labels
    assert compare_reference_labels(with_labels, record).all_exact is True


def test_reference_comparison_rejects_one_record_level_tbs_mismatch() -> None:
    record = _record(prgs=(2, 7), layer_caps=(2, 1))
    reconstructed = reconstruct_from_audit_record(record)
    bad_reference = {
        "actual_ue_mcs": 0,
        "actual_ue_layer": 1,
        "actual_ue_tbs_capacity_bytes": 985,
    }

    comparison = compare_reference_labels(reconstructed, bad_reference)

    assert comparison.mcs_exact is True
    assert comparison.layer_exact is True
    assert comparison.tbs_exact is False
    assert comparison.all_exact is False


def test_reconstruction_rejects_non_actor_order_prg_ids() -> None:
    with pytest.raises(ValueError, match="strict actor order"):
        reconstruct_from_audit_record(_record(prgs=(7, 2), layer_caps=(2, 1)))


def _audit_row(
    *,
    index: int,
    prgs: tuple[int, ...] = (0, 1, 2, 3),
    layer_caps: tuple[int, ...] = (1, 1, 1, 1),
    quality_db: float = -4.0,
    scheduled_scale: float = 1.0,
    reference_tbs_delta: int = 0,
    protocol_id: str = PHY_TBS_AUDIT_V1_PROTOCOL_ID,
) -> PhyTbsAuditUeTtiRow:
    record = _record(prgs=prgs, layer_caps=layer_caps)
    record["decision_pre_wb_sinr_first_layer_db"] = quality_db
    reconstructed = reconstruct_from_audit_record(record)
    reference_tbs = reconstructed.reconstructed_tbs_bytes + reference_tbs_delta
    record.update(
        {
            "actual_ue_mcs": reconstructed.reconstructed_mcs,
            "actual_ue_layer": reconstructed.reconstructed_layer,
            "actual_ue_tbs_capacity_bytes": reference_tbs,
            "phy_tbs_olla_source_code": PHY_TBS_OLLA_SOURCE_CPU,
            "phy_tbs_capture_phase_code": PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
        }
    )
    return PhyTbsAuditUeTtiRow.from_record(
        record,
        tti=1000 + index,
        ue=index % 36,
        cell=index % 2,
        planned_prg_ids=prgs,
        accepted_prg_ids=prgs,
        initial_backlog_bytes=100_000.0,
        raw_predicted_service_bytes=float(reference_tbs) * 1.5,
        scheduled_bytes=float(reference_tbs) * scheduled_scale,
        successful_bytes=float(reference_tbs) * scheduled_scale,
        quality_db_mean=quality_db,
        protocol_id=protocol_id,
    )


def test_config_hash_is_canonical_and_changes_with_tbs_configuration() -> None:
    first = phy_tbs_config_sha256(_config())
    same = phy_tbs_config_sha256(PhyTbsConfig.from_vector(_config().as_vector()))
    changed = phy_tbs_config_sha256(
        PhyTbsConfig.from_vector((*_config().as_vector()[:3], 8.0, 11.0, 4.0))
    )

    assert len(first) == 64
    assert first == same
    assert first != changed


def test_v2_config_hash_records_cpu_olla_reconstruction_semantics() -> None:
    v1 = phy_tbs_config_sha256(
        _config(), protocol_id=PHY_TBS_AUDIT_V1_PROTOCOL_ID
    )
    v2 = phy_tbs_config_sha256(
        _config(), protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
    )

    assert len(v2) == 64
    assert v2 != v1


def test_v2_provenance_requires_cpu_and_pre_action_pre_mcs_snapshot() -> None:
    valid = {
        "phy_tbs_olla_source_code": PHY_TBS_OLLA_SOURCE_CPU,
        "phy_tbs_capture_phase_code": PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
    }

    validate_phy_tbs_audit_provenance(
        valid, protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
    )
    with pytest.raises(ValueError, match="CPU OLLA"):
        validate_phy_tbs_audit_provenance(
            {**valid, "phy_tbs_olla_source_code": 0},
            protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
        )
    with pytest.raises(ValueError, match="pre-action, pre-MCS"):
        validate_phy_tbs_audit_provenance(
            {**valid, "phy_tbs_capture_phase_code": 0},
            protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
        )


def test_exact_parity_failure_has_priority_over_statistical_gates() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [_audit_row(index=0, reference_tbs_delta=8)]
    )

    assert report["status"] == "FAIL"
    assert report["classification"] == "decision_time_phy_path_not_exact"
    assert report["record_level_parity"]["tbs_mismatch_count"] == 1


def test_negative_prefix_marginal_has_priority_after_final_parity() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [
            _audit_row(
                index=0,
                prgs=(2, 7, 12),
                layer_caps=(4, 4, 1),
            )
        ]
    )

    assert report["record_level_parity"]["all_exact"] is True
    assert report["status"] == "FAIL"
    assert (
        report["classification"]
        == "final_tbs_reconstructable_but_not_sequentially_additive"
    )
    assert report["sequential_usability"]["negative_marginal_count"] == 1


def test_statistical_failure_is_classified_only_after_parity_and_monotonicity() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [_audit_row(index=index, scheduled_scale=0.8) for index in range(180)]
    )

    assert report["record_level_parity"]["all_exact"] is True
    assert report["sequential_usability"]["gate_passed"] is True
    assert report["status"] == "FAIL"
    assert report["classification"] == "conditional_alignment_gate_failure"
    assert report["global_alignment"]["gate_passed"] is False


def test_complete_exact_monotone_cohort_passes_frozen_double_gate() -> None:
    qualities = (-8.0, -4.0, 0.0)
    rows = [
        _audit_row(index=index, quality_db=qualities[index // 60])
        for index in range(180)
    ]

    report = summarize_phy_tbs_alignment_audit(rows)

    assert report["status"] == "PASS"
    assert report["classification"] == "phy_tbs_alignment_compatible"
    assert report["reference_label_deletion_equal"] is True
    assert report["record_level_parity"]["all_exact"] is True
    assert report["sequential_usability"]["negative_marginal_count"] == 0
    assert report["sequential_usability"]["nan_or_inf_count"] == 0
    assert report["global_alignment"]["scheduled_to_predicted_tbs_ratio"] == 1.0
    assert report["global_alignment"]["tbs_wape"] == 0.0
    assert report["global_alignment"]["raw_wape_improvement_fraction"] == 1.0
    assert report["quality_prg_count_gate"]["formal_group_count"] == 3
    assert report["cell_residual_gate"]["formal_group_count"] == 6


def test_v2_parity_failure_stops_before_prefix_gate() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [
            _audit_row(
                index=0,
                reference_tbs_delta=8,
                protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
            )
        ],
        protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    )

    assert report["classification"] == "decision_time_phy_path_not_exact"
    assert report["sequential_usability"]["formal_gate_evaluated"] is False
    assert report["sequential_usability"]["gate_passed"] is None


def test_v2_exact_parity_then_negative_marginal_stops_b4_tbs() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [
            _audit_row(
                index=0,
                prgs=(2, 7, 12),
                layer_caps=(4, 4, 1),
                protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
            )
        ],
        protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    )

    assert report["record_level_parity"]["all_exact"] is True
    assert report["sequential_usability"]["formal_gate_evaluated"] is True
    assert report["classification"] == (
        "final_tbs_reconstructable_but_not_sequentially_additive"
    )


def test_v2_exact_monotone_passes_without_statistical_gate_cohort() -> None:
    report = summarize_phy_tbs_alignment_audit(
        [
            _audit_row(
                index=0, protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
            )
        ],
        protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    )

    assert report["record_level_parity"]["all_exact"] is True
    assert report["causal_provenance"]["cpu_olla_source_exact"] is True
    assert report["causal_provenance"]["pre_action_pre_mcs_capture_exact"] is True
    assert report["sequential_usability"]["gate_passed"] is True
    assert report["quality_prg_count_gate"]["formal_group_count"] == 0
    assert report["statistical_alignment"]["formal_gate"] is False
    assert report["classification"] == "phy_tbs_alignment_compatible"


def test_online_join_uses_actor_prg_order_and_decision_pre_inputs() -> None:
    diagnostics = ServiceAccountingDecisionDiagnostics(
        selected_ue=torch.tensor([0, 0]),
        cell_index=torch.tensor([0, 0]),
        local_prg_index=torch.tensor([0, 1]),
        selected_nonblank=torch.tensor([True, True]),
        raw_service_bytes=torch.tensor([700.0, 700.0]),
        effective_service_bytes=torch.tensor([700.0, 700.0]),
        initial_demand_bytes=torch.tensor([10_000.0, 10_000.0]),
        remaining_demand_before_bytes=torch.tensor([10_000.0, 9_300.0]),
        quality_db=torch.tensor([-4.0, -4.0]),
    )
    info = {
        "harq_enabled": 0,
        "accepted_pre_pdsch_ue_prg_mask": np.asarray([[1, 1]], dtype=np.uint8),
        "actual_ue_prg_mask": np.asarray([[1, 1]], dtype=np.uint8),
        "actual_ue_served_bytes": np.asarray([123.0], dtype=np.float32),
        "actual_ue_goodput_bytes": np.asarray([123.0], dtype=np.float32),
        "decision_pre_wb_sinr_first_layer_db": np.asarray([-4.0], dtype=np.float32),
        "decision_pre_olla_delta": np.asarray([0.0], dtype=np.float32),
        "decision_pre_olla_delta_up": np.asarray([1.0], dtype=np.float32),
        "decision_pre_olla_delta_down": np.asarray([0.1], dtype=np.float32),
        "decision_pre_tb_err": np.asarray([-1], dtype=np.int32),
        "decision_pre_prg_layer_cap": np.asarray([[1, 1]], dtype=np.uint8),
        "phy_tbs_config": np.asarray(_config().as_vector(), dtype=np.float32),
        "phy_tbs_olla_source_code": PHY_TBS_OLLA_SOURCE_CPU,
        "phy_tbs_capture_phase_code": PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS,
        "actual_ue_mcs": np.asarray([0], dtype=np.int32),
        "actual_ue_layer": np.asarray([1], dtype=np.int32),
        "actual_ue_tbs_capacity_bytes": np.asarray([123], dtype=np.int32),
    }

    rows = join_phy_tbs_alignment_tti(
        diagnostics,
        info=info,
        n_ue=1,
        n_cell=1,
        n_prg=2,
        tti=1001,
        protocol_id=PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID,
    )

    assert len(rows) == 1
    assert rows[0].planned_prg_ids == (0, 1)
    assert rows[0].accepted_prg_ids == (0, 1)
    assert rows[0].actual_prg_ids == (0, 1)
    assert rows[0].raw_predicted_service_bytes == 1400.0
    assert rows[0].reconstructed.prefix_local_prgs == ((0,), (0, 1))
    assert rows[0].reference.all_exact is True
    assert rows[0].protocol_id == PHY_TBS_AUDIT_V2_CPU_OLLA_PROTOCOL_ID
    assert rows[0].olla_source_code == PHY_TBS_OLLA_SOURCE_CPU
    assert rows[0].capture_phase_code == PHY_TBS_CAPTURE_PHASE_PRE_ACTION_PRE_MCS
    assert rows[0].reference_label_deletion_equal is True

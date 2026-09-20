import numpy as np
import pytest

from training.stageb_hgraph.capacity_units_decision_probe import (
    check_capacity_row, canonical_actions, compare_actions,
)


def test_raw_bits_detect_shared_capacity_and_dequeue_inflation():
    row = dict(tbs_capacity_bits=504, tbs_capacity_bytes=63,
               scheduled_dequeue_bytes=63, successful_bytes=40, actual_prg_count=1,
               mcs=0, layer=1)
    check_capacity_row(row)
    with pytest.raises(ValueError, match="bits"):
        check_capacity_row({**row, "scheduled_dequeue_bytes": 504, "tbs_capacity_bytes": 504})
    with pytest.raises(ValueError, match="literal"):
        check_capacity_row({**row, "tbs_capacity_bits": 4032, "tbs_capacity_bytes": 504})


def test_canonical_action_agreement_counts_blank_and_first_divergence():
    rows = [dict(tti=t, ue_id=7, serving_cell=0, actual_prg_mask="0x1") for t in (1, 2)]
    a = canonical_actions(rows, total_tti=2, n_cell=1, n_prg=2)
    assert a.tolist() == [[7, -1], [7, -1]]
    b = np.array([[7, -1], [8, -1]])
    result = compare_actions(a, b, warmup=1)
    assert result["first_divergence_tti"] == 2
    assert result["measured_token_agreement"] == .5
    assert result["measured_whole_tti_agreement"] == 0
    assert result["full_token_agreement"] == .75
    with pytest.raises(ValueError, match="duplicate"):
        canonical_actions(rows + [rows[0]], total_tti=2, n_cell=1, n_prg=2)

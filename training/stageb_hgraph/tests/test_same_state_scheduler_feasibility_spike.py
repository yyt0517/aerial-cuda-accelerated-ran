import numpy as np

from training.stageb_hgraph.same_state_scheduler_feasibility_spike import (
    decode_type0_teacher_action,
    summarize_same_state_actions,
)


def test_decode_type0_teacher_action_uses_cell_major_output() -> None:
    teacher_ue = np.asarray([4, 7, 9, 11], dtype=np.int64)
    # Native Type-0 allocation is local-PRG-major, then cell.
    teacher_prg = np.asarray([0, 2, -1, 1], dtype=np.int64)

    chosen = decode_type0_teacher_action(
        teacher_ue,
        teacher_prg,
        n_cell=2,
        n_prg=2,
    )

    # Shadow actor tokens are cell-major, then local PRG.
    assert chosen.tolist() == [4, -1, 9, 7]


def test_same_state_summary_counts_changes_and_selected_quality() -> None:
    pfq = np.asarray([0, 1, 2, -1], dtype=np.int64)
    ai = np.asarray([0, -1, 3, -1], dtype=np.int64)
    post_eq = np.ones((4, 2, 1), dtype=np.float64)
    post_eq[0, 0, 0] = 1.0
    post_eq[1, 1, 0] = 10.0
    post_eq[2, 0, 0] = 0.1
    post_eq[3, 0, 0] = 100.0

    summary = summarize_same_state_actions(
        pfq,
        ai,
        post_eq_sinr=post_eq,
        n_cell=2,
        n_prg=2,
    )

    assert summary["changed_cell_prg_count"] == 2
    assert summary["action_change_ratio"] == 0.5
    assert summary["pfq_scheduled_cell_prg_count"] == 3
    assert summary["ai_scheduled_cell_prg_count"] == 2
    assert summary["scheduled_cell_prg_delta_ai_minus_pfq"] == -1
    assert summary["pfq_selected_sinr_db"] == [0.0, 10.0, -10.0]
    assert summary["ai_selected_sinr_db"] == [0.0, 20.0]

"""Opt-in end-to-end test: two real simulator executions, only slot order differs."""
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize('policy', ['pf', 'rr'])
def test_physical_result_is_invariant_to_internal_slot_permutation(policy):
    root = os.environ.get('CUMAC_PHY_SLOT_PROBE_RESULTS')
    if not root:
        pytest.skip('requires generated real-simulator slot probe (set CUMAC_PHY_SLOT_PROBE_RESULTS)')
    result = json.loads((Path(root) / 'invariance.json').read_text())[policy]
    assert result['physical_inputs_equal'], result
    assert result['changed_slot_count'] > 0, 'permutation must actually change storage'
    assert result['max_sinr_delta_cpu_db'] <= 1e-5, result
    assert result['max_sinr_delta_gpu_db'] <= 1e-5, result
    assert result['max_predicted_bler_delta_cpu'] <= 1e-6, result
    assert result['max_predicted_bler_delta_gpu'] <= 1e-6, result
    assert result['tb_error_changed_ues'] == 0, result
    assert result['goodput_changed_ues'] == 0, result

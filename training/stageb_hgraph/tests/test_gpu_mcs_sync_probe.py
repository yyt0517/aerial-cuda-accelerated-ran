"""Actual kernel replay evidence: missing init barrier must fail count/MCS checks."""
import csv
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize('policy', ['pf', 'rr'])
def test_fixed_input_gpu_count_and_mcs_match_independent_reference(policy):
    path = os.environ.get('CUMAC_MCS_SYNC_RESULTS')
    if not path:
        pytest.skip('set CUMAC_MCS_SYNC_RESULTS to real simulator probe results')
    rows = list(csv.DictReader((Path(path) / policy / 'gpu_mcs_replay.csv').open()))
    assert len({r['repeat'] for r in rows}) == 128
    assert any(int(r['expected_count']) > 0 for r in rows)
    assert all(int(r['raw_count']) == int(r['expected_count']) for r in rows)
    assert all(int(r['raw_mcs']) == int(r['expected_mcs']) for r in rows)
    assert all(abs(float(r['delta_after']) - float(r['expected_delta'])) < 1e-6 for r in rows)
    for slot in {r['slot'] for r in rows}:
        samples = [r for r in rows if r['slot'] == slot]
        assert len({(r['raw_count'], r['raw_mcs'], r['delta_after']) for r in samples}) == 1

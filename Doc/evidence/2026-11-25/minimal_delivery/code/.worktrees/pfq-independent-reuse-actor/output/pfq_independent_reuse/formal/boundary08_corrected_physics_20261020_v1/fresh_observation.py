"""Read experiment-local fresh observations after the existing CTBS response."""
from pathlib import Path
import numpy as np

ARRAYS = [
    ("obs_cell_features", "<f4"), ("obs_ue_features", "<f4"),
    ("obs_prg_features", "<f4"), ("obs_edge_index", "<i2"),
    ("obs_edge_attr", "<f4"), ("action_mask_ue", "u1"),
    ("action_mask_cell_ue", "u1"), ("action_mask_prg_cell", "u1"),
    ("post_eq_sinr", "<f4"), ("teacher_action_ue_select", "<i4"),
    ("teacher_action_prg_alloc", "<i2"),
]


def read(path, template, ctbs):
    raw = Path(path).read_bytes()
    header = np.frombuffer(raw, dtype="<u8", count=16)
    assert tuple(header[:5]) == (0x46524F31, 1, ctbs.tti, ctbs.snapshot_id,
                                  int(template["n_sched_ue"])), header[:5]
    offset = 16 * 8
    result = {"n_sched_ue": np.int32(header[4])}
    for (key, dtype), size in zip(ARRAYS, header[5:]):
        expected = np.asarray(template[key])
        assert int(size) == expected.size, (key, size, expected.shape)
        a = np.frombuffer(raw, dtype=dtype, count=int(size), offset=offset).copy()
        offset += int(size) * np.dtype(dtype).itemsize
        result[key] = a.reshape(expected.shape).astype(expected.dtype, copy=False)
    assert offset == len(raw)
    assert np.isfinite(result["obs_ue_features"]).all()
    assert result["post_eq_sinr"].shape == (36, 17, 4)
    return result

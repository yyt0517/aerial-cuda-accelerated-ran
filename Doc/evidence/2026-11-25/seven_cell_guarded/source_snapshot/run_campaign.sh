#!/usr/bin/env bash
set -euo pipefail
cd /data/aerial-cuda-accelerated-ran
experiment_root=deliverables/seven_cell_boundary08_guarded_l010_b098_20261106_v1
baseline_root=deliverables/seven_cell_boundary08_20261105_v1
multistart_root=deliverables/seven_cell_boundary08_multistart_20261105_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('deliverables/seven_cell_boundary08_guarded_l010_b098_20261106_v1')
assert json.loads((r/'preflight.json').read_text())['passed']
for f in ['smoke/analysis/paired_validation.json','smoke/analysis/search_audit.json']:
    assert json.loads((r/f).read_text())['passed']
assert json.loads((r/'smoke/campaign/completion.json').read_text())['completed']
PY
docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=2 \
 CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log \
 .venv/bin/python -u -m training.stageb_sevencell_guarded.campaign \
 --output "$experiment_root/campaign" --build "$baseline_root/build_final" --baseline "$baseline_root" --workers 3
.venv/bin/python -u -m training.stageb_sevencell_guarded.analyze --root "$experiment_root" --baseline "$baseline_root" --multistart "$multistart_root" --workers 3
.venv/bin/python -u -m training.stageb_sevencell_guarded.audit --source "$experiment_root/campaign" --output "$experiment_root/analysis/search_audit.json"
MPLCONFIGDIR=/tmp/sevencell_guarded_matplotlib .venv/bin/python -u -m training.stageb_sevencell_guarded.report --root "$experiment_root"
.venv/bin/python -u -m training.stageb_sevencell_guarded.finalize --root "$experiment_root" --baseline "$baseline_root" --multistart "$multistart_root"

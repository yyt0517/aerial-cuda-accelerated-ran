#!/usr/bin/env bash
set -euo pipefail
cd /data/aerial-cuda-accelerated-ran
experiment_root=deliverables/seven_cell_boundary08_multistart_20261105_v1
baseline_root=deliverables/seven_cell_boundary08_20261105_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('deliverables/seven_cell_boundary08_multistart_20261105_v1')
for p in ['smoke/analysis/paired_validation.json','smoke/analysis/search_audit.json']:
    assert json.loads((r/p).read_text())['passed']
assert json.loads((r/'smoke/campaign/completion.json').read_text())['completed']
PY
docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=2 \
 CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log \
 .venv/bin/python -u -m training.stageb_sevencell_multistart.campaign \
 --output "$experiment_root/campaign" --build "$baseline_root/build_final" --baseline "$baseline_root" --workers 3
.venv/bin/python -u -m training.stageb_sevencell_multistart.analyze --root "$experiment_root" --baseline "$baseline_root" --workers 3
.venv/bin/python -u -m training.stageb_sevencell_multistart.audit --source "$experiment_root/campaign" --output "$experiment_root/analysis/search_audit.json"
MPLCONFIGDIR=/tmp/sevencell_multistart_matplotlib .venv/bin/python -u -m training.stageb_sevencell_multistart.report --root "$experiment_root"

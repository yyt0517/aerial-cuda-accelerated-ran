#!/usr/bin/env bash
set -euo pipefail
cd /data/aerial-cuda-accelerated-ran
experiment_root=deliverables/seven_cell_boundary08_20261105_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
r=Path('deliverables/seven_cell_boundary08_20261105_v1')
for f in ['three_cell_parity/parity.json','topology_check.json','smoke_analysis/paired_validation.json','smoke_search_audit_all.json']:
 assert json.loads((r/f).read_text())['passed'], f
assert json.loads((r/'smoke_search_audit_all.json').read_text())['states']==6
assert json.loads((r/'smoke_final/completion.json').read_text())['completed']
PY
docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=2 \
 CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log \
 .venv/bin/python -u -m training.stageb_sevencell.campaign \
 --output "$experiment_root/campaign" --build "$experiment_root/build_final" --workers 9
.venv/bin/python -u -m training.stageb_sevencell.analyze --source "$experiment_root/campaign" --output "$experiment_root/analysis" --workers 3
.venv/bin/python -u -m training.stageb_sevencell.audit_saved --source "$experiment_root/campaign" --output "$experiment_root/analysis/search_audit.json" --all
.venv/bin/python -u -m training.stageb_sevencell.report --root "$experiment_root"

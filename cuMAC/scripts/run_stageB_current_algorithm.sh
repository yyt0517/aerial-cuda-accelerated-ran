#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="${repo_root}/training/stageb_hgraph/current_algorithm.json"
if [[ "${1:-}" == "--show-config" ]]; then
    cat -- "${config_path}"
    exit 0
fi
if [[ "${1:-}" == "--help" ]]; then
    echo 'Latest Stage-B research policy: PFQ+CTBS+deficit-weighted physical search, causal floor rho=0.90.'
    echo 'Usage: bash cuMAC/scripts/run_stageB_current_algorithm.sh --seed 41 [--gpu 0] [--smoke]'
    echo 'Use --show-config to inspect the active version. Requires the existing c_aerial_oai2 CUDA container and research worktree.'
    exit 0
fi
mapfile -t current_runtime < <(python3 - "${config_path}" <<'PY'
import json,sys
config=json.load(open(sys.argv[1]))
assert config['floor']['numerator']==90 and config['floor']['denominator']==100
assert not config['directed_coordinator_required']
runtime=config['runtime']
for key in ['container','container_workdir','container_python','module']:
    print(runtime[key])
PY
)
[[ "${#current_runtime[@]}" -eq 4 ]]
exec docker exec -w "${current_runtime[1]}" "${current_runtime[0]}" \
    "${current_runtime[2]}" -u -m "${current_runtime[3]}" "$@"

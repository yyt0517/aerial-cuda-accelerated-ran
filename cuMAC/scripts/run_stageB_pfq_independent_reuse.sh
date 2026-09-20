#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

PHASE="${1:-}"
if [[ -z "${PHASE}" ]]; then
    echo "usage: $(basename "$0") PHASE --build-dir DIR --out-dir DIR [--steps N] [--resume]" >&2
    exit 2
fi
shift

BUILD_DIR="build.reuse_actor"
OUT_DIR="output/pfq_independent_reuse"
STEPS="32"
RESUME=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "unsupported option: $1" >&2; exit 2 ;;
    esac
done

case "${PHASE}" in
    build|pfq-reference|validity|bc|pilot-A|pilot-B|pilot-C|pilot-D|select-and-eval-seed41|blank-diagnostics-seed41|hierarchical-decode-smoke|hierarchical-decode-seed41|rescue-cohort-smoke|rescue-cohort-seed41|candidate-width-compatibility|candidate-width-fp32-equivalence|formal-seed42-pfq-b4|diagnose-seed42-per-prg|audit-seed42-service-accounting|audit-seed42-phy-tbs-alignment|audit-seed42-phy-tbs-alignment-v2-cpu-olla|b4-ctbs-seed42|audit-seed42-ctbs-blank-expiry-attribution|audit-seed42-ctbs-blank-expiry-attribution-v2-full-horizon|audit-seed42-cell2-ue2-service-balance|b4-ctbs-c1-short-treatment|b4-ctbs-c1-multiseed-long|paired-seeds-41-43) ;;
    *) echo "unsupported phase: ${PHASE}" >&2; exit 2 ;;
esac
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be a positive integer" >&2; exit 2; }

if [[ "${BUILD_DIR}" != /* ]]; then BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"; fi
if [[ "${OUT_DIR}" != /* ]]; then OUT_DIR="${ROOT_DIR}/${OUT_DIR}"; fi
SIM_BIN="${BUILD_DIR}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
SIM_ARGS="-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000"
SCENARIO_ID="coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41"

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
source_revision() {
    if git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null; then
        return
    fi
    local common_root worktree_name admin_head revision ref_path
    common_root="$(cd "${ROOT_DIR}/../.." && pwd)"
    worktree_name="$(basename "${ROOT_DIR}")"
    admin_head="${common_root}/.git/worktrees/${worktree_name}/HEAD"
    [[ -f "${admin_head}" ]] || { echo "cannot resolve source revision" >&2; return 1; }
    revision="$(<"${admin_head}")"
    if [[ "${revision}" == ref:\ * ]]; then
        ref_path="${common_root}/.git/${revision#ref: }"
        [[ -f "${ref_path}" ]] || { echo "cannot resolve source ref ${revision}" >&2; return 1; }
        revision="$(<"${ref_path}")"
    fi
    printf '%s\n' "${revision}"
}
write_run_manifest() {
    local phase_dir="$1"
    local command_line="$2"
    local manifest="${phase_dir}/run_manifest.json"
    local binary_hash="missing"
    if [[ -x "${SIM_BIN}" ]]; then binary_hash="$(sha256_file "${SIM_BIN}")"; fi
    if [[ -d "${phase_dir}" && -n "$(find "${phase_dir}" -mindepth 1 -maxdepth 1 -print -quit)" && ! -e "${manifest}" ]]; then
        echo "existing output directory has no compatible run manifest: ${phase_dir}" >&2
        exit 2
    fi
    mkdir -p "${phase_dir}"
    if [[ -e "${manifest}" ]]; then
        if [[ "${RESUME}" != "1" ]]; then
            echo "output exists; pass --resume only for an identical manifest: ${phase_dir}" >&2
            exit 2
        fi
        "${PYTHON_BIN}" - "${manifest}" "${PHASE}" "${binary_hash}" "$(source_revision)" "${SCENARIO_ID}" <<'PY'
import json, sys
path, phase, binary, source, scenario = sys.argv[1:]
old = json.load(open(path, encoding="utf-8"))
expected = {"phase": phase, "binary_sha256": binary, "source_revision": source, "scenario_id": scenario}
for key, value in expected.items():
    if old.get(key) != value:
        raise SystemExit(f"resume manifest mismatch for {key}: {old.get(key)!r} != {value!r}")
PY
        return
    fi
    "${PYTHON_BIN}" - "${manifest}" "${PHASE}" "${binary_hash}" "$(source_revision)" "${SCENARIO_ID}" "${command_line}" <<'PY'
import json, sys
path, phase, binary, source, scenario, command = sys.argv[1:]
payload = {
    "phase": phase,
    "binary_sha256": binary,
    "source_revision": source,
    "scenario_id": scenario,
    "command": command,
    "canonical": {
        "cells": 3, "ues": 36, "prgs": 17, "allocation_type": 0,
        "placement": "coop_boundary", "association": "strongest",
        "precoding": "svd", "packet_bytes": 3000,
        "packets_per_ue_tti": 1.0, "packet_ttl_ms": 200.0,
    },
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}
require_binary() {
    [[ -x "${SIM_BIN}" ]] || { echo "missing simulator binary: ${SIM_BIN}; run build first" >&2; exit 2; }
}
require_file() { [[ -f "$1" ]] || { echo "missing required artifact: $1" >&2; exit 2; }; }

run_build() {
    local phase_dir="${OUT_DIR}/build"
    write_run_manifest "${phase_dir}" "cmake configure and build expiry accounting plus scheduler"
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${ROOT_DIR}/cuPHY/cmake/toolchains/native" \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native \
        2>&1 | tee "${phase_dir}/configure.log"
    cmake --build "${BUILD_DIR}" --target trafficPacketAccountingTest multiCellSchedulerUeSelection -j8 \
        2>&1 | tee "${phase_dir}/build.log"
    "${PYTHON_BIN}" - "${phase_dir}/run_manifest.json" "$(sha256_file "${SIM_BIN}")" <<'PY'
import json, sys
path, binary = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
payload["binary_sha256"] = binary
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_pfq_reference() {
    require_binary
    local phase_dir="${OUT_DIR}/pfq_seed41"
    write_run_manifest "${phase_dir}" "pfq reference: 1000 warmup + 4000 measured TTIs"
    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 41 --traffic-seed 41 --warmup-steps 1000 --measure-steps 4000 \
        --formal --out-dir "${phase_dir}"
}

run_bc() {
    require_binary
    local phase_dir="${OUT_DIR}/bc_seed41"
    write_run_manifest "${phase_dir}" "independent PFQ-label BC: 4000 teacher TTIs"
    "${PYTHON_BIN}" -m training.stageb_hgraph.online_imitation \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --socket-path "/tmp/cumac_independent_bc.sock" --online-persistent 1 \
        --episode-horizon 4000 --max-steps 4000 --episodes 1 \
        --topology-seed 41 --topology-seed-mode fixed --seed 41 --device auto \
        --out-dir "${phase_dir}" --action-head-mode independent \
        --candidate-mode all_demand_ues --include-pfq-score-anchor 0 \
        --teacher-mode pfq_passthrough --hidden-dim 128 --message-layers 2 \
        --dropout 0 --save-every-episodes 1 --print-every 500
    local checkpoint="${phase_dir}/hgraph_independent_bc_last.pt"
    require_file "${checkpoint}"
    sha256_file "${checkpoint}" > "${phase_dir}/bc_checkpoint.sha256"
}

run_pilot() {
    local treatment="$1"
    require_binary
    local phase_dir="${OUT_DIR}/pilot-${treatment}"
    local reference_dir="${OUT_DIR}/pfq_seed41"
    local scales="${reference_dir}/reward_scales.json"
    local bc_checkpoint="${OUT_DIR}/bc_seed41/hgraph_independent_bc_last.pt"
    require_file "${scales}"
    local init_args=()
    local bc_hash="none"
    if [[ "${treatment}" != "A" ]]; then
        require_file "${bc_checkpoint}"
        bc_hash="$(sha256_file "${bc_checkpoint}")"
        init_args=(--init-checkpoint "${bc_checkpoint}")
        for sibling in B C D; do
            local sibling_hash="${OUT_DIR}/pilot-${sibling}/bc_checkpoint.sha256"
            if [[ -f "${sibling_hash}" ]] && [[ "$(<"${sibling_hash}")" != "${bc_hash}" ]]; then
                echo "B--D BC checkpoint hash mismatch" >&2
                exit 2
            fi
        done
    fi
    write_run_manifest "${phase_dir}" "treatment ${treatment}: 20x512 PPO TTIs; BC SHA256=${bc_hash}"
    if [[ "${bc_hash}" != "none" ]]; then printf '%s\n' "${bc_hash}" > "${phase_dir}/bc_checkpoint.sha256"; fi
    "${PYTHON_BIN}" -m training.stageb_hgraph.online_ppo \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --socket-path "/tmp/cumac_independent_ppo_${treatment}.sock" \
        --online-persistent 1 --episode-horizon 4000 --rollout-steps 512 --iterations 20 \
        --rollout-warmup-steps 1000 --rollout-warmup-once 1 \
        --topology-seed 41 --topology-seed-mode fixed --seed 41 --device auto \
        --out-dir "${phase_dir}" --action-head-mode independent \
        --candidate-mode all_demand_ues --max-prg-candidates 13 \
        --include-pfq-score-anchor 0 --include-candidate-risk-features 0 \
        --pfq-residual-init 0 --reward-mode pfq_independent \
        --independent-treatment "${treatment}" --independent-reward-scales "${scales}" \
        --teacher-mode none --teacher-aux-coef 0 --teacher-aux-min-coef 0 \
        --imitation-episodes 0 --slot-repair-max-per-cell 0 --safe-fill-max-per-cell 0 \
        --candidate-blank-budget-max-per-cell -1 --candidate-checkpoint-interval 4 \
        --candidate-checkpoint-require-gate 0 --hidden-dim 128 --message-layers 2 \
        --ue-feat-dim 16 --dropout 0 --early-stop-patience 0 --best-metric goodput \
        --sim-env "CUMAC_PACKET_DELAY_SAMPLES=1" \
        --sim-env "CUMAC_PACKET_OUTCOME_SAMPLES_CSV=${phase_dir}/packet_outcome_samples.csv" \
        "${init_args[@]}"
}

evaluate_checkpoint() {
    local checkpoint="$1" seed="$2" warmup="$3" measure="$4" eval_dir="$5"
    local normalize_gate="${6:-1}"
    local diagnostics="${7:-0}"
    local decode_mode="${8:-argmax}"
    local -a extra_args=()
    if [[ "$#" -gt 8 ]]; then extra_args=("${@:9}"); fi
    local socket="/tmp/cumac_independent_eval_${seed}_$$.sock"
    mkdir -p "${eval_dir}"
    eval_dir="$(cd "${eval_dir}" && pwd -P)"
    local outcome_csv="${eval_dir}/packet_outcome_samples.csv"
    local packet_delay_csv="${eval_dir}/packet_delay_samples.csv"
    (
        cd "${eval_dir}"
        env CUMAC_ONLINE_BRIDGE=1 CUMAC_ONLINE_SOCKET="${socket}" CUMAC_ONLINE_PERSISTENT=1 \
            CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
            CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT="${CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT:-0}" \
            CUMAC_SERVICE_BALANCE_TRACE_CSV="${CUMAC_SERVICE_BALANCE_TRACE_CSV:-}" \
            CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI="${CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI:-602}" \
            CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="${CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV:-}" \
            CUMAC_CELL_ASSOC_MODE=strongest CUMAC_UE_PLACEMENT_MODE=coop_boundary \
            CUMAC_UE_VORONOI_CLIP=1 CUMAC_BS_TX_PATTERN=omni \
            CUMAC_TOPOLOGY_SEED="${seed}" CUMAC_TRAFFIC_SEED="${seed}" \
            CUMAC_TRAFFIC_ARRIVAL_RATE=1.0 CUMAC_PACKET_TTL_TTI=0 CUMAC_PACKET_TTL_MS=200 \
            CUMAC_COMPACT_TTI_LOG=1 CUMAC_PROGRESS_TTI_INTERVAL=0 CUMAC_COMPARE_TTI_INTERVAL=0 \
            CUMAC_PACKET_DELAY_SAMPLES=1 CUMAC_PACKET_DELAY_SAMPLES_CSV="${packet_delay_csv}" \
            CUMAC_PACKET_OUTCOME_SAMPLES_CSV="${outcome_csv}" \
            CUMAC_EXEC_MODE=both "${SIM_BIN}" ${SIM_ARGS}
    ) > "${eval_dir}/sim.log" 2>&1 &
    local sim_pid=$!
    set +e
    "${PYTHON_BIN}" -m training.stageb_hgraph.eval_checkpoint_online \
        --checkpoint "${checkpoint}" --socket-path "${socket}" \
        --episode-horizon "$((warmup + measure))" --max-steps "$((warmup + measure))" \
        --stats-warmup-steps "${warmup}" --decode-mode "${decode_mode}" --seed "${seed}" \
        --sample-seed "${seed}" --device auto --print-every-steps 500 --save-steps 0 \
        --sim-bin "${SIM_BIN}" --outcome-samples "${outcome_csv}" --out-dir "${eval_dir}" \
        --independent-policy-diagnostics "${diagnostics}" \
        "${extra_args[@]}" \
        > "${eval_dir}/eval.log" 2>&1
    local eval_rc=$?
    wait "${sim_pid}"
    local sim_rc=$?
    set -e
    if [[ "${eval_rc}" -ne 0 || "${sim_rc}" -ne 0 ]]; then
        echo "checkpoint evaluation failed: eval_rc=${eval_rc} sim_rc=${sim_rc}; ${eval_dir}" >&2
        exit 1
    fi
    if [[ "${normalize_gate}" == "1" ]]; then
        "${PYTHON_BIN}" -m training.stageb_hgraph.independent_gate normalize-eval \
            --summary "${eval_dir}/eval_summary.json" --out "${eval_dir}/gate_metrics.json"
    fi
}

run_candidate_width_compatibility() {
    require_binary
    require_file "${OUT_DIR}/seed41_selection/treatment_comparison.json"
    local winner selected_iteration checkpoint phase_dir
    winner="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_treatment"] or "")' "${OUT_DIR}/seed41_selection/treatment_comparison.json")"
    [[ -n "${winner}" ]] || { echo "seed-41 gate has no winner; compatibility smoke is forbidden" >&2; exit 2; }
    selected_iteration="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_iteration"])' "${OUT_DIR}/pilot-${winner}/checkpoint_selection.json")"
    checkpoint="${OUT_DIR}/pilot-${winner}/candidate_checkpoints/ppo_actor_iter_$(printf '%04d' "${selected_iteration}").pt"
    require_file "${checkpoint}"
    phase_dir="${OUT_DIR}/candidate_width_compatibility_v1"
    write_run_manifest "${phase_dir}" "static width gate; seed41 13-to-14 semantic equivalence; seed42 ${STEPS}-TTI integrity smoke"

    "${PYTHON_BIN}" - "${checkpoint}" "${phase_dir}/static_structure_gate.json" <<'PY'
import json
import pathlib
import sys
from dataclasses import replace

import torch

from training.stageb_hgraph.independent_actor import IndependentActorConfig, PFQIndependentReuseActor
from training.stageb_hgraph.independent_contract import state_dict_sha256

checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2]).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
raw_config = checkpoint["actor_config"]
fields = IndependentActorConfig.__dataclass_fields__
training_config = IndependentActorConfig(
    **{key: raw_config[key] for key in fields if key in raw_config}
)
training_width = int(
    raw_config.get("training_observed_max_candidates", raw_config["max_candidates"])
)
runtime_config = replace(training_config, max_candidates=14)
training_actor = PFQIndependentReuseActor(training_config)
runtime_actor = PFQIndependentReuseActor(runtime_config)
training_shapes = {name: list(value.shape) for name, value in training_actor.state_dict().items()}
runtime_shapes = {name: list(value.shape) for name, value in runtime_actor.state_dict().items()}
width_dependent = sorted(
    name for name in training_shapes if training_shapes[name] != runtime_shapes[name]
)
state = checkpoint["actor_state"]
strict_load = True
try:
    runtime_actor.load_state_dict(state, strict=True)
except Exception:
    strict_load = False
payload = {
    "passed": training_width == 13 and not width_dependent and strict_load,
    "checkpoint": str(checkpoint_path),
    "checkpoint_file_sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
    "actor_state_dict_sha256": state_dict_sha256(state),
    "training_observed_max_candidates": training_width,
    "probed_runtime_candidate_width": 14,
    "state_tensor_count": len(training_shapes),
    "parameter_count": sum(int(value.numel()) for value in state.values()),
    "width_dependent_state_tensors": width_dependent,
    "strict_state_load_at_width_14": strict_load,
}

output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not payload["passed"]:
    raise SystemExit(f"candidate width static structure gate failed: {payload}")
PY

    evaluate_checkpoint "${checkpoint}" 41 0 "${STEPS}" "${phase_dir}/seed-41-width13-vs14" 0 0 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-equivalence-reference-width 13 \
        --device cpu
    evaluate_checkpoint "${checkpoint}" 42 0 "${STEPS}" "${phase_dir}/seed-42-width14" 0 0 argmax \
        --independent-runtime-candidate-width 0 \
        --device cpu

    "${PYTHON_BIN}" - "${phase_dir}" "${STEPS}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected_steps = int(sys.argv[2])
static = json.loads((root / "static_structure_gate.json").read_text(encoding="utf-8"))
seed41 = json.loads((root / "seed-41-width13-vs14/eval_summary.json").read_text(encoding="utf-8"))
seed42 = json.loads((root / "seed-42-width14/eval_summary.json").read_text(encoding="utf-8"))
equivalence = seed41.get("candidate_width_equivalence") or {}
common = lambda summary: {
    "completed_steps": summary.get("completed_steps") == expected_steps,
    "candidate_mode": summary.get("candidate_mode") == "all_demand_ues",
    "candidate_width_policy": summary.get("candidate_width_policy") == "runtime_topology",
    "candidate_truncation_forbidden": summary.get("candidate_truncation") == "forbidden",
    "candidate_full_coverage": summary.get("candidate_full_coverage") is True,
    "zero_truncation": summary.get("candidate_truncation_count") == 0,
    "zero_replay_mismatch": summary.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal_action": summary.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_deletion_equal": summary.get("independent_deletion_equal") is True,
    "state_dict_frozen": summary.get("actor_state_dict_sha256") == static.get("actor_state_dict_sha256"),
    "training_width_is_metadata": summary.get("training_observed_max_candidates") == 13,
}
checks = {
    "static_structure": static.get("passed") is True,
    "seed41": {
        **common(seed41),
        "topology_width": seed41.get("topology_candidate_width") == 13,
        "runtime_width": seed41.get("runtime_candidate_width") == 14,
        "equivalence_steps": seed41.get("candidate_width_equivalence_steps") == expected_steps,
        "choice_equal": equivalence.get("candidate_choice_equal") is True,
        "mask_equal": equivalence.get("mask_equal") is True,
        "logprob_equal": equivalence.get("logprob_equal") is True,
        "replay_equal": equivalence.get("replay_equal") is True,
        "type0_action_equal": equivalence.get("type0_action_equal") is True,
        "extra_padding_invalid": equivalence.get("extra_padding_invalid") is True,
    },
    "seed42": {
        **common(seed42),
        "association_counts": seed42.get("association_candidate_counts") == [11, 11, 14],
        "topology_width": seed42.get("topology_candidate_width") == 14,
        "runtime_width": seed42.get("runtime_candidate_width") == 14,
    },
}
passed = bool(
    checks["static_structure"]
    and all(checks["seed41"].values())
    and all(checks["seed42"].values())
)
payload = {
    "status": "compatible" if passed else "incompatible",
    "passed": passed,
    "steps_per_online_gate": expected_steps,
    "checkpoint_file_sha256": static.get("checkpoint_file_sha256"),
    "actor_state_dict_sha256": static.get("actor_state_dict_sha256"),
    "checks": checks,
}
(root / "compatibility_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not passed:
    raise SystemExit(f"candidate width compatibility gate failed: {checks}")
PY
}

run_candidate_width_fp32_equivalence() {
    require_binary
    require_file "${OUT_DIR}/seed41_selection/treatment_comparison.json"
    local winner selected_iteration checkpoint phase_dir
    winner="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_treatment"] or "")' "${OUT_DIR}/seed41_selection/treatment_comparison.json")"
    [[ -n "${winner}" ]] || { echo "seed-41 gate has no winner; W0-v2 is forbidden" >&2; exit 2; }
    selected_iteration="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_iteration"])' "${OUT_DIR}/pilot-${winner}/checkpoint_selection.json")"
    checkpoint="${OUT_DIR}/pilot-${winner}/candidate_checkpoints/ppo_actor_iter_$(printf '%04d' "${selected_iteration}").pt"
    require_file "${checkpoint}"
    phase_dir="${OUT_DIR}/candidate_width_compatibility_w0_v2"
    write_run_manifest "${phase_dir}" "W0-v2: seed41 13-to-14 FP32 numerical equivalence; then seed42 ${STEPS}-TTI structural smoke"

    "${PYTHON_BIN}" - "${checkpoint}" "${phase_dir}/static_structure_gate.json" <<'PY'
import hashlib
import json
import pathlib
import sys
from dataclasses import replace

import torch

from training.stageb_hgraph.independent_actor import IndependentActorConfig, PFQIndependentReuseActor
from training.stageb_hgraph.independent_contract import (
    state_dict_sha256,
    state_dicts_bitwise_equal,
)

checkpoint_path = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2]).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
raw_config = checkpoint["actor_config"]
fields = IndependentActorConfig.__dataclass_fields__
training_config = IndependentActorConfig(
    **{key: raw_config[key] for key in fields if key in raw_config}
)
training_width = int(
    raw_config.get("training_observed_max_candidates", raw_config["max_candidates"])
)
runtime_config = replace(training_config, max_candidates=14)
training_actor = PFQIndependentReuseActor(training_config)
runtime_actor = PFQIndependentReuseActor(runtime_config)
training_shapes = {name: list(value.shape) for name, value in training_actor.state_dict().items()}
runtime_shapes = {name: list(value.shape) for name, value in runtime_actor.state_dict().items()}
width_dependent = sorted(
    name for name in training_shapes if training_shapes[name] != runtime_shapes[name]
)
state = checkpoint["actor_state"]
strict_load = True
try:
    runtime_actor.load_state_dict(state, strict=True)
except Exception:
    strict_load = False
loaded_state = runtime_actor.state_dict()
state_tensor_names_equal = set(state) == set(loaded_state)
state_tensor_values_equal = state_dicts_bitwise_equal(state, loaded_state)
checkpoint_state_hash = state_dict_sha256(state)
loaded_state_hash = state_dict_sha256(loaded_state)
state_dict_hash_match = checkpoint_state_hash == loaded_state_hash
payload = {
    "protocol_id": "W0-v2",
    "passed": bool(
        training_width == 13
        and not width_dependent
        and strict_load
        and state_tensor_names_equal
        and state_tensor_values_equal
        and state_dict_hash_match
    ),
    "checkpoint": str(checkpoint_path),
    "checkpoint_file_sha256_audit_only": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "actor_state_dict_sha256": checkpoint_state_hash,
    "loaded_actor_state_dict_sha256": loaded_state_hash,
    "training_observed_max_candidates": training_width,
    "probed_runtime_candidate_width": 14,
    "state_tensor_count": len(training_shapes),
    "parameter_count": sum(int(value.numel()) for value in state.values()),
    "width_dependent_state_tensors": width_dependent,
    "strict_state_load_at_width_14": strict_load,
    "state_tensor_names_equal": state_tensor_names_equal,
    "state_tensor_values_bitwise_equal": state_tensor_values_equal,
    "state_dict_hash_match_after_strict_load": state_dict_hash_match,
    "state_dict_hash_contract": "sorted tensor name, dtype, shape, and raw bytes",
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not payload["passed"]:
    raise SystemExit(f"W0-v2 state-dict structure gate failed: {payload}")
PY

    evaluate_checkpoint "${checkpoint}" 41 0 "${STEPS}" "${phase_dir}/seed-41-width13-vs14" 0 0 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-equivalence-reference-width 13 \
        --independent-equivalence-protocol fp32-w0-v2 \
        --device cpu
    evaluate_checkpoint "${checkpoint}" 42 0 "${STEPS}" "${phase_dir}/seed-42-width14" 0 0 argmax \
        --independent-runtime-candidate-width 0 \
        --device cpu

    "${PYTHON_BIN}" -m training.stageb_hgraph.independent_gate candidate-width-w0-v2 \
        --static "${phase_dir}/static_structure_gate.json" \
        --seed41 "${phase_dir}/seed-41-width13-vs14/eval_summary.json" \
        --seed42 "${phase_dir}/seed-42-width14/eval_summary.json" \
        --expected-steps "${STEPS}" \
        --out "${phase_dir}/compatibility_report.json"
}

run_validity() {
    require_binary
    local phase_dir="${OUT_DIR}/validity"
    write_run_manifest "${phase_dir}" "${STEPS}-TTI independent action-path validity"
    "${PYTHON_BIN}" -m training.stageb_hgraph.online_imitation \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --socket-path "/tmp/cumac_independent_validity_bc.sock" --online-persistent 1 \
        --episode-horizon "${STEPS}" --max-steps "${STEPS}" --episodes 1 \
        --topology-seed 41 --topology-seed-mode fixed --seed 41 --device auto \
        --out-dir "${phase_dir}/bc" --action-head-mode independent \
        --candidate-mode all_demand_ues --include-pfq-score-anchor 0 \
        --teacher-mode pfq_passthrough --hidden-dim 128 --message-layers 2 \
        --dropout 0 --print-every "${STEPS}"
    local checkpoint="${phase_dir}/bc/hgraph_independent_bc_last.pt"
    # A 32-update BC checkpoint may be blank under argmax.  This pass therefore
    # audits the deterministic action path without pretending it is a KPI run.
    evaluate_checkpoint "${checkpoint}" 41 0 "${STEPS}" "${phase_dir}/greedy" 0
    "${PYTHON_BIN}" - "${phase_dir}/greedy/eval_summary.json" "${phase_dir}/validity_gate.json" "${STEPS}" <<'PY'
import json, math, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected_steps = int(sys.argv[3])
checks = {
    "completed_steps": summary.get("completed_steps") == expected_steps,
    "action_replay_exact": summary.get("stats_sum_action_replay_mismatch_count") == 0,
    "actions_legal": summary.get("stats_sum_illegal_action_count") == 0,
    "pfq_deletion_exact": summary.get("independent_deletion_equal") is True,
    "no_fallback": summary.get("stats_mean_fallback_success_count") == 0,
    "no_safe_fill": summary.get("stats_mean_safe_fill_count") == 0,
    "no_slot_repair": summary.get("stats_mean_slot_repair_count") == 0,
}
report = {"passed": all(checks.values()), "checks": checks}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not report["passed"]: raise SystemExit(1)
PY
    local scales="${phase_dir}/sampled_calibration/reward_scales.json"
    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 41 --traffic-seed 41 --warmup-steps 16 --measure-steps 16 \
        --out-dir "${phase_dir}/sampled_calibration"
    "${PYTHON_BIN}" -m training.stageb_hgraph.online_ppo \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --socket-path "/tmp/cumac_independent_validity_ppo.sock" \
        --online-persistent 1 --episode-horizon 4000 --rollout-steps "${STEPS}" --iterations 1 \
        --rollout-warmup-steps 0 --rollout-warmup-once 1 \
        --topology-seed 41 --topology-seed-mode fixed --seed 41 --device auto \
        --out-dir "${phase_dir}/sampled" --action-head-mode independent \
        --candidate-mode all_demand_ues --include-pfq-score-anchor 0 \
        --include-candidate-risk-features 0 --pfq-residual-init 0 \
        --reward-mode pfq_independent --independent-treatment B \
        --independent-reward-scales "${scales}" --init-checkpoint "${checkpoint}" \
        --teacher-mode none --teacher-aux-coef 0 --teacher-aux-min-coef 0 \
        --imitation-episodes 0 --slot-repair-max-per-cell 0 --safe-fill-max-per-cell 0 \
        --candidate-blank-budget-max-per-cell -1 --hidden-dim 128 --message-layers 2 \
        --ue-feat-dim 16 --dropout 0 --update-epochs 1 --minibatch-size "${STEPS}" \
        --early-stop-patience 0 --write-diagnostics 1 --print-every-iters 1 \
        --sim-env "CUMAC_PACKET_DELAY_SAMPLES=1" \
        --sim-env "CUMAC_PACKET_OUTCOME_SAMPLES_CSV=${phase_dir}/sampled/packet_outcome_samples.csv"
    "${PYTHON_BIN}" - "${phase_dir}/sampled/online_ppo_metrics.csv" "${phase_dir}/sampled_validity_gate.json" <<'PY'
import csv, json, math, sys
row = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))[-1]
def f(key): return float(row[key])
zero_fields = (
    "rollout_action_replay_mismatch_count_mean",
    "rollout_native_reject_wrong_cell_slot_count_mean",
    "rollout_native_reject_empty_slot_count_mean",
    "rollout_native_reject_prg_mask_count_mean",
    "rollout_native_reject_oob_slot_count_mean",
    "rollout_native_reject_duplicate_ue_count_mean",
    "rollout_python_preflight_wrong_cell_slot_count_mean",
    "rollout_python_preflight_empty_slot_count_mean",
    "rollout_python_preflight_prg_mask_count_mean",
    "rollout_python_preflight_oob_slot_count_mean",
    "rollout_python_preflight_duplicate_ue_count_mean",
    "rollout_fallback_success_count_mean",
    "rollout_safe_fill_count_mean",
    "rollout_blank_budget_repair_count_mean",
    "rollout_coverage_repair_count_mean",
)
checks = {
    "zero_replay_reject_repair_fallback": all(f(key) == 0.0 for key in zero_fields),
    "native_diagnostics_complete": f("rollout_native_reject_diagnostics_available_mean") == 1.0,
    "python_preflight_complete": f("rollout_python_preflight_diagnostics_available_mean") == 1.0,
    "expiry_aware_ue_diagnostics_complete": f("rollout_actual_ue_packet_diagnostics_available_mean") == 1.0,
    "sampled_actions_nonempty": f("rollout_action_fill_ratio_mean") > 0.0,
    "finite_reward": math.isfinite(f("rollout_reward_mean")),
    "finite_entropy": math.isfinite(f("rollout_entropy_mean")) and f("rollout_entropy_mean") > 0.0,
    "finite_policy_gradient": math.isfinite(f("candidate_policy_grad_norm")) and f("candidate_policy_grad_norm") > 0.0,
    "finite_ea_raw_reward_terms": all(math.isfinite(f(key)) for key in (
        "rollout_reward_goodput_raw_mean", "rollout_reward_failed_raw_mean", "rollout_reward_expired_raw_mean")),
}
report = {"passed": all(checks.values()), "checks": checks, "evidence": {key: f(key) for key in zero_fields + ("rollout_action_fill_ratio_mean", "rollout_reward_mean", "rollout_entropy_mean", "candidate_policy_grad_norm")}}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not report["passed"]: raise SystemExit(1)
PY
    "${PYTHON_BIN}" - "${checkpoint}" "${phase_dir}/bc/online_imitation_summary.json" "${phase_dir}/sampled_policy_validity.json" <<'PY'
import json, math, sys, torch
from dataclasses import replace
from training.stageb_hgraph.independent_actor import IndependentActorConfig, PFQIndependentReuseActor
from training.stageb_hgraph.independent_policy import replay_independent_trace, resolve_full_candidate_width, select_independent_action
from training.stageb_hgraph.edge_generator import EdgeGeneratorConfig, SparseEdgeGenerator
from training.stageb_hgraph.feature_snapshot import FeatureSnapshotBuilder
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_observation
checkpoint = torch.load(sys.argv[1], map_location="cpu")
cfg = IndependentActorConfig(**{k: v for k, v in checkpoint["actor_config"].items() if k in IndependentActorConfig.__dataclass_fields__})
graph = SparseEdgeGenerator(EdgeGeneratorConfig(candidate_mode="all_demand_ues", include_pfq_score_anchor=False, include_candidate_risk_features=False)).build_graph(FeatureSnapshotBuilder().build(make_candidate_observation(n_prg=3, ue_feat_dim=16)))
cfg = replace(cfg, max_candidates=resolve_full_candidate_width(graph))
actor = PFQIndependentReuseActor(cfg)
actor.load_state_dict(checkpoint["actor_state"], strict=True)
packed = pack_prg_candidates(graph, max_candidates=cfg.max_candidates, fixed_width=True)
trace, action = select_independent_action(actor, graph, packed, mode="sample")
logp, entropy, replayed = replay_independent_trace(actor, graph, packed, trace)
loss = -logp
loss.backward()
grad = sum(float(p.grad.square().sum()) for n, p in actor.named_parameters() if p.grad is not None and (n.startswith("reuse_attention") or n.startswith("ue_scorer"))) ** 0.5
report = {
    "sampled_trace_replay_exact": bool(torch.equal(action.action_ue_select, replayed.action_ue_select) and torch.equal(action.action_prg_alloc, replayed.action_prg_alloc)),
    "finite_logp": math.isfinite(float(logp.detach())),
    "finite_entropy": math.isfinite(float(entropy.detach())),
    "reuse_attention_and_scorer_grad_norm": grad,
    "passed": grad > 0 and math.isfinite(float(logp.detach())) and math.isfinite(float(entropy.detach())) and torch.equal(action.action_ue_select, replayed.action_ue_select) and torch.equal(action.action_prg_alloc, replayed.action_prg_alloc),
}
open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not report["passed"]: raise SystemExit(1)
PY
}

run_select_seed41() {
    require_file "${OUT_DIR}/pfq_seed41/reference_summary.json"
    local selection_args=()
    local treatments=()
    for treatment in A B C D; do
        local pilot_dir="${OUT_DIR}/pilot-${treatment}"
        [[ -d "${pilot_dir}/candidate_checkpoints" ]] || continue
        local treatment_manifest="${pilot_dir}/treatment_manifest.json"
        "${PYTHON_BIN}" - "${OUT_DIR}/pfq_seed41/reference_manifest.json" "${pilot_dir}/run_manifest.json" "${treatment_manifest}" <<'PY'
import json, sys
reference = json.load(open(sys.argv[1], encoding="utf-8"))
run = json.load(open(sys.argv[2], encoding="utf-8"))
if reference["binary_sha256"] != run["binary_sha256"]:
    raise SystemExit("pilot binary hash differs from PFQ reference")
if reference["source_revision"] != run["source_revision"]:
    raise SystemExit("pilot source revision differs from PFQ reference")
open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(reference, indent=2) + "\n")
PY
        local candidates=()
        for iteration in 4 8 12 16 20; do
            local checkpoint="${pilot_dir}/candidate_checkpoints/ppo_actor_iter_$(printf '%04d' "${iteration}").pt"
            require_file "${checkpoint}"
            local eval_dir="${pilot_dir}/candidate_eval/iter-$(printf '%04d' "${iteration}")"
            evaluate_checkpoint "${checkpoint}" 41 1000 4000 "${eval_dir}"
            candidates+=(--candidate "${iteration}=${eval_dir}/gate_metrics.json")
        done
        local selection="${pilot_dir}/checkpoint_selection.json"
        "${PYTHON_BIN}" -m training.stageb_hgraph.independent_gate select-checkpoints \
            --pfq-metrics "${OUT_DIR}/pfq_seed41/reference_summary.json" \
            --pfq-manifest "${OUT_DIR}/pfq_seed41/reference_manifest.json" \
            --treatment-manifest "${treatment_manifest}" "${candidates[@]}" --out "${selection}"
        treatments+=("${treatment}")
        selection_args+=(--selection "${treatment}=${selection}")
    done
    [[ "${#treatments[@]}" -gt 0 ]] || { echo "no pilot outputs found" >&2; exit 2; }
    set +e
    "${PYTHON_BIN}" -m training.stageb_hgraph.independent_gate compare-treatments \
        --pfq-metrics "${OUT_DIR}/pfq_seed41/reference_summary.json" \
        "${selection_args[@]}" --out-dir "${OUT_DIR}/seed41_selection"
    local gate_rc=$?
    set -e
    return "${gate_rc}"
}

run_blank_diagnostics_seed41() {
    require_binary
    local a_selection="${OUT_DIR}/pilot-A/checkpoint_selection.json"
    local b_selection="${OUT_DIR}/pilot-B/checkpoint_selection.json"
    require_file "${a_selection}"
    require_file "${b_selection}"

    local b_iteration
    b_iteration="$("${PYTHON_BIN}" -m training.stageb_hgraph.independent_blank_diagnostics select-b \
        --selection "${b_selection}")"
    [[ "${b_iteration}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid B diagnostic iteration: ${b_iteration}" >&2; exit 2; }
    local b_iteration_padded
    printf -v b_iteration_padded '%04d' "${b_iteration}"

    local a4_checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local a8_checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
    local b_checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_${b_iteration_padded}.pt"
    local a4_formal_summary="${OUT_DIR}/pilot-A/candidate_eval/iter-0004/eval_summary.json"
    local a8_formal_summary="${OUT_DIR}/pilot-A/candidate_eval/iter-0008/eval_summary.json"
    local b_formal_summary="${OUT_DIR}/pilot-B/candidate_eval/iter-${b_iteration_padded}/eval_summary.json"
    require_file "${a4_checkpoint}"
    require_file "${a8_checkpoint}"
    require_file "${b_checkpoint}"
    require_file "${a4_formal_summary}"
    require_file "${a8_formal_summary}"
    require_file "${b_formal_summary}"

    local phase_dir="${OUT_DIR}/blank_diagnostics_seed41"
    local manifest="${phase_dir}/run_manifest.json"
    "${PYTHON_BIN}" - "${manifest}" "${RESUME}" "$(source_revision)" "${SCENARIO_ID}" \
        "${SIM_BIN}" "${a_selection}" "${b_selection}" "${b_iteration}" \
        "${a4_checkpoint}" "${a4_formal_summary}" \
        "${a8_checkpoint}" "${a8_formal_summary}" \
        "${b_checkpoint}" "${b_formal_summary}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    manifest_text, resume_text, source_revision, scenario_id, sim_text,
    a_selection_text, b_selection_text, b_iteration_text,
    a4_checkpoint_text, a4_summary_text,
    a8_checkpoint_text, a8_summary_text,
    b_checkpoint_text, b_summary_text,
) = sys.argv[1:]
manifest = pathlib.Path(manifest_text)
simulator = pathlib.Path(sim_text)
b_iteration = int(b_iteration_text)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

simulator_hash = sha256(simulator)
subjects = {}
for label, iteration, checkpoint_text, summary_text in (
    ("A4", 4, a4_checkpoint_text, a4_summary_text),
    ("A8", 8, a8_checkpoint_text, a8_summary_text),
    ("B", b_iteration, b_checkpoint_text, b_summary_text),
):
    checkpoint = pathlib.Path(checkpoint_text)
    formal_summary_path = pathlib.Path(summary_text)
    formal_summary = json.loads(formal_summary_path.read_text(encoding="utf-8"))
    if formal_summary.get("sim_binary_sha256") != simulator_hash:
        raise SystemExit(
            f"{label} formal simulator hash does not match current binary: "
            f"{formal_summary.get('sim_binary_sha256')!r} != {simulator_hash!r}"
        )
    checkpoint_hash = sha256(checkpoint)
    if formal_summary.get("checkpoint_sha256") != checkpoint_hash:
        raise SystemExit(
            f"{label} formal checkpoint hash does not match current checkpoint: "
            f"{formal_summary.get('checkpoint_sha256')!r} != {checkpoint_hash!r}"
        )
    subjects[label] = {
        "iteration": iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "formal_eval_summary": str(formal_summary_path),
    }

payload = {
    "phase": "blank-diagnostics-seed41",
    "source_revision": source_revision,
    "scenario_id": scenario_id,
    "simulator": str(simulator),
    "sim_binary_sha256": simulator_hash,
    "seed": 41,
    "warmup_ttis": 1000,
    "measure_ttis": 512,
    "subjects": subjects,
    "formal_selection_paths": {
        "A": a_selection_text,
        "B": b_selection_text,
    },
}
if manifest.parent.is_dir() and any(manifest.parent.iterdir()) and not manifest.is_file():
    raise SystemExit(f"existing output directory has no compatible run manifest: {manifest.parent}")
if manifest.is_file():
    if resume_text != "1":
        raise SystemExit(f"output exists; pass --resume only for an identical manifest: {manifest.parent}")
    old = json.loads(manifest.read_text(encoding="utf-8"))
    if old != payload:
        raise SystemExit("resume manifest differs from current diagnostic inputs")
else:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    evaluate_checkpoint "${a4_checkpoint}" 41 1000 512 "${phase_dir}/A4" 0 1
    evaluate_checkpoint "${a8_checkpoint}" 41 1000 512 "${phase_dir}/A8" 0 1
    evaluate_checkpoint "${b_checkpoint}" 41 1000 512 "${phase_dir}/B" 0 1
    "${PYTHON_BIN}" -m training.stageb_hgraph.independent_blank_diagnostics report \
        --a4 "${phase_dir}/A4/eval_summary.json" \
        --a8 "${phase_dir}/A8/eval_summary.json" \
        --b "${phase_dir}/B/eval_summary.json" \
        --b-iteration "${b_iteration}" \
        --b-selection "${b_selection}" \
        --json-out "${phase_dir}/blank_diagnostic_report.json" \
        --markdown-out "${phase_dir}/blank_diagnostic_report.md"
}

run_hierarchical_decode_smoke() {
    require_binary
    local checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
    require_file "${checkpoint}"
    local phase_dir="${OUT_DIR}/hierarchical_decode_smoke"
    write_run_manifest "${phase_dir}" "A8 hierarchical decode smoke: seed 41, 2 warmup + 4 measured TTIs"
    evaluate_checkpoint "${checkpoint}" 41 2 4 "${phase_dir}" 0 1 hierarchical_argmax
    "${PYTHON_BIN}" - "${phase_dir}/eval_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "measured_ttis": summary.get("stats_count") == 4,
    "diagnostic_decisions": summary.get("diagnostic_decision_count") == 204.0,
    "zero_replay": summary.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": summary.get("stats_sum_illegal_action_count") == 0.0,
    "zero_selected_fragmentation_blank": summary.get("diagnostic_selected_fragmentation_blank_count") == 0.0,
}
try:
    scheduled = float(summary["stats_sum_scheduled_bytes"])
    goodput = float(summary["stats_sum_ue_goodput_bytes"])
    failed = float(summary["stats_sum_failed_bytes"])
    rescues = float(summary["diagnostic_fragmentation_rescue_count"])
    counterfactual = float(summary["diagnostic_fragmentation_blank_count"])
    checks["byte_identity"] = scheduled == goodput + failed
    checks["all_counterfactual_fragmentation_rescued"] = rescues == counterfactual
except (KeyError, TypeError, ValueError):
    checks["byte_identity"] = False
    checks["all_counterfactual_fragmentation_rescued"] = False
if not all(checks.values()):
    raise SystemExit(f"hierarchical decode smoke gate failed: {checks}")
PY
}

run_hierarchical_decode_seed41() {
    require_binary
    local checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
    local control_summary="${OUT_DIR}/blank_diagnostics_seed41/A8/eval_summary.json"
    local control_manifest="${OUT_DIR}/blank_diagnostics_seed41/run_manifest.json"
    require_file "${checkpoint}"
    require_file "${control_summary}"
    require_file "${control_manifest}"

    local phase_dir="${OUT_DIR}/hierarchical_decode_seed41"
    local manifest="${phase_dir}/run_manifest.json"
    "${PYTHON_BIN}" - "${manifest}" "${RESUME}" "${SCENARIO_ID}" "${checkpoint}" "${SIM_BIN}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_text, resume_text, scenario_id, checkpoint_text, simulator_text = sys.argv[1:]
manifest = pathlib.Path(manifest_text)
checkpoint = pathlib.Path(checkpoint_text)
simulator = pathlib.Path(simulator_text)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "phase": "hierarchical-decode-seed41",
    "scenario_id": scenario_id,
    "seed": 41,
    "warmup_ttis": 1000,
    "measure_ttis": 512,
    "a8_iteration": 8,
    "checkpoint_sha256": sha256(checkpoint),
    "sim_binary_sha256": sha256(simulator),
}
if manifest.parent.is_dir() and any(manifest.parent.iterdir()) and not manifest.is_file():
    raise SystemExit(f"existing output directory has no compatible run manifest: {manifest.parent}")
if manifest.is_file():
    if resume_text != "1":
        raise SystemExit(f"output exists; pass --resume only for an identical manifest: {manifest.parent}")
    if json.loads(manifest.read_text(encoding="utf-8")) != payload:
        raise SystemExit("resume manifest differs from current hierarchical decode inputs")
else:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    evaluate_checkpoint "${checkpoint}" 41 1000 512 "${phase_dir}" 0 1 hierarchical_argmax
    "${PYTHON_BIN}" -m training.stageb_hgraph.hierarchical_decode_gate \
        --control "${control_summary}" \
        --treatment "${phase_dir}/eval_summary.json" \
        --control-manifest "${control_manifest}" \
        --treatment-manifest "${manifest}" \
        --json-out "${phase_dir}/hierarchical_decode_report.json" \
        --markdown-out "${phase_dir}/hierarchical_decode_report.md"
}

write_rescue_cohort_manifest() {
    local phase_dir="$1" checkpoint="$2" warmup="$3" measure="$4"
    local manifest="${phase_dir}/run_manifest.json"
    "${PYTHON_BIN}" - "${manifest}" "${RESUME}" "${PHASE}" "$(source_revision)" \
        "${SCENARIO_ID}" "${warmup}" "${measure}" \
        "$(sha256_file "${checkpoint}")" "$(sha256_file "${SIM_BIN}")" <<'PY'
import json
import pathlib
import sys

(
    manifest_text, resume_text, phase, source_revision, scenario_id,
    warmup_text, measure_text, checkpoint_hash, simulator_hash,
) = sys.argv[1:]
manifest = pathlib.Path(manifest_text)
payload = {
    "phase": phase,
    "scenario_id": scenario_id,
    "seed": 41,
    "warmup_ttis": int(warmup_text),
    "measure_ttis": int(measure_text),
    "a8_iteration": 8,
    "checkpoint_sha256": checkpoint_hash,
    "sim_binary_sha256": simulator_hash,
    "source_revision": source_revision,
}
if manifest.parent.is_dir() and any(manifest.parent.iterdir()) and not manifest.is_file():
    raise SystemExit(f"existing output directory has no compatible run manifest: {manifest.parent}")
if manifest.is_file():
    if resume_text != "1":
        raise SystemExit(f"output exists; pass --resume only for an identical manifest: {manifest.parent}")
    if json.loads(manifest.read_text(encoding="utf-8")) != payload:
        raise SystemExit("resume manifest differs from current rescue cohort inputs")
else:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_rescue_cohort_smoke() {
    require_binary
    local checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
    require_file "${checkpoint}"
    local phase_dir="${OUT_DIR}/rescue_cohort_smoke"
    write_rescue_cohort_manifest "${phase_dir}" "${checkpoint}" 2 4
    evaluate_checkpoint "${checkpoint}" 41 2 4 "${phase_dir}" 0 1 hierarchical_argmax
    "${PYTHON_BIN}" - "${phase_dir}/eval_summary.json" <<'PY'
import json
import math
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "measured_ttis": summary.get("stats_count") == 4,
    "diagnostic_decisions": summary.get("diagnostic_decision_count") == 204.0,
    "zero_replay": summary.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": summary.get("stats_sum_illegal_action_count") == 0.0,
    "diagnostics_enabled": summary.get("rescue_cohort_diagnostics_enabled") == 1,
}
try:
    scheduled = float(summary["stats_sum_scheduled_bytes"])
    goodput = float(summary["stats_sum_ue_goodput_bytes"])
    failed = float(summary["stats_sum_failed_bytes"])
    diagnostics = summary["rescue_cohort_diagnostics"]
    checks["byte_identity"] = scheduled == goodput + failed
    checks["schema_version"] = diagnostics.get("schema_version") == 1
    cohorts = diagnostics["cohorts"]
    checks["cohort_names"] = set(cohorts) == {"rescue_exposed", "ordinary_service"}
    raw_keys = (
        "ue_tti_count", "tti_count", "planned_prg_count", "rescued_prg_count",
        "accepted_prg_count", "actual_prg_count", "predicted_service_bytes",
        "scheduled_bytes", "goodput_bytes", "failed_bytes", "tb_tx_count",
        "tb_err_count", "early_rescue_count", "middle_rescue_count",
        "late_rescue_count",
    )
    checks["raw_denominators"] = all(
        isinstance(row, dict)
        and all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
            and float(row[key]) >= 0.0
            for key in raw_keys
        )
        for row in cohorts.values()
    )
    rescued = sum(float(row["rescued_prg_count"]) for row in cohorts.values())
    checks["rescued_prgs"] = rescued == float(summary["diagnostic_fragmentation_rescue_count"])
except (AttributeError, KeyError, TypeError, ValueError):
    checks["cohort_schema"] = False
if not all(checks.values()):
    raise SystemExit(f"rescue cohort smoke gate failed: {checks}")
PY
}

run_rescue_cohort_seed41() {
    require_binary
    local checkpoint="${OUT_DIR}/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
    local frozen_reference="${OUT_DIR}/hierarchical_decode_seed41/eval_summary.json"
    require_file "${checkpoint}"
    require_file "${frozen_reference}"
    local phase_dir="${OUT_DIR}/rescue_cohort_seed41"
    write_rescue_cohort_manifest "${phase_dir}" "${checkpoint}" 1000 512
    evaluate_checkpoint "${checkpoint}" 41 1000 512 "${phase_dir}" 0 1 hierarchical_argmax
    "${PYTHON_BIN}" -m training.stageb_hgraph.rescue_cohort_gate \
        --summary "${phase_dir}/eval_summary.json" \
        --manifest "${phase_dir}/run_manifest.json" \
        --frozen-reference "${OUT_DIR}/hierarchical_decode_seed41/eval_summary.json" \
        --json-out "${phase_dir}/rescue_cohort_report.json" \
        --markdown-out "${phase_dir}/rescue_cohort_report.md"
}

run_formal_seed42_pfq_b4() {
    require_binary
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local selection="${OUT_DIR}/seed41_selection/treatment_comparison.json"
    local checkpoint_selection="${OUT_DIR}/pilot-B/checkpoint_selection.json"
    local w0_report="${OUT_DIR}/candidate_width_compatibility_w0_v2/compatibility_report.json"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local phase_dir="${OUT_DIR}/formal_seed42_pfq_b4_v2"
    require_file "${selection}"
    require_file "${checkpoint_selection}"
    require_file "${w0_report}"
    require_file "${checkpoint}"
    if [[ -e "${phase_dir}" ]]; then
        echo "formal seed42 output already exists; use a new versioned directory: ${phase_dir}" >&2
        exit 2
    fi
    mkdir -p "${phase_dir}"
    unset CUMAC_SHADOW_SEED

    "${PYTHON_BIN}" - "${ROOT_DIR}" "${selection}" "${checkpoint_selection}" \
        "${w0_report}" "${checkpoint}" "${SIM_BIN}" "${phase_dir}/run_manifest.json" \
        "$(source_revision)" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

(
    root_arg,
    selection_arg,
    checkpoint_selection_arg,
    w0_arg,
    checkpoint_arg,
    sim_arg,
    output_arg,
    revision,
) = sys.argv[1:]
root = pathlib.Path(root_arg).resolve()
selection = json.loads(pathlib.Path(selection_arg).read_text(encoding="utf-8"))
checkpoint_selection = json.loads(
    pathlib.Path(checkpoint_selection_arg).read_text(encoding="utf-8")
)
w0 = json.loads(pathlib.Path(w0_arg).read_text(encoding="utf-8"))
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
sim_path = pathlib.Path(sim_arg).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
checkpoint_state_hash = state_dict_sha256(checkpoint["actor_state"])
preflight = {
    "selected_treatment_is_B": selection.get("selected_treatment") == "B",
    "selected_iteration_is_4": checkpoint_selection.get("selected_iteration") == 4,
    "w0_v2_passed": w0.get("passed") is True,
    "w0_v2_seed41_passed": w0.get("seed41_numerical_compatibility", {}).get("status")
    == "PASS",
    "w0_v2_seed42_passed": w0.get("seed42_structural_smoke", {}).get("status")
    == "PASS",
    "checkpoint_state_dict_frozen": checkpoint_state_hash
    == "4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571",
}
if not all(preflight.values()):
    raise SystemExit(f"formal seed42 preflight failed: {preflight}")

source_files = sorted((root / "training/stageb_hgraph").glob("*.py"))
source_files.append(root / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh")
source_files = sorted(source_files)
source_digest = hashlib.sha256()
source_entries = []
for path in source_files:
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    encoded = relative.encode("utf-8")
    file_hash = hashlib.sha256(raw).hexdigest()
    source_digest.update(struct.pack("<I", len(encoded)))
    source_digest.update(encoded)
    source_digest.update(struct.pack("<Q", len(raw)))
    source_digest.update(raw)
    source_entries.append({"path": relative, "sha256": file_hash})

payload = {
    "protocol_id": "seed42-pfq-b4-formal-v1",
    "scope": "seed42-only PFQ-B4 paired evaluation",
    "source_revision": revision,
    "source_state_sha256": source_digest.hexdigest(),
    "source_state_hash_contract": (
        "sorted runtime source path, byte length, and raw file bytes"
    ),
    "source_files": source_entries,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint_path),
    "checkpoint_file_sha256_audit_only": hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest(),
    "checkpoint_state_dict_sha256": checkpoint_state_hash,
    "checkpoint_state_dict_hash_contract": (
        "sorted tensor name, dtype, shape, and raw bytes"
    ),
    "preflight": preflight,
    "frozen": {
        "topology_seed": 42,
        "traffic_seed": 42,
        "shadow_policy": "legacy_shared_rng",
        "association_candidate_counts": [11, 11, 14],
        "runtime_candidate_width": 14,
        "warmup_steps": 1000,
        "measure_steps": 4000,
        "candidate_mode": "all_demand_ues",
        "candidate_width_policy": "runtime_topology",
        "candidate_truncation": "forbidden",
        "decode_mode": "argmax",
    },
    "forbidden_followups": ["seed43", "training", "treatment-C", "treatment-D"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"
    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 42 --traffic-seed 42 --warmup-steps 1000 --measure-steps 4000 \
        --formal --out-dir "${phase_dir}/pfq"
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"

    evaluate_checkpoint "${checkpoint}" 42 1000 4000 "${phase_dir}/actor" 1 0 argmax \
        --independent-runtime-candidate-width 14
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.independent_gate import formal_seed42_pair_gate

root = pathlib.Path(sys.argv[1]).resolve()
frozen_state_hash = sys.argv[2]
run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
pfq = json.loads((root / "pfq/reference_summary.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/gate_metrics.json").read_text(encoding="utf-8"))
actor_summary = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
native_pfq_manifest = json.loads(
    (root / "pfq/reference_manifest.json").read_text(encoding="utf-8")
)

scenario = dict(native_pfq_manifest["scenario"])
scenario["shadow_policy"] = "legacy_shared_rng"
common = {
    **native_pfq_manifest,
    "scenario": scenario,
    "source_state_sha256": run_manifest["source_state_sha256"],
    "checkpoint_state_dict_sha256": run_manifest[
        "checkpoint_state_dict_sha256"
    ],
    "warmup_steps": 1000,
    "measure_steps": 4000,
}
pfq_manifest = {
    **common,
    "arm": "PFQ",
    "command_role": "fresh formal seed42 comparator",
}
actor_manifest = {
    **common,
    "arm": "B4",
    "command_role": "frozen B4 seed42 evaluation at runtime width 14",
}
(root / "pfq/provenance_manifest.json").write_text(
    json.dumps(pfq_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "actor/provenance_manifest.json").write_text(
    json.dumps(actor_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

report = formal_seed42_pair_gate(
    pfq,
    actor,
    actor_summary,
    pfq_manifest,
    actor_manifest,
    frozen_state_dict_sha256=frozen_state_hash,
)
(root / "formal_pair_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
performance = report["performance_gate"].get("gate") or {"checks": {}}
checks = performance.get("checks", {})
lines = [
    "# Seed42 formal PFQ-B4 paired evaluation",
    "",
    report["status"],
    "",
    f"- Structural gate: {report['structural_gate']['status']}",
    f"- Performance gate: {report['performance_gate']['status']}",
]
for name in ("goodput", "tb_bler", "expiry_rate"):
    if name in checks:
        lines.append(f"- {name}: {'PASS' if checks[name]['passed'] else 'FAIL'}; {checks[name]['reason']}")
lines.extend(
    [
        "",
        "The p95 delay, UE p10 rate, Blank ratio, and PRG opportunity fields are record-only diagnostics.",
        "This result does not authorize seed43, training, or treatments C/D.",
        "",
    ]
)
(root / "formal_pair_report.md").write_text("\n".join(lines), encoding="utf-8")
print(report["classification"])
PY

    local classification
    classification="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${phase_dir}/formal_pair_report.json")"
    case "${classification}" in
        seed42_formal_gate_pass) return 0 ;;
        runtime_compatibility_path_failure)
            echo "seed42 formal structural gate failed; seed43 is forbidden" >&2
            return 20
            ;;
        seed42_performance_gate_failure)
            echo "seed42 formal performance gate failed; seed43 is forbidden and minimal diagnosis is required" >&2
            return 21
            ;;
        *) echo "unknown seed42 formal classification: ${classification}" >&2; return 22 ;;
    esac
}

verify_formal_seed42_frozen_state() {
    local manifest="$1" checkpoint="$2"
    "${PYTHON_BIN}" - "${ROOT_DIR}" "${manifest}" "${checkpoint}" "${SIM_BIN}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

root = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
checkpoint_path = pathlib.Path(sys.argv[3]).resolve()
sim_path = pathlib.Path(sys.argv[4]).resolve()
digest = hashlib.sha256()
for entry in manifest["source_files"]:
    path = root / entry["path"]
    raw = path.read_bytes()
    encoded = entry["path"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise SystemExit(f"formal source file changed during paired run: {entry['path']}")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)
checks = {
    "source_state": digest.hexdigest() == manifest["source_state_sha256"],
    "simulator_binary": hashlib.sha256(sim_path.read_bytes()).hexdigest()
    == manifest["binary_sha256"],
    "checkpoint_state": state_dict_sha256(
        torch.load(checkpoint_path, map_location="cpu")["actor_state"]
    )
    == manifest["checkpoint_state_dict_sha256"],
}
if not all(checks.values()):
    raise SystemExit(f"formal frozen-state verification failed: {checks}")
PY
}

run_diagnose_seed42_per_prg() {
    require_binary
    local formal_dir="${OUT_DIR}/formal_seed42_pfq_b4_v2"
    local formal_report="${formal_dir}/formal_pair_report.json"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local phase_dir="${OUT_DIR}/seed42_per_prg_diagnostic_v1"
    require_file "${formal_report}"
    require_file "${checkpoint}"
    if [[ -e "${phase_dir}" ]]; then
        echo "seed42 per-PRG diagnostic output already exists: ${phase_dir}" >&2
        exit 2
    fi
    mkdir -p "${phase_dir}"
    unset CUMAC_SHADOW_SEED

    "${PYTHON_BIN}" - "${formal_report}" "${checkpoint}" "${SIM_BIN}" \
        "${phase_dir}/run_manifest.json" "$(source_revision)" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

formal_path, checkpoint_arg, sim_arg, output_arg, revision = sys.argv[1:]
formal = json.loads(pathlib.Path(formal_path).read_text(encoding="utf-8"))
if formal.get("classification") != "seed42_performance_gate_failure":
    raise SystemExit("per-PRG diagnosis requires a formal seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("per-PRG diagnosis is forbidden after a structural failure")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
state_hash = state_dict_sha256(torch.load(checkpoint_path, map_location="cpu")["actor_state"])
if state_hash != formal.get("frozen_state_dict_sha256"):
    raise SystemExit("per-PRG diagnosis checkpoint differs from formal B4")
sim_path = pathlib.Path(sim_arg).resolve()
payload = {
    "protocol_id": "seed42-per-prg-phy-interference-diagnostic-v1",
    "purpose": "minimal mechanism diagnosis after formal seed42 performance failure",
    "source_revision": revision,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint_state_dict_sha256": state_hash,
    "topology_seed": 42,
    "traffic_seed": 42,
    "shadow_policy": "legacy_shared_rng",
    "runtime_candidate_width": 14,
    "warmup_steps": 1000,
    "measured_steps": 128,
    "training": False,
    "treatments_c_or_d": False,
    "seed43": False,
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 42 --traffic-seed 42 --warmup-steps 1000 --measure-steps 128 \
        --out-dir "${phase_dir}/pfq"
    evaluate_checkpoint "${checkpoint}" 42 1000 128 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14

    "${PYTHON_BIN}" - "${phase_dir}" "${formal_report}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
formal = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
pfq = json.loads((root / "pfq/reference_summary.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
local = actor.get("local_prg_phy_interference_diagnostics")
if not isinstance(local, dict) or local.get("tti_count") != 128:
    raise SystemExit("actor local-PRG diagnostic is missing or incomplete")
structural_checks = {
    "completed_steps": actor.get("completed_steps") == 1128,
    "measured_steps": actor.get("stats_count") == 128,
    "measured_interval": (actor.get("measured_tti_first"), actor.get("measured_tti_last"))
    == (1001, 1128),
    "association_counts": actor.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": actor.get("runtime_candidate_width") == 14,
    "full_coverage": actor.get("candidate_full_coverage") is True,
    "zero_truncation": actor.get("stats_sum_candidate_truncation_count") == 0.0,
    "zero_replay": actor.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": actor.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_deletion": actor.get("independent_deletion_equal") is True,
    "state_dict_frozen": actor.get("actor_state_dict_sha256")
    == formal.get("frozen_state_dict_sha256"),
}
if not all(structural_checks.values()):
    raise SystemExit(f"seed42 diagnostic structural check failed: {structural_checks}")

rows = []
for actor_row in local["local_prgs"]:
    prg = int(actor_row["local_prg"])
    pfq_opportunity = float(pfq[f"local_prg_{prg:02d}_opportunity_count_mean"])
    actor_opportunity = float(actor_row["reuse_count_mean"])
    rows.append(
        {
            **actor_row,
            "pfq_opportunity_count_mean": pfq_opportunity,
            "actor_opportunity_count_mean": actor_opportunity,
            "actor_minus_pfq_opportunity_count_mean": actor_opportunity
            - pfq_opportunity,
        }
    )
formal_performance = formal["performance_gate"]["gate"]["checks"]
payload = {
    "protocol_id": "seed42-per-prg-phy-interference-diagnostic-v1",
    "status": "COMPLETE",
    "structural_checks": structural_checks,
    "formal_failure": {
        "goodput": formal_performance["goodput"],
        "tb_bler": formal_performance["tb_bler"],
        "expiry_rate": formal_performance["expiry_rate"],
    },
    "window": {
        "warmup_ttis": 1000,
        "measured_ttis": 128,
        "seed": 42,
    },
    "blank_classification": {
        key.removeprefix("diagnostic_"): value
        for key, value in actor.items()
        if key.startswith("diagnostic_")
        and any(label in key for label in ("blank_ratio", "fragmentation", "conservative", "forced_blank", "p_blank", "p_service", "p_best_ue"))
    },
    "local_prgs": rows,
    "aggregate": {
        "pfq_opportunity_count_mean": float(pfq["prg_opportunity_count_mean"]),
        "actor_opportunity_count_mean": float(actor["stats_mean_fill_ratio"]) * 51.0,
        "actor_minus_pfq_opportunity_count_mean": sum(
            row["actor_minus_pfq_opportunity_count_mean"] for row in rows
        ),
        "actor_prg_reuse_ratio_mean": float(actor["stats_mean_prg_reuse_ratio"]),
        "pfq_prg_reuse_ratio_mean": float(pfq["prg_reuse_ratio_mean"]),
        "rescue_cohort_diagnostics": actor.get("rescue_cohort_diagnostics"),
    },
    "interpretation": (
        "Formal BLER passed while goodput and expiry failed; this trace tests where "
        "the actor removes local-PRG opportunities and what selected PHY/risk proxies it sees."
    ),
}
(root / "diagnostic_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

run_audit_seed42_service_accounting() {
    require_binary
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local formal_report="${OUT_DIR}/formal_seed42_pfq_b4_v2/formal_pair_report.json"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local phase_dir="${OUT_DIR}/seed42_service_accounting_audit_v1"
    require_file "${formal_report}"
    require_file "${checkpoint}"
    if [[ -e "${phase_dir}" ]]; then
        echo "seed42 service-accounting audit output already exists: ${phase_dir}" >&2
        exit 2
    fi
    mkdir -p "${phase_dir}"
    unset CUMAC_SHADOW_SEED

    "${PYTHON_BIN}" - "${formal_report}" "${checkpoint}" "${SIM_BIN}" \
        "${phase_dir}/run_manifest.json" "$(source_revision)" "${frozen_state_hash}" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

formal_arg, checkpoint_arg, sim_arg, output_arg, revision, frozen_hash = sys.argv[1:]
formal = json.loads(pathlib.Path(formal_arg).read_text(encoding="utf-8"))
if formal.get("classification") != "seed42_performance_gate_failure":
    raise SystemExit("service-accounting audit requires the original B4 seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("service-accounting audit is forbidden after a structural failure")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
state_hash = state_dict_sha256(torch.load(checkpoint_path, map_location="cpu")["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("service-accounting audit checkpoint differs from frozen B4")
sim_path = pathlib.Path(sim_arg).resolve()
payload = {
    "protocol_id": "seed42-service-accounting-read-only-audit-v1",
    "purpose": "test whether uncapped scheduled/raw service bias supports one global 2/3 calibration",
    "source_revision": revision,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint_state_dict_sha256": state_hash,
    "topology_seed": 42,
    "traffic_seed": 42,
    "shadow_policy": "legacy_shared_rng",
    "runtime_candidate_width": 14,
    "warmup_steps": 1000,
    "measured_steps": 512,
    "scheduled_bytes_semantics": "new_transmission_queue_dequeue",
    "eligibility": {
        "planned_accepted_actual_local_prg_id_sets_equal": True,
        "initial_backlog_at_least_raw_prediction_sum": True,
        "each_predecision_remaining_at_least_raw_prediction": True,
        "harq_enabled": 0,
    },
    "frozen_gate": {
        "target_ratio": 2.0 / 3.0,
        "global_absolute_tolerance": 0.05,
        "minimum_group_ue_ttis": 50,
        "minimum_group_actual_prgs": 200,
        "maximum_group_absolute_delta_from_global": 0.10,
        "quality_bins": "decision_pre_post_eq_quality_equal_sample_count_tertiles",
        "prg_segment_service_ratio_gate": False,
    },
    "training": False,
    "service_accounting_scale": 1.0,
    "forbidden": ["B4-SC implementation before audit PASS", "seed41", "seed43", "C", "D", "training"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    evaluate_checkpoint "${checkpoint}" 42 1000 512 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-service-accounting-audit 1

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
audit = actor.get("service_accounting_audit")
if not isinstance(audit, dict):
    raise SystemExit("service-accounting audit result is missing")
structural_checks = {
    "completed_steps": actor.get("completed_steps") == 1512,
    "measured_steps": actor.get("stats_count") == 512,
    "measured_interval": (actor.get("measured_tti_first"), actor.get("measured_tti_last"))
    == (1001, 1512),
    "association_counts": actor.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": actor.get("runtime_candidate_width") == 14,
    "full_coverage": actor.get("candidate_full_coverage") is True,
    "zero_truncation": actor.get("stats_sum_candidate_truncation_count") == 0.0,
    "zero_replay": actor.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": actor.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_deletion": actor.get("independent_deletion_equal") is True,
    "state_dict_frozen": actor.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": actor.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "harq_disabled": actor.get("runtime_harq_enabled") == 0,
}
local = actor.get("local_prg_phy_interference_diagnostics", {}).get("local_prgs", [])
segments = []
for name, first, last in (("early", 0, 5), ("middle", 6, 10), ("late", 11, 16)):
    rows = [row for row in local if first <= int(row["local_prg"]) <= last]
    decisions = sum(int(row["decision_count"]) for row in rows)
    selected = sum(int(row["selected_count"]) for row in rows)
    segments.append({
        "name": name,
        "local_prg_first": first,
        "local_prg_last": last,
        "opportunity_count_mean": sum(float(row["reuse_count_mean"]) for row in rows),
        "blank_ratio": None if decisions == 0 else 1.0 - selected / decisions,
        "forced_blank_count": sum(int(row["forced_blank_count"]) for row in rows),
        "conservative_blank_count": sum(int(row["conservative_blank_count"]) for row in rows),
        "service_ratio_gate": False,
    })
structural_passed = all(structural_checks.values())
audit_passed = audit.get("status") == "PASS"
classification = (
    "global_constant_calibration_admissible"
    if structural_passed and audit_passed
    else "service_accounting_audit_structural_failure"
    if not structural_passed
    else "global_constant_calibration_rejected"
)
payload = {
    "protocol_id": manifest["protocol_id"],
    "status": "PASS" if structural_passed and audit_passed else "FAIL",
    "classification": classification,
    "structural_checks": structural_checks,
    "service_accounting_gate": audit,
    "prg_segments_record_only": segments,
    "interpretation": (
        "PASS permits exactly one B4-SC2/3 treatment; FAIL stops before changing the policy."
    ),
}
(root / "audit_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(classification)
PY

    local classification
    classification="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${phase_dir}/audit_report.json")"
    case "${classification}" in
        global_constant_calibration_admissible) return 0 ;;
        service_accounting_audit_structural_failure) return 30 ;;
        global_constant_calibration_rejected) return 31 ;;
        *) echo "unknown service-accounting audit classification: ${classification}" >&2; return 32 ;;
    esac
}

run_audit_seed42_phy_tbs_alignment() {
    require_binary
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local formal_report="${OUT_DIR}/formal_seed42_pfq_b4_v2/formal_pair_report.json"
    local service_audit_report="${OUT_DIR}/seed42_service_accounting_audit_v1/audit_report.json"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local phase_dir="${OUT_DIR}/seed42_phy_tbs_alignment_audit_v1"
    require_file "${formal_report}"
    require_file "${service_audit_report}"
    require_file "${checkpoint}"
    if [[ -e "${phase_dir}" ]]; then
        echo "seed42 PHY/TBS alignment audit output already exists: ${phase_dir}" >&2
        exit 2
    fi
    mkdir -p "${phase_dir}"
    unset CUMAC_SHADOW_SEED

    "${PYTHON_BIN}" - "${formal_report}" "${service_audit_report}" "${checkpoint}" \
        "${SIM_BIN}" "${phase_dir}/run_manifest.json" "$(source_revision)" \
        "${frozen_state_hash}" "${ROOT_DIR}" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

(
    formal_arg,
    service_audit_arg,
    checkpoint_arg,
    sim_arg,
    output_arg,
    revision,
    frozen_hash,
    root_arg,
) = sys.argv[1:]
formal = json.loads(pathlib.Path(formal_arg).read_text(encoding="utf-8"))
service_audit = json.loads(pathlib.Path(service_audit_arg).read_text(encoding="utf-8"))
if formal.get("classification") != "seed42_performance_gate_failure":
    raise SystemExit("PHY/TBS audit requires the original B4 seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("PHY/TBS audit is forbidden after a structural failure")
if service_audit.get("classification") != "global_constant_calibration_rejected":
    raise SystemExit("PHY/TBS audit requires the rejected global service calibration audit")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
state_hash = state_dict_sha256(torch.load(checkpoint_path, map_location="cpu")["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("PHY/TBS audit checkpoint differs from frozen B4")
sim_path = pathlib.Path(sim_arg).resolve()
root = pathlib.Path(root_arg).resolve()
source_files = [
    "cuMAC/examples/multiCellSchedulerUeSelection/main.cpp",
    "cuMAC/examples/network.cu",
    "cuMAC/src/4T4R/mcsSelectionLUT.cu",
    "cuMAC/src/4T4R/multiCellLayerSel.cu",
    "training/stageb_hgraph/phy_tbs_alignment_audit.py",
]
payload = {
    "protocol_id": "seed42-phy-tbs-alignment-read-only-audit-v1",
    "purpose": "test exact decision-time reconstruction and sequential additivity of simulator TBS",
    "source_revision": revision,
    "source_file_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in source_files
    },
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint_state_dict_sha256": state_hash,
    "topology_seed": 42,
    "traffic_seed": 42,
    "shadow_policy": "legacy_shared_rng",
    "runtime_candidate_width": 14,
    "association_counts": [11, 11, 14],
    "warmup_steps": 1000,
    "measured_steps": 512,
    "scheduled_bytes_semantics": "new_transmission_queue_dequeue",
    "eligibility": {
        "planned_accepted_actual_local_prg_id_sets_equal": True,
        "pre_schedule_buffer_at_least_simulator_tbs_capacity": True,
        "harq_enabled": 0,
    },
    "causal_isolation": {
        "decision_pre_inputs": [
            "wideband_sinr_first_layer",
            "olla_delta_up_down",
            "prior_tti_tb_error",
            "per_prg_singular_value_layer_capability",
        ],
        "reference_labels": ["actual_mcs", "actual_layer", "simulator_pre_dequeue_tbs_capacity"],
        "reference_label_deletion_equality": "bitwise",
        "cell_id_is_prediction_input": False,
    },
    "prefix_definition": "first_k_actual_prgs_in_actor_local_prg_order_0_to_16",
    "frozen_gates": {
        "record_level_exact_mcs_layer_tbs": True,
        "negative_marginal_count": 0,
        "nan_or_inf_count": 0,
        "zero_marginal_allowed_and_reported": True,
        "global_scheduled_to_tbs_ratio_interval": [0.9, 1.1],
        "quality_prg_group_absolute_delta_from_one_max": 0.15,
        "cell_residual_absolute_delta_from_pooled_stratum_max": 0.10,
        "raw_wape_improvement_minimum_fraction": 0.25,
    },
    "training": False,
    "actor_modified": False,
    "forbidden": ["B4-TBS implementation", "policy smoke", "formal pair", "seed41", "seed43", "C", "D", "training"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    export CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1
    evaluate_checkpoint "${checkpoint}" 42 1000 512 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-phy-tbs-alignment-audit 1
    unset CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
audit = actor.get("phy_tbs_alignment_audit")
if not isinstance(audit, dict):
    raise SystemExit("PHY/TBS alignment audit result is missing")
config_hash = audit.get("phy_tbs_config_sha256")
ineligible = audit.get("ineligible_reason_counts", {})
structural_checks = {
    "completed_steps": actor.get("completed_steps") == 1512,
    "measured_steps": actor.get("stats_count") == 512,
    "measured_interval": (actor.get("measured_tti_first"), actor.get("measured_tti_last"))
    == (1001, 1512),
    "protocol_version": actor.get("protocol_version") == 14,
    "association_counts": actor.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": actor.get("runtime_candidate_width") == 14,
    "full_coverage": actor.get("candidate_full_coverage") is True,
    "zero_truncation": actor.get("stats_sum_candidate_truncation_count") == 0.0,
    "zero_replay": actor.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": actor.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_deletion": actor.get("independent_deletion_equal") is True,
    "state_dict_frozen": actor.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": actor.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "harq_disabled": actor.get("runtime_harq_enabled") == 0,
    "audit_enabled": actor.get("independent_phy_tbs_alignment_audit_enabled") == 1,
    "eligible_rows_present": int(audit.get("eligible_ue_tti_count", 0)) > 0,
    "no_prg_set_mismatch": not any("prg_set_mismatch" in key for key in ineligible),
    "config_hash_recorded": isinstance(config_hash, str) and len(config_hash) == 64,
    "raw_rows_recorded": (root / "actor/phy_tbs_alignment_rows.jsonl").is_file(),
}
structural_passed = all(structural_checks.values())
classification = (
    str(audit.get("classification"))
    if structural_passed
    else "phy_tbs_alignment_audit_structural_failure"
)
payload = {
    "protocol_id": manifest["protocol_id"],
    "status": "PASS" if structural_passed and audit.get("status") == "PASS" else "FAIL",
    "classification": classification,
    "structural_checks": structural_checks,
    "checkpoint_state_dict_sha256": actor.get("actor_state_dict_sha256"),
    "binary_sha256": actor.get("sim_binary_sha256"),
    "phy_tbs_config_sha256": config_hash,
    "phy_tbs_alignment_gate": audit,
    "interpretation": (
        "PASS permits a separate B4-TBS design round only; every result stops this runner without changing policy."
    ),
}
(root / "audit_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(classification)
PY

    local classification
    classification="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${phase_dir}/audit_report.json")"
    case "${classification}" in
        phy_tbs_alignment_compatible) return 0 ;;
        phy_tbs_alignment_audit_structural_failure) return 30 ;;
        decision_time_phy_path_not_exact) return 31 ;;
        final_tbs_reconstructable_but_not_sequentially_additive) return 32 ;;
        conditional_alignment_gate_failure) return 33 ;;
        *) echo "unknown PHY/TBS audit classification: ${classification}" >&2; return 34 ;;
    esac
}

run_audit_seed42_phy_tbs_alignment_v2_cpu_olla() {
    require_binary
    [[ "${STEPS}" == "32" ]] || {
        echo "v2 CPU-OLLA engineering smoke is frozen at --steps 32" >&2
        exit 2
    }
    local protocol_id="seed42-phy-tbs-alignment-read-only-audit-v2-cpu-olla"
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local formal_report="${OUT_DIR}/formal_seed42_pfq_b4_v2/formal_pair_report.json"
    local service_audit_report="${OUT_DIR}/seed42_service_accounting_audit_v1/audit_report.json"
    local v1_report="${OUT_DIR}/seed42_phy_tbs_alignment_audit_v1/audit_report.json"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local smoke_dir="${OUT_DIR}/seed42_phy_tbs_alignment_audit_v2_cpu_olla_smoke_r2"
    local phase_dir="${OUT_DIR}/seed42_phy_tbs_alignment_audit_v2_cpu_olla"
    require_file "${formal_report}"
    require_file "${service_audit_report}"
    require_file "${v1_report}"
    require_file "${checkpoint}"
    for target in "${smoke_dir}" "${phase_dir}"; do
        if [[ -e "${target}" ]]; then
            echo "v2 CPU-OLLA audit output already exists: ${target}" >&2
            exit 2
        fi
    done
    unset CUMAC_SHADOW_SEED

    prepare_phy_tbs_v2_manifest() {
        local target_dir="$1" run_kind="$2" warmup="$3" measured="$4"
        mkdir -p "${target_dir}"
        "${PYTHON_BIN}" - "${formal_report}" "${service_audit_report}" "${v1_report}" \
            "${checkpoint}" "${SIM_BIN}" "${target_dir}/run_manifest.json" \
            "$(source_revision)" "${frozen_state_hash}" "${ROOT_DIR}" "${protocol_id}" \
            "${run_kind}" "${warmup}" "${measured}" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

(
    formal_arg,
    service_arg,
    v1_arg,
    checkpoint_arg,
    sim_arg,
    output_arg,
    revision,
    frozen_hash,
    root_arg,
    protocol_id,
    run_kind,
    warmup,
    measured,
) = sys.argv[1:]
formal = json.loads(pathlib.Path(formal_arg).read_text(encoding="utf-8"))
service = json.loads(pathlib.Path(service_arg).read_text(encoding="utf-8"))
v1 = json.loads(pathlib.Path(v1_arg).read_text(encoding="utf-8"))
if formal.get("classification") != "seed42_performance_gate_failure":
    raise SystemExit("v2 audit requires the original frozen-B4 seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("v2 audit is forbidden after a seed42 structural failure")
if service.get("classification") != "global_constant_calibration_rejected":
    raise SystemExit("v2 audit requires the rejected global service calibration audit")
if (
    v1.get("protocol_id") != "seed42-phy-tbs-alignment-read-only-audit-v1"
    or v1.get("status") != "FAIL"
    or v1.get("classification") != "decision_time_phy_path_not_exact"
):
    raise SystemExit("v1 must remain an immutable decision-time-path failure")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
state_hash = state_dict_sha256(torch.load(checkpoint_path, map_location="cpu")["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("v2 audit checkpoint differs from frozen B4")
sim_path = pathlib.Path(sim_arg).resolve()
root = pathlib.Path(root_arg).resolve()
source_files = [
    "cuMAC/examples/multiCellSchedulerUeSelection/main.cpp",
    "cuMAC/examples/network.cu",
    "cuMAC/examples/onlineTrainBridge/OnlineProtocol.h",
    "cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp",
    "cuMAC/src/4T4R/mcsSelectionLUTCpu.h",
    "cuMAC/src/4T4R/mcsSelectionLUTCpu.cpp",
    "cuMAC/src/4T4R/multiCellLayerSelCpu.cpp",
    "training/stageb_hgraph/online_protocol.py",
    "training/stageb_hgraph/phy_tbs_alignment_audit.py",
]
payload = {
    "protocol_id": protocol_id,
    "run_kind": run_kind,
    "research_gate": run_kind == "formal_audit",
    "purpose": "align offline reconstruction to the causal CPU mcsSelSol OLLA path",
    "source_revision": revision,
    "source_file_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in source_files
    },
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint_state_dict_sha256": state_hash,
    "v1_report_sha256": hashlib.sha256(pathlib.Path(v1_arg).read_bytes()).hexdigest(),
    "v1_frozen_result": "FAIL: decision_time_phy_path_not_exact",
    "topology_seed": 42,
    "traffic_seed": 42,
    "shadow_policy": "legacy_shared_rng",
    "runtime_candidate_width": 14,
    "association_counts": [11, 11, 14],
    "warmup_steps": int(warmup),
    "measured_steps": int(measured),
    "scheduled_bytes_semantics": "new_transmission_queue_dequeue",
    "eligibility": {
        "planned_accepted_actual_local_prg_id_sets_equal": True,
        "pre_schedule_buffer_at_least_simulator_tbs_capacity": True,
        "harq_enabled": 0,
    },
    "causal_isolation": {
        "olla_source": "mcsSelectionLUTCpu persistent ollaParamArr",
        "snapshot_phase": "before actor recvStepReq and before current CPU MCS run",
        "current_tti_result_available": False,
        "olla_storage": "float32",
        "olla_initialization": {"delta_ini": 0.0, "delta": 0.0, "delta_up": 1.0, "delta_down": "blerTargetActUe"},
        "olla_update": "prior tbErrLast: +delta_up on 1, -delta_down on 0, unchanged on -1",
        "olla_clamp": "none",
        "sinr_quantization": "10*log10 in double, cast float32, float32 cap and OLLA subtraction",
        "mcs_rounding": "descending float32 LUT threshold comparison",
        "tbs_rounding": "cumac determineTbsPdsch integer-byte semantics",
        "reference_labels": ["actual_mcs", "actual_layer", "simulator_pre_dequeue_tbs_capacity"],
        "reference_label_deletion_equality": "bitwise",
        "cell_id_is_prediction_input": False,
    },
    "prefix_definition": "first_k_actual_prgs_in_actor_local_prg_order_0_to_16",
    "frozen_gate_order": [
        "CPU source, causal capture phase, and reference-label deletion equality",
        "record-level exact MCS, final layer, and final TBS parity",
        "only after parity: nonnegative actor-order prefix TBS marginals",
    ],
    "statistical_alignment_role": "record_only_not_a_v2_formal_gate",
    "training": False,
    "actor_modified": False,
    "forbidden": ["PFQ pair", "B4-TBS", "seed41", "seed43", "C", "D", "training"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    }

    prepare_phy_tbs_v2_manifest "${smoke_dir}" "engineering_smoke" 0 32
    export CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1
    evaluate_checkpoint "${checkpoint}" 42 0 32 "${smoke_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-phy-tbs-alignment-audit 1 \
        --independent-phy-tbs-alignment-protocol "${protocol_id}"
    unset CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT

    "${PYTHON_BIN}" - "${smoke_dir}" "${frozen_state_hash}" "${protocol_id}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
protocol_id = sys.argv[3]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
audit = actor.get("phy_tbs_alignment_audit")
provenance = audit.get("causal_provenance", {}) if isinstance(audit, dict) else {}
checks = {
    "engineering_only_not_research_gate": manifest.get("research_gate") is False,
    "completed_steps": actor.get("completed_steps") == 32,
    "measured_steps": actor.get("stats_count") == 32,
    "protocol_version": actor.get("protocol_version") == 15,
    "audit_protocol": actor.get("independent_phy_tbs_alignment_protocol") == protocol_id,
    "audit_report_protocol": isinstance(audit, dict) and audit.get("protocol_id") == protocol_id,
    "cpu_olla_source": provenance.get("cpu_olla_source_exact") is True,
    "pre_action_pre_mcs_capture": provenance.get("pre_action_pre_mcs_capture_exact") is True,
    "reference_label_deletion": isinstance(audit, dict) and audit.get("reference_label_deletion_equal") is True,
    "association_counts": actor.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": actor.get("runtime_candidate_width") == 14,
    "full_coverage": actor.get("candidate_full_coverage") is True,
    "zero_truncation": actor.get("stats_sum_candidate_truncation_count") == 0.0,
    "zero_replay": actor.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": actor.get("stats_sum_illegal_action_count") == 0.0,
    "state_dict_frozen": actor.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": actor.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "raw_rows_recorded": (root / "actor/phy_tbs_alignment_rows.jsonl").is_file(),
}
payload = {
    "protocol_id": protocol_id,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "classification": "engineering_smoke_pass" if all(checks.values()) else "engineering_smoke_failure",
    "research_gate": False,
    "checks": checks,
}
(root / "engineering_smoke_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not all(checks.values()):
    raise SystemExit("v2 CPU-OLLA engineering smoke failed")
print("engineering_smoke_pass")
PY

    prepare_phy_tbs_v2_manifest "${phase_dir}" "formal_audit" 1000 512
    export CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1
    evaluate_checkpoint "${checkpoint}" 42 1000 512 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-phy-tbs-alignment-audit 1 \
        --independent-phy-tbs-alignment-protocol "${protocol_id}"
    unset CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" "${protocol_id}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
protocol_id = sys.argv[3]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
audit = actor.get("phy_tbs_alignment_audit")
if not isinstance(audit, dict):
    raise SystemExit("v2 PHY/TBS alignment audit result is missing")
provenance = audit.get("causal_provenance", {})
ineligible = audit.get("ineligible_reason_counts", {})
config_hash = audit.get("phy_tbs_config_sha256")
checks = {
    "formal_audit": manifest.get("research_gate") is True,
    "completed_steps": actor.get("completed_steps") == 1512,
    "measured_steps": actor.get("stats_count") == 512,
    "measured_interval": (actor.get("measured_tti_first"), actor.get("measured_tti_last")) == (1001, 1512),
    "protocol_version": actor.get("protocol_version") == 15,
    "audit_protocol": actor.get("independent_phy_tbs_alignment_protocol") == protocol_id,
    "audit_report_protocol": audit.get("protocol_id") == protocol_id,
    "cpu_olla_source": provenance.get("cpu_olla_source_exact") is True,
    "pre_action_pre_mcs_capture": provenance.get("pre_action_pre_mcs_capture_exact") is True,
    "reference_label_deletion": audit.get("reference_label_deletion_equal") is True,
    "association_counts": actor.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": actor.get("runtime_candidate_width") == 14,
    "full_coverage": actor.get("candidate_full_coverage") is True,
    "zero_truncation": actor.get("stats_sum_candidate_truncation_count") == 0.0,
    "zero_replay": actor.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "zero_illegal": actor.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_deletion": actor.get("independent_deletion_equal") is True,
    "state_dict_frozen": actor.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": actor.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "harq_disabled": actor.get("runtime_harq_enabled") == 0,
    "audit_enabled": actor.get("independent_phy_tbs_alignment_audit_enabled") == 1,
    "eligible_rows_present": int(audit.get("eligible_ue_tti_count", 0)) > 0,
    "no_prg_set_mismatch": not any("prg_set_mismatch" in key for key in ineligible),
    "config_hash_recorded": isinstance(config_hash, str) and len(config_hash) == 64,
    "raw_rows_recorded": (root / "actor/phy_tbs_alignment_rows.jsonl").is_file(),
    "v2_statistics_not_formal": audit.get("statistical_alignment", {}).get("formal_gate") is False,
}
structural_passed = all(checks.values())
classification = audit.get("classification") if structural_passed else "phy_tbs_alignment_audit_structural_failure"
allowed = {
    "decision_time_phy_path_not_exact",
    "final_tbs_reconstructable_but_not_sequentially_additive",
    "phy_tbs_alignment_compatible",
    "phy_tbs_alignment_audit_structural_failure",
}
if classification not in allowed:
    raise SystemExit(f"unexpected v2 audit classification: {classification}")
payload = {
    "protocol_id": protocol_id,
    "status": "PASS" if structural_passed and audit.get("status") == "PASS" else "FAIL",
    "classification": classification,
    "structural_checks": checks,
    "checkpoint_state_dict_sha256": actor.get("actor_state_dict_sha256"),
    "binary_sha256": actor.get("sim_binary_sha256"),
    "phy_tbs_config_sha256": config_hash,
    "gate_order": ["causal provenance and deletion", "record exact parity", "prefix monotonicity after parity"],
    "phy_tbs_alignment_gate": audit,
    "interpretation": "Every result stops here; PASS only permits a separately approved B4-TBS design round.",
}
(root / "audit_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(classification)
PY

    local classification
    classification="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${phase_dir}/audit_report.json")"
    case "${classification}" in
        phy_tbs_alignment_compatible) return 0 ;;
        phy_tbs_alignment_audit_structural_failure) return 30 ;;
        decision_time_phy_path_not_exact) return 31 ;;
        final_tbs_reconstructable_but_not_sequentially_additive) return 32 ;;
        *) echo "unknown v2 PHY/TBS audit classification: ${classification}" >&2; return 34 ;;
    esac
}

run_b4_ctbs_seed42() {
    require_binary
    local protocol_id="seed42-pfq-b4-ctbs-v1-formal-v1"
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local original_formal="${OUT_DIR}/formal_seed42_pfq_b4_v2/formal_pair_report.json"
    local service_audit="${OUT_DIR}/seed42_service_accounting_audit_v1/audit_report.json"
    local alignment_v2="${OUT_DIR}/seed42_phy_tbs_alignment_audit_v2_cpu_olla/audit_report.json"
    local smoke_dir="${OUT_DIR}/seed42_b4_ctbs_v1_smoke_r3"
    local phase_dir="${OUT_DIR}/formal_seed42_pfq_b4_ctbs_v1"
    require_file "${checkpoint}"
    require_file "${original_formal}"
    require_file "${service_audit}"
    require_file "${alignment_v2}"
    for target in "${smoke_dir}" "${phase_dir}"; do
        if [[ -e "${target}" ]]; then
            echo "B4-CTBS-v1 output already exists; use a new versioned protocol: ${target}" >&2
            exit 2
        fi
    done
    unset CUMAC_SHADOW_SEED

    prepare_ctbs_manifest() {
        local target_dir="$1" run_kind="$2" warmup="$3" measured="$4"
        mkdir -p "${target_dir}"
        "${PYTHON_BIN}" - "${ROOT_DIR}" "${original_formal}" "${service_audit}" \
            "${alignment_v2}" "${checkpoint}" "${SIM_BIN}" \
            "${target_dir}/run_manifest.json" "$(source_revision)" \
            "${frozen_state_hash}" "${protocol_id}" "${run_kind}" \
            "${warmup}" "${measured}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

(
    root_arg,
    original_arg,
    service_arg,
    alignment_arg,
    checkpoint_arg,
    sim_arg,
    output_arg,
    revision,
    frozen_hash,
    protocol_id,
    run_kind,
    warmup,
    measured,
) = sys.argv[1:]
root = pathlib.Path(root_arg).resolve()
original = json.loads(pathlib.Path(original_arg).read_text(encoding="utf-8"))
service = json.loads(pathlib.Path(service_arg).read_text(encoding="utf-8"))
alignment = json.loads(pathlib.Path(alignment_arg).read_text(encoding="utf-8"))
alignment_gate = alignment.get("phy_tbs_alignment_gate", {})
parity = alignment_gate.get("record_level_parity", {})
causal = alignment_gate.get("causal_provenance", {})
sequential = alignment_gate.get("sequential_usability", {})
preflight = {
    "original_b4_seed42_performance_failed": original.get("classification")
    == "seed42_performance_gate_failure",
    "original_b4_structural_gate_passed": original.get("structural_gate", {}).get("status")
    == "PASS",
    "global_constant_calibration_rejected": service.get("classification")
    == "global_constant_calibration_rejected",
    "v2_scope_is_simulator_internal_pre_action": alignment.get("protocol_id")
    == "seed42-phy-tbs-alignment-read-only-audit-v2-cpu-olla",
    "v2_final_tbs_exact": parity.get("all_exact") is True
    and parity.get("mcs_mismatch_count") == 0
    and parity.get("layer_mismatch_count") == 0
    and parity.get("tbs_mismatch_count") == 0,
    "v2_reference_label_deletion": alignment_gate.get("reference_label_deletion_equal")
    is True,
    "v2_causal_cpu_olla": causal.get("cpu_olla_source_exact") is True
    and causal.get("pre_action_pre_mcs_capture_exact") is True,
    "v2_negative_marginal_hypothesis_rejected": alignment.get("classification")
    == "final_tbs_reconstructable_but_not_sequentially_additive"
    and int(sequential.get("negative_marginal_count", 0)) > 0,
}

checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_hash = state_dict_sha256(checkpoint["actor_state"])
preflight["checkpoint_state_dict_frozen"] = state_hash == frozen_hash
if not all(preflight.values()):
    raise SystemExit(f"B4-CTBS-v1 preflight failed: {preflight}")

source_files = sorted((root / "training/stageb_hgraph").glob("*.py"))
source_files.append(root / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh")
source_files = sorted(source_files)
source_digest = hashlib.sha256()
source_entries = []
for path in source_files:
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    encoded = relative.encode("utf-8")
    source_digest.update(struct.pack("<I", len(encoded)))
    source_digest.update(encoded)
    source_digest.update(struct.pack("<Q", len(raw)))
    source_digest.update(raw)
    source_entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})

sim_path = pathlib.Path(sim_arg).resolve()
payload = {
    "protocol_id": protocol_id if run_kind == "formal_pair" else "seed42-b4-ctbs-v1-causal-smoke-v3",
    "run_kind": run_kind,
    "hypothesis": "exact cumulative TBS headroom restores opportunity and the frozen seed42 gate",
    "source_revision": revision,
    "source_state_sha256": source_digest.hexdigest(),
    "source_state_hash_contract": "sorted runtime source path, byte length, and raw file bytes",
    "source_files": source_entries,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint_path),
    "checkpoint_state_dict_sha256": state_hash,
    "checkpoint_state_dict_hash_contract": "sorted tensor name, dtype, shape, and raw bytes",
    "preflight": preflight,
    "treatment": "B4-CTBS-v1",
    "frozen": {
        "topology_seed": 42,
        "traffic_seed": 42,
        "shadow_policy": "legacy_shared_rng",
        "association_candidate_counts": [11, 11, 14],
        "runtime_candidate_width": 14,
        "warmup_steps": int(warmup),
        "measure_steps": int(measured),
        "decode_mode": "argmax",
        "prg_order": list(range(17)),
        "headroom": "exact cumulative TBS; capacity decreases and remaining-demand rebounds allowed",
        "negative_marginal_clipping": False,
        "running_max": False,
        "safety_factor": None,
    },
    "handshake": {
        "sequence": "PreActionRsp(t,snapshot_id,causal_PHY_tuple) -> actor action -> StepReq(t,snapshot_id,action) -> simulator validation -> StepRsp(actual audit labels)",
        "pre_action_fields": [
            "cpu_olla", "wideband_sinr", "ue_prg_layer_capability",
            "ue_cell_identity", "tti_snapshot_identity", "tbs_config_hash"
        ],
        "forbidden_pre_action_fields": [
            "actual_mcs", "selected_layer", "tbs", "scheduled_bytes", "successful_bytes"
        ],
        "scorer_receives_new_fields": False,
        "q0_source": "original_B4_observation",
    },
    "training": False,
    "forbidden_followups": ["other_coefficients", "seed41", "seed43", "C", "D", "training", "lower_bound"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    }

    prepare_ctbs_manifest "${smoke_dir}" "causal_smoke" 0 128
    verify_formal_seed42_frozen_state "${smoke_dir}/run_manifest.json" "${checkpoint}"
    evaluate_checkpoint "${checkpoint}" 42 0 128 "${smoke_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1
    verify_formal_seed42_frozen_state "${smoke_dir}/run_manifest.json" "${checkpoint}"
    "${PYTHON_BIN}" - "${smoke_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.independent_gate import ctbs_seed42_smoke_gate

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
report = ctbs_seed42_smoke_gate(
    actor,
    frozen_state_dict_sha256=frozen_hash,
    frozen_binary_sha256=manifest["binary_sha256"],
)
(root / "smoke_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(report["classification"])
if not report["passed"]:
    raise SystemExit("B4-CTBS-v1 seed42 smoke failed; formal pair is forbidden")
PY

    prepare_ctbs_manifest "${phase_dir}" "formal_pair" 1000 4000
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"
    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 42 --traffic-seed 42 --warmup-steps 1000 --measure-steps 4000 \
        --formal --out-dir "${phase_dir}/pfq"
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"
    evaluate_checkpoint "${checkpoint}" 42 1000 4000 "${phase_dir}/actor" 1 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.independent_gate import ctbs_seed42_formal_pair_gate

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
pfq = json.loads((root / "pfq/reference_summary.json").read_text(encoding="utf-8"))
actor = json.loads((root / "actor/gate_metrics.json").read_text(encoding="utf-8"))
actor_summary = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
native = json.loads((root / "pfq/reference_manifest.json").read_text(encoding="utf-8"))
scenario = dict(native["scenario"])
scenario["shadow_policy"] = "legacy_shared_rng"
common = {
    **native,
    "scenario": scenario,
    "source_state_sha256": run_manifest["source_state_sha256"],
    "checkpoint_state_dict_sha256": frozen_hash,
    "warmup_steps": 1000,
    "measure_steps": 4000,
}
pfq_manifest = {**common, "arm": "PFQ", "command_role": "fresh seed42 comparator"}
actor_manifest = {
    **common,
    "arm": "B4-CTBS-v1",
    "command_role": "frozen B4 weights with exact cumulative TBS headroom",
}
(root / "pfq/provenance_manifest.json").write_text(
    json.dumps(pfq_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "actor/provenance_manifest.json").write_text(
    json.dumps(actor_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
report = ctbs_seed42_formal_pair_gate(
    pfq,
    actor,
    actor_summary,
    pfq_manifest,
    actor_manifest,
    frozen_state_dict_sha256=frozen_hash,
)
(root / "formal_pair_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
performance = report["performance_gate"].get("gate") or {"checks": {}}
lines = [
    "# Seed42 formal PFQ–B4-CTBS-v1 paired evaluation",
    "",
    report["status"],
    "",
    f"- Structural/causal gate: {report['structural_gate']['status']}",
    f"- Performance gate: {report['performance_gate']['status']}",
]
for name, value in performance.get("checks", {}).items():
    lines.append(f"- {name}: {'PASS' if value['passed'] else 'FAIL'}; {value['reason']}")
lines.extend([
    "",
    "Tail KPIs and CTBS opportunity vectors are record-only diagnostics.",
    "This result does not authorize seed41/43, C/D, training, a lower bound, or another coefficient.",
    "",
])
(root / "formal_pair_report.md").write_text("\n".join(lines), encoding="utf-8")
print(report["classification"])
PY

    local classification
    classification="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${phase_dir}/formal_pair_report.json")"
    case "${classification}" in
        seed42_ctbs_formal_gate_pass) return 0 ;;
        ctbs_runtime_or_causal_path_failure)
            echo "B4-CTBS-v1 structural/causal gate failed; all follow-ups are forbidden" >&2
            return 40
            ;;
        seed42_ctbs_performance_gate_failure)
            echo "B4-CTBS-v1 performance gate failed; stop without further experiments" >&2
            return 41
            ;;
        *) echo "unknown B4-CTBS-v1 classification: ${classification}" >&2; return 42 ;;
    esac
}

run_audit_seed42_ctbs_blank_expiry_attribution() {
    require_binary
    [[ "${STEPS}" == "512" ]] || {
        echo "Blank-expiry attribution audit is frozen at --steps 512" >&2
        exit 2
    }
    local protocol_id="seed42-ctbs-blank-expiry-attribution-read-only-audit-v1"
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local ctbs_formal="${OUT_DIR}/formal_seed42_pfq_b4_ctbs_v1/formal_pair_report.json"
    local phase_dir="${OUT_DIR}/seed42_ctbs_blank_expiry_attribution_read_only_audit_v1"
    require_file "${checkpoint}"
    require_file "${ctbs_formal}"
    if [[ -e "${phase_dir}" ]]; then
        echo "Blank-expiry audit output already exists; do not reuse or overwrite: ${phase_dir}" >&2
        exit 2
    fi
    unset CUMAC_SHADOW_SEED
    mkdir -p "${phase_dir}"

    "${PYTHON_BIN}" - "${ROOT_DIR}" "${ctbs_formal}" "${checkpoint}" \
        "${SIM_BIN}" "${phase_dir}/run_manifest.json" "$(source_revision)" \
        "${frozen_state_hash}" "${protocol_id}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

root_arg, ctbs_arg, checkpoint_arg, sim_arg, output_arg, revision, frozen_hash, protocol_id = sys.argv[1:]
root = pathlib.Path(root_arg).resolve()
ctbs = json.loads(pathlib.Path(ctbs_arg).read_text(encoding="utf-8"))
if ctbs.get("classification") != "seed42_ctbs_performance_gate_failure":
    raise SystemExit("attribution audit requires the recorded B4-CTBS seed42 performance failure")
if ctbs.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("attribution audit is forbidden after a B4-CTBS structural failure")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_hash = state_dict_sha256(checkpoint["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("attribution audit checkpoint differs from frozen B4")
source_paths = [
    *sorted((root / "training/stageb_hgraph").glob("*.py")),
    root / "cuMAC/examples/trafficModel/trafficGenerator.hpp",
    root / "cuMAC/examples/multiCellSchedulerUeSelection/main.cpp",
    root / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh",
]
source_digest = hashlib.sha256()
source_entries = []
for path in sorted(source_paths):
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    encoded = relative.encode("utf-8")
    source_digest.update(struct.pack("<I", len(encoded)))
    source_digest.update(encoded)
    source_digest.update(struct.pack("<Q", len(raw)))
    source_digest.update(raw)
    source_entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
sim_path = pathlib.Path(sim_arg).resolve()
payload = {
    "protocol_id": protocol_id,
    "research_mode": "bounded_read_only_attribution",
    "hypothesis": "one exposure category jointly dominates expiry-weighted atoms and modal UE-TTI views",
    "source_revision": revision,
    "source_state_sha256": source_digest.hexdigest(),
    "source_files": source_entries,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint_path),
    "checkpoint_state_dict_sha256": state_hash,
    "treatment": "frozen B4-CTBS-v1",
    "frozen": {
        "topology_seed": 42,
        "traffic_seed": 42,
        "association_candidate_counts": [11, 11, 14],
        "runtime_candidate_width": 14,
        "warmup_steps": 1000,
        "measured_steps": 512,
        "measured_tti": [1001, 1512],
        "trace_first_tti": 602,
        "decode_mode": "argmax",
        "headroom": "ctbs-v1",
        "training": False,
    },
    "formal_atom": "expired_packet_x_tti_x_serving_cell_x_local_prg",
    "formal_weight": "remaining_bytes_at_expiry",
    "dominance": {"minimum_share": 0.50, "minimum_lead_percentage_points": 10.0},
    "robustness_view": "UE-TTI mode over 17 local PRGs; ties are mixed/inconclusive",
    "classification_rule": "same category must dominate both weighted views",
    "interpretation": "exposure attribution only; cell-PRG rows are not causal effects",
    "forbidden": ["PFQ pair", "seed41", "seed43", "C", "D", "training", "model changes"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    evaluate_checkpoint "${checkpoint}" 42 1000 512 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1 \
        --independent-blank-expiry-attribution-audit 1 \
        --blank-expiry-trace-first-tti 602

    "${PYTHON_BIN}" - "${phase_dir}" "${frozen_state_hash}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
frozen_hash = sys.argv[2]
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
summary = json.loads((root / "actor/eval_summary.json").read_text(encoding="utf-8"))
audit = json.loads((root / "actor/blank_expiry_attribution_audit.json").read_text(encoding="utf-8"))
checks = {
    "protocol": audit.get("protocol_id") == manifest.get("protocol_id"),
    "audit_passed": audit.get("audit_status") == "PASS",
    "completed_steps": summary.get("completed_steps") == 1512,
    "measured_steps": summary.get("stats_count") == 512,
    "measured_interval": (summary.get("measured_tti_first"), summary.get("measured_tti_last")) == (1001, 1512),
    "trace_interval": summary.get("blank_expiry_trace_first_tti") == 602 and summary.get("blank_expiry_trace_count") == 911,
    "association_counts": summary.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": summary.get("runtime_candidate_width") == 14,
    "state_dict_frozen": summary.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": summary.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "lifecycle_complete": audit.get("coverage", {}).get("complete") is True,
    "unclassified_zero": audit.get("coverage", {}).get("unclassified_count") == 0,
    "raw_decisions_recorded": (root / "actor/blank_expiry_decision_rows.jsonl").is_file(),
    "packet_lifecycles_recorded": (root / "actor/blank_expiry_packet_lifecycle_rows.jsonl").is_file(),
    "atomic_exposures_recorded": (root / "actor/blank_expiry_atomic_exposures.jsonl").is_file(),
}
payload = {
    **audit,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "runner_checks": checks,
    "run_manifest_sha256": hashlib.sha256((root / "run_manifest.json").read_bytes()).hexdigest(),
}
(root / "audit_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(payload.get("formal_conclusion"))
if payload["status"] != "PASS":
    raise SystemExit(f"Blank-expiry attribution audit failed: {checks}")
PY
}

run_audit_seed42_ctbs_blank_expiry_attribution_v2_full_horizon() {
    require_binary
    [[ "${STEPS}" == "4000" ]] || {
        echo "Full-horizon Blank-expiry attribution audit is frozen at --steps 4000" >&2
        exit 2
    }
    local protocol_id="seed42-ctbs-blank-expiry-attribution-read-only-audit-v2-full-horizon"
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local formal_dir="${OUT_DIR}/formal_seed42_pfq_b4_ctbs_v1/actor"
    local formal_report="${OUT_DIR}/formal_seed42_pfq_b4_ctbs_v1/formal_pair_report.json"
    local v1_dir="${OUT_DIR}/seed42_ctbs_blank_expiry_attribution_read_only_audit_v1/actor"
    local phase_dir="${OUT_DIR}/seed42_ctbs_blank_expiry_attribution_read_only_audit_v2_full_horizon"
    require_file "${checkpoint}"
    require_file "${formal_report}"
    require_file "${formal_dir}/eval_summary.json"
    require_file "${formal_dir}/packet_outcome_samples.csv"
    require_file "${formal_dir}/output_short.txt"
    require_file "${v1_dir}/eval_summary.json"
    require_file "${v1_dir}/packet_outcome_samples.csv"
    require_file "${v1_dir}/output_short.txt"
    require_file "${v1_dir}/blank_expiry_decision_rows.jsonl"
    if [[ -e "${phase_dir}" ]]; then
        echo "Full-horizon Blank-expiry audit output already exists; do not reuse or overwrite: ${phase_dir}" >&2
        exit 2
    fi
    unset CUMAC_SHADOW_SEED
    mkdir -p "${phase_dir}"

    "${PYTHON_BIN}" - "${formal_dir}" "${v1_dir}" "${phase_dir}/prefix_consistency_report.json" \
        "${frozen_state_hash}" <<'PY'
import hashlib
import json
import pathlib
import sys

from training.stageb_hgraph.blank_expiry_attribution_audit import (
    canonical_compact_cpu_fingerprints,
    canonical_packet_outcome_fingerprint,
    canonical_type0_action_fingerprints,
)

formal = pathlib.Path(sys.argv[1]).resolve()
v1 = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3]).resolve()
frozen_hash = sys.argv[4]
formal_summary = json.loads((formal / "eval_summary.json").read_text(encoding="utf-8"))
v1_summary = json.loads((v1 / "eval_summary.json").read_text(encoding="utf-8"))

formal_packets = canonical_packet_outcome_fingerprint(
    formal / "packet_outcome_samples.csv", tti_first=1, tti_last=1512
)
v1_packets = canonical_packet_outcome_fingerprint(
    v1 / "packet_outcome_samples.csv", tti_first=1, tti_last=1512
)
formal_cpu = canonical_compact_cpu_fingerprints(
    formal / "output_short.txt", tti_first=1001, tti_last=1512,
    completed_steps=5000,
)
v1_cpu = canonical_compact_cpu_fingerprints(
    v1 / "output_short.txt", tti_first=1001, tti_last=1512,
    completed_steps=1512,
)

def decision_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)

v1_actions = canonical_type0_action_fingerprints(
    decision_rows(v1 / "blank_expiry_decision_rows.jsonl"),
    tti_first=1001,
    tti_last=1512,
    n_cell=3,
    n_prg=17,
)
formal_cpu_hashes = {
    name: row["sha256"] for name, row in formal_cpu["series"].items()
}
v1_cpu_hashes = {
    name: row["sha256"] for name, row in v1_cpu["series"].items()
}
formal_eval_prefix = (formal / "eval.log").read_text(encoding="utf-8").splitlines()[:3]
v1_eval_prefix = (v1 / "eval.log").read_text(encoding="utf-8").splitlines()[:3]
checks = {
    "state_dict_hash_exact": (
        formal_summary.get("actor_state_dict_sha256")
        == v1_summary.get("actor_state_dict_sha256")
        == frozen_hash
    ),
    "association_counts_exact": (
        formal_summary.get("association_candidate_counts")
        == v1_summary.get("association_candidate_counts")
        == [11, 11, 14]
    ),
    "packet_terminal_trajectory_exact_through_1512": formal_packets == v1_packets,
    "cpu_compact_trajectory_exact_1001_1512": formal_cpu_hashes == v1_cpu_hashes,
    "evaluator_snapshots_exact_through_1500": formal_eval_prefix == v1_eval_prefix,
    "topology_exact": (formal / "ue_topology.csv").read_bytes() == (v1 / "ue_topology.csv").read_bytes(),
    "shadow_exact": (formal / "snr.txt").read_bytes() == (v1 / "snr.txt").read_bytes(),
}
expired_rows = []
import csv
with (formal / "packet_outcome_samples.csv").open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["outcome"] == "expired":
            expired_rows.append(int(row["terminal_tti"]))
payload = {
    "protocol": "pre-experiment-common-prefix-consistency-v1",
    "status": "PASS_WITH_REFERENCE_LIMITATIONS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "formal_binary_sha256": formal_summary.get("sim_binary_sha256"),
    "instrumented_v1_binary_sha256": v1_summary.get("sim_binary_sha256"),
    "binary_hash_equal": formal_summary.get("sim_binary_sha256") == v1_summary.get("sim_binary_sha256"),
    "packet_trajectory": {"formal": formal_packets, "instrumented_v1": v1_packets},
    "cpu_compact_sha256": {"formal": formal_cpu_hashes, "instrumented_v1": v1_cpu_hashes},
    "instrumented_v1_type0_identity": v1_actions,
    "formal_first_expiry_tti": min(expired_rows) if expired_rows else None,
    "limitations": {
        "formal_direct_type0_action_stream_available": False,
        "formal_prefix_scheduled_bytes_stream_available": False,
        "interpretation": (
            "The old formal run did not persist direct Type-0 actions or per-TTI scheduled bytes. "
            "Exact packet, CPU online-path compact records, evaluator snapshots, topology, and shadow "
            "establish observable prefix reproduction; the v1 decision trace is the prospective action reference."
        ),
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] == "FAIL":
    raise SystemExit(f"common-prefix consistency failed: {checks}")
PY

    "${PYTHON_BIN}" - "${ROOT_DIR}" "${formal_report}" "${checkpoint}" \
        "${SIM_BIN}" "${phase_dir}/run_manifest.json" "$(source_revision)" \
        "${frozen_state_hash}" "${protocol_id}" "${phase_dir}/prefix_consistency_report.json" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

root_arg, formal_arg, checkpoint_arg, sim_arg, output_arg, revision, frozen_hash, protocol_id, prefix_arg = sys.argv[1:]
root = pathlib.Path(root_arg).resolve()
formal = json.loads(pathlib.Path(formal_arg).read_text(encoding="utf-8"))
if formal.get("classification") != "seed42_ctbs_performance_gate_failure":
    raise SystemExit("full-horizon audit requires the recorded B4-CTBS seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("full-horizon audit is forbidden after a B4-CTBS structural failure")
prefix_path = pathlib.Path(prefix_arg).resolve()
prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
if prefix.get("status") != "PASS_WITH_REFERENCE_LIMITATIONS":
    raise SystemExit("full-horizon audit requires the prospective common-prefix consistency Gate")
checkpoint_path = pathlib.Path(checkpoint_arg).resolve()
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_hash = state_dict_sha256(checkpoint["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("full-horizon audit checkpoint differs from frozen B4")
source_paths = [
    *sorted((root / "training/stageb_hgraph").glob("*.py")),
    root / "cuMAC/examples/trafficModel/trafficGenerator.hpp",
    root / "cuMAC/examples/multiCellSchedulerUeSelection/main.cpp",
    root / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh",
]
source_digest = hashlib.sha256()
source_entries = []
for path in sorted(source_paths):
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    encoded = relative.encode("utf-8")
    source_digest.update(struct.pack("<I", len(encoded)))
    source_digest.update(encoded)
    source_digest.update(struct.pack("<Q", len(raw)))
    source_digest.update(raw)
    source_entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
sim_path = pathlib.Path(sim_arg).resolve()
payload = {
    "protocol_id": protocol_id,
    "research_mode": "bounded_read_only_attribution_full_horizon",
    "hypothesis": "one exposure category jointly dominates expiry-weighted atoms and modal UE-TTI views",
    "source_revision": revision,
    "source_state_sha256": source_digest.hexdigest(),
    "source_files": source_entries,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint_path),
    "checkpoint_state_dict_sha256": state_hash,
    "treatment": "frozen B4-CTBS-v1",
    "prefix_consistency_report_sha256": hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
    "frozen": {
        "topology_seed": 42,
        "traffic_seed": 42,
        "association_candidate_counts": [11, 11, 14],
        "runtime_candidate_width": 14,
        "warmup_steps": 1000,
        "measured_steps": 4000,
        "measured_tti": [1001, 5000],
        "trace_first_tti": 602,
        "decode_mode": "argmax",
        "headroom": "ctbs-v1",
        "minimum_formal_expired_packets": 50,
        "training": False,
    },
    "formal_atom": "expired_packet_x_tti_x_serving_cell_x_local_prg",
    "formal_weight": "remaining_bytes_at_expiry",
    "dominance": {"minimum_share": 0.50, "minimum_lead_percentage_points": 10.0},
    "classification_rule": "same category must dominate both weighted views",
    "forbidden": ["PFQ pair", "seed41", "seed43", "C", "D", "training", "model changes"],
}
pathlib.Path(output_arg).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

    evaluate_checkpoint "${checkpoint}" 42 1000 4000 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1 \
        --independent-blank-expiry-attribution-audit 1 \
        --blank-expiry-trace-first-tti 602 \
        --independent-blank-expiry-attribution-protocol "${protocol_id}"

    "${PYTHON_BIN}" - "${phase_dir}" "${formal_dir}" "${v1_dir}" "${frozen_state_hash}" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.blank_expiry_attribution_audit import (
    V2_FULL_HORIZON_PROTOCOL_ID,
    canonical_compact_cpu_fingerprints,
    canonical_evaluator_progress_fingerprint,
    canonical_packet_outcome_fingerprint,
    canonical_type0_action_fingerprints,
    classify_v2_full_horizon_result,
)

root = pathlib.Path(sys.argv[1]).resolve()
formal = pathlib.Path(sys.argv[2]).resolve()
v1 = pathlib.Path(sys.argv[3]).resolve()
frozen_hash = sys.argv[4]
actor = root / "actor"
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
summary = json.loads((actor / "eval_summary.json").read_text(encoding="utf-8"))
audit = json.loads((actor / "blank_expiry_attribution_audit.json").read_text(encoding="utf-8"))
formal_summary = json.loads((formal / "eval_summary.json").read_text(encoding="utf-8"))

def decision_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)

reference_action_prefix = canonical_type0_action_fingerprints(
    decision_rows(v1 / "blank_expiry_decision_rows.jsonl"),
    tti_first=1001, tti_last=1512, n_cell=3, n_prg=17,
)
observed_action_prefix = canonical_type0_action_fingerprints(
    decision_rows(actor / "blank_expiry_decision_rows.jsonl"),
    tti_first=1001, tti_last=1512, n_cell=3, n_prg=17,
)
observed_action_full = canonical_type0_action_fingerprints(
    decision_rows(actor / "blank_expiry_decision_rows.jsonl"),
    tti_first=1001, tti_last=5000, n_cell=3, n_prg=17,
)
formal_packets = canonical_packet_outcome_fingerprint(
    formal / "packet_outcome_samples.csv", tti_first=1, tti_last=5000
)
observed_packets = canonical_packet_outcome_fingerprint(
    actor / "packet_outcome_samples.csv", tti_first=1, tti_last=5000
)
formal_cpu = canonical_compact_cpu_fingerprints(
    formal / "output_short.txt", tti_first=1001, tti_last=5000,
    completed_steps=5000,
)
observed_cpu = canonical_compact_cpu_fingerprints(
    actor / "output_short.txt", tti_first=1001, tti_last=5000,
    completed_steps=5000,
)
formal_cpu_hashes = {name: row["sha256"] for name, row in formal_cpu["series"].items()}
observed_cpu_hashes = {name: row["sha256"] for name, row in observed_cpu["series"].items()}
formal_progress = canonical_evaluator_progress_fingerprint(formal / "eval.log")
observed_progress = canonical_evaluator_progress_fingerprint(actor / "eval.log")
aggregate_fields = (
    "stats_sum_scheduled_bytes",
    "stats_sum_ue_goodput_bytes",
    "stats_sum_failed_bytes",
    "stats_sum_expired_bytes",
    "stats_sum_packet_completed_count",
    "stats_sum_packet_delivered_bits",
    "stats_sum_packet_expired_count",
    "measured_tb_tx_count",
    "measured_tb_err_count",
)
aggregate_comparison = {
    field: {"formal": formal_summary.get(field), "observed": summary.get(field), "equal": formal_summary.get(field) == summary.get(field)}
    for field in aggregate_fields
}
trajectory_checks = {
    "type0_identity_common_prefix_exact": reference_action_prefix["global_sha256"] == observed_action_prefix["global_sha256"],
    "packet_terminal_and_expiry_trajectory_exact": formal_packets == observed_packets,
    "cpu_compact_trajectory_exact": formal_cpu_hashes == observed_cpu_hashes,
    "evaluator_progress_snapshots_exact": formal_progress == observed_progress,
    "aggregate_scheduled_successful_expiry_exact": all(row["equal"] for row in aggregate_comparison.values()),
    "state_dict_hash_exact": summary.get("actor_state_dict_sha256") == formal_summary.get("actor_state_dict_sha256") == frozen_hash,
    "topology_exact": (formal / "ue_topology.csv").read_bytes() == (actor / "ue_topology.csv").read_bytes(),
    "shadow_exact": (formal / "snr.txt").read_bytes() == (actor / "snr.txt").read_bytes(),
}
trajectory_reproduced = all(trajectory_checks.values())
trajectory_report = {
    "status": "PASS_WITH_REFERENCE_LIMITATIONS" if trajectory_reproduced else "FAIL",
    "trajectory_reproduced": trajectory_reproduced,
    "checks": trajectory_checks,
    "type0_action": {
        "reference_common_prefix": reference_action_prefix,
        "observed_common_prefix": observed_action_prefix,
        "observed_full_horizon": observed_action_full,
        "formal_direct_full_horizon_action_stream_available": False,
    },
    "packet_trajectory": {"formal": formal_packets, "observed": observed_packets},
    "cpu_compact_sha256": {"formal": formal_cpu_hashes, "observed": observed_cpu_hashes},
    "evaluator_progress": {"formal": formal_progress, "observed": observed_progress},
    "aggregate_comparison": aggregate_comparison,
    "limitation": (
        "The original formal artifact lacks a direct Type-0 stream. Direct action identity is exact "
        "on the instrumented common prefix; full-horizon reproduction is established for packet/expiry, "
        "CPU compact online-path records, evaluator snapshots, and scheduled/successful byte aggregates."
    ),
}
(root / "formal_trajectory_reproduction_report.json").write_text(
    json.dumps(trajectory_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raw_attribution_conclusion = str(audit.get("formal_conclusion"))
formal_expired_count = int(audit.get("coverage", {}).get("formal_expired_packet_count", 0))
formal_conclusion = classify_v2_full_horizon_result(
    trajectory_reproduced=trajectory_reproduced,
    expired_packet_count=formal_expired_count,
    attribution_conclusion=raw_attribution_conclusion,
)
runner_checks = {
    "protocol": audit.get("protocol_id") == manifest.get("protocol_id") == V2_FULL_HORIZON_PROTOCOL_ID,
    "audit_runtime_passed": audit.get("audit_status") == "PASS",
    "completed_steps": summary.get("completed_steps") == 5000,
    "measured_steps": summary.get("stats_count") == 4000,
    "measured_interval": (summary.get("measured_tti_first"), summary.get("measured_tti_last")) == (1001, 5000),
    "trace_interval": summary.get("blank_expiry_trace_first_tti") == 602 and summary.get("blank_expiry_trace_count") == 4399,
    "association_counts": summary.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": summary.get("runtime_candidate_width") == 14,
    "state_dict_frozen": summary.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": summary.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "lifecycle_complete": audit.get("coverage", {}).get("complete") is True,
    "unclassified_zero": audit.get("coverage", {}).get("unclassified_count") == 0,
    "trajectory_reproduced": trajectory_reproduced,
}
payload = {
    **audit,
    "status": "PASS" if all(runner_checks.values()) else "FAIL",
    "attribution_conclusion_before_support_gates": raw_attribution_conclusion,
    "formal_conclusion": formal_conclusion,
    "minimum_formal_expired_packets": 50,
    "formal_event_support_passed": formal_expired_count >= 50,
    "trajectory_reproduction": trajectory_report,
    "runner_checks": runner_checks,
}
(root / "audit_report.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(formal_conclusion)
if payload["status"] != "PASS":
    raise SystemExit(f"full-horizon attribution runtime/trajectory Gate failed: {runner_checks}")
PY
}

run_audit_seed42_cell2_ue2_service_balance() {
    require_binary
    [[ "${STEPS}" == "4000" ]] || {
        echo "Cell2/UE2 service-balance audit is frozen at --steps 4000" >&2
        exit 2
    }
    local protocol_id="seed42-cell2-ue2-service-balance-read-only-audit-v1"
    local frozen_state_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local formal_report="${OUT_DIR}/formal_seed42_pfq_b4_ctbs_v1/formal_pair_report.json"
    local phase_dir="${OUT_DIR}/seed42_cell2_ue2_service_balance_read_only_audit_v1"
    require_file "${checkpoint}"
    require_file "${formal_report}"
    if [[ -e "${phase_dir}" ]]; then
        echo "Cell2/UE2 service-balance audit output already exists; do not reuse or overwrite: ${phase_dir}" >&2
        exit 2
    fi
    unset CUMAC_SHADOW_SEED
    write_run_manifest "${phase_dir}" \
        "fresh paired seed42 PFQ/B4-CTBS-v1; 1000 warmup + 4000 measured; trace 602--5000"

    "${PYTHON_BIN}" - "${ROOT_DIR}" "${phase_dir}/run_manifest.json" "${checkpoint}" \
        "${SIM_BIN}" "${formal_report}" "${frozen_state_hash}" "${protocol_id}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2]).resolve()
checkpoint_path = pathlib.Path(sys.argv[3]).resolve()
sim_path = pathlib.Path(sys.argv[4]).resolve()
formal = json.loads(pathlib.Path(sys.argv[5]).read_text(encoding="utf-8"))
frozen_hash = sys.argv[6]
protocol_id = sys.argv[7]
if formal.get("classification") != "seed42_ctbs_performance_gate_failure":
    raise SystemExit("service-balance audit requires the recorded B4-CTBS seed42 performance failure")
if formal.get("structural_gate", {}).get("status") != "PASS":
    raise SystemExit("service-balance audit is forbidden after a CTBS structural failure")
state_hash = state_dict_sha256(torch.load(checkpoint_path, map_location="cpu")["actor_state"])
if state_hash != frozen_hash:
    raise SystemExit("service-balance audit checkpoint differs from frozen B4")
source_paths = [
    *sorted((root / "training/stageb_hgraph").glob("*.py")),
    root / "cuMAC/examples/trafficModel/serviceBalanceTrace.hpp",
    root / "cuMAC/examples/trafficModel/trafficGenerator.hpp",
    root / "cuMAC/examples/trafficModel/trafficService.hpp",
    root / "cuMAC/examples/multiCellSchedulerUeSelection/main.cpp",
    root / "cuMAC/scripts/run_stageB_pfq_independent_reuse.sh",
]
digest = hashlib.sha256()
entries = []
for path in sorted(source_paths):
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    encoded = relative.encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)
    entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update({
    "protocol_id": protocol_id,
    "research_mode": "bounded_instrumentation_and_paired_read_only_audit",
    "hypothesis": (
        "PFQ-stable but CTBS-positive UE2 queue drift is explained by a stable material "
        "allocation, conditional-TBS, admission, or deadline-timing deficit"
    ),
    "source_state_sha256": digest.hexdigest(),
    "source_files": entries,
    "binary_sha256": hashlib.sha256(sim_path.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint_path),
    "checkpoint_state_dict_sha256": state_hash,
    "checkpoint_state_dict_hash_contract": "sorted tensor name, dtype, shape, and raw bytes",
    "frozen": {
        "topology_seed": 42,
        "traffic_seed": 42,
        "association_candidate_counts": [11, 11, 14],
        "runtime_candidate_width": 14,
        "warmup_steps": 1000,
        "measured_steps": 4000,
        "measured_tti": [1001, 5000],
        "trace_tti": [602, 5000],
        "window_tti": 100,
        "window_count": 40,
        "target_cell": 2,
        "target_ue": 2,
        "material_ratio_max": 0.90,
        "minimum_lower_windows": 30,
        "headroom": "ctbs-v1",
        "decode_mode": "argmax",
        "training": False,
    },
    "read_only_definition": (
        "instrumentation may add observations and rerun simulation but does not alter policy, "
        "checkpoint, action semantics, or reward"
    ),
    "forbidden": ["training", "other seed", "C", "D", "reward change", "policy change"],
})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"
    "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --topology-seed 42 --traffic-seed 42 --warmup-steps 1000 --measure-steps 4000 \
        --formal --service-balance-audit --service-balance-trace-first-tti 602 \
        --out-dir "${phase_dir}/pfq"
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"

    CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
    CUMAC_SERVICE_BALANCE_TRACE_CSV="${phase_dir}/actor/service_balance_trace.csv" \
    CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=602 \
    CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="${phase_dir}/actor/traffic_offered_packets.csv" \
    evaluate_checkpoint "${checkpoint}" 42 1000 4000 "${phase_dir}/actor" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1 \
        --independent-blank-expiry-attribution-audit 1 \
        --blank-expiry-trace-first-tti 602 \
        --independent-blank-expiry-attribution-protocol \
        seed42-ctbs-blank-expiry-attribution-read-only-audit-v2-full-horizon
    verify_formal_seed42_frozen_state "${phase_dir}/run_manifest.json" "${checkpoint}"

    "${PYTHON_BIN}" - "${phase_dir}/actor/eval_summary.json" "${frozen_state_hash}" \
        "${phase_dir}/run_manifest.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
frozen_hash = sys.argv[2]
manifest = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
checks = {
    "completed_steps": summary.get("completed_steps") == 5000,
    "measured_steps": summary.get("stats_count") == 4000,
    "association_counts": summary.get("association_candidate_counts") == [11, 11, 14],
    "runtime_width": summary.get("runtime_candidate_width") == 14,
    "state_dict_frozen": summary.get("actor_state_dict_sha256") == frozen_hash,
    "binary_frozen": summary.get("sim_binary_sha256") == manifest.get("binary_sha256"),
    "candidate_truncation_zero": summary.get("stats_sum_candidate_truncation_count") == 0.0,
    "replay_mismatch_zero": summary.get("stats_sum_action_replay_mismatch_count") == 0.0,
    "illegal_action_zero": summary.get("stats_sum_illegal_action_count") == 0.0,
    "pfq_field_deletion_exact": summary.get("independent_deletion_equal") is True,
    "ctbs_handshake": summary.get("ctbs_pre_action_handshake_enabled") is True,
    "ctbs_mcs_exact": summary.get("ctbs_mcs_mismatch_count") == 0,
    "ctbs_layer_exact": summary.get("ctbs_layer_mismatch_count") == 0,
    "ctbs_tbs_exact": summary.get("ctbs_tbs_mismatch_count") == 0,
    "ctbs_reference_label_deletion_exact": summary.get("ctbs_reference_label_deletion_equal") is True,
}
payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
out = pathlib.Path(sys.argv[1]).parent.parent / "actor_runtime_gate.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"service-balance actor runtime Gate failed: {checks}")
PY

    "${PYTHON_BIN}" -m training.stageb_hgraph.service_balance_audit \
        --pfq-service-trace "${phase_dir}/pfq/service_balance_trace.csv" \
        --ctbs-service-trace "${phase_dir}/actor/service_balance_trace.csv" \
        --pfq-offered-packets "${phase_dir}/pfq/traffic_offered_packets.csv" \
        --ctbs-offered-packets "${phase_dir}/actor/traffic_offered_packets.csv" \
        --pfq-outcomes "${phase_dir}/pfq/packet_outcome_samples.csv" \
        --ctbs-outcomes "${phase_dir}/actor/packet_outcome_samples.csv" \
        --ctbs-decision-rows "${phase_dir}/actor/blank_expiry_decision_rows.jsonl" \
        --out "${phase_dir}/audit_report.json"
}

run_b4_ctbs_c1_short_treatment() {
    require_binary
    [[ "${STEPS}" == "4000" ]] || {
        echo "B4-CTBS-C1 short treatment is frozen at --steps 4000" >&2
        exit 2
    }
    local protocol_id="b4-ctbs-c1-short-treatment-v1"
    local frozen_b4_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local b4_checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local reward_scales="${OUT_DIR}/pfq_seed41/reward_scales.json"
    local pfq_dir="${OUT_DIR}/seed42_cell2_ue2_service_balance_read_only_audit_v1/pfq"
    local phase_dir="${OUT_DIR}/b4_ctbs_c1_short_treatment_v1"
    local train_dir="${phase_dir}/train_seed41"
    local smoke_dir="${phase_dir}/seed42_smoke_128"
    local actor_dir="${phase_dir}/seed42_actor"
    local final_checkpoint="${train_dir}/candidate_checkpoints/ppo_actor_iter_0004.pt"
    require_file "${b4_checkpoint}"
    require_file "${reward_scales}"
    require_file "${pfq_dir}/reference_manifest.json"
    require_file "${pfq_dir}/reference_summary.json"
    require_file "${pfq_dir}/service_balance_trace.csv"
    require_file "${pfq_dir}/traffic_offered_packets.csv"
    require_file "${pfq_dir}/packet_outcome_samples.csv"
    if [[ -e "${phase_dir}" && "${RESUME}" != "1" ]]; then
        echo "B4-CTBS-C1 output already exists; pass --resume only after an engineering-stage interruption: ${phase_dir}" >&2
        exit 2
    fi
    unset CUMAC_SHADOW_SEED
    write_run_manifest "${phase_dir}" \
        "single C1 only: seed41 4x512 PPO TTIs; seed42 128 smoke then 1000+4000 evaluation; reuse exact PFQ"

    "${PYTHON_BIN}" - "${phase_dir}/run_manifest.json" "${b4_checkpoint}" \
        "${frozen_b4_hash}" "${SIM_BIN}" "${pfq_dir}/reference_manifest.json" \
        "${protocol_id}" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

manifest_path = pathlib.Path(sys.argv[1]).resolve()
b4_path = pathlib.Path(sys.argv[2]).resolve()
frozen_b4_hash = sys.argv[3]
sim_path = pathlib.Path(sys.argv[4]).resolve()
pfq_manifest_path = pathlib.Path(sys.argv[5]).resolve()
protocol_id = sys.argv[6]
actual_b4_hash = state_dict_sha256(torch.load(b4_path, map_location="cpu")["actor_state"])
if actual_b4_hash != frozen_b4_hash:
    raise SystemExit(f"frozen B4 state-dict mismatch: {actual_b4_hash}")
binary_hash = hashlib.sha256(sim_path.read_bytes()).hexdigest()
pfq_manifest = json.loads(pfq_manifest_path.read_text(encoding="utf-8"))
expected_scenario = {
    "name": "coop_boundary",
    "topology_seed": 42,
    "traffic_seed": 42,
    "n_cell": 3,
    "n_ue": 36,
    "n_prg": 17,
    "ue_placement": "coop_boundary",
    "cell_association": "strongest",
    "cell_radius_m": 500.0,
    "prbs_per_prg": 16,
    "allocation_type": 0,
    "precoding": "svd",
    "packet_size_bytes": 3000,
    "packets_per_ue_tti": 1.0,
    "packet_ttl_ms": 200.0,
}
reuse_checks = {
    "binary_sha256_exact": pfq_manifest.get("binary_sha256") == binary_hash,
    "scenario_exact": pfq_manifest.get("scenario") == expected_scenario,
    "warmup_exact": pfq_manifest.get("warmup_steps") == 1000,
    "measured_exact": pfq_manifest.get("measure_steps") == 4000,
}
reuse_gate = {
    "status": "PASS" if all(reuse_checks.values()) else "FAIL",
    "checks": reuse_checks,
    "reference_manifest": str(pfq_manifest_path),
    "binary_sha256": binary_hash,
}
(manifest_path.parent / "pfq_reuse_gate.json").write_text(
    json.dumps(reuse_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if reuse_gate["status"] != "PASS":
    raise SystemExit(f"existing PFQ cannot be reused: {reuse_checks}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update({
    "protocol_id": protocol_id,
    "research_mode": "minimal_falsifiable_single_treatment",
    "hypothesis": (
        "The existing delay-debt reward can move frozen-B4-initialized CTBS training "
        "in the correct direction within 2048 PPO TTIs and recover seed42 UE2 allocation and expiry."
    ),
    "binary_sha256": binary_hash,
    "initial_checkpoint": str(b4_path),
    "initial_checkpoint_state_dict_sha256": actual_b4_hash,
    "pfq_reference_reused": str(pfq_manifest_path.parent),
    "frozen": {
        "treatment": "C1",
        "training_seed": 41,
        "training_iterations": 4,
        "rollout_tti_per_iteration": 512,
        "measured_ppo_tti": 2048,
        "headroom": "ctbs-v1",
        "reward": "G-F-E+Delta_D",
        "checkpoint_selection": "fixed_iteration_4",
        "seed42_smoke_tti": 128,
        "seed42_warmup_tti": 1000,
        "seed42_measured_tti": 4000,
        "runtime_candidate_width": 14,
        "ue2_actual_prg_ratio_min_exclusive": 0.90,
        "ue2_cell_share_ratio_min_exclusive": 0.90,
    },
    "not_run": ["C0", "PFQ rerun", "best-of selection", "training extension", "seed41 formal", "seed43", "D"],
})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    if [[ "${RESUME}" == "1" && -f "${phase_dir}/training_gate.json" && -f "${final_checkpoint}" ]]; then
        "${PYTHON_BIN}" - "${phase_dir}/training_gate.json" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("status") != "PASS":
    raise SystemExit("cannot resume from a failed C1 training Gate")
PY
        echo "[C1] reusing the already completed fixed iteration-4 checkpoint after an engineering-stage interruption"
    else
        "${PYTHON_BIN}" -m training.stageb_hgraph.online_ppo \
        --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
        --socket-path "/tmp/cumac_b4_ctbs_c1_train.sock" \
        --online-persistent 1 --episode-horizon 4000 --rollout-steps 512 --iterations 4 \
        --rollout-warmup-steps 1000 --rollout-warmup-once 1 \
        --topology-seed 41 --topology-seed-mode fixed --seed 41 --device auto \
        --out-dir "${train_dir}" --action-head-mode independent \
        --independent-headroom-accounting ctbs-v1 \
        --candidate-mode all_demand_ues --max-prg-candidates 13 \
        --include-pfq-score-anchor 0 --include-candidate-risk-features 0 \
        --pfq-residual-init 0 --reward-mode pfq_independent \
        --independent-treatment C --independent-reward-scales "${reward_scales}" \
        --teacher-mode none --teacher-aux-coef 0 --teacher-aux-min-coef 0 \
        --imitation-episodes 0 --slot-repair-max-per-cell 0 --safe-fill-max-per-cell 0 \
        --candidate-blank-budget-max-per-cell -1 --candidate-checkpoint-interval 4 \
        --candidate-checkpoint-require-gate 0 --hidden-dim 128 --message-layers 2 \
        --ue-feat-dim 16 --dropout 0 --early-stop-patience 0 --best-metric goodput \
        --write-diagnostics 0 --init-checkpoint "${b4_checkpoint}"

        require_file "${final_checkpoint}"
        "${PYTHON_BIN}" - "${b4_checkpoint}" "${final_checkpoint}" \
        "${train_dir}/online_ppo_summary.json" "${phase_dir}/training_gate.json" \
        "${frozen_b4_hash}" <<'PY'
import json
import math
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

b4_path, final_path, summary_path, out_path = map(pathlib.Path, sys.argv[1:5])
frozen_b4_hash = sys.argv[5]
b4 = torch.load(b4_path, map_location="cpu")
final = torch.load(final_path, map_location="cpu")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
history = summary.get("history_tail", [])
reward_values = [
    float(value)
    for row in history
    for key, value in row.items()
    if key.startswith("rollout_reward_") and isinstance(value, (int, float))
]
checks = {
    "b4_hash_frozen": state_dict_sha256(b4["actor_state"]) == frozen_b4_hash,
    "fixed_four_iterations": summary.get("iterations") == 4 and summary.get("completed_iterations") == 4,
    "fixed_512_tti_rollout": summary.get("rollout_steps") == 512,
    "single_c1_reward": summary.get("independent_treatment") == "C",
    "ctbs_headroom": summary.get("independent_headroom_accounting") == "ctbs-v1",
    "fixed_final_checkpoint": final.get("iter") == 4,
    "history_complete": len(history) == 4,
    "ctbs_handshake_each_iteration": len(history) == 4 and all(
        row.get("rollout_ctbs_pre_action_handshake_enabled") is True for row in history
    ),
    "ctbs_exact_parity": len(history) == 4 and all(
        row.get("rollout_ctbs_mcs_mismatch_count") == 0.0
        and row.get("rollout_ctbs_layer_mismatch_count") == 0.0
        and row.get("rollout_ctbs_tbs_mismatch_count") == 0.0
        and row.get("rollout_ctbs_reference_label_deletion_equal") is True
        for row in history
    ),
    "rollout_replay_exact": len(history) == 4 and all(
        row.get("rollout_action_replay_mismatch_count_mean") == 0.0 for row in history
    ),
    "reward_components_finite": bool(reward_values) and all(math.isfinite(value) for value in reward_values),
    "delay_reward_executed": len(history) == 4 and all(
        row.get("rollout_reward_delay_enabled_mean") == 1.0 for row in history
    ) and any(abs(float(row.get("rollout_reward_delay_delta_raw_mean", 0.0))) > 0.0 for row in history),
}
payload = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "initial_actor_state_dict_sha256": state_dict_sha256(b4["actor_state"]),
    "final_actor_state_dict_sha256": state_dict_sha256(final["actor_state"]),
    "final_checkpoint": str(final_path.resolve()),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"C1 training validity Gate failed: {checks}")
PY
    fi

    local final_state_hash
    final_state_hash="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_actor_state_dict_sha256"])' "${phase_dir}/training_gate.json")"
    if [[ "${RESUME}" == "1" && -f "${phase_dir}/smoke_gate.json" ]]; then
        "${PYTHON_BIN}" - "${phase_dir}/smoke_gate.json" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("passed") is not True:
    raise SystemExit("cannot resume from a failed C1 smoke Gate")
PY
        echo "[C1] reusing the completed seed42 128-TTI smoke after a later engineering-stage interruption"
    else
        evaluate_checkpoint "${final_checkpoint}" 42 0 128 "${smoke_dir}" 0 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1
        "${PYTHON_BIN}" - "${smoke_dir}/eval_summary.json" "${phase_dir}/run_manifest.json" \
        "${final_state_hash}" "${phase_dir}/smoke_gate.json" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.independent_gate import ctbs_seed42_smoke_gate

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
report = ctbs_seed42_smoke_gate(
    summary,
    frozen_state_dict_sha256=sys.argv[3],
    frozen_binary_sha256=manifest["binary_sha256"],
)
report["protocol_id"] = "b4-ctbs-c1-short-treatment-v1-seed42-smoke"
pathlib.Path(sys.argv[4]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not report["passed"]:
    raise SystemExit(f"C1 seed42 smoke failed: {report['checks']}")
PY
    fi

    CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
    CUMAC_SERVICE_BALANCE_TRACE_CSV="${actor_dir}/service_balance_trace.csv" \
    CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=602 \
    CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="${actor_dir}/traffic_offered_packets.csv" \
    evaluate_checkpoint "${final_checkpoint}" 42 1000 4000 "${actor_dir}" 1 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1

    "${PYTHON_BIN}" -m training.stageb_hgraph.service_balance_audit \
        --pfq-service-trace "${pfq_dir}/service_balance_trace.csv" \
        --ctbs-service-trace "${actor_dir}/service_balance_trace.csv" \
        --pfq-offered-packets "${pfq_dir}/traffic_offered_packets.csv" \
        --ctbs-offered-packets "${actor_dir}/traffic_offered_packets.csv" \
        --pfq-outcomes "${pfq_dir}/packet_outcome_samples.csv" \
        --ctbs-outcomes "${actor_dir}/packet_outcome_samples.csv" \
        --out "${phase_dir}/service_balance_report.json"

    "${PYTHON_BIN}" - "${pfq_dir}/reference_summary.json" "${actor_dir}/gate_metrics.json" \
        "${actor_dir}/eval_summary.json" "${phase_dir}/smoke_gate.json" \
        "${phase_dir}/service_balance_report.json" "${final_state_hash}" \
        "${phase_dir}/run_manifest.json" "${phase_dir}/final_report.json" <<'PY'
import json
import pathlib
import sys

from training.stageb_hgraph.independent_gate import short_debt_treatment_gate

pfq = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
actor = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
summary = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
smoke = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
service = json.loads(pathlib.Path(sys.argv[5]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[7]).read_text(encoding="utf-8"))
report = short_debt_treatment_gate(
    pfq,
    actor,
    summary,
    smoke,
    service,
    actor_state_dict_sha256=sys.argv[6],
    binary_sha256=manifest["binary_sha256"],
)
report.update({
    "initial_checkpoint_state_dict_sha256": manifest["initial_checkpoint_state_dict_sha256"],
    "final_checkpoint_state_dict_sha256": sys.argv[6],
    "pfq_reference_reused": manifest["pfq_reference_reused"],
    "core_metrics": {
        "pfq": {key: pfq[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
        "c1": {key: actor[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
    },
})
pathlib.Path(sys.argv[8]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(report["classification"])
PY
}

run_b4_ctbs_c1_multiseed_long() {
    require_binary
    [[ "${STEPS}" == "4000" ]] || {
        echo "B4-CTBS-C1 multi-seed long training is frozen at --steps 4000" >&2
        exit 2
    }
    local protocol_id="b4-ctbs-c1-multiseed-long-v1"
    local frozen_b4_hash="4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571"
    local b4_checkpoint="${OUT_DIR}/pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt"
    local reward_scales="${OUT_DIR}/pfq_seed41/reward_scales.json"
    local seed42_pfq_dir="${OUT_DIR}/seed42_cell2_ue2_service_balance_read_only_audit_v1/pfq"
    local phase_dir="${OUT_DIR}/b4_ctbs_c1_multiseed_long_v1_attempt2"
    local train_dir="${phase_dir}/train_seeds41_42"
    local final_checkpoint="${train_dir}/candidate_checkpoints/ppo_actor_iter_0100.pt"
    local seed42_actor_dir="${phase_dir}/seed42/actor"
    require_file "${b4_checkpoint}"
    require_file "${reward_scales}"
    require_file "${seed42_pfq_dir}/reference_manifest.json"
    require_file "${seed42_pfq_dir}/reference_summary.json"
    require_file "${seed42_pfq_dir}/service_balance_trace.csv"
    require_file "${seed42_pfq_dir}/traffic_offered_packets.csv"
    require_file "${seed42_pfq_dir}/packet_outcome_samples.csv"
    if [[ -e "${phase_dir}" && "${RESUME}" != "1" ]]; then
        echo "multi-seed long-training output already exists; use --resume only after a completed stage: ${phase_dir}" >&2
        exit 2
    fi
    unset CUMAC_SHADOW_SEED
    write_run_manifest "${phase_dir}" \
        "B4 init; CTBS+C; alternating seeds41,42; 100x512 PPO TTIs; fixed iter100; sequential seed42,seed41,seed43 gates"

    "${PYTHON_BIN}" - "${phase_dir}/run_manifest.json" "${b4_checkpoint}" \
        "${frozen_b4_hash}" "${SIM_BIN}" "${seed42_pfq_dir}/reference_manifest.json" \
        "${protocol_id}" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256

manifest_path = pathlib.Path(sys.argv[1]).resolve()
b4_path = pathlib.Path(sys.argv[2]).resolve()
frozen_hash = sys.argv[3]
sim_path = pathlib.Path(sys.argv[4]).resolve()
pfq_manifest_path = pathlib.Path(sys.argv[5]).resolve()
protocol_id = sys.argv[6]
actual_hash = state_dict_sha256(torch.load(b4_path, map_location="cpu")["actor_state"])
if actual_hash != frozen_hash:
    raise SystemExit(f"frozen B4 state-dict mismatch: {actual_hash}")
binary_hash = hashlib.sha256(sim_path.read_bytes()).hexdigest()
pfq_manifest = json.loads(pfq_manifest_path.read_text(encoding="utf-8"))
scenario = pfq_manifest.get("scenario", {})
reuse_checks = {
    "binary_sha256_exact": pfq_manifest.get("binary_sha256") == binary_hash,
    "seed42_exact": scenario.get("topology_seed") == 42 and scenario.get("traffic_seed") == 42,
    "canonical_shape_exact": (
        scenario.get("n_cell") == 3
        and scenario.get("n_ue") == 36
        and scenario.get("n_prg") == 17
        and scenario.get("cell_association") == "strongest"
        and scenario.get("ue_placement") == "coop_boundary"
    ),
    "traffic_exact": (
        scenario.get("packet_size_bytes") == 3000
        and scenario.get("packets_per_ue_tti") == 1.0
        and scenario.get("packet_ttl_ms") == 200.0
    ),
    "horizon_exact": pfq_manifest.get("warmup_steps") == 1000 and pfq_manifest.get("measure_steps") == 4000,
}
reuse_gate = {
    "status": "PASS" if all(reuse_checks.values()) else "FAIL",
    "checks": reuse_checks,
    "reference_manifest": str(pfq_manifest_path),
}
(manifest_path.parent / "seed42_pfq_reuse_gate.json").write_text(
    json.dumps(reuse_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if reuse_gate["status"] != "PASS":
    raise SystemExit(f"seed42 PFQ reuse Gate failed: {reuse_checks}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.update({
    "protocol_id": protocol_id,
    "binary_sha256": binary_hash,
    "initial_checkpoint": str(b4_path),
    "initial_checkpoint_state_dict_sha256": actual_hash,
    "hypothesis": (
        "Alternating exposure to seed41 and the seed42 failure distribution allows the existing "
        "delay-debt treatment to recover seed42 UE2 allocation without breaking architecture gates."
    ),
    "frozen": {
        "training_seeds": [41, 42],
        "seed_order": "alternate_by_rollout_starting_41",
        "iterations": 100,
        "rollout_tti": 512,
        "measured_ppo_tti": 51200,
        "measured_ppo_tti_per_seed": 25600,
        "warmup_tti_before_each_rollout": 1000,
        "warmup_excluded_from_budget": True,
        "safety_prefix_iterations": 8,
        "safety_prefix_measured_tti": 4096,
        "reward": "G-F-E+Delta_D",
        "headroom": "ctbs-v1",
        "checkpoint_selection": "fixed_iteration_100",
        "evaluation_order": [42, 41, 43],
        "seed43_training_or_development_use": False,
    },
    "execution_attempt": 2,
    "supersedes_engineering_attempt": "../b4_ctbs_c1_multiseed_long_v1/attempt_failure_report.json",
    "not_run": ["C0", "best-of selection", "reward scan", "training extension", "D"],
})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    if [[ "${RESUME}" == "1" ]]; then
        require_file "${phase_dir}/training_gate.json"
        require_file "${final_checkpoint}"
        "${PYTHON_BIN}" - "${phase_dir}/training_gate.json" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("status") != "PASS":
    raise SystemExit("partial or failed long training cannot be resumed without optimizer-state continuity")
PY
        echo "[long] reusing completed fixed iteration-100 checkpoint"
    else
        "${PYTHON_BIN}" -m training.stageb_hgraph.online_ppo \
            --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
            --socket-path "/tmp/cumac_b4_ctbs_c1_multiseed_long.sock" \
            --online-persistent 1 --episode-horizon 4000 --rollout-steps 512 --iterations 100 \
            --rollout-warmup-steps 1000 --rollout-warmup-once 0 \
            --topology-seed 41 --topology-seed-mode list_cycle --seed-list 41,42 \
            --seed 41 --device auto --out-dir "${train_dir}" --action-head-mode independent \
            --independent-headroom-accounting ctbs-v1 \
            --candidate-mode all_demand_ues --max-prg-candidates 13 \
            --include-pfq-score-anchor 0 --include-candidate-risk-features 0 \
            --pfq-residual-init 0 --reward-mode pfq_independent \
            --independent-treatment C --independent-reward-scales "${reward_scales}" \
            --teacher-mode none --teacher-aux-coef 0 --teacher-aux-min-coef 0 \
            --imitation-episodes 0 --slot-repair-max-per-cell 0 --safe-fill-max-per-cell 0 \
            --candidate-blank-budget-max-per-cell -1 --candidate-checkpoint-interval 100 \
            --candidate-checkpoint-require-gate 0 --hidden-dim 128 --message-layers 2 \
            --ue-feat-dim 16 --dropout 0 --early-stop-patience 0 --best-metric goodput \
            --write-diagnostics 0 --safety-prefix-iters 8 --init-checkpoint "${b4_checkpoint}"

        require_file "${final_checkpoint}"
        require_file "${train_dir}/safety_prefix_gate.json"
        "${PYTHON_BIN}" - "${b4_checkpoint}" "${final_checkpoint}" \
            "${train_dir}/online_ppo_summary.json" "${train_dir}/online_ppo_metrics.csv" \
            "${train_dir}/safety_prefix_gate.json" "${phase_dir}/training_gate.json" \
            "${frozen_b4_hash}" <<'PY'
import csv
import json
import pathlib
import sys

import torch

from training.stageb_hgraph.independent_contract import state_dict_sha256
from training.stageb_hgraph.independent_gate import staged_multiseed_safety_prefix_gate

b4_path, final_path, summary_path, csv_path, prefix_path, out_path = map(pathlib.Path, sys.argv[1:7])
frozen_hash = sys.argv[7]

def parse(value):
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value

rows = [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(csv_path.open(encoding="utf-8"))]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
b4 = torch.load(b4_path, map_location="cpu")
final = torch.load(final_path, map_location="cpu")
full_validity = staged_multiseed_safety_prefix_gate(rows, prefix_iterations=100)
seeds = [int(row["topology_seed"]) for row in rows]
checks = {
    "initial_b4_hash_frozen": state_dict_sha256(b4["actor_state"]) == frozen_hash,
    "fixed_iteration_100_checkpoint": final.get("iter") == 100,
    "summary_complete": summary.get("iterations") == 100 and summary.get("completed_iterations") == 100,
    "fixed_rollout_and_budget": summary.get("rollout_steps") == 512 and len(rows) == 100,
    "alternate_seed_cycle_exact": seeds == [41 if i % 2 == 0 else 42 for i in range(100)],
    "per_seed_budget_exact": seeds.count(41) == 50 and seeds.count(42) == 50,
    "warmup_before_each_rollout": summary.get("rollout_warmup_steps") == 1000 and summary.get("rollout_warmup_once") == 0,
    "list_cycle_frozen": summary.get("topology_seed_mode") == "list_cycle" and summary.get("seed_list") == "41,42",
    "treatment_c_only": summary.get("independent_treatment") == "C",
    "ctbs_only": summary.get("independent_headroom_accounting") == "ctbs-v1",
    "fixed_endpoint_only": summary.get("candidate_checkpoint_interval") == 100,
    "safety_prefix_passed": prefix.get("status") == "PASS" and prefix.get("observed_iterations") == 8 and prefix.get("measured_ttis") == 4096,
    "all_100_iterations_valid": full_validity.get("status") == "PASS",
    "delay_debt_executed": all(float(row.get("rollout_reward_delay_enabled_mean", 0.0)) == 1.0 for row in rows)
        and any(abs(float(row.get("rollout_reward_delay_delta_raw_mean", 0.0))) > 0.0 for row in rows),
}
payload = {
    "protocol_id": "b4-ctbs-c1-multiseed-long-v1-training",
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "safety_prefix_gate": prefix,
    "full_training_validity_gate": full_validity,
    "initial_actor_state_dict_sha256": state_dict_sha256(b4["actor_state"]),
    "final_actor_state_dict_sha256": state_dict_sha256(final["actor_state"]),
    "fixed_final_checkpoint": str(final_path.resolve()),
    "measured_ppo_tti": sum(int(row["rollout_steps"]) for row in rows),
    "measured_ppo_tti_by_seed": {"41": seeds.count(41) * 512, "42": seeds.count(42) * 512},
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"long-training validity Gate failed: {checks}")
PY
    fi

    local final_state_hash
    final_state_hash="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_actor_state_dict_sha256"])' "${phase_dir}/training_gate.json")"

    CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
    CUMAC_SERVICE_BALANCE_TRACE_CSV="${seed42_actor_dir}/service_balance_trace.csv" \
    CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=602 \
    CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="${seed42_actor_dir}/traffic_offered_packets.csv" \
    evaluate_checkpoint "${final_checkpoint}" 42 1000 4000 "${seed42_actor_dir}" 1 1 argmax \
        --independent-runtime-candidate-width 14 \
        --independent-headroom-accounting ctbs-v1

    "${PYTHON_BIN}" -m training.stageb_hgraph.service_balance_audit \
        --pfq-service-trace "${seed42_pfq_dir}/service_balance_trace.csv" \
        --ctbs-service-trace "${seed42_actor_dir}/service_balance_trace.csv" \
        --pfq-offered-packets "${seed42_pfq_dir}/traffic_offered_packets.csv" \
        --ctbs-offered-packets "${seed42_actor_dir}/traffic_offered_packets.csv" \
        --pfq-outcomes "${seed42_pfq_dir}/packet_outcome_samples.csv" \
        --ctbs-outcomes "${seed42_actor_dir}/packet_outcome_samples.csv" \
        --out "${phase_dir}/seed42/service_balance_report.json"

    "${PYTHON_BIN}" - "${seed42_pfq_dir}/reference_summary.json" \
        "${seed42_actor_dir}/gate_metrics.json" "${seed42_actor_dir}/eval_summary.json" \
        "${phase_dir}/seed42/service_balance_report.json" "${final_state_hash}" \
        "${phase_dir}/run_manifest.json" "${phase_dir}/seed42/gate_report.json" <<'PY'
import json, pathlib, sys
from training.stageb_hgraph.independent_gate import long_multiseed_seed42_gate
pfq = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
actor = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
summary = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
service = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[6]).read_text(encoding="utf-8"))
report = long_multiseed_seed42_gate(
    pfq, actor, summary, service,
    actor_state_dict_sha256=sys.argv[5],
    binary_sha256=manifest["binary_sha256"],
)
report["metrics"] = {
    "pfq": {key: pfq[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
    "actor": {key: actor[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
}
pathlib.Path(sys.argv[7]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    if [[ "$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "${phase_dir}/seed42/gate_report.json")" != "1" ]]; then
        "${PYTHON_BIN}" - "${phase_dir}/seed42/gate_report.json" "${phase_dir}/final_report.json" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "protocol_id": "b4-ctbs-c1-multiseed-long-v1",
    "status": "COMPLETE",
    "classification": "long_multiseed_seed42_failure",
    "evaluated_seeds": [42],
    "seed42": gate,
    "stopped_before": ["seed41 evaluation", "seed43 evaluation"],
    "interpretation": "Exposure to the seed42 failure distribution and the fixed long budget did not close the seed42 Gate; the existing debt route is not supported.",
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
        return
    fi

    local seed
    for seed in 41 43; do
        local pfq_dir="${phase_dir}/seed${seed}/pfq"
        local actor_dir="${phase_dir}/seed${seed}/actor"
        "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
            --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
            --topology-seed "${seed}" --traffic-seed "${seed}" \
            --warmup-steps 1000 --measure-steps 4000 --formal --out-dir "${pfq_dir}"
        evaluate_checkpoint "${final_checkpoint}" "${seed}" 1000 4000 "${actor_dir}" 1 1 argmax \
            --independent-headroom-accounting ctbs-v1
        local expected_counts="null" expected_width="none"
        if [[ "${seed}" == "41" ]]; then
            expected_counts='[11,12,13]'
            expected_width='13'
        fi
        "${PYTHON_BIN}" - "${pfq_dir}/reference_summary.json" "${pfq_dir}/reference_manifest.json" \
            "${actor_dir}/gate_metrics.json" "${actor_dir}/eval_summary.json" \
            "${final_state_hash}" "${phase_dir}/run_manifest.json" "${expected_counts}" \
            "${expected_width}" "${phase_dir}/seed${seed}/gate_report.json" "${seed}" <<'PY'
import json, pathlib, sys
from training.stageb_hgraph.independent_gate import ctbs_formal_evaluation_gate
pfq = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pfq_manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
actor = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
summary = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
run_manifest = json.loads(pathlib.Path(sys.argv[6]).read_text(encoding="utf-8"))
counts = None if sys.argv[7] == "null" else json.loads(sys.argv[7])
width = None if sys.argv[8] == "none" else int(sys.argv[8])
seed = int(sys.argv[10])
if pfq_manifest.get("binary_sha256") != run_manifest["binary_sha256"]:
    raise SystemExit("fresh PFQ binary provenance mismatch")
scenario = pfq_manifest.get("scenario", {})
if scenario.get("topology_seed") != seed or scenario.get("traffic_seed") != seed:
    raise SystemExit("fresh PFQ seed provenance mismatch")
report = ctbs_formal_evaluation_gate(
    pfq, actor, summary,
    actor_state_dict_sha256=sys.argv[5],
    binary_sha256=run_manifest["binary_sha256"],
    expected_association_counts=counts,
    expected_runtime_width=width,
    protocol_id=f"long_multiseed_seed{seed}",
)
report["metrics"] = {
    "pfq": {key: pfq[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
    "actor": {key: actor[key] for key in ("goodput_mbps", "tb_bler", "expiry_rate")},
}
pathlib.Path(sys.argv[9]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
        if [[ "$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "${phase_dir}/seed${seed}/gate_report.json")" != "1" ]]; then
            "${PYTHON_BIN}" - "${phase_dir}/seed42/gate_report.json" \
                "${phase_dir}/seed${seed}/gate_report.json" "${phase_dir}/final_report.json" "${seed}" <<'PY'
import json, pathlib, sys
seed42 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
failed = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
seed = int(sys.argv[4])
payload = {
    "protocol_id": "b4-ctbs-c1-multiseed-long-v1",
    "status": "COMPLETE",
    "classification": "seed42_pass_seed41_regression" if seed == 41 else "seen_seeds_pass_unseen_seed43_failure",
    "evaluated_seeds": [42, seed] if seed == 41 else [42, 41, 43],
    "seed42": seed42,
    f"seed{seed}": failed,
    "stopped_before": ["seed43 evaluation"] if seed == 41 else [],
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
            return
        fi
    done

    "${PYTHON_BIN}" - "${phase_dir}/seed42/gate_report.json" \
        "${phase_dir}/seed41/gate_report.json" "${phase_dir}/seed43/gate_report.json" \
        "${phase_dir}/training_gate.json" "${phase_dir}/final_report.json" <<'PY'
import json, pathlib, sys
seed42, seed41, seed43, training = [
    json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:5]
]
payload = {
    "protocol_id": "b4-ctbs-c1-multiseed-long-v1",
    "status": "PASS",
    "classification": "long_multiseed_seed41_43_pass",
    "evaluated_seeds": [42, 41, 43],
    "training": training,
    "seed42": seed42,
    "seed41": seed41,
    "seed43": seed43,
    "interpretation": "The fixed multi-seed training passed both seen-seed gates and the untouched seed43 gate.",
}
pathlib.Path(sys.argv[5]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_paired_seeds() {
    require_file "${OUT_DIR}/seed41_selection/treatment_comparison.json"
    local winner
    winner="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_treatment"] or "")' "${OUT_DIR}/seed41_selection/treatment_comparison.json")"
    [[ -n "${winner}" ]] || { echo "seed-41 gate has no winner; paired seeds are forbidden" >&2; exit 2; }
    local selected_iteration
    selected_iteration="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_iteration"])' "${OUT_DIR}/pilot-${winner}/checkpoint_selection.json")"
    local checkpoint="${OUT_DIR}/pilot-${winner}/candidate_checkpoints/ppo_actor_iter_$(printf '%04d' "${selected_iteration}").pt"
    require_file "${checkpoint}"
    local phase_dir="${OUT_DIR}/paired_seeds_41_43"
    write_run_manifest "${phase_dir}" "paired PFQ/actor seeds 41--43; fixed winner=${winner} iteration=${selected_iteration}"
    for seed in 41 42 43; do
        "${PYTHON_BIN}" -m training.stageb_hgraph.pfq_reference_calibration \
            --sim-bin "${SIM_BIN}" --sim-args "${SIM_ARGS}" --sim-cwd "${ROOT_DIR}" \
            --topology-seed "${seed}" --traffic-seed "${seed}" --warmup-steps 1000 --measure-steps 4000 \
            --out-dir "${phase_dir}/seed-${seed}/pfq"
        evaluate_checkpoint "${checkpoint}" "${seed}" 1000 4000 "${phase_dir}/seed-${seed}/actor"
    done
    "${PYTHON_BIN}" - "${phase_dir}" "${checkpoint}" "${winner}" <<'PY'
import json, pathlib, sys
from training.stageb_hgraph.independent_gate import architecture_gate
root, checkpoint, winner = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for seed in (41, 42, 43):
    pfq = json.load(open(root / f"seed-{seed}" / "pfq" / "reference_summary.json"))
    actor = json.load(open(root / f"seed-{seed}" / "actor" / "gate_metrics.json"))
    rows.append({"seed": seed, "pfq": pfq, "actor": actor, "architecture_gate": architecture_gate(pfq, actor)})
payload = {
    "winner": winner,
    "checkpoint": checkpoint,
    "pairs": rows,
    "all_seed_architecture_gates_passed": all(row["architecture_gate"]["passed"] for row in rows),
}
open(root / "paired_seed_summary.json", "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY
}

case "${PHASE}" in
    build) run_build ;;
    pfq-reference) run_pfq_reference ;;
    validity) run_validity ;;
    bc) run_bc ;;
    pilot-A) run_pilot A ;;
    pilot-B) run_pilot B ;;
    pilot-C) run_pilot C ;;
    pilot-D) run_pilot D ;;
    select-and-eval-seed41) run_select_seed41 ;;
    blank-diagnostics-seed41) run_blank_diagnostics_seed41 ;;
    hierarchical-decode-smoke) run_hierarchical_decode_smoke ;;
    hierarchical-decode-seed41) run_hierarchical_decode_seed41 ;;
    rescue-cohort-smoke) run_rescue_cohort_smoke ;;
    rescue-cohort-seed41) run_rescue_cohort_seed41 ;;
    candidate-width-compatibility) run_candidate_width_compatibility ;;
    candidate-width-fp32-equivalence) run_candidate_width_fp32_equivalence ;;
    formal-seed42-pfq-b4) run_formal_seed42_pfq_b4 ;;
    diagnose-seed42-per-prg) run_diagnose_seed42_per_prg ;;
    audit-seed42-service-accounting) run_audit_seed42_service_accounting ;;
    audit-seed42-phy-tbs-alignment) run_audit_seed42_phy_tbs_alignment ;;
    audit-seed42-phy-tbs-alignment-v2-cpu-olla) run_audit_seed42_phy_tbs_alignment_v2_cpu_olla ;;
    b4-ctbs-seed42) run_b4_ctbs_seed42 ;;
    audit-seed42-ctbs-blank-expiry-attribution) run_audit_seed42_ctbs_blank_expiry_attribution ;;
    audit-seed42-ctbs-blank-expiry-attribution-v2-full-horizon) run_audit_seed42_ctbs_blank_expiry_attribution_v2_full_horizon ;;
    audit-seed42-cell2-ue2-service-balance) run_audit_seed42_cell2_ue2_service_balance ;;
    b4-ctbs-c1-short-treatment) run_b4_ctbs_c1_short_treatment ;;
    b4-ctbs-c1-multiseed-long) run_b4_ctbs_c1_multiseed_long ;;
    paired-seeds-41-43) run_paired_seeds ;;
esac

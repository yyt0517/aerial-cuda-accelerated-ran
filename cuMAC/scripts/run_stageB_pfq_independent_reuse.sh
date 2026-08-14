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
    build|pfq-reference|validity|bc|pilot-A|pilot-B|pilot-C|pilot-D|select-and-eval-seed41|blank-diagnostics-seed41|paired-seeds-41-43) ;;
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
    local socket="/tmp/cumac_independent_eval_${seed}_$$.sock"
    mkdir -p "${eval_dir}"
    eval_dir="$(cd "${eval_dir}" && pwd -P)"
    local outcome_csv="${eval_dir}/packet_outcome_samples.csv"
    local packet_delay_csv="${eval_dir}/packet_delay_samples.csv"
    (
        cd "${eval_dir}"
        env CUMAC_ONLINE_BRIDGE=1 CUMAC_ONLINE_SOCKET="${socket}" CUMAC_ONLINE_PERSISTENT=1 \
            CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
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
        --stats-warmup-steps "${warmup}" --decode-mode argmax --seed "${seed}" \
        --sample-seed "${seed}" --device auto --print-every-steps 500 --save-steps 0 \
        --sim-bin "${SIM_BIN}" --outcome-samples "${outcome_csv}" --out-dir "${eval_dir}" \
        --independent-policy-diagnostics "${diagnostics}" \
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
    paired-seeds-41-43) run_paired_seeds ;;
esac

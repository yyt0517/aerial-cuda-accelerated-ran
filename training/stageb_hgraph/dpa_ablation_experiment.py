"""Fixed, equal-budget DPA experiment. No tuning, checkpoint selection, or resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np

from .dpa_v1_historical_comparison import WindowSpec
from .pfq_dpa_aligned_comparison import summarize_aligned_window, validate_seed_artifacts
from .pfq_reference_calibration import _extend_ld_library_path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_reuse_attention", "no_dimensionless")
TRAIN_SEEDS = (701, 702, 703)
TEST_SEEDS = (201, 202, 203, 204, 205)
SIM_ARGS = "-d 1 -b 2 -x 0 -f 0 -g 100 -r 3000"
REFERENCE_SIM_ARGS = {"pfq": SIM_ARGS, "rrq": SIM_ARGS.replace("-b 2", "-b 3")}
METRICS = ("goodput_mbps", "tb_bler", "expiry_drop_rate", "outcome_delay_mean_ms",
           "outcome_delay_p95_ms", "ue_expiry_aware_rate_p10_mbps",
           "ue_expiry_aware_rate_min_mbps", "prg_utilization_ratio",
           "successful_bytes_per_active_prg", "end_backlog_bytes")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def cli(module, options):
    result = [sys.executable, "-m", "training.stageb_hgraph." + module]
    for key, value in options.items():
        result.extend(("--" + key, str(value)))
    return result


def train_command(sim, out, variant, seed, *, smoke):
    return cli("online_ppo", {
        "sim-bin": sim, "sim-args": SIM_ARGS, "sim-cwd": out,
        "out-dir": out, "socket-path": f"/tmp/dpa_ab_{os.getpid()}_{seed}_{variant}.sock",
        "connect-timeout-s": 60, "device": "cuda", "seed": seed,
        "action-head-mode": "independent", "independent-ablation": variant,
        "independent-random-init": 1, "independent-headroom-accounting": "ctbs-v1",
        "candidate-mode": "all_demand_ues", "max-prg-candidates": 13,
        "include-pfq-score-anchor": 0, "include-candidate-risk-features": 0,
        "pfq-residual-init": 0, "teacher-mode": "none", "teacher-aux-coef": 0,
        "teacher-aux-min-coef": 0, "imitation-episodes": 0,
        "reward-mode": "deadline_tail", "independent-treatment": "E",
        "slot-repair-max-per-cell": 0, "safe-fill-max-per-cell": 0,
        "candidate-blank-budget-max-per-cell": -1,
        "hidden-dim": 128, "message-layers": 2, "ue-feat-dim": 16, "dropout": 0,
        "online-persistent": 1, "episode-horizon": 4000,
        "iterations": 1 if smoke else 48, "rollout-steps": 64 if smoke else 512,
        "rollout-warmup-steps": 32 if smoke else 1000, "rollout-warmup-once": 0,
        "topology-seed": 41, "topology-seed-mode": "list_cycle", "seed-list": "41,42,45",
        "lr": 1e-4, "actor-lr": 1e-4, "critic-lr": 3e-4, "weight-decay": 1e-5,
        "gamma": .99, "gae-lambda": .95, "clip-eps": .2, "entropy-coef": .01,
        "value-coef": .5, "value-loss": "huber", "value-huber-beta": 10,
        "normalize-state": 1, "normalize-reward": 1, "normalize-adv": 1,
        "max-grad-norm": 1, "target-kl": 0, "early-stop-patience": 0,
        "update-epochs": 4, "minibatch-size": 64,
        "candidate-checkpoint-interval": 0, "best-metric": "goodput",
        "write-diagnostics": 0, "safety-prefix-iters": 0,
        "rollout-log-every-steps": 128, "update-log-every-minibatches": 8,
    })


def paired_deltas(matrix):
    """Rows are training seeds, columns are fixed evaluation seeds (not iid cells)."""
    matrix = np.asarray(matrix, dtype=float)
    means = matrix.mean(axis=1)
    n = len(means)
    ci = None
    if n == 3:
        half = 4.3026527299 * float(means.std(ddof=1)) / np.sqrt(n)
        ci = [float(means.mean() - half), float(means.mean() + half)]
    return {"raw_deltas": matrix.tolist(), "training_seed_means": means.tolist(),
            "test_seed_means": matrix.mean(axis=0).tolist(),
            "n_training_replicates": n, "mean_delta": float(means.mean()),
            "conditional_ci95": ci}


def train_audit(out, variant, smoke):
    import torch
    rows = read_csv(out / "online_ppo_metrics.csv")
    niter, rollout, warmup = (1, 64, 32) if smoke else (48, 512, 1000)
    if len(rows) != niter:
        raise ValueError(f"incomplete training: {out}")
    required = {"rollout_steps": rollout, "rollout_warmup_steps": warmup,
                "rollout_total_env_steps": rollout + warmup,
                "num_updates": 4 * rollout // 64, "epochs_completed": 4,
                "stopped_by_target_kl": 0, "rollout_ctbs_mcs_mismatch_count": 0,
                "rollout_ctbs_layer_mismatch_count": 0, "rollout_ctbs_tbs_mismatch_count": 0}
    for i, row in enumerate(rows):
        if row["rollout_ctbs_reference_label_deletion_equal"] != "True":
            raise ValueError("CTBS reference-label deletion equality failed")
        for key, value in required.items():
            if float(row[key]) != value:
                raise ValueError(f"budget/CTBS violation: {out} iter={i+1} {key}={row[key]}")
        if int(float(row["topology_seed"])) != (41, 42, 45)[i % 3]:
            raise ValueError("training topology sequence mismatch")
        for key in ("policy_loss", "value_loss", "approx_kl_mean", "rollout_reward_mean"):
            if not np.isfinite(float(row[key])):
                raise ValueError(f"nonfinite training metric {key}")
    checkpoint = out / "ppo_actor_last.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["iter"] != niter or ckpt["actor_config"]["ablation"] != variant:
        raise ValueError("last checkpoint identity mismatch")
    if ckpt["initialization"]["mode"] != "random":
        raise ValueError("learned initialization invalidates this campaign")
    if not all(torch.isfinite(t).all().item() for t in ckpt["actor_state"].values()):
        raise ValueError("nonfinite checkpoint")
    return {"iterations": niter, "training_transitions": niter * rollout,
            "total_env_steps": niter * (rollout + warmup),
            "actor_optimizer_updates": niter * 4 * rollout // 64,
            "checkpoint_sha256": sha(checkpoint), "initialization": ckpt["initialization"]}


def base_env(gpu):
    # Do not inherit stale simulator overrides from an interactive shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUMAC_")}
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "CUMAC_TORCH_NUM_THREADS": "1", "CUMAC_TORCH_NUM_INTEROP_THREADS": "1",
                "CUMAC_ONLINE_EXPORT_TEACHER_ACTION": "1",
                "CUMAC_ONLINE_EXPORT_POST_EQ_SINR": "1",
                "CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT": "1",
                "CUMAC_CELL_ASSOC_MODE": "strongest", "CUMAC_UE_PLACEMENT_MODE": "coop_boundary",
                "CUMAC_UE_VORONOI_CLIP": "1", "CUMAC_TOPOLOGY_DUMP": "1",
                "CUMAC_BS_TX_PATTERN": "omni", "CUMAC_TRAFFIC_ARRIVAL_RATE": "1.0",
                "CUMAC_TRAFFIC_PROFILE": "homogeneous", "CUMAC_PACKET_TTL_TTI": "0",
                "CUMAC_PACKET_TTL_MS": "200", "CUMAC_COMPACT_TTI_LOG": "1",
                "CUMAC_PROGRESS_TTI_INTERVAL": "0", "CUMAC_COMPARE_TTI_INTERVAL": "0",
                "CUMAC_EXEC_MODE": "both"})
    return env


def eval_env(base, out, seed, socket):
    return {**base, "CUMAC_ONLINE_BRIDGE": "1", "CUMAC_ONLINE_SOCKET": socket,
            "CUMAC_ONLINE_PERSISTENT": "1", "CUMAC_TOPOLOGY_SEED": str(seed),
            "CUMAC_TRAFFIC_SEED": str(seed), "CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI": "1",
            "CUMAC_SERVICE_BALANCE_TRACE_CSV": str(out / "service_balance_trace.csv"),
            "CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV": str(out / "traffic_offered_packets.csv"),
            "CUMAC_PACKET_DELAY_SAMPLES": "1",
            "CUMAC_PACKET_DELAY_SAMPLES_CSV": str(out / "packet_delay_samples.csv"),
            "CUMAC_PACKET_OUTCOME_SAMPLES_CSV": str(out / "packet_outcome_samples.csv")}


def run_command(command, out, env, timeout):
    write_json(out / "command.json", {"command": command, "env": {
        k: v for k, v in env.items() if k.startswith("CUMAC_") or k in
        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}})
    with (out / "run.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def evaluate(sim, out, checkpoint, seed, base, smoke, policy=None):
    out.mkdir(parents=True, exist_ok=False)
    total, warmup = (96, 32) if smoke else (4000, 1000)
    socket = f"/tmp/dpa_eval_{os.getpid()}.sock"
    env = eval_env(base, out, seed, socket)
    _extend_ld_library_path(env, sim)
    if policy:
        command = cli("pfq_reference_calibration", {
            "sim-bin": sim, "sim-args": REFERENCE_SIM_ARGS[policy], "sim-cwd": out,
            "topology-seed": seed, "traffic-seed": seed, "traffic-profile": "homogeneous",
            "reference-policy": policy, "warmup-steps": 0, "measure-steps": total,
            "service-balance-trace-first-tti": 1,
            "out-dir": out}) + ["--service-balance-audit"]
        run_command(command, out, env, 4 * 3600)
        return
    command = cli("eval_checkpoint_online", {
        "checkpoint": checkpoint, "socket-path": socket, "episode-horizon": total,
        "max-steps": total, "stats-warmup-steps": warmup, "decode-mode": "argmax",
        "seed": seed, "sample-seed": seed, "device": "cuda", "print-every-steps": 500,
        "save-steps": 1, "sim-bin": sim, "out-dir": out,
        "outcome-samples": out / "packet_outcome_samples.csv",
        "independent-policy-diagnostics": 1, "independent-runtime-candidate-width": 0,
        "independent-headroom-accounting": "ctbs-v1", "independent-non-starvation-guard": "off",
        "slot-repair-max-per-cell": 0, "safe-fill-max-per-cell": 0,
        "candidate-blank-budget-max-per-cell": -1,
        "candidate-decode-demand-slack-bytes": 3000,
        "candidate-marginal-headroom-slack-bytes": -1,
        "candidate-headroom-rank-mode": "stable", "candidate-budgeted-select-mode": "off"})
    with (out / "sim.log").open("w") as log:
        proc = subprocess.Popen([str(sim), *shlex.split(SIM_ARGS)], cwd=out, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            run_command(command, out, env, 4 * 3600)
            if proc.wait(timeout=60) != 0:
                raise RuntimeError("evaluation simulator failed")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def summarize_run(out, smoke):
    service = read_csv(out / "service_balance_trace.csv")
    outcomes = read_csv(out / "packet_outcome_samples.csv")
    decision_file = out / ("eval_steps.jsonl" if (out / "eval_steps.jsonl").exists() else "tti_trace.jsonl")
    decisions = [json.loads(line) for line in decision_file.read_text().splitlines() if line.strip()]
    if (out / "eval_summary.json").exists():
        audit_eval(json.loads((out / "eval_summary.json").read_text()), decisions)
    windows = (("full", 1, 96), ("steady", 33, 96)) if smoke else (("full", 1, 4000), ("steady", 1001, 4000))
    result = {}
    for label, first, last in windows:
        summary = summarize_aligned_window(service_rows=service, outcome_rows=outcomes,
                    decision_rows=decisions, window=WindowSpec(label, first, last))
        selected = [r for r in service if first <= int(r["tti"]) <= last]
        active = sum(int(r["actual_prg_count"]) for r in selected)
        summary["successful_bytes_per_active_prg"] = (sum(int(r["successful_bytes"]) for r in selected) / active if active else 0.)
        result[label] = summary
    write_json(out / "aligned_metrics.json", result)
    return result


def audit_eval(summary, decisions):
    for key in ("independent_deletion_equal", "candidate_full_coverage", "ctbs_reference_label_deletion_equal"):
        if summary[key] is not True:
            raise ValueError(f"evaluation invariant failed: {key}")
    for key in ("ctbs_mcs_mismatch_count", "ctbs_layer_mismatch_count", "ctbs_tbs_mismatch_count"):
        if summary[key] != 0:
            raise ValueError(f"evaluation CTBS mismatch: {key}")
    for row in decisions:
        for key in ("illegal_action_count", "candidate_truncation_count", "action_replay_mismatch_count", "native_reject_count"):
            if float(row[key]) != 0:
                raise ValueError(f"evaluation invariant failed: {key}")


def aggregate(root, sim, train_seeds, test_seeds, smoke):
    result = {"status": "smoke_passed" if smoke else "complete", "smoke_only": smoke,
              "training_seeds": train_seeds, "test_seeds": test_seeds,
              "delta_convention": "full minus comparator (raw units; negative favors full for delay/expiry/BLER)",
              "training": {}, "evaluations": {}, "paired": {}}
    for seed in train_seeds:
        initial = []
        for variant in VARIANTS:
            audit = train_audit(root / "train" / str(seed) / variant, variant, smoke)
            result["training"][f"{seed}/{variant}"] = audit
            initial.append(audit["initialization"])
        for field in ("actor_initial_tensor_sha256", "critic_initial_tensor_sha256"):
            if len({item[field] for item in initial}) != 1:
                raise ValueError(f"unpaired random initialization: {seed} {field}")
    for test in test_seeds:
        reference_dirs = {p: root / "eval" / str(test) / p for p in ("pfq", "rrq")}
        for policy, folder in reference_dirs.items():
            manifest = json.loads((folder / "reference_manifest.json").read_text())
            command = manifest["command"]
            if command[command.index("-b") + 1] != {"pfq": "2", "rrq": "3"}[policy]:
                raise ValueError(f"reference label does not match native scheduler: {policy}")
            result["evaluations"][f"{test}/{policy}"] = summarize_run(folder, smoke)
        for seed in train_seeds:
            for variant in VARIANTS:
                folder = root / "eval" / str(test) / str(seed) / variant
                for reference in reference_dirs.values():
                    validate_seed_artifacts(seed=test, dpa_dir=folder, pfq_dir=reference,
                        expected_sim_sha256=sha(sim), expected_tti_count=96 if smoke else 4000)
                result["evaluations"][f"{test}/{seed}/{variant}"] = summarize_run(folder, smoke)
    for variant in (*VARIANTS[1:], "pfq", "rrq"):
        result["paired"][variant] = {}
        for metric in METRICS:
            matrix = []
            for seed in train_seeds:
                deltas = []
                for test in test_seeds:
                    full = result["evaluations"][f"{test}/{seed}/full"]["steady"][metric]
                    key = f"{test}/{variant}" if variant in reference_dirs else f"{test}/{seed}/{variant}"
                    deltas.append(full - result["evaluations"][key]["steady"][metric])
                matrix.append(deltas)
            result["paired"][variant][metric] = paired_deltas(matrix)
    write_json(root / "final_report.json", result)
    lines = ["# DPA 等预算消融" + ("烟测（不是性能结论）" if smoke else "实验结果"), "",
             "固定 last checkpoint；差值 = Full − comparator。时延、expiry、BLER 的负值有利于 Full。",
             "区间仅基于 3 个训练 seed、条件于固定测试拓扑；15 个单元不视为独立样本。",
             "所有组保留 CTBS/sequential decode；本实验不单独验证解码收益。", "",
             "| comparator | metric | mean Δ | conditional 95% CI |",
             "|---|---|---:|---|"]
    for variant, metrics in result["paired"].items():
        for metric, stats in metrics.items():
            lines.append(f"| {variant} | {metric} | {stats['mean_delta']:.6g} | {stats['conditional_ci95']} |")
    lines += ["", "完整单次结果、训练预算、初始化 hash 与配对差值见 `final_report.json`。",
              "结果需检查 Full 学习质量和跨训练 seed 一致性后解释；单个指标的正差值不等于整体先进性。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-bin", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root, sim = args.out_root.resolve(), args.sim_bin.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT / "training").rglob("*.py"))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    binary_hash = sha(sim)
    env = base_env(args.gpu)
    _extend_ld_library_path(env, sim)
    ldd = subprocess.check_output(["ldd", str(sim)], text=True, env=env)
    local_libraries = {}
    for line in ldd.splitlines():
        tokens = line.split()
        if len(tokens) >= 3 and tokens[1] == "=>" and tokens[2].startswith(str(ROOT) + "/"):
            local_libraries[tokens[2]] = sha(tokens[2])
    (root / "ldd.txt").write_text(ldd)
    write_json(root / "manifest.json", {"argv": sys.argv, "source_sha256": sources,
               "sim_binary": str(sim), "sim_sha256": binary_hash, "started_at": time.time(),
               "local_library_sha256": local_libraries,
               "protocol": "docs/superpowers/plans/2026-09-08-dpa-equal-budget-ablation.md"})
    status = {"status": "running", "pid": os.getpid(), "completed_jobs": [], "started_at": time.time()}
    def progress(job):
        if (sha(sim) != binary_hash or any(sha(ROOT / p) != h for p, h in sources.items())
                or any(sha(p) != h for p, h in local_libraries.items())):
            raise RuntimeError("source/binary changed during campaign; refusing mixed-version evidence")
        status.update(current_job=job, updated_at=time.time())
        write_json(root / "status.json", status)
        print(f"[{time.strftime('%F %T')}] {job}", flush=True)
    seeds = (701,) if args.smoke else TRAIN_SEEDS
    tests = (201,) if args.smoke else TEST_SEEDS
    try:
        for seed in seeds:
            for variant in VARIANTS:
                job = f"train/{seed}/{variant}"
                progress(job)
                out = root / job
                out.mkdir(parents=True, exist_ok=False)
                run_command(train_command(sim, out, variant, seed, smoke=args.smoke), out, env, 24 * 3600)
                write_json(out / "budget_audit.json", train_audit(out, variant, args.smoke))
                status["completed_jobs"].append(job)
        for test in tests:
            for policy in ("pfq", "rrq"):
                job = f"eval/{test}/{policy}"
                progress(job)
                evaluate(sim, root / job, None, test, env, args.smoke, policy)
                status["completed_jobs"].append(job)
            for seed in seeds:
                for variant in VARIANTS:
                    job = f"eval/{test}/{seed}/{variant}"
                    progress(job)
                    ckpt = root / "train" / str(seed) / variant / "ppo_actor_last.pt"
                    evaluate(sim, root / job, ckpt, test, env, args.smoke)
                    status["completed_jobs"].append(job)
        progress("aggregate")
        aggregate(root, sim, seeds, tests, args.smoke)
        status.update(status="smoke_passed" if args.smoke else "complete", current_job=None)
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status["updated_at"] = time.time()
        write_json(root / "status.json", status)


if __name__ == "__main__":
    main()

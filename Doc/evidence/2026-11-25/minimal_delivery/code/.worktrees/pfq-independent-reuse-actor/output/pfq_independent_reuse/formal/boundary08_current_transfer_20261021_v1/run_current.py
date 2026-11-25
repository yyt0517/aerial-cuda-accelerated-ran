"""Minimal frozen-current transfer to the original Boundary08 native environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import torch

from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1 import run as historical
from output.pfq_independent_reuse.formal.boundary08_fast_multiseed_20261010_v1.build import FIELDS
from output.pfq_independent_reuse.formal.boundary75_pfq_ctbs_replacement_20261016_v1 import policy as replacement
from output.pfq_independent_reuse.formal.boundary75_search_acceleration_20261020_v1 import policy as optimized
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1.causal_phy import CausalPhy
from training.stageb_hgraph.directed_reuse import DirectedReuseCoordinator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
OLD = OUT.parent / "boundary08_fast_multiseed_20261010_v1"
ACCEL = OUT.parent / "boundary75_search_acceleration_20261020_v1"
p = historical.p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sources():
    replacement.frozen_check()
    manifest = json.loads((OLD / "build_manifest.json").read_text())
    assert sha(OLD / "simulator") == manifest["simulator_sha256"]
    assert sha(ROOT / "build.corrected_uniform_v1/cuMAC/src/libcumac.so") == manifest["lib_sha256"]
    confirmation = json.loads((ACCEL / "confirmation.json").read_text())
    assert confirmation["passed"] and confirmation["full_trace_and_every_timed_outcome_exact"]
    for name, digest in confirmation["source_sha256"].items():
        assert sha(name) == digest, name
    files = [
        __file__, OUT / "protocol.json", OLD / "simulator", OLD / "run.py",
        ROOT / "build.corrected_uniform_v1/cuMAC/src/libcumac.so",
        replacement.__file__, optimized.__file__, ACCEL / "phy.py",
        ACCEL / "candidates.py", ACCEL / "physics_kernel.so",
        replacement.prior.CHECKPOINT,
    ]
    assert sha(replacement.prior.CHECKPOINT) == sha(historical.CHECKPOINT)
    return {str(Path(name)): sha(name) for name in files}


def verify_sources(record):
    replacement.frozen_check()
    for name, digest in record.items():
        assert sha(name) == digest, name


def reference_smoke(seed, gpu):
    """Run the unchanged historical PFQ runner with process-local observation tracing."""
    source_hashes = sources()
    for key in list(os.environ):
        if key.startswith("CUMAC_"):
            del os.environ[key]
    name = f"smoke_reference_seed{seed}_128"
    alias = OUT / "simulator"
    if not alias.exists():
        alias.symlink_to(OLD / "simulator")
    assert alias.resolve() == (OLD / "simulator").resolve()
    real_client = p.BridgeEnvClient
    rows = {key: [] for key in ("hashes", "actions", "ue", "prg", "goodput")}

    class TracedClient:
        def __init__(self, *args, **kwargs):
            self.client = real_client(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.client, name)

        def reset(self, *args, **kwargs):
            answer = self.client.reset(*args, **kwargs)
            rows["hashes"].append(p.digest_arrays(answer[0]))
            return answer

        def step(self, ue, prg):
            rows["ue"].append(np.asarray(ue).copy())
            rows["prg"].append(np.asarray(prg).copy())
            rows["actions"].append(p.decode_type0_teacher_action(
                ue, prg, n_cell=3, n_prg=17).reshape(-1).copy())
            answer = self.client.step(ue, prg)
            rows["hashes"].append(p.digest_arrays(answer[0]))
            rows["goodput"].append(float(np.asarray(answer[3]["actual_ue_goodput_bytes"]).sum()))
            return answer

    historical.OUT = OUT
    p.BridgeEnvClient = TracedClient
    try:
        historical.run(argparse.Namespace(
            arm="pfq0", name=name, seed=seed, gpu=gpu, warmup=0,
            measured=128, followup=0, reference="", summarize=False))
    finally:
        p.BridgeEnvClient = real_client
    np.savez_compressed(OUT / name / "prefix.npz", **{k: np.asarray(v) for k, v in rows.items()})
    expected = np.fromfile(OLD / f"seed{seed}_pfq0_2000/metrics.bin", dtype="<f8")
    actual = np.fromfile(OUT / name / "metrics.bin", dtype="<f8")
    np.testing.assert_array_equal(actual, expected[:128 * 36 * len(FIELDS)])
    verify_sources(source_hashes)
    write(OUT / name / "validation.json", dict(
        passed=True, original_runner_unchanged=True, historical_service_prefix_exact=128,
        source_sha256=source_hashes, trace_states=len(rows["hashes"]),
        trace_actions=len(rows["actions"])))
    print("REFERENCE_SMOKE_PASSED", name, flush=True)


def run(seed, gpu, smoke=False):
    source_hashes = sources()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    horizon, warmup, measured, followup = (136, 128, 8, 0) if smoke else (6400, 4000, 2000, 400)
    name = f"smoke_current_seed{seed}_136" if smoke else f"seed{seed}_current_2000"
    out = OUT / name
    out.mkdir(exist_ok=False)
    reference = OLD / f"seed{seed}_pfq0_2000"
    launch = json.loads((reference / "launch.json").read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUMAC_")}
    env.update(launch["env"])
    origin = str(Path(launch["env"]["CUMAC_DIRECTED_SNAPSHOT_PATH"]).parent)
    for key, value in list(env.items()):
        if key.startswith("CUMAC_") and value.startswith(origin):
            env[key] = str(out) + value[len(origin):]
    assert not any(key.startswith("CUMAC_LAST_SLOT_DROP") for key in env)
    assert env["CUMAC_TRAFFIC_PROFILE"] == "homogeneous"
    assert float(env["CUMAC_TRAFFIC_ARRIVAL_RATE"]) == .8
    assert env["CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES"] == "0"
    socket = f"/tmp/b08_current_{os.getpid()}.sock"
    env.update(CUMAC_ONLINE_SOCKET=socket, CUDA_VISIBLE_DEVICES=str(gpu),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(OLD / "simulator")] + launch["command"][1:]
    linked = subprocess.check_output(["ldd", str(OLD / "simulator")], env=env, text=True)
    cumac_resolution = next(line.strip() for line in linked.splitlines() if "libcumac.so =>" in line)
    assert cumac_resolution.startswith(
        "libcumac.so => " + str(ROOT / "build.corrected_uniform_v1/cuMAC/src/libcumac.so") + " (")
    write(out / "launch.json", dict(
        seed=seed, arm="current", horizon=horizon, warmup=warmup, measured=measured,
        followup=followup, command=command,
        env={k: v for k, v in env.items() if k.startswith("CUMAC_") or k == "CUDA_VISIBLE_DEVICES"},
        source_sha256=source_hashes, target_success_bytes_per_tti=[2400.] * 36,
        inherited_environment=str(reference / "launch.json"),
        runtime_libcumac_resolution=cumac_resolution,
        proactive_drop=False, timing_scope="Shared-host decision wall; includes snapshot read and frozen current policy/search; excludes CTBS IPC and native client.step."))
    coordinator = DirectedReuseCoordinator()
    coordinator.load_state_dict(torch.load(
        replacement.prior.CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    coordinator.eval()
    builder = p.FeatureSnapshotBuilder(p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges = p.SparseEdgeGenerator(p.EdgeGeneratorConfig(
        ue_prg_topk=4, ue_prg_diversity_extra=0, candidate_mode="all_demand_ues",
        include_pfq_score_anchor=False, include_candidate_risk_features=False))
    ema = np.zeros(36)
    target = np.full(36, 2400.)
    rows = {key: [] for key in ("hashes", "actions", "ue", "prg", "goodput")}
    weights_history, decisions, rounds = [], [], []
    started = time.monotonic()
    client = None
    with (out / "sim.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = p.BridgeEnvClient(p.BridgeEnvConfig(
                socket_path=socket, connect_timeout_s=60, enable_ctbs=True))
            obs, _, info = client.reset(seed=seed, episode_horizon=horizon)
            assert info["harq_enabled"] == 0
            config = p.PhyTbsConfig.from_vector(info["ctbs_config"])
            if smoke:
                class SmokePhy(optimized.OptimizedPhy):
                    def evaluate(self, actions):
                        predicted = super().evaluate(actions)
                        original = CausalPhy.evaluate(self, actions)
                        for key in predicted:
                            np.testing.assert_array_equal(predicted[key], original[key], err_msg=key)
                        self.audit_calls += 1
                        return predicted
                phy = SmokePhy(config)
            else:
                phy = optimized.OptimizedPhy(config)
            rows["hashes"].append(p.digest_arrays(obs))
            for tti in range(1, horizon + 1):
                ctbs = p._ctbs_inputs(client, info, config)
                assert ctbs.tti == tti - 1
                prediction = None
                if tti <= warmup:
                    builder.build(obs)
                    ue = obs["teacher_action_ue_select"].copy()
                    prg = obs["teacher_action_prg_alloc"].copy()
                    chosen = p.decode_type0_teacher_action(ue, prg, n_cell=3, n_prg=17).reshape(3, 17)
                else:
                    start = time.monotonic()
                    snapshot = historical.old.read_snapshot(out / "pre_action.bin", ctbs.tti)
                    draft, keep, ue, prg, *_ = replacement.policy(
                        obs, ctbs, snapshot, builder, edges, None, coordinator, phy)
                    weights = optimized.service_weights(ema, target)
                    chosen, prediction, record, selected_rounds = optimized.select(
                        draft, keep, phy, obs, ctbs, weights)
                    weights_history.append(weights.copy())
                    if not np.array_equal(chosen, keep):
                        ue, prg = replacement.prior.encode(chosen, obs)
                    record.update(tti=tti, decision_wall_s=time.monotonic() - start)
                    decisions.append(record)
                    rounds.extend(dict(tti=tti, **item) for item in selected_rounds)
                    assert record["converged"] and not record["cap_reached"]
                rows["actions"].append(chosen.reshape(-1).copy())
                rows["ue"].append(np.asarray(ue).copy())
                rows["prg"].append(np.asarray(prg).copy())
                obs, _, _, info = client.step(ue, prg)
                actual = p._actual_chosen_from_mask(
                    info["actual_ue_prg_mask"], np.asarray(ctbs.serving_cell), n_cell=3, n_prg=17)
                np.testing.assert_array_equal(actual, chosen.reshape(-1))
                if prediction is not None:
                    active = prediction["count"] > 0
                    for key, label in (("mcs", "actual_ue_mcs"), ("layer", "actual_ue_layer"),
                                       ("capacity", "actual_ue_tbs_capacity_bytes")):
                        np.testing.assert_array_equal(prediction[key][active], np.asarray(info[label])[active])
                    np.testing.assert_array_equal(prediction["dequeue"], info["actual_ue_served_bytes"])
                good = np.asarray(info["actual_ue_goodput_bytes"])
                ema = .995 * ema + .005 * good
                rows["goodput"].append(float(good.sum()))
                rows["hashes"].append(p.digest_arrays(obs))
                if tti % 100 == 0 or tti in (warmup + 1, horizon):
                    write(out / "progress.json", dict(
                        tti=tti, horizon=horizon, elapsed_s=time.monotonic() - started,
                        phase="warmup" if tti <= warmup else "current",
                        decisions=len(decisions), phy_audit_calls=phy.audit_calls))
                    pd.DataFrame(decisions).to_csv(out / "decisions.csv", index=False)
                    pd.DataFrame(rounds).to_csv(out / "rounds.csv", index=False)
                    print(name, tti, round(time.monotonic() - started, 2), flush=True)
        finally:
            if client:
                client.close()
            try:
                exit_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                exit_code = proc.wait(timeout=15)
            assert exit_code == 0, exit_code
    np.savez_compressed(out / "prefix.npz", **{k: np.asarray(v) for k, v in rows.items()})
    np.save(out / "weights.npy", np.asarray(weights_history))
    pd.DataFrame(replacement.AUDIT).to_csv(out / "alignment_audit.csv", index=False)
    data = np.fromfile(out / "metrics.bin", dtype="<f8").reshape(-1, len(FIELDS))
    old_data = np.fromfile(reference / "metrics.bin", dtype="<f8").reshape(-1, len(FIELDS))
    assert len(data) == horizon * 36
    np.testing.assert_array_equal(data[:warmup * 36], old_data[:warmup * 36])
    service = pd.DataFrame(data, columns=FIELDS)
    np.testing.assert_array_equal(service.groupby("tti").successful_bytes.sum(), rows["goodput"])
    exogenous = {name: sha(out / name) for name in ("traffic_offered_packets.csv", "ue_topology.csv")}
    smoke_exact = None
    if smoke:
        saved = np.load(OUT / f"smoke_reference_seed{seed}_128/prefix.npz")
        for key in rows:
            count = 129 if key == "hashes" else 128
            np.testing.assert_array_equal(np.asarray(rows[key])[:count], saved[key], err_msg=key)
        np.testing.assert_array_equal(data[:128 * 36], np.fromfile(
            OUT / f"smoke_reference_seed{seed}_128/metrics.bin", dtype="<f8").reshape(-1, len(FIELDS)))
        assert len(decisions) == 8 and phy.audit_calls > 0
        smoke_exact = True
    else:
        for name, digest in exogenous.items():
            assert digest == sha(reference / name), name
        assert len(decisions) == 2400 and phy.audit_calls > 0
        assert [item["tti"] for item in replacement.AUDIT] == list(range(4001, 6401))
    verify_sources(source_hashes)
    execution = dict(
        seed=seed, arm="current", measured_ttis=measured, warmup_ttis=warmup,
        followup_ttis=followup, horizon=horizon, elapsed_native_s=time.monotonic() - started,
        actions_exact=horizon, exogenous_sha256=exogenous, timing={},
        interference_tx_mean_scaled=None, interference_prg_weighted_scaled=None,
        source_sha256=source_hashes, original_phy_audit_calls=phy.audit_calls)
    write(out / "execution.json", execution)
    write(out / "validation.json", dict(
        passed=True, native_actions_exact=horizon, predicted_tbs_mcs_layer_dequeue_exact=horizon - warmup,
        historical_pfq_service_prefix_exact_ttis=warmup,
        original_fast_reference_smoke_exact=smoke_exact,
        traffic_and_topology_exact_to_historical_pfq=not smoke,
        no_proactive_drop=True, uniform_configured_target_bytes_per_tti=2400.,
        frozen_policy_and_sources_unchanged=True, full_search_converged=True,
        full_search_decisions=len(decisions),
        original_phy_audit_calls=phy.audit_calls,
        caveat="Historical DPA used its own BC warmup; this arm uses PFQ warmup. The same4001 policy state is not claimed. Historical full state/action/channel traces were not saved.",
    ))
    print("CURRENT_TRANSFER_VALIDATED", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reference-smoke", "current-smoke", "pilot"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    assert args.seed in range(41, 51)
    if args.mode == "reference-smoke":
        reference_smoke(args.seed, args.gpu)
    else:
        run(args.seed, args.gpu, smoke=args.mode == "current-smoke")

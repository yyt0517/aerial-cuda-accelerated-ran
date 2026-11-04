def run(args):
    source_hashes = sources()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    warmup, settling, measured, followup = (128, 0, 8, 0) if args.smoke else (4000, 4000, 20000, 400)
    horizon = warmup + settling + measured + followup
    name = f"smoke_seed{args.seed}_{args.arm}_136" if args.smoke else f"seed{args.seed}_{args.arm}_20000"
    out = OUT / name
    out.mkdir(exist_ok=False)
    original = r.OLD / f"seed{args.seed}_pfq0_2000"
    launch = json.loads((original / "launch.json").read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUMAC_")}
    env.update(launch["env"])
    origin = str(Path(launch["env"]["CUMAC_DIRECTED_SNAPSHOT_PATH"]).parent)
    for key, value in list(env.items()):
        if key.startswith("CUMAC_") and value.startswith(origin):
            env[key] = str(out) + value[len(origin):]
    env.update(CUMAC_ONLINE_SOCKET=f"/tmp/b08_corrected_{os.getpid()}.sock",
               CUDA_VISIBLE_DEVICES=str(args.gpu), CUMAC_SHADOW_FADING_STD_DB="3.0",
               CUMAC_FRESH_OBSERVATION_PATH=str(out / "fresh_observation.bin"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    assert not any(k.startswith("CUMAC_LAST_SLOT_DROP") for k in env)
    assert env["CUMAC_TRAFFIC_PROFILE"] == "homogeneous"
    assert env["CUMAC_TRAFFIC_ARRIVAL_RATE"] == ".8"
    assert env["CUMAC_PFQ_DEMAND_CAP_SLACK_BYTES"] == "0"
    command = [str(OUT / "simulator")] + launch["command"][1:]
    linked = subprocess.check_output(["ldd", command[0]], env=env, text=True)
    cumac = next(x.strip() for x in linked.splitlines() if "libcumac.so =>" in x)
    assert cumac.startswith("libcumac.so => " + str(r.ROOT / "build.corrected_uniform_v1/cuMAC/src/libcumac.so") + " (")
    r.write(out / "launch.json", dict(
        seed=args.seed, arm=args.arm, horizon=horizon, warmup=warmup, measured=measured,
        followup=followup, settling=settling, measurement_first_tti=warmup+settling+1,
        measurement_last_tti=warmup+settling+measured, args=vars(args), command=command,
        env={k: v for k, v in env.items() if k.startswith("CUMAC_") or k == "CUDA_VISIBLE_DEVICES"},
        source_sha256=source_hashes, simulator_sha256=r.sha(command[0]),
        runtime_libcumac_resolution=cumac, build_manifest=str(OUT / "native_build_manifest.json"),
        common_pfq_warmup=True, fresh_preaction_all_inputs=True,
        target_success_bytes_per_tti=[2400.] * 36, proactive_drop=False))
    actor = coordinator = None
    if False:  # No learned modules in component ablation.
        coordinator = r.DirectedReuseCoordinator()
        coordinator.load_state_dict(torch.load(
            r.replacement.prior.CHECKPOINT, map_location="cpu", weights_only=False)["model"])
        coordinator.eval()
    if args.arm == "full":
        actor, _ = r.historical.old._load_actor_checkpoint(r.historical.old.CHECKPOINT, torch.device("cpu"))
    builder = r.p.FeatureSnapshotBuilder(r.p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
    edges = r.p.SparseEdgeGenerator(r.p.EdgeGeneratorConfig(
        ue_prg_topk=4, ue_prg_diversity_extra=0, candidate_mode="all_demand_ues",
        include_pfq_score_anchor=False, include_candidate_risk_features=False))
    ema = np.zeros(36)
    rows = {k: [] for k in ["input_hashes", "fresh_input_tti", "ctbs_hashes",
                            "channel_fingerprints", "precoder_fingerprints",
                            "actions", "ue", "prg", "goodput"]}
    weights_history, decisions, rounds, states = [], [], [], []
    sample_ttis = {129, 132, 136} if args.smoke else {4000, 4001, 8000, 8001, 18000, 28000}
    client = None
    started = time.monotonic()
    with (out / "sim.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = r.p.BridgeEnvClient(r.p.BridgeEnvConfig(
                socket_path=env["CUMAC_ONLINE_SOCKET"], connect_timeout_s=60, enable_ctbs=True))
            previous_obs, _, info = client.reset(seed=args.seed, episode_horizon=horizon)
            assert info["harq_enabled"] == 0
            config = r.p.PhyTbsConfig.from_vector(info["ctbs_config"])
            phy = r.optimized.OptimizedPhy(config)
            for tti in range(1, horizon + 1):
                ctbs = r.p._ctbs_inputs(client, info, config)
                assert ctbs.tti == tti - 1
                obs = fresh_observation.read(out / "fresh_observation.bin", previous_obs, ctbs)
                snapshot = r.historical.old.read_snapshot(out / "pre_action.bin", ctbs.tti)
                np.testing.assert_array_equal(obs["obs_ue_features"][:, 0], snapshot["queue"])
                # Actual cell association must agree in CTBS, fresh masks and fresh UE features.
                serving = np.asarray(ctbs.serving_cell)
                np.testing.assert_array_equal(
                    obs["action_mask_cell_ue"], np.arange(3)[:, None] == serving[None, :])
                np.testing.assert_array_equal(obs["action_mask_ue"], snapshot["queue"] > 0)
                canonical_ctbs = asdict(ctbs)
                canonical_ctbs.pop("snapshot_id")
                rows["input_hashes"].append(r.p.digest_arrays(obs))
                rows["fresh_input_tti"].append(ctbs.tti)
                rows["ctbs_hashes"].append(hashlib.sha256(json.dumps(
                    canonical_ctbs, sort_keys=True).encode()).hexdigest())
                rows["channel_fingerprints"].append(fingerprint(snapshot["h"]))
                rows["precoder_fingerprints"].append(fingerprint(snapshot["w"]))
                start = time.monotonic()
                record = {}
                search_prediction = None
                if tti <= warmup or args.arm == "pfq0":
                    builder.build(obs)
                    ue = obs["teacher_action_ue_select"].copy()
                    prg = obs["teacher_action_prg_alloc"].copy()
                    chosen = r.p.decode_type0_teacher_action(ue, prg, n_cell=3, n_prg=17).reshape(3, 17)
                elif args.arm == "full":
                    graph = edges.build_graph(builder.build(obs))
                    packed = r.historical.old.pack_prg_candidates(
                        graph, max_candidates=r.historical.old.resolve_full_candidate_width(graph), fixed_width=True)
                    with torch.no_grad():
                        _, decoded = r.historical.old.select_independent_action(
                            actor, graph, packed, mode="greedy", ctbs_headroom_inputs=ctbs)
                    ue = decoded.action_ue_select.numpy()
                    prg = decoded.action_prg_alloc.numpy().copy()
                    draft = r.p.decode_type0_teacher_action(ue, prg, n_cell=3, n_prg=17).reshape(3, 17)
                    _, chosen, ue, prg, _, token = r.replacement.coordinate(
                        draft, ue, prg, obs, ctbs, snapshot, coordinator, phy)
                    record["coordinator_token"] = token
                else:
                    draft, keep, ue, prg, *_ = r.replacement.policy(
                        obs, ctbs, snapshot, builder, edges, None, coordinator, phy)
                    weights = r.optimized.service_weights(ema, np.full(36, 2400.))
                    chosen, search_prediction, record, selected_rounds = r.optimized.select(
                        draft, keep, phy, obs, ctbs, weights)
                    weights_history.append(weights.copy())
                    if not np.array_equal(chosen, keep):
                        ue, prg = r.replacement.prior.encode(chosen, obs)
                    rounds.extend(dict(tti=tti, **x) for x in selected_rounds)
                    assert record["converged"] and not record["cap_reached"]
                if tti > warmup:
                    record.update(tti=tti, decision_wall_s=time.monotonic() - start)
                    decisions.append(record)
                prediction = discrete_prediction(chosen, ctbs, snapshot, phy.tbs)
                if search_prediction is not None:
                    for key in prediction:
                        np.testing.assert_array_equal(prediction[key], search_prediction[key])
                sample = None
                if tti in sample_ttis:
                    sample = dict(tti=tti, obs=obs, ctbs=ctbs, snapshot=snapshot, chosen=chosen.copy(),
                                  ema=ema.copy(), prediction=prediction)
                    (out / f"snapshot_{tti}.bin").write_bytes((out / "pre_action.bin").read_bytes())
                rows["actions"].append(chosen.reshape(-1).copy())
                rows["ue"].append(np.asarray(ue).copy())
                rows["prg"].append(np.asarray(prg).copy())
                previous_obs, _, _, info = client.step(ue, prg)
                actual = r.p._actual_chosen_from_mask(info["actual_ue_prg_mask"], serving, n_cell=3, n_prg=17)
                np.testing.assert_array_equal(actual, chosen.reshape(-1))
                active = prediction["count"] > 0
                for key, label in [("mcs", "actual_ue_mcs"), ("layer", "actual_ue_layer"),
                                   ("capacity", "actual_ue_tbs_capacity_bytes")]:
                    np.testing.assert_array_equal(prediction[key][active], np.asarray(info[label])[active])
                np.testing.assert_array_equal(prediction["dequeue"], info["actual_ue_served_bytes"])
                good = np.asarray(info["actual_ue_goodput_bytes"])
                ema = .995 * ema + .005 * good
                rows["goodput"].append(float(good.sum()))
                if sample is not None:
                    sample["native_info"] = info
                    states.append(sample)
                    torch.save(states, out / "states.pt")
                if tti % 100 == 0 or tti in (warmup + 1, horizon):
                    r.write(out / "progress.json", dict(
                        tti=tti, horizon=horizon, elapsed_s=time.monotonic() - started,
                        decisions=len(decisions), phy_audit_calls=phy.audit_calls))
                    if tti % 1000 == 0 or tti == horizon:
                        pd.DataFrame(decisions).to_csv(out / "decisions.csv", index=False)
                        pd.DataFrame(rounds).to_csv(out / "rounds.csv", index=False)
                    print(name, tti, round(time.monotonic() - started, 3), flush=True)
        finally:
            if client:
                client.close()
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                code = proc.wait(timeout=15)
            assert code == 0, code
    rows["hashes"] = rows["input_hashes"]
    np.savez_compressed(out / "prefix.npz", **{k: np.asarray(v) for k, v in rows.items()})
    np.save(out / "weights.npy", np.asarray(weights_history))
    pd.DataFrame(r.replacement.AUDIT).to_csv(out / "alignment_audit.csv", index=False)
    data = np.fromfile(out / "metrics.bin", dtype="<f8").reshape(horizon, 36, len(FIELDS))
    np.testing.assert_array_equal(data[:, :, FIELDS.index("successful_bytes")].sum(1), rows["goodput"])
    np.testing.assert_array_equal(rows["fresh_input_tti"], np.arange(horizon))
    assert len(states) == len(sample_ttis) and len(decisions) == horizon - warmup
    physical_checks = check_samples(out, states, data)
    r.verify_sources(source_hashes)
    exogenous = {n: r.sha(out / n) for n in ["traffic_offered_packets.csv", "ue_topology.csv"]}
    r.write(out / "execution.json", dict(
        seed=args.seed, arm=args.arm, measured_ttis=measured, warmup_ttis=warmup, followup_ttis=followup,
        settling_ttis=settling, measurement_first_tti=warmup+settling+1,
        measurement_last_tti=warmup+settling+measured, horizon=horizon, elapsed_native_s=time.monotonic() - started, actions_exact=horizon,
        timing={}, exogenous_sha256=exogenous, source_sha256=source_hashes,
        interference_tx_mean_scaled=None, interference_prg_weighted_scaled=None,
        original_phy_audit_calls=phy.audit_calls))
    r.write(out / "validation.json", dict(
        passed=True, native_actions_exact=horizon, native_mcs_layer_tbs_dequeue_exact=horizon,
        fresh_observation_snapshot_queue_exact=horizon, fresh_ctbs_tti_binding_exact=horizon,
        current_association_mask_exact=horizon, common_pfq_warmup=warmup,
        frozen_sources_unchanged=True, full_search_decisions=horizon - warmup if args.arm == "current" else 0,
        full_search_converged=True if args.arm == "current" else None,
        physical_samples=physical_checks, no_proactive_drop=True,
        corrected_actual_serving_precoder=True, shadow_fading_std_db=3.0))
    print("CORRECTED_RUN_VALIDATED", name, flush=True)

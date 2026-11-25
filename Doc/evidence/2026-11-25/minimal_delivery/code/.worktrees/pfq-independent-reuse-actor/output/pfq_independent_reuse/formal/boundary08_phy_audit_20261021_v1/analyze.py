"""Independent direct-filter PHY audit on captured, fixed pre-action states."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from output.pfq_independent_reuse.formal.boundary08_current_transfer_20261021_v1 import run_current as r
from output.pfq_independent_reuse.formal.boundary08_directed_coordination_20261008_v1.causal_phy import CausalPhy
from output.pfq_independent_reuse.formal.boundary08_directed_attention_20261008_v1.precision import native_bler

OUT = Path(__file__).resolve().parent


def direct_filter(snapshot, action, serving, layer_caps, force_neighbors=False):
    """Wiener receiver, explicit desired / intra-layer / inter-cell / noise powers.

    This does not call sinr_bank, use the 8-mask bank, or the cached C++ kernel.
    Each layer's g solves (S S^H + R)g = s_l; SINR is computed from output powers.
    As in native, all four serving precoder columns enter this receiver; averaging
    then includes only the selected minimum layer-cap columns.
    """
    per_ue = np.full(36, np.nan)
    details = []
    for u in range(36):
        c = int(serving[u])
        allocated = np.flatnonzero(action[c] == u)
        if not len(allocated):
            continue
        layers = int(np.min(np.asarray(layer_caps)[u, allocated]))
        gammas = []
        for p in allocated:
            h = snapshot["h"][u, p]
            s = h[c] @ snapshot["w"][u, p]
            inter_cov = np.zeros((4, 4), dtype=np.complex128)
            neighbors = []
            for other in range(3):
                if other != c and (force_neighbors or action[other, p] >= 0):
                    inter_cov += h[other] @ h[other].conj().T
                    neighbors.append(other)
            noise_cov = snapshot["noise"] * np.eye(4)
            receive_cov = s @ s.conj().T + inter_cov + noise_cov
            receiver = np.linalg.solve(receive_cov, s)
            for layer in range(layers):
                g = receiver[:, layer]
                all_layer_power = np.abs(g.conj() @ s) ** 2
                desired = float(all_layer_power[layer])
                intra = float(np.delete(all_layer_power, layer).sum())
                inter = float(np.real(g.conj() @ inter_cov @ g))
                noise = float(np.real(g.conj() @ noise_cov @ g))
                gamma = desired / (intra + inter + noise)
                gammas.append(gamma)
                details.append(dict(
                    ue_id=u, serving_cell=c, prg=int(p), layer=layer,
                    active_other_cells=len(neighbors), desired_output_power=desired,
                    intralayer_output_interference=intra, intercell_output_interference=inter,
                    output_noise=noise, layer_linear_sinr=gamma,
                    serving_channel_trace=float(np.vdot(h[c], h[c]).real),
                    serving_signal_trace=float(np.vdot(s, s).real),
                    interferer_covariance_trace=float(np.trace(inter_cov).real),
                    noise_covariance_trace=float(np.trace(noise_cov).real)))
        per_ue[u] = 10 * np.log10(np.mean(gammas))
    return per_ue, details


def main():
    states = torch.load(OUT / "states.pt", map_location="cpu", weights_only=False)
    assert [x["tti"] for x in states] == [4001, 4100, 5000]
    assert json.loads((OUT / "capture_validation.json").read_text())["passed"]
    native = np.fromfile(OUT / "replay_seed41/metrics.bin", dtype="<f8").reshape(-1, 36, len(r.FIELDS))
    hist = np.fromfile(r.OLD / "seed41_pfq0_2000/metrics.bin", dtype="<f8").reshape(-1, 36, len(r.FIELDS))
    fields = list(r.FIELDS)
    summaries, ue_rows, power_rows, checks, ctbs_rows, sensitivity, freshness = [], [], [], [], [], [], []
    phy = r.optimized.OptimizedPhy(states[0]["ctbs"].config)
    for state in states:
        tti, snapshot, ctbs = state["tti"], state["snapshot"], state["ctbs"]
        assert snapshot["tti"] == ctbs.tti == tti - 1
        np.testing.assert_array_equal(snapshot["queue"], native[tti - 1, :, fields.index("pre_queue_bytes")])
        phy.prepare(snapshot, ctbs)
        actions = np.stack(list(state["actions"].values()))
        before = {k: v.copy() for k, v in snapshot.items() if isinstance(v, np.ndarray)}
        pred = phy.evaluate(actions)
        independent_original = CausalPhy(ctbs.config)
        independent_original.prepare(snapshot, ctbs)
        original = independent_original.evaluate(actions)
        for key in pred:
            np.testing.assert_array_equal(pred[key], original[key])
        repeated = phy.evaluate(actions[::-1])
        physical_keys = ["count", "layer", "mcs", "capacity", "dequeue", "sinr_db", "interference"]
        for key in physical_keys:
            np.testing.assert_array_equal(pred[key], repeated[key][::-1])
        for k, v in before.items():
            np.testing.assert_array_equal(v, snapshot[k])
        all_mcs = [ctbs.reconstructed_mcs_for_ue(u) for u in range(36)]
        obs_queue = np.asarray(state["obs"]["obs_ue_features"])[:, 0]
        freshness.append(dict(
            tti=tti, previous_returned_obs_queue_sum=float(obs_queue.sum()),
            current_snapshot_queue_sum=float(snapshot["queue"].sum()),
            queue_difference_sum=float((snapshot["queue"] - obs_queue).sum()),
            changed_ues=int(np.count_nonzero(snapshot["queue"] != obs_queue)),
            prior_zero_now_positive_ues=int(np.sum((obs_queue == 0) & (snapshot["queue"] > 0)))))
        for u in range(36):
            ctbs_rows.append(dict(tti=tti, ue_id=u, serving_cell=ctbs.serving_cell[u],
                                  wb_sinr_first_layer_db=ctbs.wb_sinr_first_layer_db[u],
                                  olla_delta=ctbs.olla_delta[u], olla_delta_up=ctbs.olla_delta_up[u],
                                  olla_delta_down=ctbs.olla_delta_down[u], prior_tb_err=ctbs.prior_tb_err[u],
                                  reconstructed_mcs=all_mcs[u], queue_bytes=snapshot["queue"][u],
                                  returned_obs_queue_bytes=float(obs_queue[u]),
                                  layer_caps=json.dumps(ctbs.prg_layer_cap[u])))
        direct_results = {}
        for i, (label, action) in enumerate(state["actions"].items()):
            direct, details = direct_filter(snapshot, action, ctbs.serving_cell, ctbs.prg_layer_cap)
            active = pred["count"][i] > 0
            error = np.abs(direct[active] - pred["sinr_db"][i, active])
            assert float(error.max(initial=0)) < 1e-7, (tti, label, error.max())
            direct_results[label] = direct
            power_rows.extend(dict(tti=tti, action=label, **row) for row in details)
            direct_bler = native_bler(direct[active], np.asarray(all_mcs)[active])
            native_rows = native[tti - 1] if label == "final" else (
                hist[tti - 1] if tti == 4001 and label == "pfq" else None)
            check = dict(tti=tti, action=label, direct_vs_bank_max_abs_sinr_db=float(error.max(initial=0)),
                         original_vs_optimized_all_exact=True, reverse_candidate_physical_exact=True,
                         reverse_candidate_bler_max_difference=float(np.abs(pred["bler"][i] - repeated["bler"][::-1][i]).max()),
                         native_comparison_available=native_rows is not None)
            if native_rows is not None:
                for key, column in [("count", "actual_prg_count"), ("mcs", "mcs"), ("layer", "layer")]:
                    np.testing.assert_array_equal(pred[key][i, active], native_rows[active, fields.index(column)])
                np.testing.assert_array_equal(pred["dequeue"][i], native_rows[:, fields.index("scheduled_dequeue_bytes")])
                native_sinr = native_rows[active, fields.index("phy_sinr_db_cpu")]
                native_b = native_rows[active, fields.index("predicted_bler_cpu")]
                check.update(
                    native_discrete_fields_exact=True,
                    direct_vs_native_max_abs_sinr_db=float(np.abs(direct[active] - native_sinr).max(initial=0)),
                    direct_vs_native_max_abs_bler=float(np.abs(direct_bler - native_b).max(initial=0)),
                    native_recorded_curve_max_abs_error=float(np.abs(native_bler(native_sinr, np.asarray(all_mcs)[active]) - native_b).max(initial=0)))
                if label == "final":
                    for key, column in [("capacity", "actual_ue_tbs_capacity_bytes"),
                                        ("mcs", "actual_ue_mcs"), ("layer", "actual_ue_layer")]:
                        np.testing.assert_array_equal(pred[key][i, active], np.asarray(state["native_info"][column])[active])
            checks.append(check)
            for u in range(36):
                row = dict(tti=tti, action=label, ue_id=u, serving_cell=ctbs.serving_cell[u],
                           queue_bytes=float(snapshot["queue"][u]),
                           **{k: float(v[i, u]) for k, v in pred.items()},
                           direct_sinr_db=float(direct[u]) if active[u] else None)
                ue_rows.append(row)
            summaries.append(dict(
                tti=tti, action=label, used_cell_prgs=int((action >= 0).sum()),
                used_frequency_prgs=int((action >= 0).any(axis=0).sum()), active_ues=int(active.sum()),
                mean_active_ue_sinr_db=float(direct[active].mean()), mean_active_mcs=float(np.asarray(all_mcs)[active].mean()),
                capacity_bytes=float(pred["capacity"][i].sum()), dequeue_bytes=float(pred["dequeue"][i].sum()),
                native_curve_expected_bytes=float((pred["dequeue"][i, active] * (1 - direct_bler)).sum()),
                mean_interference_trace=float(pred["interference"][i, active].mean()),
                mean_active_neighbors=float(np.mean([x["active_other_cells"] for x in details]))))
        forced, details = direct_filter(snapshot, state["actions"]["final"], ctbs.serving_cell,
                                        ctbs.prg_layer_cap, force_neighbors=True)
        use = np.isfinite(forced)
        gain = direct_results["final"][use] - forced[use]
        checks.append(dict(
            tti=tti, action="final_vs_all_other_cells_forced_on", own_prgs_ue_layers_mcs_channel_queue_fixed=True,
            mean_sinr_gain_db=float(gain.mean()), min_gain_db=float(gain.min()), max_gain_db=float(gain.max()),
            final_mean_db=float(direct_results["final"][use].mean()), forced_mean_db=float(forced[use].mean())))
        power_rows.extend(dict(tti=tti, action="final_all_neighbors_on", **row) for row in details)
        # Offline one-variable sensitivity requested after the native-faithful audit.
        # Preserve actual action and original selected layers/caps, MCS, queue and H.
        corrected = dict(snapshot)
        serving_h = snapshot["h"][np.arange(36), :, np.asarray(ctbs.serving_cell)]
        _, _, vh = np.linalg.svd(serving_h)
        corrected["w"] = vh.conj().swapaxes(-1, -2)
        for i, (label, action) in enumerate(state["actions"].items()):
            value, _ = direct_filter(corrected, action, ctbs.serving_cell, ctbs.prg_layer_cap)
            use = pred["count"][i] > 0
            b = native_bler(value[use], np.asarray(all_mcs)[use])
            old_b = native_bler(direct_results[label][use], np.asarray(all_mcs)[use])
            sensitivity.append(dict(
                tti=tti, action=label, active_ues=int(use.sum()),
                native_w_mean_sinr_db=float(direct_results[label][use].mean()),
                actual_serving_svd_mean_sinr_db=float(value[use].mean()),
                same_ue_mean_delta_sinr_db=float((value[use] - direct_results[label][use]).mean()),
                native_w_expected_bytes=float((pred["dequeue"][i, use] * (1 - old_b)).sum()),
                actual_serving_svd_expected_bytes=float((pred["dequeue"][i, use] * (1 - b)).sum()),
                actual_serving_svd_ue_sinr_db=json.dumps([None if not np.isfinite(x) else float(x) for x in value])))
    pd.DataFrame(summaries).to_csv(OUT / "same_state_actions.csv", index=False)
    pd.DataFrame(ue_rows).to_csv(OUT / "same_state_per_ue.csv", index=False)
    pd.DataFrame(power_rows).to_csv(OUT / "direct_receiver_powers.csv", index=False)
    pd.DataFrame(ctbs_rows).to_csv(OUT / "ctbs_all_ues.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "actual_serving_svd_sensitivity.csv", index=False)
    pd.DataFrame(freshness).to_csv(OUT / "observation_freshness.csv", index=False)
    r.write(OUT / "phy_checks.json", dict(
        passed_discrete_and_independent_formula_checks=True,
        native_float_precision_reported_separately=True, checks=checks,
        scope="Three actual seed41 states; four fixed-state actions; native final at all three, native PFQ only at common-prefix TTI4001. No native counterfactual claim for other actions.",
        model_boundary="Checks simulator consistency, not independent validation of link abstraction or deployment CSI availability.",
        source_sha256={__file__: r.sha(__file__), str(OUT / "states.pt"): r.sha(OUT / "states.pt")}))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

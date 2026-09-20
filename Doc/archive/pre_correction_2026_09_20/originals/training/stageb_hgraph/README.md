# Stage-B Sparse Entity Graph / DPA-v1

This directory contains the active Stage-B research stack. The current frozen policy is **DPA-v1 (Deadline-Pressure Actor v1)**; its compatibility class/family names are `PFQIndependentReuseActor` and `pfq_independent_reuse_v1`.

Canonical documents:

- [DPA-v1 main design](../../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)
- [DPA/PFQ/RRQ integrated experiment report](../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)
- [Effective configuration and metric semantics](../../Doc/current_stageB_effective_configuration.md)
- [Document index](../../Doc/README.md)

This README describes code ownership and the current execution contract. It does not duplicate experiment tables.

## Current frozen identity

```text
reporting name:       DPA-v1
actor class:          PFQIndependentReuseActor
actor family:         pfq_independent_reuse_v1
checkpoint:           output/pfq_independent_reuse/formal/
                      deadline_dimensionless_features_seed45_v1/
                      train_attempt2/ppo_actor_last.pt
file SHA-256:         94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a
tensor SHA-256:       6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be
selection:            deterministic argmax
headroom:             CTBS-v1 cumulative TBS
non-starvation guard: off
online PFQ features:  none
```

Changing feature slots, CTBS semantics, reuse-group update order, guard mode, or candidate truncation policy creates a different treatment even if the checkpoint file is unchanged.

## Current runtime scope

The validated experiments use:

- `3cell / 36UE / 17 local PRGs`
- `coop_boundary` UE placement and `strongest` association
- `cell_radius=500 m`, `site_spacing=1000 m`
- legacy shared shadow RNG
- Rayleigh + SVD + Type-0 allocation
- packet TTL `200 ms`
- continuous `4000 TTI` runs

The older 7cell/TAR-GNN material is retained as historical design context. DPA-v1 has not yet established a 7cell/84UE performance result.

## Online decision contract

For every TTI, the independent policy:

1. consumes one causal pre-action snapshot;
2. builds Cell, UE, and Cell-PRG entities plus sparse relations;
3. encodes the graph and applies cross-cell attention within each local-PRG reuse group;
4. visits local PRGs in fixed order `0..16`;
5. scores all cells in a reuse group from the same pre-update state;
6. chooses one associated UE or Blank per `(cell, local PRG)`;
7. updates cumulative CTBS headroom only after all cells in that reuse group have been scored;
8. materializes the result into a legal native Type-0 action.

Candidate packing covers the full runtime association width. Demand and legality are applied through fail-closed masks; silent candidate truncation is forbidden. The actor may choose the same UE on multiple local PRGs in one TTI.

PFQ score, rank, action, and teacher labels must not enter online actor-visible tensors. They are allowed only as offline supervision ancestry or baseline outputs.

## Feature layout

### Cell `[C, 5]`

```text
0 cell_load_bytes
1 active_ue_count
2 mean_wb_sinr_lin
3 mean_avg_rate_mbps
4 tb_err_rate
```

Slot 4 is the bridge's current-snapshot cell TB-error rate. `FeatureSnapshotBuilder` currently updates a private `_cell_tb_err_ema`, but does not substitute that EMA into `cell_values`; do not describe the actor input as an EMA without changing and re-evaluating the implementation.

### UE `[U, 16]`

```text
 0 buffer_bytes
 1 avg_rate_mbps
 2 wb_sinr_lin
 3 cqi
 4 ri
 5 tb_err_last
 6 new_data_flag
 7 stale_slots
 8 hol_delay_ms
 9 ttl_slack_ms
10 recent_scheduled_ratio
11 recent_goodput_deficit_norm
12 log1p_drain_pressure
13 scheduled_dequeue_adequacy
14 ttl_normalized_service_age
15 reserved_zero
```

The DPA-v1 deadline/service slots are:

```text
S_hat(t+1) = 0.95*S_hat(t) + 0.05*scheduled_dequeue_bytes(t)
slack      = max(TTL slack in TTI, 1)
pressure   = Q / (S_hat*slack + 1)

f12 = log1p(pressure)
f13 = clip(S_hat*slack/max(Q,1), 0, 1)
f14 = clip(service_age_tti/packet_ttl_tti, 0, 1)
f15 = 0
```

The current `TTL=200 ms` and `TTI=0.5 ms` give `packet_ttl_tti=400`. All four slots are zero when `Q=0`. The previous meanings `critical_ttl_backlog`, `near_ttl_backlog`, `age_weighted_backlog`, and raw `service_gap` are obsolete for DPA-v1 checkpoints.

### Cell-PRG `[C*P, 12]`

```text
 0 top1_sinr_db
 1 top2_gap_db
 2 prev_prg_assigned
 3 prev_prg_reuse_ratio
 4 neighbor_max_top1_sinr_db
 5 neighbor_mean_top1_sinr_db
 6 same_prg_conflict_ratio
 7 ici_proxy
 8 post_eq_p10_sinr_db
 9 post_eq_mean_sinr_db
10 backlog_weighted_expected_goodput_mbps
11 low_sinr_hol_ttl_weight
```

### Sequential candidate features `[C, K, 3]`

```text
remaining_demand / initial_demand
allocated_prgs / 17
predicted_or_cumulative_service / initial_demand
```

Under current DPA-v1, remaining demand is governed by the causal CTBS-v1 state.

## Active modules

- `independent_actor.py`
  - `IndependentActorConfig`: feature dims `5/16/12`, hidden `128`, 2 message layers, 4 reuse-attention heads, dropout 0.
  - `PFQIndependentReuseActor`: graph encoding, same-local-PRG reuse attention, UE/Blank scoring.
- `independent_policy.py`
  - full-width candidate validation;
  - fixed reuse-group argmax decode;
  - sequential state and CTBS integration;
  - deterministic replay and Type-0 materialization.
- `feature_snapshot.py`
  - exact actor-visible feature names and runtime topology metadata;
  - exclusion of teacher actions from actor-visible tensors.
- `ctbs_headroom.py`
  - causal MCS reconstruction from pre-action OLLA/WB-SINR/prior-TB state;
  - cumulative TBS recomputation as local PRGs are assigned;
  - post-action reference comparison for audit only.
- `independent_reward.py`
  - independent treatments A--E;
  - frozen DPA-v1 checkpoint uses treatment E (deadline-tail potential + tail-rate utility).
- `independent_objectives.py`
  - deadline pressure and expiry-aware UE-window trackers.
- `graph_ops.py`, `edge_generator.py`, `mask_checker.py`
  - candidate packing, graph construction, action legality, and legacy/shared utilities.
- `bridge_env.py`
  - Python side of the online bridge observation/action exchange.
- `pfq_dpa_aligned_comparison.py`
  - strict same-binary/same-stream DPA/PFQ aggregation.
- `granularity_experiment.py`
  - fixed-expected-load homogeneous/mixed DPA/PFQ/RRQ aggregation.
- `dpa_v1_historical_comparison.py`
  - windowed expiry-aware metrics and historical-report utilities.

`estimate_candidate_service_bytes()` remains available for legacy/diagnostic paths. It is not the current CTBS-v1 cumulative-TBS contract.

The older `StageBHGraphPolicy`, joint `slot -> UE / PRG -> slot` head, behavior-cloning path, and fixed-shape ONNX export remain in this directory for historical experiments. Their README-era commands must not be interpreted as DPA-v1 reproduction instructions.

## Training identity

The frozen checkpoint is iter8 initialized from the earlier deadline-tail-rate checkpoint. Its base reward is normalized goodput minus failed service minus twice expired bytes, with treatment-E deadline-tail and tail-rate terms:

```text
goodput_weight                 = 1.0
failed_weight                  = 1.0
expired_weight                 = 2.0
deadline_tail_eta              = 0.25
deadline_tail_window_steps     = 100
deadline_tail_tail_fraction    = 0.10
deadline_tail_rate_weight      = 0.25
deadline_tail_rate_tail_fraction = 0.10
```

Treatments A--D are retained for ablation and compatibility; they are not the frozen DPA-v1 treatment.

## Evaluation evidence

Primary current outputs:

```text
output/pfq_independent_reuse/formal/
  dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/
    final_report.json
    aggregate_comparison.{json,csv}
    per_seed_comparison.{json,csv}

  granularity_newseed_stage1_v1/
    campaign_report_10seed.json
    seed101..seed110/

  iter8_dimensionless_frozen_generalization_seed48_seed51_v1/
    final_report.json

  iter8_ctbs_legal_non_starvation_guard_seed51_v1/
    final_report.json
```

The packet-granularity `campaign_report.json` and `campaign_report_3seed.json` files are intermediate stage-1 snapshots. Current conclusions use `campaign_report_10seed.json`.

The seed51 guard experiment produced zero overrides because no qualifying high-pressure UE had a legally replaceable Blank opportunity. The guard is therefore off in current formal evaluations.

## Focused validation

Run the report/aggregation checks inside the experiment container:

```bash
python -m pytest -q \
  training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
  training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py \
  training/stageb_hgraph/tests/test_granularity_experiment.py
```

Last documentation audit on 2026-09-07: `19 passed in 1.49s`.

For implementation changes, also select the smallest relevant fail-closed tests from:

```text
test_independent_actor.py
test_independent_policy.py
test_independent_online_path.py
test_independent_pfq_deletion.py
test_independent_materializer.py
test_independent_reward.py
```

## Research boundary

Supported today:

- PFQ-independent online actions;
- 3cell/36UE legality and deterministic replay;
- standard seeds 41--50 DPA/PFQ paired evidence;
- unseen seeds 101--110 fixed-load DPA/PFQ/RRQ evidence.

Not supported today:

- zero expiry for every seed/profile;
- an effective non-starvation safety guard;
- 7cell/84UE generalization;
- production-grade robustness or real-time deployment qualification.

The next minimal research task is a read-only causal audit of mixed seeds104/107 before adding a new guard, retraining, or expanding the architecture.

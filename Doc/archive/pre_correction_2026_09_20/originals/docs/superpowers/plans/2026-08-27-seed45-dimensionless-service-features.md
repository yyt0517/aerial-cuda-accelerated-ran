# Seed45 Dimensionless Service Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw per-UE deadline tail features with causal dimensionless service features, run a fixed seed45 feature-only PPO experiment, and add a CTBS-legal blank-only non-starvation guard only if the frozen starvation condition is observed.

**Architecture:** Preserve the 16-D actor contract by replacing UE slots 12--15 and feed scheduled dequeue from the already available post-action service accounting into the next observation. The feature-only run keeps the actor and decoder unchanged. A conditional evaluator-only guard, if needed, overrides at most one blank per cell using the current sequential CTBS-valid mask.

**Tech Stack:** C++17 simulator/observation bridge, Python 3/PyTorch PPO and evaluator, pytest, standalone C++ assertion test, Aerial container with `.venv`.

**Spec:** `docs/superpowers/specs/2026-08-27-seed45-dimensionless-service-features-design.md`

## Global Constraints

- Work only in `.worktrees/pfq-independent-reuse-actor` and preserve all pre-existing dirty changes.
- Enter the container with `./cuPHY-CP/container/attach_aerial.sh`, then run `source .venv/bin/activate`.
- Do not change reward coefficients, network shape, candidate construction, PRG order, Type-0 mapping, or CTBS-v1 accounting.
- Keep UE feature dimension 16 and use slots 12--15 exactly as frozen in the spec.
- Train seed45 for exactly 8 iterations and select the last checkpoint.
- Implement the guard only if measured feature-only starvation meets the frozen condition.
- Do not commit from the dirty shared worktree; report the scoped diff instead.

---

### Task 1: Dimensionless Feature Formula and Causal State

**Files:**
- Create: `cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h`
- Create: `cuMAC/examples/onlineTrainBridge/deadlineServiceFeaturesTest.cpp`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`

**Interfaces:**
- Consumes: pre-action queue bytes, TTL slack TTI, TTL horizon, current TTI, prior scheduled-dequeue EWMA, and prior positive scheduled-dequeue TTI.
- Produces: `DeadlineServiceFeatures { log1pDrainPressure, scheduledDequeueAdequacy, ttlNormalizedServiceAge }` and a next-snapshot scheduled-dequeue EWMA.

- [ ] **Step 1: Write the failing standalone C++ behavior test**

Add literal cases that require: empty queue maps to zeros; `Q=12000`,
`S_hat=3000`, `slack=4`, `age=100`, `TTL=400` maps to `log(2), 1, 0.25`;
zero service with `Q=1200000` maps to high finite pressure, zero adequacy,
and clipped age 1; EWMA `1000 -> 1100` for dequeue `3000` at alpha `0.05`.

- [ ] **Step 2: Run the test and verify RED**

Run inside the container:

```bash
cd /opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
g++ -std=c++17 -IcuMAC/examples/onlineTrainBridge \
  cuMAC/examples/onlineTrainBridge/deadlineServiceFeaturesTest.cpp \
  -o /tmp/deadlineServiceFeaturesTest
```

Expected: compilation fails because `DeadlineServiceFeatures.h` does not exist.

- [ ] **Step 3: Implement the pure formula helper**

Create a header-only helper with finite clamps and the exact frozen formulas.
Do not add runtime tuning parameters beyond the fixed alpha and one-byte
epsilon.

- [ ] **Step 4: Run the standalone test and verify GREEN**

Run the compile command from Step 2, then:

```bash
/tmp/deadlineServiceFeaturesTest
```

Expected: exit code 0.

- [ ] **Step 5: Wire scheduled dequeue into the next observation**

Change `observeTtiResult` to consume `servedBytesThisTti`. Maintain an EWMA in
bytes/TTI and reset last-service TTI only when scheduled dequeue is positive.
Populate extras 4--7 with the three new features and zero reserved slot. Pass
`servedBytesThisTti` from `main.cpp`; keep goodput history unchanged for slots
10--11.

- [ ] **Step 6: Re-run the standalone test and compile the simulator**

Use the existing `run_stageB_pfq_independent_reuse.sh` build path inside the
container. Expected: build exit code 0 with no signature mismatch.

### Task 2: Python Feature Contract

**Files:**
- Modify: `training/stageb_hgraph/tests/test_feature_snapshot.py`
- Modify: `training/stageb_hgraph/feature_snapshot.py`

**Interfaces:**
- Consumes: unchanged 16-column online observation tensor.
- Produces: names `log1p_drain_pressure`, `scheduled_dequeue_adequacy`, `ttl_normalized_service_age`, `reserved_zero` for columns 12--15.

- [ ] **Step 1: Change the existing snapshot test first**

Use literal tail values `[0.6931472, 1.0, 0.25, 0.0]` and assert the four new
names and exact preserved values.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m pytest training/stageb_hgraph/tests/test_feature_snapshot.py -q
```

Expected: failure showing the old raw-tail names.

- [ ] **Step 3: Replace the Python feature names**

Update only the four constants/names. Do not transform values again in Python.

- [ ] **Step 4: Run focused and independent-actor tests**

```bash
python -m pytest \
  training/stageb_hgraph/tests/test_feature_snapshot.py \
  training/stageb_hgraph/tests/test_independent_actor.py -q
```

Expected: all pass.

### Task 3: Fixed Seed45 Feature-Only Experiment

**Files:**
- Create output under: `output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/`
- Modify only after evidence: `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`

**Interfaces:**
- Consumes: iter21 actor checkpoint and the new simulator binary.
- Produces: 8-iteration seed45 training metrics, fixed last checkpoint, 1000-TTI measured evaluation, per-UE starvation report.

- [ ] **Step 1: Verify initialization and frozen arguments**

Load `ppo_actor_absolute_best.pt` in the container, assert checkpoint iter is
21 and UE feature dimension is 16, and copy the existing deadline-tail-rate
arguments while overriding only seed sequence to `45`, iterations to `8`,
checkpoint interval to `8`, and the output directory.

- [ ] **Step 2: Run fixed short PPO adaptation**

Use `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh` through its existing
deadline-tail-rate path with seed45, `rollout_steps=512`,
`rollout_warmup_steps=1000`, `iterations=8`, and iter21 initialization.
Expected: 8 completed rows, zero replay/illegal/CTBS mismatch, and a last
checkpoint marked iteration 8.

- [ ] **Step 3: Run the fixed last-checkpoint rollout**

Use `training.stageb_hgraph.eval_checkpoint_online` with seed45, argmax,
CTBS-v1, `warmup=1000`, and `measured=1000`. Enable service-balance and packet
outcome traces. Do not select a different checkpoint based on seed45 metrics.

- [ ] **Step 4: Produce the starvation decision**

Compute per UE: measured scheduled-dequeue sum, maximum consecutive
positive-backlog zero-dequeue TTIs, delivered/expired outcomes, PRG allocation,
expiry-aware rate, and outcome delay. Write `feature_only_gate.json` with the
two frozen starvation conditions and an explicit `guard_required` boolean.

- [ ] **Step 5: Stop if feature-only does not starve**

If `guard_required=false`, do not modify policy or evaluator code. Record the
feature-only result and proceed directly to final verification.

### Task 4: Conditional CTBS-Legal Blank Guard

**Files:**
- Modify: `training/stageb_hgraph/tests/test_independent_policy.py`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`
- Modify: `training/stageb_hgraph/independent_policy.py`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Create output under: `output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/guarded_eval/`

**Interfaces:**
- Consumes: actor-selected class, current `valid_main`, packed candidate UE IDs, UE features 12--14, and CTBS-v1 state.
- Produces: at most one blank-to-urgent override per cell/TTI plus an explicit guard override count.

- [ ] **Step 1: Write failing policy behavior tests**

Test with real packed candidates that: an urgent CTBS-valid candidate replaces
a blank; a nonblank actor choice is untouched; a pressure-at-most-one,
adequacy-one, or age-below-0.25 candidate is untouched; at most one override
occurs per cell; an invalid candidate is never selected; and enabling the
guard without CTBS-v1 raises an error.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest \
  training/stageb_hgraph/tests/test_independent_policy.py \
  training/stageb_hgraph/tests/test_independent_online_path.py -q
```

Expected: failures because the guard option and telemetry do not exist.

- [ ] **Step 3: Implement the minimum evaluator-only guard**

Add an opt-in guard configuration to greedy independent selection. Apply it
only when CTBS inputs exist, only to selected blanks, and only using the
current `valid_main`. Track per-cell use so one TTI cannot exceed one override
per cell. Store the override count in the trace and evaluation summary.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all pass.

- [ ] **Step 5: Run guarded seed45 evaluation**

Evaluate the same fixed feature-only last checkpoint with identical seed,
warmup, measured horizon, CTBS, and trace settings, changing only the guard
enable flag. Produce the same per-UE starvation report and a paired comparison
against feature-only.

### Task 5: Verification and Research Record

**Files:**
- Modify: `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`
- Modify: `Doc/README.md`
- Preserve: `Doc/deadline_tail_rate_iter21_seed42_formal_brief.md`

**Interfaces:**
- Consumes: feature-only and optional guarded result artifacts.
- Produces: evidence-backed conclusion, exact hashes/commands, and the main remaining uncertainty.

- [ ] **Step 1: Run the focused Python suite**

```bash
python -m pytest \
  training/stageb_hgraph/tests/test_feature_snapshot.py \
  training/stageb_hgraph/tests/test_independent_actor.py \
  training/stageb_hgraph/tests/test_independent_policy.py \
  training/stageb_hgraph/tests/test_independent_online_path.py \
  training/stageb_hgraph/tests/test_ctbs_headroom.py -q
```

Expected: zero failures.

- [ ] **Step 2: Re-run C++ formula test and binary verification**

Compile/run `deadlineServiceFeaturesTest`, record the simulator SHA-256, and
verify the simulator/evaluator exit codes are zero.

- [ ] **Step 3: Verify experiment validity**

Assert 16-D feature shape, finite/ranged new feature values, complete candidate
coverage, zero illegal/replay mismatch, zero CTBS MCS/layer/TBS mismatch, and
the frozen seed/checkpoint/horizon in generated JSON.

- [ ] **Step 4: Update documentation**

Record what changed, what was tested, whether feature-only starved, whether
the conditional guard ran, business/resource metric deltas, the output paths,
and the remaining uncertainty. Do not generalize beyond seed45.

- [ ] **Step 5: Inspect the scoped diff**

```bash
git diff --check -- \
  cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h \
  cuMAC/examples/onlineTrainBridge/deadlineServiceFeaturesTest.cpp \
  cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h \
  cuMAC/examples/multiCellSchedulerUeSelection/main.cpp \
  training/stageb_hgraph/feature_snapshot.py \
  training/stageb_hgraph/tests/test_feature_snapshot.py \
  training/stageb_hgraph/independent_policy.py \
  training/stageb_hgraph/eval_checkpoint_online.py \
  Doc/stageB_boundary_intelligent_algorithm_experiment_report.md \
  Doc/README.md
```

Expected: no whitespace errors. Do not commit unrelated dirty changes.


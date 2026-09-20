> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-07-30-stageb-hgraph-ppo-delivery-roadmap.md)保留当时状态。

# Stage-B HGraph+PPO Delivery Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a causally aligned, PFQ-residual HGraph policy that passes packet/worst-seed gates and has identical Python, ONNX, and C++ action semantics before adaptive bucket training is promoted.

**Architecture:** The work is split into five gated sub-projects because policy-gradient alignment, observation/reward changes, deployment parity, experiment orchestration, and control-granularity selection can each be reviewed and rejected independently. Each sub-project produces testable software and a checkpoint gate; the next plan starts only after the prior gate passes.

**Tech Stack:** Python 3, PyTorch, pytest, NumPy, ONNX/ONNX Runtime, C++17, CUDA/cuMAC, CMake, Bash, JSON/CSV.

## Global Constraints

- Use one shared policy; do not introduce per-seed experts or expose seed/bucket identity to the actor.
- Keep the initial control shape at 3 cells, 36 UE, 17 PRG × 16 PRB, and `max_candidates=12`.
- Treat `(cell, PRG) -> candidate UE/blank` as the PPO-owned action; UE slots are deterministic materialization.
- Training uses constrained stochastic sampling; evaluation and deployment use the matching greedy policy.
- Hard legality and demand-cap masks constrain candidate selection. After the
  canonical discrete action is produced, native code may only drop illegal
  entries; non-hard decoder override rate must stay below 5%, with 0 as the
  target.
- Initialize candidate logits as the existing log-scaled PFQ anchor plus a
  near-zero HGraph residual.
- Final seed41--50 gates: full and post-1000 mean goodput above PFQ, worst-seed goodput delta at least -1.0%, pooled expiry no worse than PFQ, pooled completion no worse than PFQ, and cross-seed mean p95 delay no worse than PFQ.
- Python, ONNX, and C++ must produce identical discrete UE/PRG actions for golden observations.
- Target deployment latency is p99 at or below 0.5 ms, with 0.4 ms as the engineering target.
- Preserve unrelated dirty-worktree changes; execute implementation in an isolated worktree created with `superpowers:using-git-worktrees`.

---

## Plan Map

| Order | Detailed plan | Deliverable | Blocking gate |
|---|---|---|---|
| 1 | `2026-07-30-stageb-hgraph-causal-action-alignment.md` | Candidate action owns PPO log-probability and deterministically materializes Type-0 slots | Non-zero candidate gradient, exact rollout/replay action, override <5%, native rejects 0 |
| 2 | `2026-07-30-stageb-hgraph-pfq-residual-tail-credit.md` | PFQ-residual initialization plus dense unfinished-packet pressure | Initial logits match PFQ anchor; potential reward is numerically correct; short online smoke is stable |
| 3 | `2026-07-30-stageb-hgraph-canonical-deployment-parity.md` | One tensor canonical action operator used by Python export and respected by C++ | Python/ONNX/C++ golden actions exact; native ONNX pilot matches Python utilization semantics |
| 4 | `2026-07-30-stageb-hgraph-adaptive-bucket-training.md` | G/U/T/R curriculum, macrocycle evaluation, hard KPI gates | Uniform seed41--50 final gate and unseen-seed directional check |
| 5 | `2026-07-30-stageb-hgraph-prg-granularity-ablation.md` | Matched 17×16 versus 68×4 PRG experiment | Promote 68 only with at least 0.2% paired mean gain, all packet/worst-seed gates, parity, and p99 ≤0.5 ms |

The optional action-conditioned cross-cell interference head is not scheduled.
Create its plan only if all five plans above pass their architecture gates and
the remaining failures are shown to depend on neighbor actions that the
current state cannot distinguish.

### Task 1: Execute causal action alignment

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-stageb-hgraph-causal-action-alignment.md`
- Produce: the code, tests, metrics, and commit series specified by that plan

**Interfaces:**
- Consumes: current `StageBHGraphPolicy`, packed PRG candidates, online bridge rollouts
- Produces: `CandidatePolicyConfig`, `CandidatePolicyTrace`, constrained sample/replay functions, and deterministic candidate-to-Type-0 materialization

- [ ] **Step 1: Create an isolated implementation worktree**

Use `superpowers:using-git-worktrees`, select a feature branch named `stageb-hgraph-causal-action`, and verify the worktree starts from commit `e856e8f` or a later commit containing the approved design.

- [ ] **Step 2: Execute every checkbox in the causal alignment plan**

Run each specified failing test before implementation, make the minimum implementation pass, and commit at every task boundary.

- [ ] **Step 3: Run the causal gate**

Run:

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_candidate_policy.py \
  training/stageb_hgraph/tests/test_candidate_type0_materialization.py \
  training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -v
```

Expected: all tests pass; the report contains finite non-zero candidate-head gradient, positive training entropy, exact replayed log-probability, override ratio below 0.05, and zero preflight/native rejects.

- [ ] **Step 4: Stop if the gate fails**

Do not begin PFQ-residual work while candidate gradient is zero, action replay differs, or non-hard override remains at or above 5%.

### Task 2: Execute PFQ residual and tail credit

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-stageb-hgraph-pfq-residual-tail-credit.md`
- Produce: the code, tests, metrics, and commit series specified by that plan

**Interfaces:**
- Consumes: direct candidate policy and deterministic Type-0 materialization from Task 1
- Produces: PFQ-residual initialization, 16-dimensional UE observations, queue-pressure potential, and the causal residual training preset

- [ ] **Step 1: Confirm the Task 1 gate artifact**

Check that the causal gate JSON records `"passed": true` and that its checkpoint config contains `"candidate_policy_mode": "direct"`.

- [ ] **Step 2: Execute every checkbox in the residual/tail plan**

Keep 17 PRG and the current graph topology fixed while changing PFQ initialization and packet-pressure features.

- [ ] **Step 3: Run the residual/tail gate**

Run:

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_pfq_residual.py \
  training/stageb_hgraph/tests/test_feature_snapshot.py \
  training/stageb_hgraph/tests/test_queue_pressure_reward.py -v
```

Expected: all tests pass; zero-residual main logits equal the existing
log-scaled PFQ anchor logits; the 16 UE feature names and values are stable;
reducing queue pressure produces positive shaping credit.

- [ ] **Step 4: Stop if the gate fails**

Do not export or start bucket training if the residual initialization does not reproduce the PFQ base ordering or if pressure shaping rewards increasing unfinished-packet pressure.

### Task 3: Execute canonical deployment parity

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-stageb-hgraph-canonical-deployment-parity.md`
- Produce: the code, tests, ONNX artifact checks, C++ parity report, and commit series specified by that plan

**Interfaces:**
- Consumes: direct PFQ-residual candidate policy with 16 UE features
- Produces: shared tensor greedy/materialization operator, authoritative action-mode ONNX, C++ legality-only finalizer, parity and latency reports

- [ ] **Step 1: Confirm Python canonical stability**

Run the Task 1 and Task 2 unit suites together and require a clean pass before modifying export/runtime code.

- [ ] **Step 2: Execute every checkbox in the deployment parity plan**

Use the same synthetic golden observations for PyTorch, ONNX Runtime, and the C++ action harness.

- [ ] **Step 3: Run the deployment gate**

Run:

```bash
.venv/bin/python training/stageb_hgraph/check_action_operator_parity.py \
  --checkpoint training/stageb_hgraph/runs/causal_residual/ppo_actor_best.pt \
  --onnx training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --cpp-runner build.$(uname -m)/cuMAC/examples/multiCellSchedulerUeSelection/gnnrlActionGolden
```

Expected: exit 0 with exact UE and PRG action equality for every case.

- [ ] **Step 4: Stop if the gate fails**

Do not treat wrapper-level ONNX numeric equality as deployment parity; the final discrete Type-0 action must match.

### Task 4: Execute adaptive bucket training

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-stageb-hgraph-adaptive-bucket-training.md`
- Produce: curriculum implementation, experiment manifests, uniform validation reports, and final candidate

**Interfaces:**
- Consumes: parity-passing PFQ-residual checkpoint and canonical deployment action
- Produces: macrocycle checkpoints, bucket weight history, seed41--50 paired gate report, unseen-seed directional report

- [ ] **Step 1: Confirm all architecture gates**

Require the causal, residual/tail, and deployment gate JSON files to contain `"passed": true`.

- [ ] **Step 2: Execute fixed-bucket C1**

Use G/U/T/R weights 25/30/25/20 for at least one macrocycle and run a uniform seed41--50 validation.

- [ ] **Step 3: Execute adaptive-bucket C2**

Allow weight updates only within floors G20/U20/T15/R15 and a 40% per-bucket ceiling.

- [ ] **Step 4: Run the final KPI gate**

Run:

```bash
.venv/bin/python cuMAC/scripts/check_stageB_hgraph_gates.py \
  --pfq-manifest output/pfq_seed41_50/scenario_seed_manifest.csv \
  --ai-manifest output/hgraph_bucket_final/scenario_seed_manifest.csv \
  --warmup-tti 1000 \
  --worst-goodput-delta-pct -1.0 \
  --out output/hgraph_bucket_final/gate_report.json
```

Expected: exit 0 and `"passed": true`.

- [ ] **Step 5: Run the unseen-seed directional check**

Evaluate seeds51--60 without updating bucket weights. Report mean/worst deltas and packet gates separately; do not use these seeds to retroactively select the seed41--50 checkpoint.

### Task 5: Execute the PRG granularity ablation

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-stageb-hgraph-prg-granularity-ablation.md`
- Produce: matched 17/68-PRG training/evaluation manifests, parity/latency reports, and `granularity_decision.json`

**Interfaces:**
- Consumes: the passing aligned 17-PRG checkpoint, frozen bucket schedule, and canonical deployment operator
- Produces: an evidence-backed decision to retain 17 PRG or promote a separately trained/exported 68-PRG actor

- [ ] **Step 1: Confirm the 17-PRG reference is stable**

Require the final seed41--50 gate, action parity report, and latency report to pass before launching the matched A/B.

- [ ] **Step 2: Execute every checkbox in the granularity plan**

Hold total PRBs, training TTIs, seeds, optimizer, reward, and checkpoint gates fixed; vary only PRBs per group and the resulting action shape.

- [ ] **Step 3: Apply the promotion rule**

Promote 68 PRG only if its paired mean goodput is at least 0.2% above the aligned 17-PRG reference, every packet/worst-seed/parity gate passes, and p99 deployment latency is at most 0.5 ms. Otherwise retain 17 PRG and record the failed checks.

## Completion Definition

The roadmap is complete only when:

- all five detailed plans are implemented and their unit/integration tests pass;
- causal, residual/tail, deployment, and final KPI gate reports are archived with config and commit hashes;
- the promoted ONNX artifact uses the same discrete action semantics as the selected PyTorch checkpoint;
- no result mixes PRG granularities, simulator lifecycles, or action operators;
- the optional joint interference plan remains deferred unless its evidence conditions are met.

> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-09-01-dpa-v1-pfq-aligned-expiry-aware-comparison.md)保留当时状态。

# DPA-v1 / PFQ Aligned Expiry-Aware Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run PFQ on seeds 41--50 under the exact DPA-v1 4000-TTI protocol and generate fully paired expiry-aware overall and per-seed comparisons.

**Architecture:** Reuse `pfq_reference_calibration` for native PFQ execution and the existing DPA raw service/outcome traces. Add one experiment-specific aligned aggregator whose canonical window function is shared by both arms; do not change either scheduling algorithm.

**Tech Stack:** Python 3, NumPy, CSV/JSON, pytest, existing Aerial simulator and online bridge.

**Spec:** `docs/superpowers/specs/2026-09-01-dpa-v1-pfq-aligned-expiry-aware-comparison-design.md`

## Global Constraints

- Work only in `/opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor` after `./cuPHY-CP/container/attach_aerial.sh` and `source .venv/bin/activate`.
- Freeze seeds `41--50`, simulator SHA-256 from the DPA manifest, one uninterrupted 4000-TTI rollout, full `1--4000`, and steady `1001--4000`.
- PFQ uses native teacher actions. DPA-v1 checkpoint, argmax, CTBS-v1, Guard OFF, feature set, candidates, decoder, Type-0 mapping, and PRG order remain unchanged.
- Every headline delay/rate calculation includes expired outcomes. All 36 active UEs enter UE macro/p10/min; a UE without a terminal outcome contributes zero rate.
- Preserve unrelated dirty-worktree changes. Do not delete or overwrite the existing DPA or historical-comparison campaigns.

---

### Task 1: Canonical expiry-aware metric contract

**Files:**
- Modify: `training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py`
- Modify: `training/stageb_hgraph/dpa_v1_historical_comparison.py`
- Create: `training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py`
- Create: `training/stageb_hgraph/pfq_dpa_aligned_comparison.py`

**Interfaces:**
- Produces: `summarize_aligned_window(service_rows, outcome_rows, decision_rows, window)` returning identical canonical metrics for either arm.

- [ ] **Step 1: Write the failing active-UE denominator test**

  Add a three-UE fixture with one delivered outcome, one expired-only outcome,
  and one active UE with no terminal outcome. Assert rates `[24, 0, 0]` Mbps,
  macro `8`, p10/min `0`, and evaluated UE count `3`.

- [ ] **Step 2: Run the focused test and verify RED**

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py::test_expiry_aware_rate_includes_all_active_ues -q
  ```

  Expected: FAIL because the current implementation iterates only over UEs
  present in terminal outcomes.

- [ ] **Step 3: Write the failing shared-arm window test**

  Use literal service/outcome rows containing one delivered packet and one
  expired packet. Assert that `outcome_delay_mean_ms` is `100.5`, UE macro rate
  includes the expired-only and no-outcome UEs as zero, goodput comes from
  `successful_bytes`, served rate comes from `scheduled_dequeue_bytes`, and
  legacy `packet_delay_*` / `packet_effective_service_rate_*` keys are absent.

- [ ] **Step 4: Run the aligned test and verify RED**

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py -q
  ```

  Expected: FAIL because the aligned module does not exist.

- [ ] **Step 5: Implement the minimum shared window aggregator**

  The implementation validates exact UE/TTI coverage, filters service rows by
  `tti` and outcomes by inclusive `terminal_tti`, and emits only the metrics
  frozen in the spec. It uses all observed active UE IDs as the rate population.

- [ ] **Step 6: Run focused and existing metric tests**

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_pfq_reference_calibration.py -q
  ```

### Task 2: Frozen paired extractor and report writer

**Files:**
- Modify: `training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py`
- Modify: `training/stageb_hgraph/pfq_dpa_aligned_comparison.py`

**Interfaces:**
- Produces CLI commands `preflight`, `extract-seed`, and `aggregate`.
- Writes per-seed JSON plus normalized CSV/JSON and `final_report.json`.

- [ ] **Step 1: Write failing tests for alignment and paired aggregation**

  Test exact seed/window coverage, simulator hash equality, offered-stream hash
  equality, rejection of missing outcome columns, paired delta direction, and
  absence of legacy served-only metrics from headline/win-count structures.

- [ ] **Step 2: Run tests and verify RED**

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py -q
  ```

- [ ] **Step 3: Implement the minimal CLI and deterministic writers**

  `extract-seed` loads DPA `eval_steps.jsonl`, PFQ `tti_trace.jsonl`, both
  service traces and outcome CSVs, then runs the same canonical window function
  for both arms. `aggregate` writes 10-seed descriptive statistics, paired
  deltas/win counts, stress-seed callouts, and per-seed tables.

- [ ] **Step 4: Run tests and CLI help checks**

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py -q
  python -m training.stageb_hgraph.pfq_dpa_aligned_comparison --help
  ```

### Task 3: Seed41 minimum falsifiable PFQ replay

**Files:**
- Create outputs under: `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/seed41/pfq/`

**Interfaces:**
- Consumes: current DPA simulator binary and seed41 DPA raw traces.
- Produces: one structurally aligned seed41 PFQ run and paired metrics.

- [ ] **Step 1: Enter the container, activate the environment, and freeze paths**

  ```bash
  ./cuPHY-CP/container/attach_aerial.sh
  cd /opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
  source .venv/bin/activate
  ```

- [ ] **Step 2: Run PFQ seed41 for exactly 4000 TTIs**

  ```bash
  python -m training.stageb_hgraph.pfq_reference_calibration \
    --sim-bin "$SIM_BIN" --sim-cwd "$PFQ_DIR" \
    --topology-seed 41 --traffic-seed 41 \
    --warmup-steps 0 --measure-steps 4000 \
    --service-balance-audit --service-balance-trace-first-tti 1 \
    --out-dir "$PFQ_DIR"
  ```

- [ ] **Step 3: Verify and extract the paired seed**

  Require 4000 PFQ decision rows, 144000 service rows, terminal outcome fields,
  matching simulator/topology/offered-stream hashes, then run `extract-seed`.
  Stop before seed42 if any structural/alignment check fails.

### Task 4: Sequential PFQ seeds42--50

**Files:**
- Create outputs under: `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/seed42/` through `seed50/`

**Interfaces:**
- Consumes: the seed41-validated command and immutable protocol manifest.
- Produces: ten complete paired seed artifacts.

- [ ] **Step 1: Run seeds42--50 sequentially on one GPU**

  Repeat the exact seed41 command with topology and traffic seed both equal to
  the current seed. Do not run simulator instances concurrently.

- [ ] **Step 2: Extract and structurally validate each seed immediately**

  A metric regression is recorded and does not stop the campaign. Execution,
  trace, provenance, or alignment failure stops immediately.

- [ ] **Step 3: Verify cohort completeness**

  Require ten PFQ `reference_manifest.json`, ten `tti_trace.jsonl`, ten
  `packet_outcome_samples.csv`, ten 144000-row service traces, and ten paired
  `seed_metrics.json` files.

### Task 5: Aggregate comparison and reporting brief

**Files:**
- Create: `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/final_report.json`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/per_seed_comparison.csv`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/aggregate_comparison.csv`
- Create: `Doc/dpa_v1_pfq_seed41_50_aligned_expiry_aware_brief.md`

**Interfaces:**
- Consumes: ten paired seed artifacts.
- Produces: overall statistics first, followed by detailed seed41--50 comparisons.

- [ ] **Step 1: Run aggregate and validate report structure**

  Require both full and steady windows, all canonical metrics, paired deltas and
  win counts, stress-seed 45/48 callouts, and no legacy served-only metric in
  headline conclusions.

- [ ] **Step 2: Write the reporting brief from machine-readable results**

  Present experiment alignment, 10-seed averages, paired wins, per-seed table,
  worst cases, and the supported research conclusion without starting another
  experiment.

- [ ] **Step 3: Run fresh verification**

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_pfq_reference_calibration.py -q
  python -m json.tool "$CAMPAIGN_ROOT/final_report.json" >/dev/null
  test "$(find "$CAMPAIGN_ROOT" -mindepth 2 -maxdepth 2 -name seed_metrics.json | wc -l)" -eq 10
  ```

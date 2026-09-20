# PFQ-Independent Rescue-Cohort Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether A8 hierarchical-decoder rescues lose goodput because they have lower service value, lower reliability, or lower Type-0/PDSCH acceptance, without changing scheduling behavior.

**Architecture:** Extend the existing independent-policy trace with detached decision-time evidence, then join it after each measured simulator step to the existing UE-level transmission arrays. Summarize two UE-by-TTI cohorts with sum-derived ratios and allocation-size strata, validate the rerun against the frozen hierarchical treatment, and classify exactly one mechanism before stopping.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, Bash, JSON/Markdown reporting, the existing cuMAC simulator, Docker container `c_aerial_oai2`, and virtual environment `/opt/nvidia/cuBB/.venv`.

## Global Constraints

- This is a research-first, behavior-invariant diagnostic. Do not change actor parameters, graph contents, logits, masks, decode choices, sequential demand updates, reward, simulator, or random-number consumption.
- Execute source work in `/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor`, after safely fast-forwarding it from `main`. The worktree contains the ignored frozen checkpoint and formal evidence.
- Run Python and simulator commands only through `docker exec c_aerial_oai2 bash -lc 'cd /opt/nvidia/cuBB && source .venv/bin/activate && ...'`. Host-side Git inspection and `apply_patch` remain on the host.
- Preserve the five user-owned worktree-root artifacts byte-for-byte: `output.txt`, `output_short.txt`, `packet_delay_samples.csv`, `snr.txt`, and `ue_topology.csv`.
- Reuse frozen A8 checkpoint SHA-256 `c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc` and simulator SHA-256 `3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c`.
- The formal scenario remains cooperative boundary, topology/traffic seed `41`, `hierarchical_argmax`, `1000` warmup TTIs, `512` measured TTIs (`1001..1512`), and `26112` decisions.
- The cohort universe is every UE x TTI with at least one planned non-Blank PRG. Retain `actual_prg_count == 0` as an acceptance outcome; never invent service-per-PRG values for it.
- All aggregate ratios use summed numerators and denominators. Represent unavailable ratios as JSON `null`; never turn an empty denominator into zero evidence.
- Treat outcomes as UE x TTI associations. Do not claim an individual rescued PRG caused aggregate UE bytes or TB errors.
- Reuse the existing flat A8 and hierarchical formal outputs. Do not rerun A4, B4, PFQ, flat A8, another checkpoint, or another seed.
- Stop on the first invalid smoke/formal gate. After one valid formal classification, stop without training, weighting logits, changing reward, adding simulator telemetry, or starting a follow-up experiment.
- Use TDD for source changes and commit after each independently reviewable source task. Do not commit live output evidence.

---

### Task 0: Freeze the experiment context and baseline

**Files:**

- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt`
- Read only: `.worktrees/pfq-independent-reuse-actor/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection`
- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/hierarchical_decode_seed41/eval_summary.json`
- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/hierarchical_decode_seed41/run_manifest.json`

**Interfaces:**

- Consumes: approved design commit `9757d22` and this plan commit from `main`.
- Produces: a clean experiment worktree at the approved revision and verified frozen identities.

- [ ] **Step 1: Invoke the worktree safety skill and inspect tracked state**

Use `superpowers:using-git-worktrees` before changing the existing experiment worktree. Then run on the host:

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" diff --quiet
git -C "$research_wt" diff --cached --quiet
test -z "$(git -C "$research_wt" ls-files -u)"
git -C "$research_wt" status --short
```

Expected: the first three checks return `0`; status may contain only the known untracked root artifacts/caches. Stop if any tracked, staged, or conflicted path exists.

- [ ] **Step 2: Fast-forward the experiment branch**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" merge --ff-only main
git -C "$research_wt" log -2 --oneline
git -C "$research_wt" diff --quiet
git -C "$research_wt" diff --cached --quiet
```

Expected: the approved design and plan are present and tracked/staged diffs remain empty. Do not apply or drop the retained safety stash.

- [ ] **Step 3: Verify checkpoint, simulator, manifest, and frozen treatment**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
sha256sum \
  "$research_wt/output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt" \
  "$research_wt/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor")
phase = root / "output/pfq_independent_reuse/formal/hierarchical_decode_seed41"
summary = json.loads((phase / "eval_summary.json").read_text())
manifest = json.loads((phase / "run_manifest.json").read_text())
assert summary["checkpoint_sha256"] == "c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc"
assert summary["sim_binary_sha256"] == "3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c"
assert manifest["seed"] == 41
assert manifest["warmup_ttis"] == 1000
assert manifest["measure_ttis"] == 512
assert summary["decode_mode"] == "hierarchical_argmax"
assert summary["stats_count"] == 512
assert (summary["measured_tti_first"], summary["measured_tti_last"]) == (1001, 1512)
assert summary["diagnostic_decision_count"] == 26112.0
assert summary["stats_sum_scheduled_bytes"] == 54960288.0
assert summary["stats_sum_ue_goodput_bytes"] == 49104376.0
assert summary["stats_sum_failed_bytes"] == 5855912.0
assert summary["stats_sum_scheduled_bytes"] == summary["stats_sum_ue_goodput_bytes"] + summary["stats_sum_failed_bytes"]
print("frozen hierarchical treatment verified")
PY
```

Expected: the two printed hashes equal the frozen values and Python prints `frozen hierarchical treatment verified`.

- [ ] **Step 4: Verify protected-root artifact hashes**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
cd "$research_wt"
sha256sum output.txt output_short.txt packet_delay_samples.csv snr.txt ue_topology.csv
```

Expected, in order:

```text
489338736f2fc7fff488b91a02c3e6ab3b015e855541d230cfffeb9fee865d99
1ce42f8bbbb4815307c0b59ccfd822278fd56c4fbf84adc9c100192815d09db6
50e19e76179e6f5a38eac690980cfa2fd8767738bf5ab82713c44d627f54bbf0
2551411c05a411a7d5fc77bd07d1b238051c4f1055e7ec34803cb8dc9a90c4d8
8f20a1675e238196d7eddf91920acb6099649e44b59e5ef315cab3ff15ad47b5
```

- [ ] **Step 5: Run the focused baseline in the container**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    training/stageb_hgraph/tests/test_hierarchical_decode_gate.py
'
```

Expected: `130 passed`. Any failure blocks source work.

---

### Task 1: Capture detached rescue-decision evidence without changing policy behavior

**Files:**

- Create: `training/stageb_hgraph/rescue_cohort_diagnostics.py`
- Modify: `training/stageb_hgraph/independent_policy.py`
- Create: `training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_policy.py`

**Interfaces:**

- Consumes: `IndependentPolicyDiagnostics`, `PackedPrgCandidates`, `SparseEntityGraph`, existing selected classes, service estimates, and sequential features.
- Produces: `RescueDecisionDiagnostics`, `build_rescue_decision_diagnostics(...)`, `concatenate_rescue_decision_diagnostics(...)`, and optional `IndependentPolicyTrace.rescue_diagnostics`.

- [ ] **Step 1: Write RED tests for exact selected-candidate evidence**

In `test_rescue_cohort_diagnostics.py`, construct one local reuse group with two cells, literal selected classes, base edge attributes, UE features, service estimates, and an `IndependentPolicyDiagnostics` row. Require this exact public dataclass shape:

```python
@dataclass(frozen=True)
class RescueDecisionDiagnostics:
    selected_ue: torch.Tensor
    cell_index: torch.Tensor
    local_prg_index: torch.Tensor
    selected_nonblank: torch.Tensor
    fragmentation_rescued: torch.Tensor
    selected_forced_blank: torch.Tensor
    selected_conservative_blank: torch.Tensor
    estimated_service_bytes: torch.Tensor
    quality_db: torch.Tensor
    quality_gap_db: torch.Tensor
    prg_risk: torch.Tensor
    ue_tb_err_last: torch.Tensor
    remaining_demand_ratio: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    success_proxy: torch.Tensor
```

Assert a non-Blank row gathers the selected UE/class values, a Blank row has `selected_ue == -1` and zero-valued selected-candidate fields, all fields are detached one-dimensional CPU tensors of equal length, and:

```python
expected_success = torch.sigmoid(
    (quality_db - 0.35 * quality_gap_db - 7.5 * prg_risk - 6.0 * ue_tb_err_last) / 4.0
)
assert torch.allclose(result.success_proxy[selected], expected_success)
```

- [ ] **Step 2: Write RED policy-path invariance tests**

In `test_independent_policy.py`, run the existing fixture through diagnostics off/on for `greedy`, `hierarchical_greedy`, and seeded `sample`. Assert:

```python
assert torch.equal(trace_off.choice, trace_on.choice)
assert torch.equal(action_off.action_ue_select, action_on.action_ue_select)
assert torch.equal(action_off.action_prg_alloc, action_on.action_prg_alloc)
assert torch.equal(trace_off.logp, trace_on.logp)
assert torch.equal(trace_off.entropy, trace_on.entropy)
assert trace_off.rescue_diagnostics is None
assert trace_on.rescue_diagnostics is not None
assert trace_on.rescue_diagnostics.selected_ue.numel() == graph.snapshot.n_cell * graph.snapshot.n_prg
assert torch.equal(
    trace_on.rescue_diagnostics.fragmentation_rescued,
    trace_on.diagnostics.fragmentation_rescued,
)
```

For `sample`, reset equal `torch.Generator` states before both calls and assert the states remain equal afterward. This catches hidden RNG consumption.

- [ ] **Step 3: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    -k "rescue_decision or rescue_diagnostics or diagnostic_invariance"
'
```

Expected: collection/import failures because the rescue diagnostic API and trace field do not exist.

- [ ] **Step 4: Implement the detached decision builder**

In `rescue_cohort_diagnostics.py`, add the dataclass above and these exact functions:

```python
def build_rescue_decision_diagnostics(
    *,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    token_indices: torch.Tensor,
    local_prg: int,
    selected: torch.Tensor,
    service_bytes: torch.Tensor,
    sequential_features: torch.Tensor,
    policy_diagnostics: IndependentPolicyDiagnostics,
) -> RescueDecisionDiagnostics:
    """Gather selected-class evidence for one reuse group without affecting actions."""


def concatenate_rescue_decision_diagnostics(
    rows: Sequence[RescueDecisionDiagnostics],
) -> RescueDecisionDiagnostics:
    """Concatenate fixed-schema detached decision rows in policy execution order."""
```

Validate shapes, base edge width `>=5`, finite gathered numeric values, UE bounds, and equal field lengths. Gather edge indices `0`, `1`, and `4`; UE feature index `5`; service estimates; sequential feature index `0`; and policy probabilities. Mask all selected-candidate numeric fields to zero for Blank. Compute only the frozen success-proxy formula from the spec. Call `.detach().reshape(-1).cpu()` on every output tensor.

- [ ] **Step 5: Attach evidence to the existing policy trace**

In `independent_policy.py`, extend the trace with a defaulted field:

```python
@dataclass(frozen=True)
class IndependentPolicyTrace:
    choice: torch.Tensor
    local_prg_order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor
    diagnostics: IndependentPolicyDiagnostics | None = None
    rescue_diagnostics: RescueDecisionDiagnostics | None = None
```

Inside `_run_policy`, only when `collect_diagnostics` is true:

1. build the existing `IndependentPolicyDiagnostics` row once;
2. pass that row plus the already-computed `service`, `sequence_features`, tokens, and selected classes to `build_rescue_decision_diagnostics`;
3. append both diagnostic rows;
4. concatenate rescue rows in the returned trace.

Do not move selection, log-probability, entropy, or sequential-state mutation statements. Do not compute a second policy distribution.

- [ ] **Step 6: Prove GREEN and full path invariance**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py
'
```

Expected: all selected tests pass, including sample RNG, flat/hierarchical actions, replay, and PFQ deletion equality.

- [ ] **Step 7: Review and commit Task 1**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" diff --check
git -C "$research_wt" add \
  training/stageb_hgraph/rescue_cohort_diagnostics.py \
  training/stageb_hgraph/independent_policy.py \
  training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
  training/stageb_hgraph/tests/test_independent_policy.py
git -C "$research_wt" commit -m "feat: capture rescue decision cohorts"
```

Expected: one source commit and no output artifact staged.

---

### Task 2: Join decisions to UE-by-TTI outcomes and summarize the measured window

**Files:**

- Modify: `training/stageb_hgraph/rescue_cohort_diagnostics.py`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Modify: `training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`

**Interfaces:**

- Consumes: `RescueDecisionDiagnostics`, materialized `chosen_ue_per_prg`, measured-step `next_info`, `n_ue`, `n_cell`, `n_prg`, and TTI.
- Produces: `RescueUeTtiRow`, `RescueCellTtiRow`, `join_rescue_cohort_tti(...)`, `summarize_rescue_cohorts(...)`, and nested evaluator key `rescue_cohort_diagnostics`.

- [ ] **Step 1: Write RED tests for the UE-by-TTI join**

Add these dataclass contracts to the test before production implementation:

```python
@dataclass(frozen=True)
class RescueUeTtiRow:
    tti: int
    ue: int
    rescue_exposed: bool
    planned_prg_count: int
    rescued_prg_count: int
    accepted_prg_count: int
    actual_prg_count: int
    predicted_service_bytes: float
    scheduled_bytes: float
    goodput_bytes: float
    tb_tx_count: int
    tb_err_count: int
    quality_db_sum: float
    quality_gap_db_sum: float
    prg_risk_sum: float
    ue_tb_err_last_sum: float
    remaining_demand_ratio_sum: float
    success_proxy_sum: float
    p_blank_sum: float
    p_service_sum: float
    p_best_ue_sum: float
    early_rescue_count: int
    middle_rescue_count: int
    late_rescue_count: int


@dataclass(frozen=True)
class RescueCellTtiRow:
    tti: int
    cell: int
    rescue_count: int
    first_rescue_local_prg: int
    decisions_before_or_without_prior_rescue: int
    decisions_after_prior_rescue: int
    forced_before_or_without_prior_rescue: int
    forced_after_prior_rescue: int
    conservative_before_or_without_prior_rescue: int
    conservative_after_prior_rescue: int
```

Use a two-UE literal where UE 0 has one rescued and one ordinary planned PRG, UE 1 has ordinary service only, and UE 0 has `actual_prg_count == 0`. Assert UE 0 remains exactly one `rescue_exposed` row, retains its zero-actual acceptance outcome, and has no fabricated service-per-PRG value. Assert planned selected UE identities reconstruct the supplied materialized chosen-UE vector exactly.

Add fail-closed cases for wrong array shapes, negative/nonfinite bytes, `goodput > scheduled`, `tb_err > tb_tx`, `accepted > planned`, `actual > accepted`, and nonzero actual evidence for an unplanned UE.

- [ ] **Step 2: Write RED tests for sum-derived cohort metrics and strata**

Build literal `RescueUeTtiRow` objects across two TTIs. Require this schema:

```python
summary = summarize_rescue_cohorts(ue_rows, cell_rows)
assert summary["schema_version"] == 1
assert summary["unit"] == "ue_tti"
assert set(summary["cohorts"]) == {"rescue_exposed", "ordinary_service"}
assert set(summary["actual_prg_strata"]) == {"zero", "1_2", "3_5", "ge_6"}
```

For each group, assert these exact raw keys:

```text
ue_tti_count
tti_count
planned_prg_count
rescued_prg_count
accepted_prg_count
actual_prg_count
predicted_service_bytes
scheduled_bytes
goodput_bytes
failed_bytes
tb_tx_count
tb_err_count
early_rescue_count
middle_rescue_count
late_rescue_count
```

Also assert these exact derived keys:

```text
predicted_service_bytes_per_planned_prg
scheduled_bytes_per_actual_prg
goodput_bytes_per_actual_prg
failed_byte_ratio
tb_error_ratio
planned_to_accepted_ratio
accepted_to_actual_ratio
quality_db_mean
quality_gap_db_mean
prg_risk_mean
ue_tb_err_last_mean
remaining_demand_ratio_mean
success_proxy_mean
p_blank_mean
p_service_mean
p_best_ue_mean
```

Choose literal values proving ratios are computed from total numerators and denominators, not the mean of row ratios. Assert unavailable values are `None`; `zero`, `1_2`, `3_5`, and `ge_6` are disjoint/exhaustive; and per-TTI comparison entries contain `mean_delta_rescue_minus_ordinary`, `paired_tti_count`, and `direction_agreement_fraction`.

- [ ] **Step 3: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
    -k "join or summarize or stratum or depletion"
'
```

Expected: failures for missing row types and join/summary functions.

- [ ] **Step 4: Implement the pure join and summarizer**

In `rescue_cohort_diagnostics.py`, implement:

```python
def join_rescue_cohort_tti(
    decisions: RescueDecisionDiagnostics,
    *,
    chosen_ue_per_prg: torch.Tensor,
    info: Mapping[str, object],
    n_ue: int,
    n_cell: int,
    n_prg: int,
    tti: int,
) -> tuple[list[RescueUeTtiRow], list[RescueCellTtiRow]]:
    """Join one action's planned decisions to the simulator's UE-level outcome."""


def summarize_rescue_cohorts(
    ue_rows: Sequence[RescueUeTtiRow],
    cell_rows: Sequence[RescueCellTtiRow],
) -> dict[str, object]:
    """Summarize measured-window cohorts using sums and explicit denominators."""
```

Reconstruct cell-major PRG tokens with `token = cell * n_prg + local_prg` before comparing decision-selected UE IDs to `chosen_ue_per_prg`. Read these required outcome arrays: `accepted_pre_pdsch_ue_prg_count`, `actual_ue_prg_count`, `actual_ue_served_bytes`, `actual_ue_goodput_bytes`, `actual_ue_tb_tx_count`, and `actual_ue_tb_err_count`. Reject missing or inconsistent evidence.

For each planned UE, sum decision-time quantities, including `p_blank`, `p_service`, and `p_best_ue`, across all selected PRGs and assign `rescue_exposed = rescued_prg_count > 0`. Count early `0..5`, middle `6..11`, and late `12..16` rescued PRGs. For each cell, a Blank at local PRG `j` is "after prior rescue" only if the same cell has a rescued selection at an index `< j`.

Return the nested schema exercised by Step 2. Use a small `_ratio(numerator, denominator) -> float | None` helper and never average row ratios. Summarize `actual_prg_count == 0` only as the `zero` acceptance bucket.

- [ ] **Step 5: Wire only the measured evaluator window**

In the independent branch of `_select_joint_action`, add the already-materialized vector without changing the action:

```python
"decoded_chosen_ue_per_prg": decoded.chosen_ue_per_prg.detach().cpu(),
```

In evaluator initialization, create `rescue_ue_rows` and `rescue_cell_rows` only when `independent_policy_diagnostics` is enabled. Inside the existing `step_idx >= stats_warmup_steps` block, require `trace.rescue_diagnostics`, call `join_rescue_cohort_tti`, and extend the lists. Do not collect warmup cohort rows.

After the existing independent byte and decision-count validation, add:

```python
transmission_summary["rescue_cohort_diagnostics_enabled"] = 1
transmission_summary["rescue_cohort_diagnostics"] = summarize_rescue_cohorts(
    rescue_ue_rows,
    rescue_cell_rows,
)
```

Require summed rescued PRGs to equal `diagnostic_fragmentation_rescue_count`, and summed planned non-Blank PRGs to equal the non-Blank choices reconstructed from the trace. A mismatch raises `ValueError`.

- [ ] **Step 6: Prove GREEN and evaluator non-regression**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py
'
```

Expected: all selected tests pass, including existing diagnostic counts, decode routing, action replay, and PFQ deletion.

- [ ] **Step 7: Review and commit Task 2**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" diff --check
git -C "$research_wt" add \
  training/stageb_hgraph/rescue_cohort_diagnostics.py \
  training/stageb_hgraph/eval_checkpoint_online.py \
  training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
  training/stageb_hgraph/tests/test_independent_online_path.py
git -C "$research_wt" commit -m "feat: summarize rescue cohort outcomes"
```

Expected: one focused commit and no generated evaluation output staged.

---

### Task 3: Add one fail-closed scientific gate and report

**Files:**

- Create: `training/stageb_hgraph/rescue_cohort_gate.py`
- Create: `training/stageb_hgraph/tests/test_rescue_cohort_gate.py`

**Interfaces:**

- Consumes: new treatment `eval_summary.json`, its run manifest, and frozen hierarchical treatment `eval_summary.json`.
- Produces: `build_rescue_cohort_report(summary, *, manifest, frozen_reference) -> dict[str, object]`, CLI JSON, and Markdown report.

- [ ] **Step 1: Write literal RED tests for validity gates**

Create compact literal mappings with the required summary, manifest, nested cohort schema, and frozen reference. Test that the report rejects each of:

```text
wrong decode/checkpoint/simulator/seed/window/decision count
PFQ deletion false
nonzero replay or illegal actions
scheduled != goodput + failed
missing/null cohort denominators needed by a claimed mechanism
cohort rescued PRG sum != diagnostic_fragmentation_rescue_count
changed frozen scheduled/goodput/failed/TB totals
```

The behavior-invariance reference comparison must require exact equality for measured scheduled/goodput/failed byte sums, TB tx/error counts, selected Blank subclass counts, and rescue count. A mismatch is invalid evidence, not an inconclusive mechanism.

- [ ] **Step 2: Write RED classification tests**

Require exactly four classifications with literal cohort values:

```python
assert reliability_report["classification"] == "reliability_supported"
assert service_report["classification"] == "service_value_supported"
assert acceptance_report["classification"] == "acceptance_supported"
assert conflicting_report["classification"] == "inconclusive"
```

Apply this ordered rule:

1. `reliability_supported` when rescue `success_proxy_mean` is lower and either failed-byte ratio or TB-error ratio is worse overall and in at least two populated positive-PRG strata;
2. `service_value_supported` when rule 1 is false, predicted bytes/planned PRG and scheduled bytes/actual PRG are both lower overall and in at least two populated positive-PRG strata;
3. `acceptance_supported` when rules 1-2 are false, at least one retention ratio is lower overall, and its per-TTI direction agreement is greater than `0.5`;
4. `inconclusive` otherwise.

Also assert the next-action strings are exactly:

```text
reliability: test reliability-weighted aggregate Serve mass with the frozen actor
service value: test service-value-weighted aggregate Serve mass with the frozen actor
acceptance: inspect the Type-0/PDSCH acceptance path before any policy change
inconclusive: consider per-PRG simulator tracing under a separate approved design
```

- [ ] **Step 3: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_rescue_cohort_gate.py
'
```

Expected: import failure because the gate does not exist.

- [ ] **Step 4: Implement the gate and renderer**

Implement this exact API and a `python -m` CLI:

```python
def build_rescue_cohort_report(
    summary: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    frozen_reference: Mapping[str, object],
) -> dict[str, object]:
    """Validate one formal diagnostic and classify its rescue-quality mechanism."""
```

The returned report must contain `valid: true`, frozen identities/window, cohort raw denominators and deltas, populated-stratum direction counts, per-TTI consistency, auxiliary depletion counts, `classification`, `hypothesis_supported`, `evidence_boundary`, and exactly one `next_experiment`. Set `hypothesis_supported` true only for the reliability or service-value classifications; acceptance is a pipeline mechanism, not support for low expected successful service.

The CLI accepts:

```text
--summary PATH
--manifest PATH
--frozen-reference PATH
--json-out PATH
--markdown-out PATH
```

Render a compact Markdown report containing validity/identity, the two overall cohort rows, positive-PRG strata, per-TTI direction consistency, demand-depletion auxiliary counts, classification, evidence limitation, and next experiment. Do not add a generic report framework.

- [ ] **Step 5: Prove GREEN and CLI identity**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_rescue_cohort_gate.py &&
  python -m py_compile \
    training/stageb_hgraph/rescue_cohort_diagnostics.py \
    training/stageb_hgraph/rescue_cohort_gate.py
'
```

Expected: all gate tests pass and compilation exits `0`.

- [ ] **Step 6: Review and commit Task 3**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" diff --check
git -C "$research_wt" add \
  training/stageb_hgraph/rescue_cohort_gate.py \
  training/stageb_hgraph/tests/test_rescue_cohort_gate.py
git -C "$research_wt" commit -m "feat: gate rescue cohort diagnostic"
```

Expected: one focused commit.

---

### Task 4: Add isolated runner phases and run the minimum smoke

**Files:**

- Modify: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh`
- Create: `training/stageb_hgraph/tests/test_rescue_cohort_runner.py`
- Evidence only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/rescue_cohort_smoke/`

**Interfaces:**

- Consumes: existing `evaluate_checkpoint`, frozen A8 checkpoint, new evaluator summary, and `rescue_cohort_gate` CLI.
- Produces: runner phases `rescue-cohort-smoke` and `rescue-cohort-seed41` plus an isolated 2+4-TTI smoke summary.

- [ ] **Step 1: Write a RED runner-routing test**

In `test_rescue_cohort_runner.py`, invoke the Bash runner with each new phase and nonexistent temporary build/output directories. Assert both phases pass option parsing and reach `missing simulator binary`, rather than `unsupported phase`. Also assert source text contains exactly one call with `2 4 ... hierarchical_argmax` in the smoke function and one call with `1000 512 ... hierarchical_argmax` in the formal function.

- [ ] **Step 2: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_rescue_cohort_runner.py
'
```

Expected: failures because both phase names are unsupported.

- [ ] **Step 3: Add only the two diagnostic phases**

Extend the phase allowlist/case with `rescue-cohort-smoke` and `rescue-cohort-seed41`.

`run_rescue_cohort_smoke` must:

```text
require the frozen A8 checkpoint and simulator
refuse an existing non-identical/non-resumed output directory
evaluate seed 41, 2 warmup, 4 measured, diagnostics=1, hierarchical_argmax
require stats_count=4 and diagnostic_decision_count=204
require zero replay/illegal and exact byte identity
require rescue_cohort_diagnostics_enabled=1 and schema_version=1
require both cohort dictionaries and all raw count/byte denominators
require summed rescued PRGs == diagnostic_fragmentation_rescue_count
```

`run_rescue_cohort_seed41` must create a manifest containing phase, scenario ID, seed `41`, warmup `1000`, measure `512`, A8 iteration `8`, checkpoint SHA-256, simulator SHA-256, and source revision; then run the same evaluator with `1000 512` and call:

```bash
"${PYTHON_BIN}" -m training.stageb_hgraph.rescue_cohort_gate \
  --summary "${phase_dir}/eval_summary.json" \
  --manifest "${phase_dir}/run_manifest.json" \
  --frozen-reference "${OUT_DIR}/hierarchical_decode_seed41/eval_summary.json" \
  --json-out "${phase_dir}/rescue_cohort_report.json" \
  --markdown-out "${phase_dir}/rescue_cohort_report.md"
```

Do not add configuration knobs, alternate checkpoints, seeds, thresholds, or resume behavior beyond the existing exact-manifest rule.

- [ ] **Step 4: Prove runner GREEN and run the focused suite**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_rescue_cohort_diagnostics.py \
    training/stageb_hgraph/tests/test_rescue_cohort_gate.py \
    training/stageb_hgraph/tests/test_rescue_cohort_runner.py \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    training/stageb_hgraph/tests/test_hierarchical_decode_gate.py &&
  bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
'
```

Expected: every selected test passes and `bash -n` exits `0`.

- [ ] **Step 5: Review and commit Task 4 before live execution**

```bash
research_wt=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$research_wt" diff --check
git -C "$research_wt" add \
  cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
  training/stageb_hgraph/tests/test_rescue_cohort_runner.py
git -C "$research_wt" commit -m "feat: run rescue cohort diagnostic"
git -C "$research_wt" diff --quiet
git -C "$research_wt" diff --cached --quiet
```

Expected: one runner commit and clean tracked/staged state.

- [ ] **Step 6: Verify hashes immediately before smoke**

Run the same checkpoint, simulator, and five protected-root `sha256sum` commands from Task 0. Every hash must equal its frozen constant. Confirm `output/pfq_independent_reuse/formal/rescue_cohort_smoke` does not exist; do not delete an existing directory.

- [ ] **Step 7: Run exactly one isolated 2+4-TTI smoke**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  bash cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
    rescue-cohort-smoke \
    --build-dir build.reuse_actor \
    --out-dir output/pfq_independent_reuse/formal
'
```

Expected: evaluator and simulator exit `0`, the embedded smoke gate passes, and `eval_summary.json` contains four measured TTIs and `204` decisions. Do not rerun on failure.

- [ ] **Step 8: Verify smoke hygiene and pause at the formal gate**

Recompute the five protected-root hashes and require exact equality with Task 0. Require tracked/staged diffs to remain clean. Report smoke cohort counts and any validity failure. Launch Task 5 only after a valid smoke and explicit formal-run approval.

---

### Task 5: Run one formal diagnostic, classify the mechanism, and stop

**Files:**

- Evidence only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/rescue_cohort_seed41/`
- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/hierarchical_decode_seed41/eval_summary.json`

**Interfaces:**

- Consumes: approved clean Task-4 revision, valid smoke, frozen A8 checkpoint/simulator/reference, and explicit formal-run approval.
- Produces: one formal `eval_summary.json`, `run_manifest.json`, `rescue_cohort_report.json`, `rescue_cohort_report.md`, and one research classification.

- [ ] **Step 1: Invoke verification-before-completion and run formal preflight**

Use `superpowers:verification-before-completion`. Re-run all Task-0 identity/window assertions, the focused suite from Task 4, `bash -n`, tracked/staged cleanliness, protected-root hashes, and assert the destination `output/pfq_independent_reuse/formal/rescue_cohort_seed41` is absent. Stop if any check fails.

- [ ] **Step 2: Launch exactly one formal treatment**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  bash cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
    rescue-cohort-seed41 \
    --build-dir build.reuse_actor \
    --out-dir output/pfq_independent_reuse/formal
'
```

Expected: exactly one `1000 + 512` hierarchical treatment; runner and report gate exit `0`. Do not launch another phase or rerun on failure.

- [ ] **Step 3: Independently verify JSON/report identities**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python - <<'PY'
import json
from pathlib import Path

root = Path("output/pfq_independent_reuse/formal")
phase = root / "rescue_cohort_seed41"
summary = json.loads((phase / "eval_summary.json").read_text())
report = json.loads((phase / "rescue_cohort_report.json").read_text())
reference = json.loads((root / "hierarchical_decode_seed41/eval_summary.json").read_text())
assert report["valid"] is True
assert summary["stats_count"] == 512
assert summary["diagnostic_decision_count"] == 26112.0
assert summary["independent_deletion_equal"] is True
assert summary["stats_sum_action_replay_mismatch_count"] == 0.0
assert summary["stats_sum_illegal_action_count"] == 0.0
for key in (
    "stats_sum_scheduled_bytes",
    "stats_sum_ue_goodput_bytes",
    "stats_sum_failed_bytes",
    "measured_tb_tx_count",
    "measured_tb_err_count",
    "diagnostic_fragmentation_rescue_count",
    "diagnostic_selected_forced_blank_count",
    "diagnostic_selected_fragmentation_blank_count",
    "diagnostic_selected_conservative_blank_count",
):
    assert summary[key] == reference[key], key
assert summary["stats_sum_scheduled_bytes"] == summary["stats_sum_ue_goodput_bytes"] + summary["stats_sum_failed_bytes"]
assert report["classification"] in {
    "reliability_supported",
    "service_value_supported",
    "acceptance_supported",
    "inconclusive",
}
print(report["classification"], report["next_experiment"])
PY
'
```

Expected: all assertions pass and exactly one classification/next experiment prints.

- [ ] **Step 4: Verify final hygiene**

Recompute checkpoint, simulator, and protected-root hashes; require all frozen constants. Require tracked/staged diffs clean and confirm output evidence remains ignored/untracked. Do not delete the output or safety stash.

- [ ] **Step 5: Report the research result and stop**

Report only:

```text
what changed: behavior-invariant UE x TTI diagnostic only
what was tested: frozen A8 hierarchical seed41, 1000+512
validity: identities, action invariance, byte/TB equality, cohort denominators
result: both cohort rows, overall deltas, populated-stratum consistency, auxiliary depletion counts
classification: exactly one predeclared mechanism
hypothesis: supported or not supported under the gate definition
next experiment: the single report-selected branch
main remaining uncertainty: UE x TTI association is not individual-PRG causality and is one seed
```

Do not implement or launch the next experiment in this plan.

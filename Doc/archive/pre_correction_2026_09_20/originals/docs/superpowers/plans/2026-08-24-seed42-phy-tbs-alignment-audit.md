# Seed42 PHY/TBS Alignment Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether causal, decision-time PHY state exactly reconstructs simulator new-TX TBS and whether the reconstructed cumulative TBS has non-negative actor-order prefix marginals.

**Architecture:** Extend the existing online diagnostic protocol with audit-only causal inputs and post-decision reference labels, then join them with the existing exact PRG-set and raw-service trace in one offline Python gate. The actor, checkpoint, decoder, reward, and action path remain unchanged; the single seed42 run stops after a versioned read-only report.

**Tech Stack:** C++17/CUDA online simulator bridge, Python 3, NumPy, PyTorch, pytest, Bash runner.

**Spec:** `docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md`

## Global Constraints

- Run only frozen B4 on seed42 with 1000 warmup + 512 measured TTI.
- Use runtime candidate width 14 and checkpoint tensor hash `4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571`.
- Causal inputs are captured before the current action; OLLA state excludes the current TTI result.
- Actual MCS, layer, and simulator TBS are reference labels only and are deleted for the deletion-equality test.
- Prefix `k` is the first `k` PRGs actually awarded to a UE in actor order `0..16`.
- No actor, weight, reward, PRG order, candidate order, decoder, or Type-0 mapping change.
- Do not run a policy smoke, formal pair, seed41/43, C/D, or training.
- Do not commit automatically: this linked worktree contains pre-existing user/research changes.

---

### Task 1: Add causally separated PHY/TBS audit diagnostics

**Files:**
- Modify: `cuMAC/src/4T4R/mcsSelectionLUT.cuh`
- Modify: `cuMAC/src/4T4R/mcsSelectionLUT.cu`
- Modify: `cuMAC/src/4T4R/multiCellLayerSel.cuh`
- Modify: `cuMAC/src/4T4R/multiCellLayerSel.cu`
- Modify: `cuMAC/examples/network.h`
- Modify: `cuMAC/examples/network.cu`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineProtocol.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Modify: `training/stageb_hgraph/online_protocol.py`
- Modify: `training/stageb_hgraph/bridge_env.py`
- Test: `training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py`

**Interfaces:**
- Produces protocol-v14 audit arrays keyed by active UE: decision-pre MCS SINR input, decision-pre OLLA delta, prior TB result actually applied after deterministic slot mapping, per-PRG layer capability, actual MCS, actual layer, and simulator pre-dequeue TBS capacity.
- Produces fixed configuration metadata: MCS LUT type/cap, singular-value threshold, PRBs per PRG, data symbols, effective-layer cap, and a deterministic configuration hash.

- [x] **Step 1: Write failing protocol tests**

Add a complete synthetic v14 state payload and assert that causal arrays and reference arrays decode separately, with exact shapes `[U]` and `[U,P]`. Name the mutation caught: swapping causal/reference serialization order or silently accepting a wrong length.

```python
assert obs["decision_pre_olla_delta"].shape == (n_ue,)
assert obs["decision_pre_prg_layer_cap"].shape == (n_ue, n_prg)
assert obs["actual_ue_mcs"].shape == (n_ue,)
assert obs["actual_ue_tbs_capacity_bytes"].tolist() == [3824, 0]
```

- [x] **Step 2: Run the focused test in the container and verify RED**

Run:

```bash
./cuPHY-CP/container/attach_aerial.sh
source .venv/bin/activate
pytest training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py -q
```

Expected: failure because protocol v14 and PHY/TBS audit fields do not exist.

- [x] **Step 3: Implement the minimum protocol and simulator capture**

Capture causal state before MCS/layer execution. Map the already-captured prior TB result through the deterministic final slot mapping, but never read the current PHY result. Capture reference labels only after layer/MCS execution. Export simulator capacity before `min(TBS, buffer)` through the same `determineTbsPdsch` semantics used for dequeue.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the same focused pytest command and require zero failures.

### Task 2: Implement independent cumulative-TBS reconstruction and exact gates

**Files:**
- Create: `training/stageb_hgraph/phy_tbs_alignment_audit.py`
- Create: `training/stageb_hgraph/tests/test_phy_tbs_alignment_audit.py`

**Interfaces:**
- Produces `determine_tbs_bytes(rb_size, data_symbols, layers, code_rate, qam_order) -> int` using literal MCS tables and simulator rounding semantics.
- Produces `reconstruct_ue_tti(...) -> PhyTbsAuditRow`, including reconstructed MCS, final layer, final TBS, actor-order prefix TBS, marginals, and reference parity fields.
- Produces `build_phy_tbs_alignment_report(rows, ...) -> dict` with exact parity, deletion equality, monotonicity, ratio, WAPE, cross-stratum, and controlled-cell gates.

- [x] **Step 1: Write hand-derived failing TBS and prefix tests**

Use literal expected values independent of the implementation. Cover the small-TBS table branch, large-TBS branch, prefix order `{2, 7, 12}` rather than local range `0..2`, a zero marginal, and a constructed negative marginal.

```python
assert determine_tbs_bytes(16, 11, 1, 120 / 1024, 2) == 504
assert row.prefix_local_prgs == ((2,), (2, 7), (2, 7, 12))
assert zero_marginal_row.marginals == (984, 0)
assert negative_marginal_row.marginals[-1] == -2496
```

- [x] **Step 2: Run the new tests and verify RED**

Expected: import failure because `phy_tbs_alignment_audit.py` does not exist.

- [x] **Step 3: Implement minimal reconstruction**

Reconstruct MCS from decision-pre SINR, prior OLLA delta, and prior TB result; reconstruct each prefix layer from the selected PRGs' causal layer capabilities; then call the independent TBS function. Do not accept cell ID or any reference label as a reconstruction input.

- [x] **Step 4: Add failing exact-parity and deletion-equality tests**

Assert record-level MCS/layer/TBS equality; then delete actual MCS/layer/TBS and verify the reconstructed tuple remains bitwise equal. Assert one mismatching record fails before statistical gates.

- [x] **Step 5: Implement report classification and frozen gates**

Implement these ordered classifications:

```text
decision_time_phy_path_not_exact
final_tbs_reconstructable_but_not_sequentially_additive
conditional_alignment_gate_failure
phy_tbs_alignment_compatible
```

Use exact actual-PRG count crossed with pre-frozen equal-count quality tertiles. A formal stratum needs at least 50 UE-TTI and 200 PRGs. A controlled cell×stratum residual is formal at 30 UE-TTI and 100 PRGs and is measured against its pooled stratum; its maximum absolute residual is 0.10. Require global ratio `[0.9,1.1]`, stratum ratio absolute error at most `0.15`, and raw-to-TBS WAPE improvement at least `25%`.

- [x] **Step 6: Run the new test file and verify GREEN**

Run:

```bash
pytest training/stageb_hgraph/tests/test_phy_tbs_alignment_audit.py -q
```

### Task 3: Join the actor trace and enforce reference-label isolation

**Files:**
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Modify: `training/stageb_hgraph/service_accounting_audit.py`
- Modify: `training/stageb_hgraph/tests/test_service_accounting_audit.py`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`

**Interfaces:**
- Consumes the v14 bridge arrays and existing `ServiceAccountingDecisionDiagnostics` rows.
- Produces `phy_tbs_alignment_audit.json` and embeds the same report in `eval_summary.json` only when `--independent-phy-tbs-alignment-audit 1` is set.

- [x] **Step 1: Write failing join tests**

Assert exact planned/accepted/actual UE-local PRG set equality, actor-order prefix construction, HARQ=0, and eligibility based on `pre_schedule_buffer >= simulator_tbs_capacity` rather than raw prediction.

- [x] **Step 2: Run focused join tests and verify RED**

Run the two focused test files; require failures specifically for missing PHY/TBS join behavior.

- [x] **Step 3: Implement the minimum evaluation path**

Preserve the original raw-service rows for same-cohort WAPE. Join causal and reference fields only after the action response, invoke reconstruction once, then rerun reconstruction with the reference dictionary removed and compare serialized outputs exactly.

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
pytest training/stageb_hgraph/tests/test_service_accounting_audit.py \
       training/stageb_hgraph/tests/test_independent_online_path.py -q
```

### Task 4: Version, execute once, and stop

**Files:**
- Modify: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh`
- Modify: `docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md`
- Create at runtime: `output/pfq_independent_reuse/formal/seed42_phy_tbs_alignment_audit_v1/`

**Interfaces:**
- Adds phase `audit-seed42-phy-tbs-alignment`.
- Produces `run_manifest.json` with source/binary/checkpoint/config hashes and `audit_report.json` with the frozen ordered classification.

- [x] **Step 1: Write a failing runner contract test**

Extend the existing runner test to execute the manifest/report validation logic against a temporary fixture and assert that an existing output directory fails closed and no forbidden phase is launched.

- [x] **Step 2: Verify the runner test is RED, then implement the phase**

The phase creates a new directory, validates the original seed42 structural pass and frozen checkpoint hash, runs exactly 1512 TTI, and rejects any missing causal/reference/config field.

- [x] **Step 3: Run all Python tests before building**

Run inside the container and virtual environment:

```bash
pytest training/stageb_hgraph/tests -q
```

- [x] **Step 4: Build the simulator and rerun protocol tests**

Use the existing runner build phase, then rerun the online protocol and PHY/TBS audit tests against the newly built binary provenance.

- [x] **Step 5: Run the one authorized audit**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh audit-seed42-phy-tbs-alignment formal
```

Do not launch any subsequent phase regardless of result.

- [x] **Step 6: Verify the versioned report and frozen boundaries**

Check 1512 completed / 512 measured TTI, seed42, runtime width 14, exact checkpoint and binary hashes, reference deletion equality, record-level parity, prefix monotonicity, statistical gates, and absence of seed41/43, C/D, training, smoke, or formal-pair output under the new directory.

- [x] **Step 7: Update only the design status paragraph**

Record the audit protocol, exact result classification, evidence counts, and the mandated stop. Do not design or implement B4-TBS.

## Self-Review

- Spec coverage: causal timing, deletion equality, configuration hash, actor-order prefix, record exactness, monotonicity, zero marginals, statistical corroboration, ordered classification, and stop scope are each assigned above.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: protocol arrays are active-UE keyed; per-PRG layer capability is `[U,P]`; references are never passed to reconstruction; report classifications are identical across tests, runner, and documentation.

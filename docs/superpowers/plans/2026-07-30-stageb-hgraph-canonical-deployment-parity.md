> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-07-30-stageb-hgraph-canonical-deployment-parity.md)保留当时状态。

# Stage-B HGraph Canonical Deployment Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make direct candidate greedy selection and Type-0 materialization identical in Python evaluation, action-mode ONNX, and C++ execution, then measure end-to-end inference latency.

**Architecture:** Implement an ONNX-exportable tensor operator for constrained greedy choice plus deterministic slot construction and use it from both the Python evaluator and `FixedHGraphActionOnnxWrapper`. Treat ONNX action outputs as authoritative in C++; C++ performs legality rejection only and never re-ranks, fills, repairs, or reapplies demand heuristics.

**Tech Stack:** PyTorch, ONNX, ONNX Runtime, NumPy, pytest, C++17, TensorRT, CMake, cuMAC.

## Global Constraints

- Phase 1 and Phase 2 unit/online gates must pass before export changes.
- Canonical deployment mode is 3 cells, 36 UE, 17 PRG, 12 candidates, 16 UE features.
- Training samples stochastically; export and C++ use constrained greedy with the same masks and demand-cap arithmetic.
- The discrete `action_ue_select` and `action_prg_alloc` tensors must match exactly, not merely the logits.
- C++ may reject illegal output but may not replace UE, add PRGs, apply fallback, or recompute candidate ranking.
- Golden parity failure blocks bucket training.
- End-to-end observation-build, inference, decode, and writeback p99 must be at most 0.5 ms; 0.4 ms is the target.

---

## File Structure

- Create `training/stageb_hgraph/canonical_action.py`: tensor-only greedy constrained selection and Type-0 materialization.
- Create `training/stageb_hgraph/tests/test_canonical_action.py`: tensor operator versus graph reference implementation.
- Modify `training/stageb_hgraph/eval_checkpoint_online.py`: use tensor canonical action in direct greedy mode.
- Modify `training/stageb_hgraph/export_onnx.py`: use the tensor operator and 16-dimensional inputs.
- Modify `training/stageb_hgraph/check_onnx_consistency.py`: compare final discrete actions over multiple golden cases.
- Create `training/stageb_hgraph/tests/test_onnx_action_parity.py`: PyTorch wrapper/ONNX Runtime final action equality.
- Modify `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`: remove semantic postprocessing in action mode.
- Modify `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.h`: expose latency counters and action-mode diagnostics.
- Create `cuMAC/examples/customScheduler/gnnrlActionGolden.cpp`: standalone JSON/NPZ-like binary-input action runner.
- Modify `cuMAC/examples/multiCellSchedulerUeSelection/CMakeLists.txt`: build `gnnrlActionGolden`.
- Create `training/stageb_hgraph/check_action_operator_parity.py`: orchestrate Python, ONNX Runtime, and C++ exact checks.
- Create `training/stageb_hgraph/tests/test_action_operator_parity_cli.py`: CLI pass/fail behavior.
- Modify `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`: log latency stages.
- Create `cuMAC/scripts/summarize_gnnrl_inference_latency.py`: p50/p95/p99 and gate report.
- Create `cuMAC/scripts/tests/test_summarize_gnnrl_inference_latency.py`: latency aggregation.
- Modify `training/stageb_hgraph/README.md`: export, parity, and latency commands.

### Task 1: Build one tensor canonical action operator

**Files:**
- Create: `training/stageb_hgraph/canonical_action.py`
- Create: `training/stageb_hgraph/tests/test_canonical_action.py`

**Interfaces:**
- Consumes fixed-shape tensors:
  - `candidate_logits [B,T,K+1]`
  - `candidate_ue_ids [B,T,K]`
  - `candidate_mask [B,T,K]`
  - `ue_prg_cap [B,U]`
  - `prg_owner_cell [T]`
  - `prg_local_index [T]`
  - `action_mask_ue [B,U]`
  - `action_mask_cell_ue [B,C,U]`
  - `slot_to_cell [B,S]`
  - `slot_valid_mask [B,S]`
- Produces `canonical_candidate_action_tensors(candidate_logits: torch.Tensor, candidate_ue_ids: torch.Tensor, candidate_mask: torch.Tensor, ue_prg_cap: torch.Tensor, prg_owner_cell: torch.Tensor, prg_local_index: torch.Tensor, action_mask_ue: torch.Tensor, action_mask_cell_ue: torch.Tensor, slot_to_cell: torch.Tensor, slot_valid_mask: torch.Tensor, usage_penalty: float, first_service_bonus: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write the failing reference-equivalence test**

For eight deterministic random seeds, build a fixture graph, run Phase 1 greedy selection plus materialization, and compare:

```python
def graph_reference_action(graph, packed, logits):
    cfg = CandidatePolicyConfig(
        mode="greedy",
        demand_cap_slack_bytes=0.0,
        hard_cap=True,
        usage_penalty=0.4,
        first_service_bonus=0.01,
    )
    trace = select_candidate_policy_action(
        graph, packed, packed.mask, logits, cfg
    )
    action = materialize_candidate_choices_to_type0(
        graph, packed, trace.choice
    )
    return trace, action


def tensor_action_for_graph(graph, packed, logits):
    cap = _estimate_ue_prg_demand_cap(graph, slack_bytes=0.0)
    action_mask_ue = graph.snapshot.static_meta.action_mask_ue
    layout = build_type0_slot_layout(
        graph.snapshot.static_meta.action_mask_cell_ue.unsqueeze(0),
        n_sched_ue=graph.snapshot.static_meta.n_sched_ue,
        action_mask_ue=action_mask_ue.unsqueeze(0),
    )
    return canonical_candidate_action_tensors(
        candidate_logits=logits.unsqueeze(0),
        candidate_ue_ids=packed.ue_ids.unsqueeze(0),
        candidate_mask=packed.mask.unsqueeze(0),
        ue_prg_cap=cap.unsqueeze(0),
        prg_owner_cell=graph.snapshot.static_meta.prg_owner_cell,
        prg_local_index=graph.snapshot.static_meta.prg_local_index,
        action_mask_ue=action_mask_ue.unsqueeze(0),
        action_mask_cell_ue=graph.snapshot.static_meta.action_mask_cell_ue.unsqueeze(0),
        slot_to_cell=layout.slot_to_cell,
        slot_valid_mask=layout.slot_valid_mask,
        usage_penalty=0.4,
        first_service_bonus=0.01,
    )


@pytest.mark.parametrize("seed", range(8))
def test_tensor_canonical_matches_graph_reference(seed):
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    logits = torch.randn(
        (packed.num_prg_tokens, packed.max_candidates + 1),
        generator=torch.Generator().manual_seed(seed),
    )
    reference_trace, reference = graph_reference_action(graph, packed, logits)
    tensor_ue, tensor_prg, tensor_choice = tensor_action_for_graph(
        graph, packed, logits
    )
    assert torch.equal(tensor_choice[0], reference_trace.choice)
    assert torch.equal(tensor_ue[0], reference.action_ue_select)
    assert torch.equal(tensor_prg[0], reference.action_prg_alloc)
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_canonical_action.py -v
```

Expected: FAIL because `canonical_action.py` is missing.

- [ ] **Step 3: Implement fixed-shape constrained greedy**

Use tensor operations and loops bounded by fixed `T`, `C`, `U`, and `S`. The greedy visitation order is descending detached best valid main logit, matching Phase 1. At every step:

- compute headroom mask from `ue_prg_cap - ue_prg_used`;
- apply usage penalty and first-service bonus to main candidates;
- append the unchanged blank logit as a valid class;
- choose `argmax`;
- increment usage only for a non-blank valid choice.

Do not use Python graph objects, dictionaries, NumPy, `.item()`-driven data-dependent branches, or custom autograd.

- [ ] **Step 4: Implement deterministic slot tensors**

For each cell, derive each UE's first selected token index, rank valid UEs by that index, map ranks to valid cell slots, then scatter slot IDs into native PRG order. Produce `-1` for blank or capacity failure.

- [ ] **Step 5: Add edge-case tests**

Cover all blank, one UE repeated over all PRGs, demand cap zero, exactly full slot capacity, and one more chosen UE than slot capacity. The tensor and graph reference outputs must match exactly.

- [ ] **Step 6: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_canonical_action.py -v
git add training/stageb_hgraph/canonical_action.py training/stageb_hgraph/tests/test_canonical_action.py
git commit -m "feat: add canonical tensor candidate action"
```

### Task 2: Route Python evaluation through the canonical tensor operator

**Files:**
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Modify: `training/stageb_hgraph/tests/test_canonical_action.py`

**Interfaces:**
- Consumes: direct checkpoint actor outputs and graph masks/layout
- Produces: evaluator action tensors from `canonical_candidate_action_tensors`

- [ ] **Step 1: Write the failing evaluator equivalence test**

Patch the graph reference materializer to raise, invoke direct greedy `_select_joint_action`, and assert its UE/PRG action equals a direct call to the tensor operator.

- [ ] **Step 2: Run the focused test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_canonical_action.py -k evaluator -v
```

Expected: FAIL because evaluator still calls the graph reference path.

- [ ] **Step 3: Replace the direct greedy branch**

Use `pack_prg_candidates`, build candidate masks and demand caps, build Type-0 slot layout, and call `canonical_candidate_action_tensors`. Keep the graph reference path in tests as an oracle but not in production direct evaluation.

- [ ] **Step 4: Preserve diagnostic semantics**

Derive `chosen_ue_per_prg`, blank ratio, unique UE count, preflight reject counts, and override count from tensor outputs. Set direct non-hard override count to zero; report capacity/legality failures separately.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_canonical_action.py \
  training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -v
git add training/stageb_hgraph/eval_checkpoint_online.py \
  training/stageb_hgraph/tests/test_canonical_action.py
git commit -m "refactor: use canonical tensor action in evaluation"
```

### Task 3: Export the same final action to ONNX

**Files:**
- Modify: `training/stageb_hgraph/export_onnx.py`
- Modify: `training/stageb_hgraph/check_onnx_consistency.py`
- Create: `training/stageb_hgraph/tests/test_onnx_action_parity.py`

**Interfaces:**
- Consumes: `FixedHGraphActionOnnxWrapper` candidate logits and fixed masks
- Produces action-mode ONNX outputs `action_ue_select [1,36]` and `action_prg_alloc [1,51]`

- [ ] **Step 1: Write the failing wrapper parity test**

Instantiate the action wrapper from a synthetic direct PFQ-residual checkpoint, feed `make_dummy_inputs` with 16 UE features, and assert wrapper outputs equal Python evaluator outputs for the same tensors.

- [ ] **Step 2: Run the PyTorch wrapper test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_onnx_action_parity.py::test_action_wrapper_matches_python_canonical_action -v
```

Expected: FAIL because the wrapper owns a separate budget/materialization implementation.

- [ ] **Step 3: Replace wrapper-specific action logic**

In `FixedHGraphActionOnnxWrapper.forward`, retain fixed graph encoding and candidate packing, but replace `_budgeted_candidate_choice` and `_actions_from_candidate_logits` with one call to `canonical_candidate_action_tensors`. Delete the duplicate methods after parity tests pass.

Set default action export input metadata to:

```python
"n_cell": 3,
"n_ue": 36,
"n_sched_ue": 36,
"n_prg": 17,
"ue_feat_dim": 16,
"max_prg_candidates": 12,
"candidate_policy_mode": "direct",
```

- [ ] **Step 4: Add ONNX Runtime exact-action test**

Export to a pytest temporary directory, load with ONNX Runtime, and compare both integer-valued outputs using `np.array_equal`. Test at least four dummy observations with changed backlog, TTL pressure, masks, and post-EQ SINR.

- [ ] **Step 5: Run ONNX checks**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_onnx_action_parity.py -v
.venv/bin/python training/stageb_hgraph/export_onnx.py \
  --checkpoint training/stageb_hgraph/runs/causal_residual/ppo_actor_best.pt \
  --output-mode action \
  --out training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --check
```

Expected: pytest passes, ONNX checker reports `ok`, and ONNX Runtime final action max absolute difference is 0.

- [ ] **Step 6: Commit**

```bash
git add training/stageb_hgraph/export_onnx.py \
  training/stageb_hgraph/check_onnx_consistency.py \
  training/stageb_hgraph/tests/test_onnx_action_parity.py
git commit -m "feat: export canonical HGraph Type-0 actions"
```

Do not commit generated ONNX unless repository policy explicitly tracks deployment artifacts.

### Task 4: Make C++ action mode legality-only

**Files:**
- Modify: `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.h`
- Modify: `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`
- Create: `cuMAC/examples/customScheduler/gnnrlActionGolden.cpp`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/CMakeLists.txt`

**Interfaces:**
- Consumes: action-mode ONNX output tensors
- Produces:
  - authoritative Type-0 action with illegal entries dropped, never replaced
  - `gnnrlActionGolden --model <onnx> --input <binary> --output <json>`
  - packed little-endian golden input with these exact headers:

```cpp
#pragma pack(push, 1)
struct GoldenFileHeader {
    uint32_t magic;        // 0x48474143, ASCII "HGAC"
    uint16_t version;      // 1
    uint16_t tensorCount;
};

struct GoldenTensorHeader {
    char name[64];         // NUL-terminated ONNX input name
    uint8_t dtype;         // 1=float32, 2=int64, 3=uint8
    uint8_t rank;          // 1 through 8
    uint16_t reserved;     // zero
    uint64_t shape[8];     // unused entries are zero
    uint64_t byteCount;    // immediately following payload size
};
#pragma pack(pop)
```

Tensor records follow in ONNX model-input order. Each tensor header is followed
immediately by one contiguous C-order payload. A JSON sidecar repeats name,
dtype, shape, byte offset, and SHA-256 for diagnostic inspection.

The runtime legality-only interface uses:

```cpp
struct ActionFinalizeDiagnostics {
    uint32_t droppedIllegalCount = 0;
    uint32_t replacedCount = 0;
    uint32_t filledCount = 0;
};

struct ActionLegalityView {
    int32_t nCell;
    int32_t nUe;
    const std::vector<uint8_t>& actionMaskUe;       // [U]
    const std::vector<uint8_t>& actionMaskCellUe;   // row-major [C,U]
    const std::vector<int16_t>& slotToCell;         // [S]
    const std::vector<int16_t>& prgOwnerCell;       // native [P*C], value=index % C
};
```

- [ ] **Step 1: Add a failing C++ golden case**

Create a small golden executable test case containing an ONNX-produced action where one slot is intentionally illegal. Expected JSON:

```json
{
  "action_ue_select": [-1, 1, -1],
  "action_prg_alloc": [-1, 1, -1],
  "dropped_illegal_count": 1,
  "replaced_count": 0,
  "filled_count": 0
}
```

The executable test must fail against the current runtime if it performs demand-cap re-ranking or fallback fill.

- [ ] **Step 2: Build and run the failing harness**

```bash
cmake --build build.$(uname -m) --target gnnrlActionGolden -j2
ctest --test-dir build.$(uname -m) -R gnnrl_action_golden --output-on-failure
```

Expected before implementation: the test is absent or fails.

- [ ] **Step 3: Split action finalization from inference**

Add a testable method:

```cpp
ActionFinalizeDiagnostics finalizeAuthoritativeAction(
    const std::vector<int32_t>& modelUe,
    const std::vector<int16_t>& modelPrg,
    const ActionLegalityView& legality,
    std::vector<int32_t>& finalUe,
    std::vector<int16_t>& finalPrg) const;
```

It copies legal entries, drops invalid/duplicate/wrong-cell/out-of-range references, and increments diagnostics. It must never choose another UE or allocate a blank PRG.

- [ ] **Step 4: Remove action-mode semantic postprocessing**

For action output mode, bypass native candidate re-ranking, demand-cap fill, and fallback logic. Preserve those routines only for logits/legacy mode and print an explicit runtime mode line:

```text
GNNRL_ACTION_SEMANTICS canonical_authoritative legality_only=1
```

- [ ] **Step 5: Run C++ tests and build**

```bash
cmake --build build.$(uname -m) --target gnnrlActionGolden multiCellSchedulerUeSelection -j2
ctest --test-dir build.$(uname -m) -R gnnrl_action_golden --output-on-failure
```

Expected: golden test passes and both targets build.

- [ ] **Step 6: Commit**

```bash
git add cuMAC/examples/customScheduler/GnnRlPolicyRuntime.h \
  cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp \
  cuMAC/examples/customScheduler/gnnrlActionGolden.cpp \
  cuMAC/examples/multiCellSchedulerUeSelection/CMakeLists.txt
git commit -m "feat: make ONNX Type-0 actions authoritative in C++"
```

### Task 5: Add three-way golden parity

**Files:**
- Create: `training/stageb_hgraph/check_action_operator_parity.py`
- Create: `training/stageb_hgraph/tests/test_action_operator_parity_cli.py`

**Interfaces:**
- Consumes: checkpoint, ONNX, C++ golden executable, deterministic generated cases
- Produces: JSON report with per-case Python/ONNX/C++ actions and mismatch counts

- [ ] **Step 1: Write failing CLI tests**

Mock the three runners with equal outputs and one mismatching output. Assert equal returns exit 0/`passed=true`; mismatch returns exit 1 and identifies `case_id`, tensor name, and first differing index.

- [ ] **Step 2: Run the tests**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_action_operator_parity_cli.py -v
```

Expected: FAIL because the CLI is missing.

- [ ] **Step 3: Implement deterministic case generation**

Generate at least 16 cases using seeds 0--15. Vary:

- one/all UE demand masks;
- critical and near-expiry backlog;
- post-EQ SINR ties and clear winners;
- blank-dominant logits;
- per-UE caps of 0, 1, and multiple PRGs.

Serialize each C++ input as a documented little-endian binary record with a JSON sidecar containing shapes and dtypes.

- [ ] **Step 4: Implement exact comparison**

Convert all actions to `int64`, use `np.array_equal`, and emit:

```json
{
  "passed": true,
  "case_count": 16,
  "python_onnx_mismatch_count": 0,
  "python_cpp_mismatch_count": 0
}
```

- [ ] **Step 5: Run the real parity command**

```bash
.venv/bin/python training/stageb_hgraph/check_action_operator_parity.py \
  --checkpoint training/stageb_hgraph/runs/causal_residual/ppo_actor_best.pt \
  --onnx training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx \
  --cpp-runner build.$(uname -m)/cuMAC/examples/multiCellSchedulerUeSelection/gnnrlActionGolden \
  --out training/stageb_hgraph/exports/action_parity_report.json
```

Expected: exit 0 and all mismatch counts are zero.

- [ ] **Step 6: Commit**

```bash
git add training/stageb_hgraph/check_action_operator_parity.py \
  training/stageb_hgraph/tests/test_action_operator_parity_cli.py
git commit -m "test: verify Python ONNX and C++ action parity"
```

Do not commit generated binary cases or reports.

### Task 6: Instrument and gate inference latency

**Files:**
- Modify: `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.h`
- Modify: `cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Create: `cuMAC/scripts/summarize_gnnrl_inference_latency.py`
- Create: `cuMAC/scripts/tests/test_summarize_gnnrl_inference_latency.py`
- Modify: `training/stageb_hgraph/README.md`

**Interfaces:**
- Consumes per-TTI timings in microseconds
- Produces CSV columns `tti,feature_us,inference_us,finalize_us,writeback_us,total_us` and a JSON gate report

- [ ] **Step 1: Write the failing latency summary test**

```python
def write_latency_csv(tmp_path: Path, total_us: list[float]) -> Path:
    path = tmp_path / "latency.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tti", "feature_us", "inference_us", "finalize_us", "writeback_us", "total_us"],
        )
        writer.writeheader()
        for tti, total in enumerate(total_us):
            writer.writerow({
                "tti": tti,
                "feature_us": total * 0.2,
                "inference_us": total * 0.6,
                "finalize_us": total * 0.1,
                "writeback_us": total * 0.1,
                "total_us": total,
            })
    return path


def test_latency_summary_reports_percentiles_and_gate(tmp_path):
    csv_path = write_latency_csv(tmp_path, total_us=[300.0, 350.0, 400.0, 450.0])
    report = summarize_latency(csv_path, budget_us=500.0, warmup_rows=1)
    assert report["p99_total_us"] <= 500.0
    assert report["passed"] is True
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest cuMAC/scripts/tests/test_summarize_gnnrl_inference_latency.py -v
```

Expected: FAIL because the summarizer is missing.

- [ ] **Step 3: Add stage timers**

Use `std::chrono::steady_clock` around:

1. observation/feature tensor preparation;
2. TensorRT enqueue and synchronization;
3. authoritative action finalization;
4. action writeback.

Write one CSV row per TTI when `CUMAC_GNNRL_LATENCY_CSV` is set. Do not enable per-TTI console prints in the timed path.

- [ ] **Step 4: Implement summary and hard gate**

Read finite non-negative values, drop the configured warm-up rows, compute NumPy p50/p95/p99/max/mean, and exit non-zero if p99 total exceeds `--budget-us`.

- [ ] **Step 5: Run tests and a 4000-TTI pilot**

```bash
.venv/bin/python -m pytest cuMAC/scripts/tests/test_summarize_gnnrl_inference_latency.py -v
CUMAC_GNNRL_LATENCY_CSV=output/hgraph_onnx_seed41/latency.csv \
  ./cuMAC/scripts/run_stageB_main_experiment.sh \
    --topology-seed 41 \
    --tti-count 4000 \
    --custom-policy gnnrl \
    --gnnrl-model-output-mode action \
    --gnnrl-model-path training/stageb_hgraph/exports/hgraph_3cell_36ue_17prg_action_posteq.onnx
.venv/bin/python cuMAC/scripts/summarize_gnnrl_inference_latency.py \
  --csv output/hgraph_onnx_seed41/latency.csv \
  --warmup-rows 1000 \
  --budget-us 500 \
  --out output/hgraph_onnx_seed41/latency_summary.json
```

Expected: unit test passes; pilot produces a report. If p99 exceeds 500 µs, mark deployment gate failed while retaining research-mode functionality.

- [ ] **Step 6: Run the native semantic pilot**

Compare the ONNX seed41 action-mode run with the Python canonical seed41 run using identical traffic/topology and post-1000 window. Require action mismatch count 0 and no large utilization collapse attributable to action semantics.

- [ ] **Step 7: Document and commit**

```bash
git add cuMAC/examples/customScheduler/GnnRlPolicyRuntime.h \
  cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp \
  cuMAC/examples/multiCellSchedulerUeSelection/main.cpp \
  cuMAC/scripts/summarize_gnnrl_inference_latency.py \
  cuMAC/scripts/tests/test_summarize_gnnrl_inference_latency.py \
  training/stageb_hgraph/README.md
git commit -m "feat: gate canonical HGraph deployment latency"
```

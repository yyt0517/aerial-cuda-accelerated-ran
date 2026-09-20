# Stage-B HGraph PRG Granularity Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether 68 PRG × 4 PRB provides enough frequency-selective gain over 17 PRG × 16 PRB to justify its larger action space and deployment cost.

**Architecture:** Parameterize the already aligned candidate policy, canonical tensor action, exporter, and parity fixtures by PRG count without changing model semantics. Train and evaluate matched 17/68-PRG variants with identical traffic, seeds, optimizer budget measured in environment TTIs, and KPI gates; choose 68 PRG only if it improves paired performance without violating latency or packet gates.

**Tech Stack:** Python 3, PyTorch, pytest, ONNX, C++17/TensorRT, existing Stage-B experiment runners.

## Global Constraints

- Start only after causal, residual/tail, deployment parity, and seed41--50 bucket gates pass at 17 PRG.
- Keep total scheduled bandwidth constant at 272 PRB: 17 × 16 PRB versus 68 × 4 PRB.
- Keep TTI 0.5 ms, 3 cells, 36 UE, `max_candidates=12`, HGraph hidden size 128, and two message layers.
- Use the same training TTIs, seed schedule, reward weights, actor/critic learning rates, PPO epochs, and checkpoint selection gate.
- Do not reuse a 17-PRG ONNX artifact as a 68-PRG deployment artifact.
- Promote 68 PRG only if mean goodput improves by at least 0.2% relative to the aligned 17-PRG actor, worst-seed and packet gates remain satisfied, and deployment p99 remains at or below 0.5 ms.

---

## File Structure

- Modify `training/stageb_hgraph/tests/graph_fixtures.py`: parameterized 17/68 PRG fixtures.
- Modify `training/stageb_hgraph/tests/test_canonical_action.py`: reference equivalence at both granularities.
- Modify `training/stageb_hgraph/export_onnx.py`: explicit `--n-prg`/shape metadata validation.
- Modify `training/stageb_hgraph/check_action_operator_parity.py`: parity cases at selected shape.
- Modify `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`: explicit `--prbs-per-group` frozen in run metadata.
- Create `training/stageb_hgraph/run_prg_granularity_ablation.py`: matched train/eval orchestration.
- Create `training/stageb_hgraph/tests/test_run_prg_granularity_ablation.py`: command and decision tests.
- Create `cuMAC/scripts/compare_stageB_hgraph_granularity.py`: paired KPI/latency comparison.
- Create `cuMAC/scripts/tests/test_compare_stageB_hgraph_granularity.py`: promotion rules.
- Modify `training/stageb_hgraph/README.md`: A/B commands and final decision.

### Task 1: Parameterize action and parity tests by PRG count

**Files:**
- Modify: `training/stageb_hgraph/tests/graph_fixtures.py`
- Modify: `training/stageb_hgraph/tests/test_canonical_action.py`
- Modify: `training/stageb_hgraph/export_onnx.py`
- Modify: `training/stageb_hgraph/check_action_operator_parity.py`

**Interfaces:**
- Consumes: `n_prg` in `{17, 68}`
- Produces: exact Python/tensor/ONNX action parity for `T = 3 * n_prg`

- [ ] **Step 1: Write the failing dual-shape test**

```python
@pytest.mark.parametrize("n_prg", [17, 68])
@pytest.mark.parametrize("seed", range(8))
def test_tensor_canonical_matches_graph_reference(n_prg, seed):
    graph = make_candidate_graph(n_prg=n_prg)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    logits = torch.randn(
        (packed.num_prg_tokens, packed.max_candidates + 1),
        generator=torch.Generator().manual_seed(seed),
    )
    reference_trace, reference = graph_reference_action(graph, packed, logits)
    tensor_ue, tensor_prg, tensor_choice = tensor_action_for_graph(graph, packed, logits)
    assert torch.equal(tensor_choice[0], reference_trace.choice)
    assert torch.equal(tensor_ue[0], reference.action_ue_select)
    assert torch.equal(tensor_prg[0], reference.action_prg_alloc)
    assert tensor_ue.shape == (1, graph.snapshot.static_meta.n_sched_ue)
    assert tensor_prg.shape == (1, 3 * n_prg)
    assert tensor_choice.shape == (1, 3 * n_prg)
```

During the canonical deployment plan, extract its complete reference setup into `graph_reference_action(graph: SparseEntityGraph, packed: PackedPrgCandidates, logits: torch.Tensor) -> tuple[CandidatePolicyTrace, DecodedType0Action]` and its tensor call into `tensor_action_for_graph(graph: SparseEntityGraph, packed: PackedPrgCandidates, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`. Both helpers live in `test_canonical_action.py`, so this task only adds the `n_prg` parameter and shape assertions.

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_canonical_action.py \
  -k tensor_canonical_matches_graph_reference -v
```

Expected: the 68-PRG case fails at the first fixed-shape assumption.

- [ ] **Step 3: Remove hard-coded 17/51 constants**

Derive token count as `n_cell * n_prg` in Python tensor operators and parity generation. Keep action-mode ONNX fixed per artifact, but require `--n-prg` to match checkpoint/run metadata and write it to the metadata JSON.

- [ ] **Step 4: Run both shapes**

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_canonical_action.py \
  training/stageb_hgraph/tests/test_onnx_action_parity.py -v
```

Expected: all 17- and 68-PRG parameterized tests pass.

- [ ] **Step 5: Commit**

```bash
git add training/stageb_hgraph/tests/graph_fixtures.py \
  training/stageb_hgraph/tests/test_canonical_action.py \
  training/stageb_hgraph/export_onnx.py \
  training/stageb_hgraph/check_action_operator_parity.py
git commit -m "refactor: parameterize canonical HGraph PRG shape"
```

### Task 2: Build matched granularity experiment orchestration

**Files:**
- Modify: `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`
- Create: `training/stageb_hgraph/run_prg_granularity_ablation.py`
- Create: `training/stageb_hgraph/tests/test_run_prg_granularity_ablation.py`

**Interfaces:**
- Consumes: passing 17-PRG checkpoint/config and bucket seed schedule
- Produces matched `prg17/` and `prg68/` manifests with equal environment TTI budgets

- [ ] **Step 1: Write the failing command test**

Mock subprocess and assert the orchestrator emits two training commands whose
normalized algorithm/training arguments are identical and whose PRG arguments
are:

```text
prg17: --prbs-per-group 16
prg68: --prbs-per-group 4
```

Both commands must contain identical iteration count, rollout steps, seed
schedule, preset, optimizer arguments, and input checkpoint. Output
directories/run labels and the explicit PRG shape fields are excluded from the
normalization comparison.

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_run_prg_granularity_ablation.py -v
```

Expected: FAIL because the orchestrator is missing.

- [ ] **Step 3: Freeze PRG metadata**

Pass `--prbs-per-group` through the training wrapper, derive `n_prg=272/prbs_per_group`, reject non-divisors, and store both values in checkpoint args and run manifest.

- [ ] **Step 4: Implement matched orchestration**

For each variant:

1. load the same aligned actor and critic weights;
2. reset optimizer state for both variants;
3. run the same number of training TTIs;
4. run uniform seed41--50 4000-TTI evaluation with warm-up 1000;
5. export a shape-specific ONNX;
6. run action parity and latency gates.

Persist command arrays, config hashes, checkpoint hashes, and environment TTI counts.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_run_prg_granularity_ablation.py -v
git add cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  training/stageb_hgraph/run_prg_granularity_ablation.py \
  training/stageb_hgraph/tests/test_run_prg_granularity_ablation.py
git commit -m "feat: orchestrate matched HGraph PRG ablation"
```

### Task 3: Implement promotion rules and run the A/B

**Files:**
- Create: `cuMAC/scripts/compare_stageB_hgraph_granularity.py`
- Create: `cuMAC/scripts/tests/test_compare_stageB_hgraph_granularity.py`
- Modify: `training/stageb_hgraph/README.md`

**Interfaces:**
- Consumes: paired seed41--50 manifests and latency reports for 17/68 PRG
- Produces: `granularity_decision.json` with `promoted_prg_count` in `{17,68}`

- [ ] **Step 1: Write failing decision tests**

```python
def test_promotes_68_only_with_material_gain_and_all_gates():
    report = decide_granularity(
        mean_goodput_delta_pct=0.25,
        worst_seed_gate=True,
        packet_gates=True,
        pfq_gate=True,
        parity_gate=True,
        p99_latency_us=480.0,
    )
    assert report["promoted_prg_count"] == 68


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_goodput_delta_pct": 0.19},
        {"worst_seed_gate": False},
        {"packet_gates": False},
        {"pfq_gate": False},
        {"parity_gate": False},
        {"p99_latency_us": 501.0},
    ],
)
def test_keeps_17_when_any_68_gate_fails(kwargs):
    inputs = {
        "mean_goodput_delta_pct": 0.25,
        "worst_seed_gate": True,
        "packet_gates": True,
        "pfq_gate": True,
        "parity_gate": True,
        "p99_latency_us": 480.0,
    }
    inputs.update(kwargs)
    assert decide_granularity(**inputs)["promoted_prg_count"] == 17
```

- [ ] **Step 2: Run and implement**

```bash
.venv/bin/python -m pytest cuMAC/scripts/tests/test_compare_stageB_hgraph_granularity.py -v
```

Expected first run: FAIL due to missing script. Compute
`mean_goodput_delta_pct` from paired 68-versus-17 full-window seed results.
Set `worst_seed_gate` only when the minimum paired goodput delta is at least
-1.0%. Set `packet_gates` only when pooled expiry is non-worse, pooled
completion is non-worse, and cross-seed mean p95 delay is non-worse versus
17 PRG. Set `pfq_gate` from the existing seed41--50 hard-gate report for the
68-PRG run. Implement the exact rules in the tests, plus paired per-seed
goodput/BLER/utilization/p95/expiry tables. Expected second run: PASS.

- [ ] **Step 3: Run the matched experiment**

```bash
.venv/bin/python training/stageb_hgraph/run_prg_granularity_ablation.py \
  --checkpoint output/stageB_hgraph_bucket_training_fixed/best_uniform_checkpoint.pt \
  --bucket-config training/stageb_hgraph/configs/bucket_seed41_50_v1.json \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --training-tti-budget 102400 \
  --eval-tti 4000 \
  --warmup-tti 1000 \
  --out output/stageB_hgraph_prg_ablation
```

- [ ] **Step 4: Generate the decision**

```bash
.venv/bin/python cuMAC/scripts/compare_stageB_hgraph_granularity.py \
  --prg17-root output/stageB_hgraph_prg_ablation/prg17 \
  --prg68-root output/stageB_hgraph_prg_ablation/prg68 \
  --min-mean-goodput-gain-pct 0.2 \
  --latency-budget-us 500 \
  --out output/stageB_hgraph_prg_ablation/granularity_decision.json
```

Expected: exit 0; the report either promotes 68 with every gate true or retains 17 with explicit failed checks.

- [ ] **Step 5: Document and commit**

```bash
git add cuMAC/scripts/compare_stageB_hgraph_granularity.py \
  cuMAC/scripts/tests/test_compare_stageB_hgraph_granularity.py \
  training/stageb_hgraph/README.md
git commit -m "test: decide HGraph PRG granularity from paired evidence"
```

Do not commit generated experiment output or checkpoints.

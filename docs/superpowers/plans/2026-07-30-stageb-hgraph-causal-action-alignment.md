# Stage-B HGraph Causal Candidate Action Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the actually executed `(cell, PRG) -> candidate UE/blank` action own a differentiable PPO log-probability and deterministically materialize legal Type-0 UE slots without coverage repair or ranking overrides.

**Architecture:** Extract the sequential constrained candidate distribution into a focused module that returns a replayable trace. Use the same trace for rollout log-probability and PPO updates, then convert selected UE identities to Type-0 slots by first occurrence per cell; retain the legacy path behind an explicit mode for regression comparison.

**Tech Stack:** Python 3, PyTorch, pytest, NumPy, existing Stage-B online bridge and Type-0 graph utilities.

## Global Constraints

- Keep 3 cells, 36 UE, 17 PRG × 16 PRB, `max_candidates=12`, and the existing HGraph encoders unchanged.
- Direct candidate mode includes blank in every non-forced categorical distribution.
- Training mode is stochastic sample; evaluation mode is greedy over the same adjusted logits.
- Demand cap, UE legality, cell ownership, slot capacity, and PRG mask are hard constraints.
- Tail, rescue, completion, safe-fill, blank-budget repair, and coverage repair are disabled in direct mode.
- The UE slot selector log-probability is excluded from PPO in direct mode.
- Non-hard decoder override rate must be below 5%, target 0; native and Python preflight rejects must be 0.
- Preserve the legacy action path and checkpoint loading behavior unless `candidate_policy_mode=direct`.

---

## File Structure

- Create `training/stageb_hgraph/candidate_policy.py`: constrained sampling, greedy selection, replay trace, and log-probability.
- Create `training/stageb_hgraph/tests/graph_fixtures.py`: deterministic 3-cell sparse graph fixture shared by candidate tests.
- Create `training/stageb_hgraph/tests/test_candidate_policy.py`: sampling, blank, cap, replay, entropy, and gradient tests.
- Create `training/stageb_hgraph/tests/test_candidate_type0_materialization.py`: candidate UE identities to unique Type-0 slots.
- Create `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`: rollout/PPO ownership, metric, and checkpoint regression tests.
- Modify `training/stageb_hgraph/graph_ops.py`: add deterministic materialization without heuristic repair.
- Modify `training/stageb_hgraph/online_ppo.py`: store trace, use direct candidate log-probability, log gradient/override gates.
- Modify `training/stageb_hgraph/eval_checkpoint_online.py`: select greedy direct candidate actions and bypass legacy repair.
- Modify `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`: add the causal-direct preset and pass the direct-mode flag.
- Modify `training/stageb_hgraph/README.md`: document direct versus legacy action ownership and gate commands.

### Task 1: Build deterministic graph fixtures

**Files:**
- Create: `training/stageb_hgraph/tests/__init__.py`
- Create: `training/stageb_hgraph/tests/graph_fixtures.py`
- Create: `training/stageb_hgraph/tests/test_candidate_policy.py`

**Interfaces:**
- Consumes: `FeatureSnapshotBuilder`, `SparseEdgeGenerator`, `pack_prg_candidates`
- Produces: `make_candidate_observation(n_prg: int = 2, ue_feat_dim: int = 12, backlog_bytes: tuple[float, float, float, float, float, float] = (3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0)) -> dict[str, np.ndarray]`
- Produces: `make_candidate_graph(n_prg: int = 2, backlog_bytes: tuple[float, float, float, float, float, float] = (3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0)) -> SparseEntityGraph`

- [ ] **Step 1: Write the failing fixture-shape test**

```python
from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_candidate_fixture_has_three_cells_and_all_demand_ues():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    assert graph.snapshot.static_meta.n_cell == 3
    assert graph.snapshot.static_meta.n_prg == 2
    assert packed.ue_ids.shape == (6, 2)
    assert packed.mask.all()
```

- [ ] **Step 2: Run the test and verify import failure**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_candidate_policy.py::test_candidate_fixture_has_three_cells_and_all_demand_ues -v
```

Expected: FAIL because `graph_fixtures` does not exist.

- [ ] **Step 3: Implement the fixture**

Build an observation with 3 cells, 2 legal UE per cell, configurable PRG count, positive buffer/new-data flags, all legal PRGs, directed all-pairs cell adjacency, and post-EQ SINR shaped `[6, n_prg, 1]`. Use `SnapshotBuilderConfig()` and:

```python
def make_candidate_observation(
    n_prg: int = 2,
    ue_feat_dim: int = 12,
    backlog_bytes: tuple[float, float, float, float, float, float] = (
        3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0,
    ),
) -> dict[str, np.ndarray]:
    ue = np.zeros((6, ue_feat_dim), dtype=np.float32)
    ue[:, 0] = np.asarray(backlog_bytes, dtype=np.float32)
    ue[:, 1] = 1.0
    ue[:, 2] = 10.0
    ue[:, 3] = 10.0
    ue[:, 4] = 1.0
    ue[:, 6] = 1.0
    return {
        "obs_cell_features": np.zeros((3, 5), dtype=np.float32),
        "obs_ue_features": ue,
        "obs_prg_features": np.zeros((3, n_prg, 12), dtype=np.float32),
        "action_mask_ue": np.ones((6,), dtype=np.uint8),
        "action_mask_cell_ue": np.array(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]],
            dtype=np.uint8,
        ),
        "action_mask_prg_cell": np.ones((3, n_prg), dtype=np.uint8),
        "post_eq_sinr": np.full((6, n_prg, 1), 10.0, dtype=np.float32),
        "n_sched_ue": np.int32(6),
    }


def make_candidate_graph(
    n_prg: int = 2,
    backlog_bytes: tuple[float, float, float, float, float, float] = (
        3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0,
    ),
) -> SparseEntityGraph:
    obs = make_candidate_observation(n_prg=n_prg, backlog_bytes=backlog_bytes)
    snapshot = FeatureSnapshotBuilder().build(obs)
    return SparseEdgeGenerator(
        EdgeGeneratorConfig(
            candidate_mode="all_demand_ues",
            ue_prg_topk=2,
        )
    ).build_graph(snapshot)
```

Use the actual `EdgeGeneratorConfig` field names from `edge_generator.py`; keep the fixture candidate width at two.

- [ ] **Step 4: Run the fixture test**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit the fixture**

```bash
git add training/stageb_hgraph/tests/__init__.py \
  training/stageb_hgraph/tests/graph_fixtures.py \
  training/stageb_hgraph/tests/test_candidate_policy.py
git commit -m "test: add deterministic HGraph candidate fixtures"
```

### Task 2: Implement a replayable constrained candidate distribution

**Files:**
- Create: `training/stageb_hgraph/candidate_policy.py`
- Modify: `training/stageb_hgraph/tests/test_candidate_policy.py`

**Interfaces:**
- Consumes: `SparseEntityGraph`, `PackedPrgCandidates`, candidate-valid mask, candidate logits
- Produces:
  - `CandidatePolicyConfig`
  - `CandidatePolicyTrace`
  - `select_candidate_policy_action(graph: SparseEntityGraph, packed: PackedPrgCandidates, candidate_valid: torch.Tensor, candidate_logits: torch.Tensor, cfg: CandidatePolicyConfig, generator: torch.Generator | None = None) -> CandidatePolicyTrace`
  - `replay_candidate_policy_trace(graph: SparseEntityGraph, packed: PackedPrgCandidates, candidate_logits: torch.Tensor, trace: CandidatePolicyTrace, cfg: CandidatePolicyConfig) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing sample/replay/blank tests**

Add tests that use two logits tensors requiring gradients:

```python
def fixture_policy_inputs():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    return graph, packed, packed.mask.clone()


def test_sample_trace_replays_exact_logp_and_has_entropy():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.zeros((packed.num_prg_tokens, packed.max_candidates + 1), requires_grad=True)
    generator = torch.Generator().manual_seed(7)
    trace = select_candidate_policy_action(
        graph, packed, valid, logits,
        CandidatePolicyConfig(mode="sample", demand_cap_slack_bytes=0.0),
        generator=generator,
    )
    replay_logp, replay_entropy = replay_candidate_policy_trace(
        graph, packed, logits, trace,
        CandidatePolicyConfig(mode="sample", demand_cap_slack_bytes=0.0),
    )
    assert torch.equal(trace.choice, trace.choice.clamp(0, packed.blank_class_index))
    assert torch.allclose(replay_logp, trace.logp, atol=1e-6)
    assert replay_entropy.item() > 0.0


def test_blank_is_selectable_when_its_logit_is_largest():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.zeros((packed.num_prg_tokens, packed.max_candidates + 1))
    logits[:, packed.blank_class_index] = 20.0
    trace = select_candidate_policy_action(
        graph, packed, valid, logits,
        CandidatePolicyConfig(mode="greedy", demand_cap_slack_bytes=0.0),
    )
    assert torch.equal(
        trace.choice,
        torch.full_like(trace.choice, packed.blank_class_index),
    )
```

- [ ] **Step 2: Run the focused tests**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_candidate_policy.py -k 'sample_trace or blank_is_selectable' -v
```

Expected: FAIL because `candidate_policy.py` does not exist.

- [ ] **Step 3: Add the exact dataclasses**

```python
@dataclass(frozen=True)
class CandidatePolicyConfig:
    mode: Literal["sample", "greedy"] = "sample"
    demand_cap_slack_bytes: float = 0.0
    hard_cap: bool = True
    usage_penalty: float = 0.4
    overcap_penalty: float = 8.0
    first_service_bonus: float = 0.01


@dataclass(frozen=True)
class CandidatePolicyTrace:
    choice: torch.Tensor
    order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor
    forced_step_count: int
```

`choice` and `order` are `[T]`; `valid_at_step` is `[T, K+1]` indexed by visit step. Blank is valid at every step. Hard demand caps mask only main candidates.

- [ ] **Step 4: Implement one shared selection/replay kernel**

Factor adjusted logits into:

```python
def _step_logits(
    logits_row: torch.Tensor,
    candidate_ue_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    used: torch.Tensor,
    cap: torch.Tensor,
    cfg: CandidatePolicyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = cap.gather(0, candidate_ue_ids) - used.gather(0, candidate_ue_ids)
    main_valid = candidate_valid & (remaining > 0)
    main = logits_row[:-1] - cfg.usage_penalty * used.gather(0, candidate_ue_ids).float()
    main = main + cfg.first_service_bonus * (
        (used.gather(0, candidate_ue_ids) == 0) & (cap.gather(0, candidate_ue_ids) > 0)
    ).to(main.dtype)
    if cfg.hard_cap:
        main = main.masked_fill(~main_valid, -1.0e9)
    else:
        main = main - (~main_valid).to(main.dtype) * cfg.overcap_penalty
        main = main.masked_fill(~candidate_valid, -1.0e9)
    adjusted = torch.cat((main, logits_row[-1:]), dim=0)
    valid = torch.cat((main > -1.0e8, torch.ones(1, dtype=torch.bool, device=main.device)))
    return adjusted, valid
```

Selection stores detached `order` and `valid_at_step`; sample calls `torch.multinomial` with the supplied generator argument, while greedy uses `argmax`. Replay uses the stored order and choices, reconstructs usage from preceding stored choices, applies the stored mask, and averages the categorical log-probabilities using the same reduction as rollout.

Construct `order` once before the sequential loop:

```python
initial_main = candidate_logits[:, :-1].detach().masked_fill(
    ~candidate_valid, -1.0e9
)
best_initial_main = initial_main.max(dim=-1).values
order = torch.argsort(best_initial_main, descending=True, stable=True)
```

Stable sorting makes native PRG-token index the tie breaker. The blank logit
does not affect visit order; it remains a valid categorical choice at each
visited token.

- [ ] **Step 5: Add the gradient assertion**

```python
def test_replayed_logp_backpropagates_to_selected_candidate_logits():
    graph, packed, valid = fixture_policy_inputs()
    logits = torch.randn(
        packed.num_prg_tokens,
        packed.max_candidates + 1,
        generator=torch.Generator().manual_seed(3),
        requires_grad=True,
    )
    trace = select_candidate_policy_action(
        graph, packed, valid, logits,
        CandidatePolicyConfig(mode="sample"),
        generator=torch.Generator().manual_seed(5),
    )
    logp, _ = replay_candidate_policy_trace(
        graph, packed, logits, trace, CandidatePolicyConfig(mode="sample")
    )
    (-logp).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0
```

- [ ] **Step 6: Run and commit**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_candidate_policy.py -v
```

Expected: PASS.

```bash
git add training/stageb_hgraph/candidate_policy.py training/stageb_hgraph/tests/test_candidate_policy.py
git commit -m "feat: add replayable constrained candidate policy"
```

### Task 3: Deterministically materialize candidate actions into Type-0

**Files:**
- Modify: `training/stageb_hgraph/graph_ops.py`
- Create: `training/stageb_hgraph/tests/test_candidate_type0_materialization.py`

**Interfaces:**
- Consumes: `graph`, `PackedPrgCandidates`, selected candidate classes
- Produces: `materialize_candidate_choices_to_type0(graph, packed, choice) -> DecodedType0Action`

- [ ] **Step 1: Write failing materialization tests**

Test that repeated PRGs selecting the same UE use one slot, first occurrence determines per-cell slot order, blank stays unallocated, and every PRG allocation references a slot containing the chosen UE:

```python
def test_materialization_uses_unique_slots_and_preserves_chosen_ue():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.tensor([0, 0, 1, 1, 0, packed.blank_class_index])
    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)
    selected = decoded.action_ue_select[decoded.action_ue_select >= 0]
    assert selected.numel() == torch.unique(selected).numel()
    assert decoded.non_hard_override_count == 0
    for token, ue in enumerate(decoded.chosen_ue_per_prg.tolist()):
        if ue < 0:
            continue
        native_idx = int(graph.snapshot.static_meta.prg_local_index[token]) * 3
        native_idx += int(graph.snapshot.static_meta.prg_owner_cell[token])
        slot = int(decoded.action_prg_alloc[native_idx])
        assert int(decoded.action_ue_select[slot]) == ue
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_candidate_type0_materialization.py -v
```

Expected: FAIL because `materialize_candidate_choices_to_type0` is missing.

- [ ] **Step 3: Implement deterministic slot construction**

In `graph_ops.py`:

1. Map candidate class to global UE ID with `prg_choice_to_ue_ids`.
2. For each cell, scan local PRG tokens in ascending local PRG order.
3. Record each legal chosen UE only at first occurrence.
4. Assign those UEs to the cell's `build_type0_slot_layout` slots in first-occurrence order.
5. Write the corresponding slot ID into native `[local_prg * n_cell + owner_cell]` allocation.
6. Leave illegal, over-capacity, and blank tokens at `-1`; count each reason separately.
7. Add `non_hard_override_count`, `illegal_choice_count`, and `slot_capacity_drop_count` integer fields to `DecodedType0Action`; populate its existing `final_blank_count` field.

Do not call coverage repair, safe fill, blank budget, tail quota, or fallback selection from this function.

- [ ] **Step 4: Add the legality test**

```python
def test_materialized_action_passes_type0_preflight():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    choice = torch.zeros((packed.num_prg_tokens,), dtype=torch.long)
    decoded = materialize_candidate_choices_to_type0(graph, packed, choice)
    reject = type0_action_preflight_reject_counts(
        graph, decoded.action_ue_select, decoded.action_prg_alloc
    )
    assert sum(float(value.sum()) for value in reject.values()) == 0.0
```

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_candidate_type0_materialization.py -v
git add training/stageb_hgraph/graph_ops.py training/stageb_hgraph/tests/test_candidate_type0_materialization.py
git commit -m "feat: materialize candidate actions without heuristic repair"
```

Expected: tests pass and commit succeeds.

### Task 4: Give direct candidate actions exclusive PPO ownership

**Files:**
- Modify: `training/stageb_hgraph/online_ppo.py`
- Create: `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`

**Interfaces:**
- Consumes: `CandidatePolicyTrace`, `materialize_candidate_choices_to_type0`
- Produces: direct-mode rollout steps whose `old_logp` and PPO `new_logp` refer only to the executed candidate trace

- [ ] **Step 1: Write the failing PPO ownership test**

Construct one fixture graph, actor output with `candidate_prg_logits`, a sampled trace, and a `RolloutStep`. Assert:

```python
def test_direct_mode_excludes_ue_selector_logp_and_updates_candidate_head():
    torch.manual_seed(11)
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(action_head_mode="candidate", max_prg_candidates=2)
    )
    actor_out = actor(graph)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    trace = select_candidate_policy_action(
        graph,
        packed,
        packed.mask,
        actor_out["candidate_prg_logits"],
        CandidatePolicyConfig(mode="sample"),
        generator=torch.Generator().manual_seed(11),
    )
    new_logp, entropy = _direct_candidate_logp_entropy(
        actor_out, graph, trace, CandidatePolicyConfig(mode="sample")
    )
    loss = -new_logp
    loss.backward()
    grad = actor.action_head.candidate_pair_score[-1].weight.grad
    assert torch.isfinite(loss)
    assert grad is not None and grad.abs().sum().item() > 0.0
    assert entropy.item() > 0.0
```

Add `_direct_candidate_logp_entropy(actor_out: Dict[str, torch.Tensor], graph: SparseEntityGraph, trace: CandidatePolicyTrace, cfg: CandidatePolicyConfig) -> tuple[torch.Tensor, torch.Tensor]` as the smallest replay seam used by both this test and `_ppo_update`; do not launch the simulator.

- [ ] **Step 2: Run the test**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py::test_direct_mode_excludes_ue_selector_logp_and_updates_candidate_head -v
```

Expected: FAIL because rollout steps do not store a trace and direct mode does not exist.

- [ ] **Step 3: Extend rollout state and CLI**

Append these exact fields to the existing `RolloutStep` dataclass without
changing its other fields:

```python
candidate_policy_trace: CandidatePolicyTrace | None = None
candidate_policy_mode: str = "legacy"
non_hard_override_count: float = 0.0
```

Add CLI:

```python
p.add_argument(
    "--candidate-policy-mode",
    choices=["legacy", "direct"],
    default="legacy",
)
```

Reject `direct` unless `action_head_mode == "candidate"` and `candidate_budgeted_select_mode == "sample"` during training.

- [ ] **Step 4: Replace direct rollout action selection**

In `_sample_joint_action`, branch before UE coverage repair:

```python
trace = select_candidate_policy_action(
    graph,
    packed,
    candidate_valid,
    actor_out["candidate_prg_logits"],
    CandidatePolicyConfig(
        mode="sample",
        demand_cap_slack_bytes=float(candidate_decode_demand_slack_bytes),
        hard_cap=bool(candidate_budgeted_hard_cap),
        usage_penalty=float(candidate_budgeted_usage_penalty),
        overcap_penalty=float(candidate_budgeted_overcap_penalty),
        first_service_bonus=float(candidate_budgeted_first_service_bonus),
    ),
)
decoded = materialize_candidate_choices_to_type0(graph, packed, trace.choice)
return {
    "ue_action_class": torch.full_like(actor_out["slot_prior_ue_class"], graph.snapshot.static_meta.n_ue),
    "prg_action_class": trace.choice,
    "candidate_policy_trace": trace,
    "logp": trace.logp,
    "entropy": trace.entropy,
    "action_ue_select": decoded.action_ue_select.cpu().numpy(),
    "action_prg_alloc": decoded.action_prg_alloc.cpu().numpy(),
    "chosen_ue_per_prg": decoded.chosen_ue_per_prg.cpu(),
    "non_hard_override_count": float(decoded.non_hard_override_count),
}
```

Set direct-mode tail/rescue/completion/safe-fill/blank-budget/slot-repair configuration to zero and raise `ValueError` if incompatible non-zero values are supplied.

- [ ] **Step 5: Replay traces during PPO update**

For direct-mode minibatches, evaluate the actor, then replay each stored trace against its corresponding candidate logits. Do not add UE selector log-probability:

```python
new_logp, entropy = replay_candidate_policy_trace(
    graph, packed, candidate_logits, step.candidate_policy_trace, direct_cfg
)
ratio = torch.exp(new_logp - old_logp)
policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
```

Record candidate-head gradient norm after `backward()` and before gradient clipping. Report `ue_selector_policy_logp_included=0.0`.

- [ ] **Step 6: Run direct and legacy regression tests**

Add a test that loads a synthetic legacy checkpoint dictionary without `candidate_policy_mode` and asserts it resolves to `legacy`. Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add training/stageb_hgraph/online_ppo.py training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py
git commit -m "feat: align PPO with executed candidate actions"
```

### Task 5: Align deterministic evaluation and add the causal preset

**Files:**
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Modify: `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`
- Modify: `training/stageb_hgraph/README.md`
- Modify: `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`

**Interfaces:**
- Consumes: direct-mode checkpoint args and candidate logits
- Produces: matching greedy evaluation and preset `goodput_delay_rate_v20_causal_direct_sample_fast`

- [ ] **Step 1: Write the failing evaluation-path test**

Monkeypatch legacy repair functions to raise, call `_select_joint_action` with a direct-mode checkpoint config, and assert the call succeeds with:

```python
assert metrics["slot_repair_count"] == 0.0
assert metrics["non_hard_override_count"] == 0.0
assert metrics["candidate_policy_mode_direct"] == 1.0
```

- [ ] **Step 2: Run the test**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -k evaluation -v
```

Expected: FAIL because evaluation still runs repair/legacy budget decode.

- [ ] **Step 3: Implement greedy direct evaluation**

Resolve `candidate_policy_mode` from checkpoint args. For `direct`, call `select_candidate_policy_action(graph, packed, candidate_valid, candidate_logits, CandidatePolicyConfig(mode="greedy", demand_cap_slack_bytes=demand_slack, hard_cap=hard_cap, usage_penalty=usage_penalty, overcap_penalty=overcap_penalty, first_service_bonus=first_service_bonus))`, then `materialize_candidate_choices_to_type0`. Reject CLI overrides that enable slot repair, safe fill, tail quota, rescue bias, completion bonus, or blank-budget repair.

- [ ] **Step 4: Add the causal preset**

In `run_stageB_hgraph_online_ppo.sh`, add preset `goodput_delay_rate_v20_causal_direct_sample_fast` with:

```bash
ACTION_HEAD_MODE="candidate"
CANDIDATE_MODE="all_demand_ues"
MAX_PRG_CANDIDATES=12
CANDIDATE_POLICY_MODE="direct"
CANDIDATE_BUDGETED_SELECT_MODE="sample"
CANDIDATE_BUDGETED_HARD_CAP=1
CANDIDATE_BUDGETED_USAGE_PENALTY=0.4
CANDIDATE_BUDGETED_OVERCAP_PENALTY=8.0
CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.01
SLOT_REPAIR_MAX_PER_CELL=0
SAFE_FILL_MAX_PER_CELL=0
CANDIDATE_BLANK_BUDGET_MAX_PER_CELL=-1
CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.0
CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.0
CANDIDATE_BUDGETED_COMPLETION_BONUS=0.0
CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE=0
```

Pass `--candidate-policy-mode` to `online_ppo.py` and print it in the frozen configuration.

- [ ] **Step 5: Run shell and Python regressions**

```bash
bash -n cuMAC/scripts/run_stageB_hgraph_online_ppo.sh
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -v
```

Expected: both commands succeed.

- [ ] **Step 6: Document and commit**

Document that direct mode owns PPO and legacy mode exists only for reproducibility. Then:

```bash
git add training/stageb_hgraph/eval_checkpoint_online.py \
  cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  training/stageb_hgraph/README.md \
  training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py
git commit -m "feat: add direct candidate training and evaluation preset"
```

### Task 6: Run the causal online smoke gate

**Files:**
- Create: `training/stageb_hgraph/causal_gate.py`
- Create: `training/stageb_hgraph/tests/test_causal_gate.py`
- Produce: `training/stageb_hgraph/runs/causal_direct_smoke/causal_gate.json`

**Interfaces:**
- Consumes: online PPO summary/history JSON and per-TTI diagnostics
- Produces: `evaluate_causal_gate(summary: Mapping[str, float]) -> dict[str, object]`

- [ ] **Step 1: Write the failing gate test**

```python
def test_causal_gate_requires_gradient_entropy_and_action_parity():
    report = evaluate_causal_gate({
        "candidate_policy_grad_norm": 0.2,
        "candidate_policy_entropy": 0.4,
        "non_hard_override_ratio": 0.0,
        "python_preflight_reject_count": 0.0,
        "native_reject_count": 0.0,
        "action_replay_mismatch_count": 0.0,
    })
    assert report["passed"] is True


def test_causal_gate_rejects_zero_candidate_gradient():
    report = evaluate_causal_gate({
        "candidate_policy_grad_norm": 0.0,
        "candidate_policy_entropy": 0.4,
        "non_hard_override_ratio": 0.0,
        "python_preflight_reject_count": 0.0,
        "native_reject_count": 0.0,
        "action_replay_mismatch_count": 0.0,
    })
    assert report["passed"] is False
```

- [ ] **Step 2: Run and implement the gate**

Run:

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_causal_gate.py -v
```

Expected first run: FAIL due to missing module.

Implement exact conditions:

```python
checks = {
    "candidate_gradient_nonzero": math.isfinite(grad) and grad > 0.0,
    "training_entropy_positive": math.isfinite(entropy) and entropy > 0.0,
    "non_hard_override_below_5pct": override_ratio < 0.05,
    "python_preflight_zero": python_reject == 0.0,
    "native_reject_zero": native_reject == 0.0,
    "action_replay_exact": replay_mismatch == 0.0,
}
```

Expected second run: PASS.

- [ ] **Step 3: Run the 2-iteration online smoke**

```bash
./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --ppo-preset goodput_delay_rate_v20_causal_direct_sample_fast \
  --iterations 2 \
  --rollout-steps 64 \
  --episode-horizon 128 \
  --topology-seed 41 \
  --out-dir training/stageb_hgraph/runs/causal_direct_smoke
```

Expected: two PPO iterations complete, sample entropy is positive, and candidate gradient norm is non-zero.

- [ ] **Step 4: Generate and inspect the gate report**

```bash
.venv/bin/python -m training.stageb_hgraph.causal_gate \
  --summary training/stageb_hgraph/runs/causal_direct_smoke/online_ppo_summary.json \
  --history training/stageb_hgraph/runs/causal_direct_smoke/online_ppo_metrics.csv \
  --out training/stageb_hgraph/runs/causal_direct_smoke/causal_gate.json
```

Expected: exit 0 and `"passed": true`.

- [ ] **Step 5: Run the full Phase 1 suite and commit**

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_candidate_policy.py \
  training/stageb_hgraph/tests/test_candidate_type0_materialization.py \
  training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py \
  training/stageb_hgraph/tests/test_causal_gate.py -v
git add training/stageb_hgraph/causal_gate.py training/stageb_hgraph/tests/test_causal_gate.py
git commit -m "test: gate causal HGraph action alignment"
```

Expected: all tests pass. Do not commit generated run artifacts.

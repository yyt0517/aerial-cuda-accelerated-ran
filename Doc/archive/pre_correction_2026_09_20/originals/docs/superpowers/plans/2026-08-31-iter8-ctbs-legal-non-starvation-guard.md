# Iter8 CTBS-Legal Non-Starvation Guard V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the single frozen, blank-only, CTBS-legal non-starvation guard to the iter8 independent actor evaluation path and answer on seed51 whether it restores UE4/UE6 service without violating the existing formal Gate.

**Architecture:** Keep the actor, logits, network, reward, observations, candidate construction, PRG order, and Type-0 materialization unchanged. A small pure selector evaluates only raw actor Blank decisions inside the existing 17-step loop, replaces at most one Blank per cell and TTI when the frozen three-feature predicate and exact positive CTBS marginal-capacity check pass, and emits a separate causal trace. A focused audit then combines that trace with the reused seed51 PFQ reference, the frozen unguarded actor result, and one guarded rollout; no training or additional seed is part of this plan.

**Tech Stack:** Python 3, PyTorch, pytest, the existing Aerial online bridge and C++ simulator binary, JSON/CSV research artifacts.
1
**Spec:** `docs/superpowers/specs/2026-08-28-iter8-ctbs-legal-non-starvation-guard-design.md`

## Global Constraints

- Work only in `/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor` on the existing linked worktree.
- Before Python or simulator commands, enter the container from `/home/oai2/aerial-cuda-accelerated-ran` with `./cuPHY-CP/container/attach_aerial.sh`; inside it run `source /opt/nvidia/cuBB/.venv/bin/activate` and `cd /opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor`.
- Preserve the checkpoint file SHA-256 `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`, actor-state SHA-256 `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`, and simulator SHA-256 `4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`.
- Preserve UE feature dimension `16`; slots 12--14 must be exactly `log1p_drain_pressure`, `scheduled_dequeue_adequacy`, and `ttl_normalized_service_age`.
- Freeze pressure `> log(2)`, adequacy `< 1.0`, age `>= 0.25`, one override per cell per TTI, Blank-only replacement, and strictly positive exact cumulative-TBS delta.
- Preserve fixed local PRG order `0..16`, complete runtime candidate width `17` on seed51, argmax decoding, CTBS-v1, Type-0 mapping, all PHY/topology/traffic/TTL settings, and all disabled repair/fill/budget mechanisms.
- Do not add threshold CLI arguments. The evaluator exposes only `off` or the exact `ctbs-blank-v1` configuration.
- Do not expose the guard through PPO or imitation code. Guarded decisions are evaluation/deployment actions, not policy-gradient or teacher targets.
- Phase A is exactly topology seed `51`, traffic seed `51`, `1000` warmup TTIs, and measured TTIs `1001..5000`; do not run seed48 or another seed in this plan.
- Reuse the frozen PFQ and unguarded seed51 artifacts only if the simulator/feature/checkpoint provenance and canonical offered-packet hash remain exact. A mismatch requires one fresh paired PFQ seed51 run before Gate calculation.
- Preserve unrelated dirty-worktree changes. Because core target files already contain uncommitted research changes, record per-task diffs and test evidence but do not create implementation commits unless the user separately authorizes selective staging.

## File Structure

- Create `training/stageb_hgraph/ctbs_non_starvation_guard.py`: frozen constants, pure per-PRG guard selector, detached diagnostics, CSV row conversion, and summary aggregation.
- Modify `training/stageb_hgraph/independent_policy.py`: place the guard between raw actor choice and the existing sequential demand/CTBS update; carry raw and executed actions separately through replay.
- Modify `training/stageb_hgraph/eval_checkpoint_online.py`: add the single guard mode, fail-closed validation, full guard trace export, and summary fields.
- Create `training/stageb_hgraph/ctbs_non_starvation_guard_audit.py`: validate trace invariants, provenance, business Gate, UE4/UE6 directionality, offered-stream identity, and the frozen stop classification.
- Create `training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py`: pure selector and boundary tests.
- Modify `training/stageb_hgraph/tests/test_independent_policy.py`: sequential insertion, disabled equivalence, log-probability, Type-0, and replay tests.
- Modify `training/stageb_hgraph/tests/test_independent_online_path.py`: evaluator configuration, trace, and fail-closed tests.
- Create `training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py`: audit and Phase-A classification tests.
- Create artifacts only under `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/`.

---

### Task 1: Frozen Pure Guard Selector

**Files:**
- Create: `training/stageb_hgraph/ctbs_non_starvation_guard.py`
- Create: `training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py`

**Interfaces:**
- Consumes: raw per-cell actor classes, the current packed candidate rows, current `valid_main`, remaining demand, feature tensor/names, current `CtbsHeadroomState` objects, the frozen per-UE/per-PRG layer caps, local PRG index, and per-cell used budget.
- Produces: `CtbsNonStarvationGuardConfig`, `GuardReason`, `GuardStepDiagnostics`, `GuardStepResult`, `apply_ctbs_non_starvation_guard_step(...)`, `concatenate_guard_diagnostics(...)`, `guard_diagnostics_to_rows(...)`, and `summarize_guard_diagnostics(...)`.

- [ ] **Step 1: Write selector fixtures and the first failing eligible-Blank test**

  In `test_ctbs_non_starvation_guard.py`, create a six-UE, three-cell test input with two candidates per cell. Use a real `PhyTbsConfig` and `CtbsHeadroomState`; define the production contract explicitly and set UE0 to pressure `log(3)`, adequacy `0.5`, age `0.5`:

  ```python
  FEATURE_NAMES = (
      "buffer_bytes",
      "avg_rate_mbps",
      "wb_sinr_lin",
      "cqi",
      "ri",
      "tb_err_last",
      "new_data_flag",
      "stale_slots",
      "hol_delay_ms",
      "ttl_slack_ms",
      "recent_scheduled_ratio",
      "recent_goodput_deficit_norm",
      "log1p_drain_pressure",
      "scheduled_dequeue_adequacy",
      "ttl_normalized_service_age",
      "reserved_zero",
  )
  ```

  The first test must call the final interface:

  ```python
  result = apply_ctbs_non_starvation_guard_step(
      config=CtbsNonStarvationGuardConfig(enabled=True),
      blank_class_index=2,
      local_prg=0,
      actor_choice=torch.tensor([2, 0, 2]),
      candidate_ue_ids=torch.tensor([[0, 1], [2, 3], [4, 5]]),
      valid_main=torch.ones((3, 2), dtype=torch.bool),
      remaining_demand_bytes=torch.full((6,), 12000.0),
      ue_features=features,
      ue_feature_names=FEATURE_NAMES,
      ctbs_states=states,
      prg_layer_cap=((4,) * 3,) * 6,
      override_used_by_cell=torch.zeros(3, dtype=torch.bool),
  )

  assert result.executed_choice.tolist() == [0, 0, 2]
  assert result.override_used_by_cell.tolist() == [True, False, False]
  assert result.diagnostics.guard_override.tolist() == [True, False, False]
  assert result.diagnostics.selected_ue.tolist() == [0, -1, -1]
  assert result.diagnostics.ctbs_capacity_after_bytes[0] > result.diagnostics.ctbs_capacity_before_bytes[0]
  ```

- [ ] **Step 2: Run the focused test and observe the intended failure**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py::test_eligible_blank_uses_unique_urgent_ctbs_positive_candidate -q
  ```

  Expected: collection fails because `training.stageb_hgraph.ctbs_non_starvation_guard` does not exist.

- [ ] **Step 3: Implement the frozen types and exact selector**

  Implement these public definitions, with no configurable numerical fields other than `enabled`:

  ```python
  LOG1P_DRAIN_PRESSURE_INDEX = 12
  SCHEDULED_DEQUEUE_ADEQUACY_INDEX = 13
  TTL_NORMALIZED_SERVICE_AGE_INDEX = 14
  MINIMUM_LOG1P_DRAIN_PRESSURE = math.log(2.0)
  MAXIMUM_SCHEDULED_DEQUEUE_ADEQUACY = 1.0
  MINIMUM_TTL_NORMALIZED_SERVICE_AGE = 0.25
  MAXIMUM_OVERRIDES_PER_CELL_PER_TTI = 1

  @dataclass(frozen=True)
  class CtbsNonStarvationGuardConfig:
      enabled: bool = False
      protocol_id: str = field(default="ctbs-blank-v1", init=False)

  class GuardReason(IntEnum):
      ACTOR_NONBLANK = 0
      CELL_BUDGET_USED = 1
      NO_PRESSURE_CANDIDATE = 2
      NO_POSITIVE_CTBS_HEADROOM = 3
      OVERRIDDEN = 4

  @dataclass(frozen=True)
  class GuardStepDiagnostics:
      cell_index: torch.Tensor
      local_prg_index: torch.Tensor
      blank_class_index: torch.Tensor
      actor_choice: torch.Tensor
      executed_choice: torch.Tensor
      guard_override: torch.Tensor
      selected_class: torch.Tensor
      selected_ue: torch.Tensor
      candidate_was_valid: torch.Tensor
      log1p_drain_pressure: torch.Tensor
      scheduled_dequeue_adequacy: torch.Tensor
      ttl_normalized_service_age: torch.Tensor
      ctbs_capacity_before_bytes: torch.Tensor
      ctbs_capacity_after_bytes: torch.Tensor
      reason_code: torch.Tensor

  @dataclass(frozen=True)
  class GuardStepResult:
      executed_choice: torch.Tensor
      override_used_by_cell: torch.Tensor
      diagnostics: GuardStepDiagnostics

  ```

  Declare `concatenate_guard_diagnostics(rows: Sequence[GuardStepDiagnostics]) -> GuardStepDiagnostics`, `guard_diagnostics_to_rows(*, tti: int, diagnostics: GuardStepDiagnostics) -> list[dict[str, object]]`, and `summarize_guard_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]` with those exact parameter and return types.

  Implement `apply_ctbs_non_starvation_guard_step(...)` with the exact signature used in Step 1; type `ctbs_states` and `prg_layer_cap` as optional because disabled execution must not require either input. When disabled, return an unchanged cloned choice, unchanged budget, and zero overrides without accessing CTBS or feature slots. When enabled, validate rank/shape/device, require exact feature names at 12--14, require CTBS states and layer caps, and reject modes that cannot make the exact capacity test.

  For each cell, handle the raw choice in this order: nonblank, budget already used, no pressure-eligible candidate, no CTBS-positive candidate, override. Rank CTBS-positive candidates by:

  ```python
  key = (
      float(ttl_normalized_service_age),
      float(log1p_drain_pressure),
      -int(ue_id),
  )
  ```

  Set `candidate_was_valid=True` only on an override and retain the exact before/after cumulative TBS values from `with_selected_prg(...)`.

- [ ] **Step 4: Run the first test and verify it passes**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py::test_eligible_blank_uses_unique_urgent_ctbs_positive_candidate -q
  ```

  Expected: `1 passed`.

- [ ] **Step 5: Add failing boundary, rank, legality, budget, and fail-closed tests**

  Add `@pytest.mark.parametrize` coverage with argument names `pressure`, `adequacy`, `age`, and `expected_override`, using these exact four rows:

  | pressure | adequacy | age | expected override |
  |---:|---:|---:|:---:|
  | `math.log(2.0)` | `0.5` | `0.50` | false |
  | `math.log(2.0) + 1e-6` | `1.0` | `0.50` | false |
  | `math.log(2.0) + 1e-6` | `0.5` | `0.25 - 1e-6` | false |
  | `math.log(2.0) + 1e-6` | `0.5` | `0.25` | true |

  Add separate tests asserting:

  - a nonblank raw choice is unchanged and reason is `ACTOR_NONBLANK`;
  - `override_used_by_cell=True` prevents a second replacement and reports `CELL_BUDGET_USED`;
  - a candidate with no increase in cumulative TBS is rejected as `NO_POSITIVE_CTBS_HEADROOM`;
  - age wins over pressure, pressure breaks equal-age ties, and the smaller UE ID breaks the final tie;
  - disabled mode is bitwise identity and accepts `ctbs_states=None`;
  - enabled mode raises `ValueError` for missing CTBS state, fewer than 15 UE features, or any wrong name in slots 12--14;
  - diagnostics concatenate in local-PRG order and `guard_diagnostics_to_rows(tti=7, ...)` emits three rows with `tti`, `cell_index`, `local_prg_index`, raw/executed class, reason string, feature values, and CTBS before/after values;
  - `summarize_guard_diagnostics(...)` reports total/per-cell/per-UE overrides, nonblank override count, maximum per-cell-per-TTI count, and all four no-action classifications.

- [ ] **Step 6: Run the complete selector test file**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py -q
  ```

  Expected: all selector, boundary, trace-conversion, and summary tests pass.

- [ ] **Step 7: Inspect the isolated Task 1 diff**

  Run:

  ```bash
  git diff --check -- training/stageb_hgraph/ctbs_non_starvation_guard.py training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py
  git diff --stat -- training/stageb_hgraph/ctbs_non_starvation_guard.py training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py
  ```

  Expected: no whitespace errors; only the two Task 1 files are listed.

---

### Task 2: Integrate Raw and Executed Choices into the 17-Step Policy

**Files:**
- Modify: `training/stageb_hgraph/independent_policy.py:48-68`
- Modify: `training/stageb_hgraph/independent_policy.py:190-566`
- Modify: `training/stageb_hgraph/independent_policy.py:570-614`
- Modify: `training/stageb_hgraph/tests/test_independent_policy.py`

**Interfaces:**
- Consumes: `CtbsNonStarvationGuardConfig`, `apply_ctbs_non_starvation_guard_step(...)`, and diagnostic concatenation from Task 1.
- Produces: `IndependentPolicyTrace.actor_choice`, `IndependentPolicyTrace.choice`, `IndependentPolicyTrace.guard_override`, `IndependentPolicyTrace.guard_diagnostics`, and optional `non_starvation_guard` on `select_independent_action(...)`.

- [ ] **Step 1: Add a failing disabled-equivalence test**

  In `test_independent_policy.py`, run the same graph, actor, packed candidates, and CTBS tuple once with the existing default and once with `CtbsNonStarvationGuardConfig(enabled=False)`. Assert bitwise equality of:

  ```python
  assert torch.equal(default_trace.actor_choice, disabled_trace.actor_choice)
  assert torch.equal(default_trace.choice, disabled_trace.choice)
  assert torch.equal(default_trace.guard_override, torch.zeros_like(default_trace.choice, dtype=torch.bool))
  assert torch.equal(default_trace.choice, disabled_trace.choice)
  assert torch.equal(default_trace.logp, disabled_trace.logp)
  assert torch.equal(default_trace.entropy, disabled_trace.entropy)
  assert torch.equal(default_action.action_ue_select, disabled_action.action_ue_select)
  assert torch.equal(default_action.action_prg_alloc, disabled_action.action_prg_alloc)
  ```

- [ ] **Step 2: Run the disabled-equivalence test and observe failure**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_independent_policy.py::test_disabled_non_starvation_guard_is_bitwise_identity -q
  ```

  Expected: failure because the trace fields and function parameter do not exist.

- [ ] **Step 3: Extend trace and policy signatures without changing the default path**

  Extend the trace and public function exactly as follows:

  ```python
  @dataclass(frozen=True)
  class IndependentPolicyTrace:
      actor_choice: torch.Tensor
      choice: torch.Tensor
      guard_override: torch.Tensor
      local_prg_order: torch.Tensor
      valid_at_step: torch.Tensor
      logp: torch.Tensor
      entropy: torch.Tensor
      guard_diagnostics: GuardStepDiagnostics
      # retain every existing optional diagnostic field unchanged

  def select_independent_action(
      actor: PFQIndependentReuseActor,
      graph: SparseEntityGraph,
      packed: PackedPrgCandidates,
      mode: Literal["sample", "greedy", "hierarchical_greedy"],
      generator: torch.Generator | None = None,
      collect_diagnostics: bool = False,
      collect_numerical_trace: bool = False,
      collect_blank_expiry_attribution: bool = False,
      ctbs_headroom_inputs: CtbsHeadroomInputs | None = None,
      non_starvation_guard: CtbsNonStarvationGuardConfig = CtbsNonStarvationGuardConfig(),
  ) -> tuple[IndependentPolicyTrace, DecodedType0Action]:
  ```

  In `_run_policy`, fail closed if the guard is enabled with a mode other than `greedy`, without CTBS inputs, or with teacher actions. Allocate separate token-length tensors for `actor_choices`, executed `choices`, and `guard_overrides`, plus `override_used_by_cell = torch.zeros(meta.n_cell, dtype=torch.bool, device=device)` once before the 17-step loop. Feed the updated budget returned by each guard step into the next local PRG.

- [ ] **Step 4: Insert the guard before existing CTBS/demand state mutation**

  For every local PRG, preserve the current actor selection code as `actor_selected`. Build actor policy diagnostics and blank-expiry attribution from `actor_selected`, then call the pure selector. Use `executed_selected` for `choices[tokens]`, Type-0 output, service-accounting/rescue diagnostics, remaining demand, allocated PRGs, predicted service, and `CtbsHeadroomState` updates.

  The ordering in `_run_policy` must be:

  ```python
  actor_selected = adjusted.argmax(dim=-1)  # existing greedy branch
  guard_result = apply_ctbs_non_starvation_guard_step(
      config=non_starvation_guard,
      blank_class_index=packed.blank_class_index,
      local_prg=local_prg,
      actor_choice=actor_selected,
      candidate_ue_ids=packed.ue_ids.to(device=device).index_select(0, tokens),
      valid_main=valid_main,
      remaining_demand_bytes=remaining,
      ue_features=graph.snapshot.ue_features.values.to(device=device, dtype=dtype),
      ue_feature_names=graph.snapshot.ue_features.names,
      ctbs_states=ctbs_states,
      prg_layer_cap=(
          ctbs_headroom_inputs.prg_layer_cap
          if ctbs_headroom_inputs is not None
          else None
      ),
      override_used_by_cell=override_used_by_cell,
  )
  executed_selected = guard_result.executed_choice

  actor_choices[tokens] = actor_selected
  choices[tokens] = executed_selected
  guard_overrides[tokens] = guard_result.diagnostics.guard_override

  chosen_logp = logp_all.gather(1, actor_selected.unsqueeze(-1)).squeeze(-1)
  # existing sequential state update uses executed_selected
  ```

  This preserves policy likelihood semantics while ensuring guarded actions causally affect later PRGs. Do not change logits or masks.

- [ ] **Step 5: Make replay exact for both action layers**

  In stored-trace replay, obtain raw choices from `stored_trace.actor_choice` and executed choices from `stored_trace.choice`. Reuse the stored detached `guard_diagnostics` after validating its actor/executed/override columns against those tensors. Validate for every token:

  ```python
  expected_override = stored_actor_choice != stored_executed_choice
  assert torch.equal(expected_override, stored_trace.guard_override)
  ```

  Require each raw choice and executed choice to be valid in the stored/current mask; require every override to originate from Blank and end nonblank. Gather replay log-probability from the stored raw actor choice, update sequential state from the stored executed choice, and materialize Type-0 from the executed choice. Existing unguarded/sample/teacher replay behavior remains unchanged because raw and executed choices are equal.

- [ ] **Step 6: Add sequential and replay tests**

  Add focused tests with a `RecordingActor` that chooses Blank on PRG0 and would normally see unchanged headroom on PRG1:

  - eligible guard execution at PRG0 changes the selected UE's PRG1 remaining-demand and CTBS sequential state;
  - the same cell cannot be overridden again on PRG1;
  - raw `actor_choice` stays Blank, executed `choice` is the candidate, `guard_override=True`, and actor log-probability equals the Blank log-probability;
  - materialized Type-0 action has zero preflight rejects;
  - `replay_independent_trace(...)` reproduces log-probability, entropy, `action_ue_select`, and `action_prg_alloc` exactly;
  - an enabled guard with `sample`, `hierarchical_greedy`, teacher actions, missing CTBS inputs, or a broken feature-name contract raises `ValueError`.

- [ ] **Step 7: Run focused and existing policy/CTBS tests**

  Run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_ctbs_headroom.py -q
  ```

  Expected: all tests pass, including all pre-existing candidate-width, sequential-mask, and CTBS tests.

- [ ] **Step 8: Verify training callers remain guard-free**

  Run:

  ```bash
  grep -R "non_starvation_guard" -n training/stageb_hgraph/online_ppo.py training/stageb_hgraph/online_imitation.py
  ```

  Expected: no matches. Then run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_independent_online_path.py -q
  git diff --check -- training/stageb_hgraph/independent_policy.py training/stageb_hgraph/tests/test_independent_policy.py
  ```

  Expected: existing online training/replay tests pass and no whitespace errors are reported.

---

### Task 3: Evaluator Mode, Complete Trace, and Frozen Summary

**Files:**
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:298-435`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:705-805`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:900-1010`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:1460-1735`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:2217-2592`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`

**Interfaces:**
- Consumes: guard configuration and diagnostics from Tasks 1--2.
- Produces: `--independent-non-starvation-guard {off,ctbs-blank-v1}`, `non_starvation_guard_trace.csv`, and exact guard fields in `eval_summary.json`.

- [ ] **Step 1: Add failing evaluator configuration tests**

  Add tests that call `_select_joint_action(...)` with `independent_non_starvation_guard="ctbs-blank-v1"` and a real CTBS tuple. Assert the returned trace contains raw/executed actions and at least one override in a pressure-controlled fixture. Add fail-closed cases for:

  ```python
  (decode_mode="hierarchical_argmax", guard="ctbs-blank-v1")
  (decode_mode="argmax", guard="ctbs-blank-v1", ctbs_headroom_inputs=None)
  (non_independent_actor, guard="ctbs-blank-v1")
  ```

  Each case must raise `ValueError` with the rejected frozen contract named in the message.

- [ ] **Step 2: Run the evaluator tests and observe the missing-argument failure**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_independent_online_path.py -k non_starvation_guard -q
  ```

  Expected: failure because the evaluator argument and selector parameter do not exist.

- [ ] **Step 3: Add one non-tunable evaluator mode**

  Add exactly one CLI option:

  ```python
  p.add_argument(
      "--independent-non-starvation-guard",
      choices=("off", "ctbs-blank-v1"),
      default="off",
      help="frozen blank-only CTBS non-starvation guard; evaluation only",
  )
  ```

  Resolve it to `CtbsNonStarvationGuardConfig(enabled=mode == "ctbs-blank-v1")`. When enabled, require an independent actor, `--decode-mode argmax`, `--independent-headroom-accounting ctbs-v1`, and `--independent-policy-diagnostics 1`. Pass the resolved config into `_select_joint_action(...)` and then `select_independent_action(...)`.

- [ ] **Step 4: Accumulate every guard decision for all 5000 TTIs**

  Immediately after `_select_joint_action(...)`, convert `trace.guard_diagnostics` to rows using the action TTI `step_idx + 1`. Append all `n_cell * n_prg = 51` rows for every TTI, including warmup, so the complete trace has `255000` rows. Aggregate a second measured-window view for TTIs `1001..5000`.

  Fail during evaluation if any enabled step lacks guard diagnostics, contains a nonblank override, exceeds one override in a cell/TTI, or has a nonpositive capacity delta on an override.

- [ ] **Step 5: Add exact evaluator summary fields and CSV output**

  Add these fields to `eval_summary.json`:

  ```python
  "independent_non_starvation_guard": guard_mode,
  "non_starvation_guard_config": {
      "minimum_log1p_drain_pressure": math.log(2.0),
      "maximum_scheduled_dequeue_adequacy": 1.0,
      "minimum_ttl_normalized_service_age": 0.25,
      "maximum_overrides_per_cell_per_tti": 1,
      "override_nonblank": False,
      "require_positive_ctbs_capacity_delta": True,
  },
  "guard_full_summary": summarize_guard_diagnostics(full_rows),
  "guard_measured_summary": summarize_guard_diagnostics(measured_rows),
  ```

  When guard mode is enabled, write `out_dir/non_starvation_guard_trace.csv` with a fixed header from `guard_diagnostics_to_rows(...)`. When mode is `off`, do not create the CSV; record the mode and zero-valued summaries so the default evaluator remains behaviorally identical.

- [ ] **Step 6: Add output-contract tests**

  Test the summary builder/row writer with two TTIs and assert:

  - raw and executed vectors are both retained;
  - total, per-cell, per-UE, reason, and measured override counts are exact;
  - the frozen thresholds and Boolean rules appear verbatim;
  - the maximum per-cell/per-TTI override count is one;
  - mode `off` reports all-zero guard counts;
  - no threshold CLI flag exists in `parse_args()`.

- [ ] **Step 7: Run evaluator, policy, and feature-contract tests**

  Run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_feature_snapshot.py -q
  ```

  Expected: all tests pass; existing guard-off deterministic evaluation remains unchanged.

- [ ] **Step 8: Inspect Task 3 diff and CLI surface**

  Run:

  ```bash
  python -m training.stageb_hgraph.eval_checkpoint_online --help | grep -A2 "independent-non-starvation-guard"
  git diff --check -- training/stageb_hgraph/eval_checkpoint_online.py training/stageb_hgraph/tests/test_independent_online_path.py
  ```

  Expected: help lists only `off` and `ctbs-blank-v1`; no whitespace errors.

---

### Task 4: Phase-A Audit and Stop Classification

**Files:**
- Create: `training/stageb_hgraph/ctbs_non_starvation_guard_audit.py`
- Create: `training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py`

**Interfaces:**
- Consumes: `frozen_manifest.json`, guarded `eval_summary.json` and normalized Gate metrics, complete guard CSV, guarded starvation audit, frozen unguarded starvation audit, PFQ summary/starvation audit, paired outcome CSVs, and paired offered-packet CSVs.
- Produces: `audit_guard_trace(...)`, `build_phase_a_report(...)`, CLI-written `guard_audit.json`, `offered_stream_comparison.json`, and `final_report.json`.

- [ ] **Step 1: Add a failing exact-invariant audit test**

  Create a synthetic 2-TTI, 3-cell, 2-PRG CSV fixture with one legal override. Call:

  ```python
  audit = audit_guard_trace(
      rows,
      measured_tti_first=1,
      measured_tti_last=2,
      expected_tti_first=1,
      expected_tti_last=2,
      expected_n_cell=3,
      expected_n_prg=2,
  )
  assert audit["row_count"] == 12
  assert audit["override_count"] == 1
  assert all(audit["checks"].values())
  ```

  Parameterize corrupt rows that violate raw-Blank origin, one-per-cell budget, strict pressure, strict adequacy, inclusive age, current validity, and positive CTBS delta; each must make its named check false.

- [ ] **Step 2: Run the audit test and observe the missing-module failure**

  Run:

  ```bash
  python -m pytest training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py::test_guard_trace_audit_accepts_one_exact_legal_override -q
  ```

  Expected: collection fails because `ctbs_non_starvation_guard_audit.py` does not exist.

- [ ] **Step 3: Implement strict CSV loading and invariant audit**

  Implement a fixed `GUARD_TRACE_FIELDS` tuple matching Task 3 and `load_guard_trace(path)`. Reject missing/extra columns, non-finite feature/capacity values, duplicate `(tti, cell_index, local_prg_index)` keys, noncanonical coverage, or rows outside the expected full horizon.

  Implement `audit_guard_trace(...)` and calculate the named checks directly from the parsed rows:

  ```python
  {
      "complete_decision_coverage": len(rows)
      == (expected_tti_last - expected_tti_first + 1) * expected_n_cell * expected_n_prg,
      "nonblank_override_count_zero": all(
          not row["guard_override"] or row["actor_choice"] == row["blank_class_index"]
          for row in rows
      ),
      "max_one_override_per_cell_tti": max(overrides_per_cell_tti.values(), default=0) <= 1,
      "pressure_strictly_above_log2": all(
          not row["guard_override"] or row["log1p_drain_pressure"] > math.log(2.0)
          for row in rows
      ),
      "adequacy_strictly_below_one": all(
          not row["guard_override"] or row["scheduled_dequeue_adequacy"] < 1.0
          for row in rows
      ),
      "age_at_least_quarter_ttl": all(
          not row["guard_override"] or row["ttl_normalized_service_age"] >= 0.25
          for row in rows
      ),
      "override_candidate_valid": all(
          not row["guard_override"] or row["candidate_was_valid"]
          for row in rows
      ),
      "positive_ctbs_capacity_delta": all(
          not row["guard_override"]
          or row["ctbs_capacity_after_bytes"] > row["ctbs_capacity_before_bytes"]
          for row in rows
      ),
      "reason_codes_known": all(row["reason"] in KNOWN_REASON_NAMES for row in rows),
  }
  ```

  Include total/per-cell/per-UE overrides and counts for `actor_nonblank`, `cell_budget_used`, `no_pressure_candidate`, and `no_positive_ctbs_headroom`.

  Add `summarize_outcome_delays(path, measured_tti_first, measured_tti_last)`. Parse positive integer `terminal_tti`, retain rows inside the measured interval, and calculate `sample_count`, `mean_ms`, `p50_ms`, `p90_ms`, `p95_ms`, and `max_ms` from `system_time_ms` with NumPy. Raise `ValueError` on an empty measured window, a non-finite delay, or an invalid terminal TTI.

- [ ] **Step 4: Add failing Phase-A Gate tests**

  Build compact PFQ, guarded actor, unguarded actor, and per-UE fixtures. A passing fixture must satisfy:

  ```python
  guarded_ue4["scheduled_dequeue_bytes"] > unguarded_ue4["scheduled_dequeue_bytes"]
  guarded_ue4["expiry_aware_rate_mbps"] > unguarded_ue4["expiry_aware_rate_mbps"]
  guarded_ue4["expired_bytes"] < unguarded_ue4["expired_bytes"]
  guarded_ue4["max_backlogged_zero_dequeue_streak_tti"] < unguarded_ue4["max_backlogged_zero_dequeue_streak_tti"]
  ```

  Repeat all four assertions for UE6. Also test all backlogged UEs require positive scheduled dequeue/rate and streak `< 400`, offered packet fingerprints match, override count is positive, CTBS/replay/legality/truncation/reference-deletion checks pass, and business thresholds are goodput ratio `>=0.99`, BLER delta `<=0.002`, expiry delta `<=0.0005`.

- [ ] **Step 5: Implement the combined Gate with deterministic classification**

  `build_phase_a_report(...)` must first call the existing `ctbs_formal_evaluation_gate(...)` with seed51 association counts `(8, 17, 11)`, runtime width `17`, expected `5000/4000` step counts, and the frozen state/binary hashes. Add provenance checks for checkpoint file, actor-state, simulator, CTBS config, three feature-source hashes, feature names/dimension, decoder/repair settings, and canonical offered stream.

  Compute `passed` only when provenance/mechanism, existing business Gate, target-UE mechanism Gate, and all-backlogged-UE Gate pass. Use this exact failure precedence:

  ```python
  if not provenance_or_invariants_pass:
      classification = "invalid_experiment"
  elif measured_override_count == 0:
      classification = (
          "no_positive_ctbs_headroom"
          if no_positive_ctbs_headroom_count > 0
          else "insufficient_blank_opportunities"
      )
  elif not target_ue_gate_pass:
      classification = (
          "one_prg_budget_insufficient"
          if cell_budget_used_count > 0
          else (
              "no_positive_ctbs_headroom"
              if no_positive_ctbs_headroom_count > 0
              else "insufficient_blank_opportunities"
          )
      )
  elif not business_gate_pass:
      classification = "business_metric_cost"
  else:
      classification = "ctbs_blank_v1_seed51_pass"
  ```

  Record all required outcome-delay mean/p50/p90/p95/max, UE macro/p10/minimum expiry-aware rate, goodput, BLER, expiry, PRG utilization, Blank ratio, scheduled bytes, failed bytes, per-UE expiry, and UE4/UE6 comparison fields even when the Gate fails.

- [ ] **Step 6: Implement one fail-closed CLI**

  Add CLI arguments:

  ```text
  --worktree-root
  --frozen-manifest
  --pfq-metrics
  --pfq-starvation
  --pfq-outcomes
  --pfq-offered-packets
  --unguarded-starvation
  --actor-summary
  --actor-metrics
  --actor-starvation
  --actor-outcomes
  --actor-offered-packets
  --guard-trace
  --guard-audit-out
  --offered-comparison-out
  --final-report-out
  ```

  The CLI recomputes file SHA-256 values, canonical offered fingerprints via `service_balance_audit.canonical_offered_packet_stream(...)`, and both complete outcome-delay summaries via `summarize_outcome_delays(...)`. It writes all three JSON files with sorted keys and exits `0` only on a Phase-A PASS; a scientifically valid negative result writes all reports then exits `2`; invalid provenance/trace exits `3` after writing the failure reports.

- [ ] **Step 7: Run all audit and Gate tests**

  Run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py \
    training/stageb_hgraph/tests/test_independent_gate.py \
    training/stageb_hgraph/tests/test_seed45_starvation_audit.py \
    training/stageb_hgraph/tests/test_service_balance_audit.py -q
  ```

  Expected: all tests pass, including each negative classification fixture.

- [ ] **Step 8: Inspect the Task 4 diff**

  Run:

  ```bash
  git diff --check -- training/stageb_hgraph/ctbs_non_starvation_guard_audit.py training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py
  python -m training.stageb_hgraph.ctbs_non_starvation_guard_audit --help
  ```

  Expected: no whitespace errors and every fixed input/output argument is listed.

---

### Task 5: Freeze Provenance and Run the One Authorized Seed51 Experiment

**Files:**
- Create: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/frozen_manifest.json`
- Create: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/actor/*`
- Create only on offered-stream mismatch: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/pfq/*`
- Create: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/guard_audit.json`
- Create: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/offered_stream_comparison.json`
- Create: `output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/final_report.json`

**Interfaces:**
- Consumes: the tested evaluator/audit path, frozen iter8 checkpoint, frozen simulator, existing seed51 PFQ and unguarded actor artifacts.
- Produces: one immutable Phase-A evidence package and an explicit stop decision; no seed48 run.

- [ ] **Step 1: Run the full pre-experiment regression set**

  Inside the activated container, run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_ctbs_non_starvation_guard.py \
    training/stageb_hgraph/tests/test_ctbs_non_starvation_guard_audit.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_ctbs_headroom.py \
    training/stageb_hgraph/tests/test_feature_snapshot.py \
    training/stageb_hgraph/tests/test_independent_gate.py \
    training/stageb_hgraph/tests/test_seed45_starvation_audit.py \
    training/stageb_hgraph/tests/test_service_balance_audit.py -q
  ```

  Expected: all selected tests pass. Do not launch the simulator if any test fails.

- [ ] **Step 2: Verify frozen binary, checkpoint, feature, and actor hashes**

  Run:

  ```bash
  sha256sum \
    build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection \
    output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt \
    cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h \
    cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h \
    training/stageb_hgraph/feature_snapshot.py
  ```

  Expected hashes, in order:

  ```text
  4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10
  94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a
  4a353601a8378888d089e34d3f04225770ae757ac07d5bc4108533a1a384975a
  ccf58c6c07e5ab65da373870d7d615b8048611508e4972910c4a4871744f876b
  e86976c6208ff6e23e350d1480beaac011d694870d4a68e4e6fb9f7077aa7f7f
  ```

  Load the checkpoint with the existing `state_dict_sha256(...)` helper and require `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`.

- [ ] **Step 3: Write the frozen manifest before simulator launch**

  Use a short Python command with `json.dumps(..., indent=2, sort_keys=True)` to write the manifest. It must include:

  ```json
  {
    "protocol_id": "iter8-ctbs-legal-non-starvation-guard-seed51-v1",
    "seed": 51,
    "warmup_tti": 1000,
    "measured_tti": 4000,
    "measured_tti_first": 1001,
    "measured_tti_last": 5000,
    "packet_ttl_tti": 400,
    "packet_ttl_ms": 200,
    "decode_mode": "argmax",
    "independent_headroom_accounting": "ctbs-v1",
    "guard_mode": "ctbs-blank-v1",
    "runtime_candidate_width": 17,
    "association_candidate_counts": [8, 17, 11],
    "training_or_adaptation_allowed": false,
    "additional_seed_allowed": false,
    "checkpoint_file_sha256": "94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a",
    "actor_state_dict_sha256": "6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be",
    "simulator_sha256": "4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10",
    "ctbs_config_sha256": "be277403fbfff0eb6359b94e3b000344deac00bfe5acbbd91004060dd3336631",
    "performance_thresholds": {
      "actor_over_pfq_goodput_min": 0.99,
      "actor_minus_pfq_bler_max": 0.002,
      "actor_minus_pfq_expiry_max": 0.0005
    }
  }
  ```

  Add exact file hashes for all three feature sources plus the four Python implementation/audit files. Record every frozen guard constant and all disabled repair/fill/budget settings.

- [ ] **Step 4: Launch the guarded seed51 actor evaluation**

  In the activated container, run the simulator and evaluator with these exact values. Use a fresh socket and the new output root:

  ```bash
  WORKTREE=/opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
  GUARD_ROOT="$WORKTREE/output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1"
  ACTOR_DIR="$GUARD_ROOT/actor"
  SIM_BIN="$WORKTREE/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
  CHECKPOINT="$WORKTREE/output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt"
  SOCKET=/tmp/cumac_iter8_ctbs_guard_seed51_v1.sock
  mkdir -p "$ACTOR_DIR"

  (
    cd "$ACTOR_DIR"
    env \
      CUMAC_ONLINE_BRIDGE=1 \
      CUMAC_ONLINE_SOCKET="$SOCKET" \
      CUMAC_ONLINE_PERSISTENT=1 \
      CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 \
      CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
      CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
      CUMAC_SERVICE_BALANCE_TRACE_CSV="$ACTOR_DIR/service_balance_trace.csv" \
      CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=602 \
      CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="$ACTOR_DIR/traffic_offered_packets.csv" \
      CUMAC_CELL_ASSOC_MODE=strongest \
      CUMAC_UE_PLACEMENT_MODE=coop_boundary \
      CUMAC_UE_VORONOI_CLIP=1 \
      CUMAC_BS_TX_PATTERN=omni \
      CUMAC_TOPOLOGY_SEED=51 \
      CUMAC_TRAFFIC_SEED=51 \
      CUMAC_TRAFFIC_ARRIVAL_RATE=1.0 \
      CUMAC_PACKET_TTL_TTI=0 \
      CUMAC_PACKET_TTL_MS=200 \
      CUMAC_COMPACT_TTI_LOG=1 \
      CUMAC_PROGRESS_TTI_INTERVAL=0 \
      CUMAC_COMPARE_TTI_INTERVAL=0 \
      CUMAC_PACKET_DELAY_SAMPLES=1 \
      CUMAC_PACKET_DELAY_SAMPLES_CSV="$ACTOR_DIR/packet_delay_samples.csv" \
      CUMAC_PACKET_OUTCOME_SAMPLES_CSV="$ACTOR_DIR/packet_outcome_samples.csv" \
      CUMAC_EXEC_MODE=both \
      "$SIM_BIN" -d 1 -b 2 -x 0 -f 0 -g 100 -r 3000
  ) > "$ACTOR_DIR/sim.log" 2>&1 &
  SIM_PID=$!

  python -m training.stageb_hgraph.eval_checkpoint_online \
    --checkpoint "$CHECKPOINT" \
    --socket-path "$SOCKET" \
    --episode-horizon 5000 \
    --max-steps 5000 \
    --stats-warmup-steps 1000 \
    --decode-mode argmax \
    --seed 51 \
    --sample-seed 51 \
    --device auto \
    --print-every-steps 500 \
    --save-steps 0 \
    --sim-bin "$SIM_BIN" \
    --outcome-samples "$ACTOR_DIR/packet_outcome_samples.csv" \
    --out-dir "$ACTOR_DIR" \
    --independent-policy-diagnostics 1 \
    --independent-runtime-candidate-width 17 \
    --independent-headroom-accounting ctbs-v1 \
    --independent-non-starvation-guard ctbs-blank-v1 \
    > "$ACTOR_DIR/eval.log" 2>&1
  EVAL_RC=$?
  wait "$SIM_PID"
  SIM_RC=$?
  test "$EVAL_RC" -eq 0
  test "$SIM_RC" -eq 0
  ```

  Expected: evaluator and simulator exit zero, `completed_steps=5000`, `stats_count=4000`, and `non_starvation_guard_trace.csv` has one header plus `255000` rows.

- [ ] **Step 5: Normalize metrics and generate the guarded starvation audit**

  Run:

  ```bash
  python -m training.stageb_hgraph.independent_gate normalize-eval \
    --summary "$ACTOR_DIR/eval_summary.json" \
    --out "$ACTOR_DIR/gate_metrics.json"

  set +e
  python -m training.stageb_hgraph.seed45_starvation_audit \
    --seed 51 \
    --service-trace "$ACTOR_DIR/service_balance_trace.csv" \
    --measured-first-tti 1001 \
    --measured-last-tti 5000 \
    --expected-ue-count 36 \
    --streak-threshold-tti 400 \
    --eval-summary "$ACTOR_DIR/eval_summary.json" \
    --outcome-samples "$ACTOR_DIR/packet_outcome_samples.csv" \
    --out "$ACTOR_DIR/starvation_audit.json"
  STARVATION_RC=$?
  set -e
  test "$STARVATION_RC" -eq 0 -o "$STARVATION_RC" -eq 1
  test -f "$ACTOR_DIR/starvation_audit.json"
  ```

  The starvation command may exit nonzero for a valid negative result; retain its JSON and continue to the combined audit.

- [ ] **Step 6: Compare the offered stream before deciding PFQ reuse**

  Compare:

  ```text
  existing PFQ: output/pfq_independent_reuse/formal/iter8_dimensionless_frozen_generalization_seed48_seed51_v1/seed51/pfq/traffic_offered_packets.csv
  guarded actor: output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/actor/traffic_offered_packets.csv
  ```

  Use `canonical_offered_packet_stream(...)`. The required frozen fingerprint is:

  ```text
  packet_count = 180016
  offered_bytes = 540048000
  sha256 = 4abe2d4ac1606b3a635cc4eaf3db506055b04fc2c8e56dc4542782995037a404
  ```

  If exact, reference the existing PFQ summary, outcome, service, starvation, and offered-stream files by path and SHA-256. If not exact, run `training.stageb_hgraph.pfq_reference_calibration` once with seed51, `--warmup-steps 1000`, `--measure-steps 4000`, `--formal`, `--service-balance-audit`, `--service-balance-trace-first-tti 602`, and the same simulator arguments/environment into `$GUARD_ROOT/pfq/`; require the fresh PFQ and actor fingerprints to match before continuing.

- [ ] **Step 7: Generate the guard audit and combined final report**

  With the exact existing PFQ/unguarded paths, run:

  ```bash
  OLD_ROOT="$WORKTREE/output/pfq_independent_reuse/formal/iter8_dimensionless_frozen_generalization_seed48_seed51_v1/seed51"

  set +e
  python -m training.stageb_hgraph.ctbs_non_starvation_guard_audit \
    --worktree-root "$WORKTREE" \
    --frozen-manifest "$GUARD_ROOT/frozen_manifest.json" \
    --pfq-metrics "$OLD_ROOT/pfq/reference_summary.json" \
    --pfq-starvation "$OLD_ROOT/pfq/starvation_audit.json" \
    --pfq-outcomes "$OLD_ROOT/pfq/packet_outcome_samples.csv" \
    --pfq-offered-packets "$OLD_ROOT/pfq/traffic_offered_packets.csv" \
    --unguarded-starvation "$OLD_ROOT/actor/starvation_gate.json" \
    --actor-summary "$ACTOR_DIR/eval_summary.json" \
    --actor-metrics "$ACTOR_DIR/gate_metrics.json" \
    --actor-starvation "$ACTOR_DIR/starvation_audit.json" \
    --actor-outcomes "$ACTOR_DIR/packet_outcome_samples.csv" \
    --actor-offered-packets "$ACTOR_DIR/traffic_offered_packets.csv" \
    --guard-trace "$ACTOR_DIR/non_starvation_guard_trace.csv" \
    --guard-audit-out "$GUARD_ROOT/guard_audit.json" \
    --offered-comparison-out "$GUARD_ROOT/offered_stream_comparison.json" \
    --final-report-out "$GUARD_ROOT/final_report.json"
  AUDIT_RC=$?
  set -e
  test "$AUDIT_RC" -eq 0 -o "$AUDIT_RC" -eq 2 -o "$AUDIT_RC" -eq 3
  ```

  Accept CLI exit `0` as a positive hypothesis result and exit `2` as a valid falsification. Exit `3` invalidates the experiment and requires fixing provenance/trace collection before interpreting business metrics.

- [ ] **Step 8: Verify artifacts and apply the protocol stop rule**

  Run:

  ```bash
  python -m json.tool "$GUARD_ROOT/frozen_manifest.json" >/dev/null
  python -m json.tool "$GUARD_ROOT/guard_audit.json" >/dev/null
  python -m json.tool "$GUARD_ROOT/offered_stream_comparison.json" >/dev/null
  python -m json.tool "$GUARD_ROOT/final_report.json" >/dev/null
  git diff --check
  ```

  Confirm the final report contains the complete metric table, UE4/UE6 directional comparison, guard reason distribution, exact provenance, and one classification from Task 4.

  - If classification is `ctbs_blank_v1_seed51_pass`, stop and request separate authorization for seed48 non-degeneration evaluation.
  - If classification is `insufficient_blank_opportunities`, `no_positive_ctbs_headroom`, `one_prg_budget_insufficient`, or `business_metric_cost`, stop; reject Guard V1 without tuning thresholds or adding another rule.
  - If classification is `invalid_experiment`, stop interpreting performance and repair only the invalid measurement/provenance path before rerunning the same frozen seed51 protocol.

---

## Final Verification Checklist

- [ ] Guard-off selection, log-probability, replay, and Type-0 outputs are bitwise identical to the pre-guard path.
- [ ] Guard-on changes only raw Blank choices and at most once per cell/TTI.
- [ ] Every override passes strict pressure, strict adequacy, inclusive age, current validity, positive demand, and exact positive cumulative-TBS delta.
- [ ] Executed guarded choices update sequential demand and CTBS state before the next local PRG.
- [ ] PPO and imitation entry points expose no guard parameter and receive no guarded training target.
- [ ] The full 5000-TTI trace contains exactly `255000` decision rows and the measured audit covers exactly TTI `1001..5000`.
- [ ] Replay mismatch, illegal action, truncation, CTBS MCS/layer/TBS mismatch, and mask reactivation counts are zero; reference-label deletion equality is true.
- [ ] Offered-packet identity and all frozen file/state/config hashes match.
- [ ] `final_report.json` applies the goodput/BLER/expiry Gate and both UE4/UE6 directionality Gates without coefficient or threshold selection.
- [ ] No seed48 or other follow-up experiment has been run automatically.

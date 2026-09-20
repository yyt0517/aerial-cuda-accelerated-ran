# PFQ-Independent Blank Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default, behavior-invariant diagnostic that explains deterministic Blank decisions as forced, probability-mass fragmentation, or learned conservatism, then run the minimum seed-41 boundary experiment on A4, A8, and one formally selected B checkpoint.

**Architecture:** Observe the already-masked independent-policy distribution at each existing Cell-PRG decision, detach compact tensors into the existing trace, and aggregate them only at rollout/evaluation boundaries. Reuse the current evaluator, simulator binary, checkpoints, warmup handling, and formal metrics. Add exact scheduled/goodput/failed-byte accounting from protocol arrays, plus one small report/selection module; do not change logits, masks, sampling, reward, PPO replay, Type-0 materialization, simulator code, or deployment mode.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, Bash runner, existing cuMAC online bridge and corrected simulator binary.

## Global Constraints

- The approved design is [2026-08-13-pfq-independent-blank-diagnostics-design.md](../specs/2026-08-13-pfq-independent-blank-diagnostics-design.md). It is authoritative if this plan is ambiguous.
- Do not edit `independent_policy.py`, `online_ppo.py`, or `eval_checkpoint_online.py` until the currently running `select-and-eval-seed41` process has exited and its already-started formal evaluations are complete. A gate failure is an experimental result, not permission to rerun or replace formal outputs.
- Run every Python test, evaluator, and runner command inside `c_aerial_oai2`, after `cd /opt/nvidia/cuBB && source .venv/bin/activate && cd .worktrees/pfq-independent-reuse-actor`. Host commands are limited to editing and Git inspection/commits.
- Keep `--independent-policy-diagnostics` defaulted to `0`. When it is `0`, do not allocate or serialize diagnostic probability tensors.
- The enabled flag may change observation only. It must not change action classes, Type-0 arrays, log-probability, entropy, sequential state, RNG consumption, replay, reward, loss, or optimizer behavior.
- A sampled-run shadow is a one-step argmax of the sampled trajectory's current adjusted logits. Do not advance a second greedy sequential state and do not submit the shadow action.
- Deterministic evaluator mode remains `argmax`; no stochastic deployment result is promotion evidence.
- Keep PFQ independence exact: `requires_pfq_features=false`, PFQ score fields deleted, no PFQ residual/base logits, no repair/fallback/safe-fill policy.
- Reuse `build.reuse_actor/.../multiCellSchedulerUeSelection`; do not rebuild or modify the simulator for this diagnostic. Verify its SHA-256 is identical across the three diagnostic subjects and the completed formal runs.
- Store all new experimental outputs below `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/`. Never modify `pfq_seed41`, `pilot-A`, `pilot-B`, or `seed41_selection` results.
- Preserve the user's unrelated untracked files (`output.txt`, `output_short.txt`, `packet_delay_samples.csv`, `snr.txt`, `ue_topology.csv`, and `__pycache__` directories).
- Stop after the diagnostic report identifies Blank composition, exact failed Mbps/TB failures, whether `H_frag` is supported for A8, and one next experiment. Do not implement the next policy.

---

## Task 0: Close the formal-run boundary and freeze its evidence

**Files:**

- Read: `output/pfq_independent_reuse/formal/pilot-A/candidate_eval/*/eval_summary.json`
- Read: `output/pfq_independent_reuse/formal/pilot-B/candidate_eval/*/eval_summary.json`
- Read: `output/pfq_independent_reuse/formal/pilot-A/checkpoint_selection.json`
- Read: `output/pfq_independent_reuse/formal/pilot-B/checkpoint_selection.json`
- Read: `output/pfq_independent_reuse/formal/seed41_selection/treatment_comparison.json`
- Do not modify code in this task.

- [ ] **Step 1: Wait for the already-running formal process to exit**

Run on the host without interrupting it:

```bash
ps -eo pid,etime,args | rg '[r]un_stageB_pfq_independent_reuse.sh select-and-eval-seed41|[e]val_checkpoint_online'
```

Expected: while output is present, wait; do not start implementation. Repeat at a reasonable interval until no matching formal process remains.

- [ ] **Step 2: Verify that formal checkpoint evaluation is complete enough to choose the diagnostic subjects**

Run on the host:

```bash
for treatment in A B; do
  for iteration in 4 8 12 16 20; do
    test -s "output/pfq_independent_reuse/formal/pilot-${treatment}/candidate_eval/iter-$(printf '%04d' "${iteration}")/eval_summary.json"
    test -s "output/pfq_independent_reuse/formal/pilot-${treatment}/candidate_eval/iter-$(printf '%04d' "${iteration}")/gate_metrics.json"
  done
  test -s "output/pfq_independent_reuse/formal/pilot-${treatment}/checkpoint_selection.json"
done
test -s output/pfq_independent_reuse/formal/seed41_selection/treatment_comparison.json
```

Expected: all commands exit `0`. Only A and B are required because they are the completed treatments in this experiment; do not invent C/D requirements.

- [ ] **Step 3: Record the immutable subject inputs before source changes**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python - <<"PY"
import hashlib, json
from pathlib import Path

root = Path("output/pfq_independent_reuse/formal")
binary = Path("build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection")
payload = {
    "sim_binary": str(binary.resolve()),
    "sim_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "a4_checkpoint": str((root / "pilot-A/candidate_checkpoints/ppo_actor_iter_0004.pt").resolve()),
    "a8_checkpoint": str((root / "pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt").resolve()),
    "formal_a_selection": json.loads((root / "pilot-A/checkpoint_selection.json").read_text()),
    "formal_b_selection": json.loads((root / "pilot-B/checkpoint_selection.json").read_text()),
}
print(json.dumps(payload, indent=2))
PY
'
```

Expected: the binary exists; A4/A8 paths resolve; both formal selection JSON objects load. Save the console evidence in the work log, not in a formal output file yet.

---

## Task 1: Implement and validate the decision-level diagnostic algebra

**Files:**

- Create: `training/stageb_hgraph/independent_blank_diagnostics.py`
- Create: `training/stageb_hgraph/tests/test_independent_blank_diagnostics.py`

- [ ] **Step 1: Write failing classification and identity tests**

Add synthetic cases that cover all three Blank causes and one service decision. Use Blank as the last class, matching `PackedPrgCandidates.blank_class_index`:

```python
def test_classifies_fragmentation_conservatism_forced_and_service() -> None:
    adjusted = torch.tensor([
        [0.0, 0.0, 0.1],
        [-2.0, -2.0, 0.0],
        [-1.0e9, -1.0e9, 0.0],
        [2.0, 0.0, -1.0],
    ])
    valid_main = torch.tensor([
        [True, True],
        [True, True],
        [False, False],
        [True, True],
    ])
    probabilities = torch.softmax(adjusted, dim=-1)
    selected = adjusted.argmax(dim=-1)

    diag = compute_independent_policy_diagnostics(
        adjusted_logits=adjusted,
        probabilities=probabilities,
        valid_main=valid_main,
        selected=selected,
        blank_class_index=2,
    )

    assert diag.greedy_blank.tolist() == [True, True, True, False]
    assert diag.forced_blank.tolist() == [False, False, True, False]
    assert diag.learned_blank.tolist() == [True, True, False, False]
    assert diag.fragmentation_blank.tolist() == [True, False, False, False]
    assert diag.conservative_blank.tolist() == [False, True, False, False]
    assert diag.margin_valid.tolist() == [True, True, False, True]
    assert diag.blank_margin[2].item() == 0.0
```

Also assert:

```python
assert torch.allclose(diag.p_blank + diag.p_service, torch.ones(4))
assert bool((diag.p_best_ue <= diag.p_service + 1.0e-7).all())
assert torch.equal(diag.learned_blank, diag.fragmentation_blank | diag.conservative_blank)
assert not bool((diag.fragmentation_blank & diag.conservative_blank).any())
```

Add a summary test asserting all-decision denominators, a zero (not NaN) conditional share when learned count is zero, exclusion of forced rows from margin percentiles, and exact `decision_count`/`valid_margin_count`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q training/stageb_hgraph/tests/test_independent_blank_diagnostics.py
'
```

Expected: failure because `independent_blank_diagnostics` and its public functions do not exist.

- [ ] **Step 3: Implement the compact detached tensor record**

Implement this public data shape, keeping all tensors one-dimensional after collection:

```python
@dataclass(frozen=True)
class IndependentPolicyDiagnostics:
    valid_ue_count: torch.Tensor
    p_blank: torch.Tensor
    p_service: torch.Tensor
    p_best_ue: torch.Tensor
    blank_margin: torch.Tensor
    margin_valid: torch.Tensor
    selected_blank: torch.Tensor
    greedy_blank: torch.Tensor
    forced_blank: torch.Tensor
    learned_blank: torch.Tensor
    fragmentation_blank: torch.Tensor
    conservative_blank: torch.Tensor
```

Implement only these policy-facing helpers:

```python
def compute_independent_policy_diagnostics(
    *,
    adjusted_logits: torch.Tensor,
    probabilities: torch.Tensor,
    valid_main: torch.Tensor,
    selected: torch.Tensor,
    blank_class_index: int,
) -> IndependentPolicyDiagnostics: ...

def concatenate_independent_policy_diagnostics(
    rows: Sequence[IndependentPolicyDiagnostics],
) -> IndependentPolicyDiagnostics: ...

def summarize_independent_policy_diagnostics(
    diagnostics: IndependentPolicyDiagnostics,
    *,
    classification_prefix: str,
) -> dict[str, float]: ...
```

The compute helper must:

- validate shapes and that `blank_class_index` is the final class;
- compute service mass only across `valid_main` UE classes;
- use the exact provided `adjusted_logits` and `probabilities`, without another softmax;
- compute `greedy_blank` with `adjusted_logits.argmax(dim=-1)` (no RNG);
- classify forced/learned/fragmentation/conservative from greedy, not sampled, Blank;
- put zero in `blank_margin` when no UE is valid and set `margin_valid=False` there;
- validate finiteness, `[0,1]` ranges, `p_blank + p_service == 1`, partition identities, and forced-Blank probability one within tolerance;
- return `tensor.detach().reshape(-1).cpu()` for every field.

The summary helper must expose:

```text
decision_count
selected_blank_ratio
{prefix}greedy_blank_ratio
{prefix}forced_blank_ratio
{prefix}learned_blank_ratio
{prefix}fragmentation_blank_ratio
{prefix}conservative_blank_ratio
{prefix}fragmentation_share_among_learned_blank
{prefix}learned_blank_count
mean_valid_ue_count
p_blank_mean, p_blank_p10, p_blank_p50, p_blank_p90
p_service_mean, p_service_p10, p_service_p50, p_service_p90
p_best_ue_mean, p_best_ue_p10, p_best_ue_p50, p_best_ue_p90
valid_margin_count
blank_margin_mean, blank_margin_p10, blank_margin_p50, blank_margin_p90
```

Use `classification_prefix="shadow_"` for sampled rollout and `classification_prefix=""` for deterministic evaluation. All ratios except `fragmentation_share_among_learned_blank` use total decisions as denominator.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run the same focused pytest command.

Expected: all tests in `test_independent_blank_diagnostics.py` pass.

- [ ] **Step 5: Commit the algebra in isolation**

Run on the host:

```bash
git add training/stageb_hgraph/independent_blank_diagnostics.py training/stageb_hgraph/tests/test_independent_blank_diagnostics.py
git commit -m "feat: add independent blank diagnostic algebra"
```

---

## Task 2: Attach diagnostics to the existing independent trace without changing behavior

**Files:**

- Modify: `training/stageb_hgraph/independent_policy.py:20-27,120-129,165-168,200-262,282-306`
- Modify: `training/stageb_hgraph/tests/test_independent_policy.py:89-117`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py:63-78`

- [ ] **Step 1: Write failing disabled/enabled invariance tests**

Extend `test_independent_policy.py` with one parametrized test for `sample` and `greedy`. For sampled mode, construct two independent generators with the same seed; for greedy, pass `None`:

```python
@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_diagnostic_collection_is_action_and_rng_invariant(mode: str) -> None:
    graph = _small_demand_graph(n_prg=3)
    actor = RecordingActor(n_prg=3)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    generator_off = torch.Generator().manual_seed(29) if mode == "sample" else None
    generator_on = torch.Generator().manual_seed(29) if mode == "sample" else None

    trace_off, action_off = select_independent_action(
        actor, graph, packed, mode=mode, generator=generator_off,
        collect_diagnostics=False,
    )
    trace_on, action_on = select_independent_action(
        actor, graph, packed, mode=mode, generator=generator_on,
        collect_diagnostics=True,
    )

    assert trace_off.diagnostics is None
    assert trace_on.diagnostics is not None
    assert torch.equal(trace_off.choice, trace_on.choice)
    assert torch.equal(trace_off.valid_at_step, trace_on.valid_at_step)
    assert torch.equal(trace_off.logp, trace_on.logp)
    assert torch.equal(trace_off.entropy, trace_on.entropy)
    assert torch.equal(action_off.action_ue_select, action_on.action_ue_select)
    assert torch.equal(action_off.action_prg_alloc, action_on.action_prg_alloc)
    if mode == "sample":
        assert torch.equal(generator_off.get_state(), generator_on.get_state())
```

Assert `trace_on.diagnostics.valid_ue_count.numel() == 3 * 3`, and assert replay of the enabled trace still returns exact actions and exact logp/entropy.

In `test_independent_online_path.py`, reset `torch.manual_seed` before each `_sample_joint_action` call and assert the returned NumPy Type-0 arrays are exactly equal with the flag off/on. This catches plumbing regressions at the production call boundary.

- [ ] **Step 2: Run the two focused suites and confirm RED**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py
'
```

Expected: failures because `collect_diagnostics` and `trace.diagnostics` are not defined.

- [ ] **Step 3: Add one optional trace field and one opt-in policy argument**

Extend the trace and policy signatures exactly at the independent path:

```python
@dataclass(frozen=True)
class IndependentPolicyTrace:
    choice: torch.Tensor
    local_prg_order: torch.Tensor
    valid_at_step: torch.Tensor
    logp: torch.Tensor
    entropy: torch.Tensor
    diagnostics: IndependentPolicyDiagnostics | None = None
```

```python
def _run_policy(..., collect_diagnostics: bool = False, ...) -> IndependentPolicyTrace: ...

def select_independent_action(
    ...,
    generator: torch.Generator | None = None,
    collect_diagnostics: bool = False,
) -> tuple[IndependentPolicyTrace, DecodedType0Action]: ...
```

Inside each existing local-PRG loop, after `adjusted`, `logp_all`, and `probabilities` exist and after `selected` has been chosen, append a diagnostic row only when enabled:

```python
if collect_diagnostics:
    diagnostic_rows.append(
        compute_independent_policy_diagnostics(
            adjusted_logits=adjusted,
            probabilities=probabilities,
            valid_main=valid_main,
            selected=selected,
            blank_class_index=packed.blank_class_index,
        )
    )
```

Do not move or duplicate `torch.multinomial`. Do not pass `collect_diagnostics=True` from `replay_independent_trace` or teacher tracing. Set the trace field to `None` when disabled, otherwise concatenate the 17 rows (each containing three cells) into 51 decision records.

- [ ] **Step 4: Plumb the observation flag through `_sample_joint_action` only**

Add a keyword argument:

```python
independent_policy_diagnostics: bool = False,
```

Pass it only to the independent actor's `select_independent_action(..., collect_diagnostics=...)`. The returned dict already carries `independent_policy_trace`, so do not add a parallel mutable diagnostics store.

- [ ] **Step 5: Run focused tests and the existing PFQ-deletion/replay validity tests**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py
'
```

Expected: all pass; enabled and disabled sampled generator states and submitted Type-0 arrays are identical.

- [ ] **Step 6: Commit the behavior-invariant policy hook**

Run on the host:

```bash
git add training/stageb_hgraph/independent_policy.py training/stageb_hgraph/online_ppo.py training/stageb_hgraph/tests/test_independent_policy.py training/stageb_hgraph/tests/test_independent_online_path.py
git commit -m "feat: observe independent blank decisions"
```

---

## Task 3: Aggregate sampled shadows and deterministic evaluation metrics

**Files:**

- Modify: `training/stageb_hgraph/online_ppo.py:127-303,3063-3185,3897-5070,5140-5235,7570-7665,9050-9225`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py:208-285,577-662,974-1210,1310-1496`
- Modify: `training/stageb_hgraph/independent_blank_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_blank_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`

- [ ] **Step 1: Write failing exact failed-byte tests**

Add a pure helper test using current-TTI arrays:

```python
def test_extract_failed_transmission_bytes_is_exact() -> None:
    metrics = extract_failed_transmission_bytes(
        {
            "actual_ue_served_bytes": np.array([3000.0, 6000.0]),
            "actual_ue_goodput_bytes": np.array([3000.0, 3000.0]),
        },
        expected_n_ue=2,
    )
    assert metrics == {
        "scheduled_bytes": 9000.0,
        "goodput_bytes": 6000.0,
        "failed_bytes": 3000.0,
    }
```

Also test fail-closed behavior for missing arrays, wrong shapes, negative values, non-finite values, and goodput greater than scheduled beyond `1e-6` bytes. The helper must not substitute zeros for missing protocol evidence when diagnostics are enabled.

- [ ] **Step 2: Run the focused test and confirm RED**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q training/stageb_hgraph/tests/test_independent_blank_diagnostics.py
'
```

Expected: failure because `extract_failed_transmission_bytes` is absent.

- [ ] **Step 3: Implement exact per-TTI transmission accounting**

Add:

```python
def extract_failed_transmission_bytes(
    info: Mapping[str, object],
    *,
    expected_n_ue: int,
) -> dict[str, float]: ...
```

It must require both arrays, reshape each to exactly `(expected_n_ue,)`, reject non-finite/negative values, compute sums in `np.float64`, reject `goodput_bytes > scheduled_bytes + 1e-6`, and return `failed_bytes=max(scheduled-goodput, 0.0)`. This describes failed transmission attempts; do not relabel them as terminal packet failures or expiry.

- [ ] **Step 4: Add the sampled-rollout diagnostic flag and aggregate stored trace observations**

Add the CLI option to the PPO parser:

```python
p.add_argument(
    "--independent-policy-diagnostics",
    type=int,
    choices=(0, 1),
    default=0,
    help="collect detached PFQ-independent Blank diagnostics without changing actions",
)
```

Pass `bool(args.independent_policy_diagnostics)` through `_collect_rollout` to `_sample_joint_action`. Do not add one field per metric to `RolloutStep`; its existing `independent_policy_trace` already owns the optional detached diagnostic.

At rollout summary construction, collect all non-`None` `step.independent_policy_trace.diagnostics`, concatenate once, and merge:

```python
rollout_diag = summarize_independent_policy_diagnostics(
    concatenate_independent_policy_diagnostics(diagnostics),
    classification_prefix="shadow_",
)
summary.update({f"rollout_independent_{key}": value for key, value in rollout_diag.items()})
```

When the flag is `0`, add no diagnostic columns. When it is `1` for an independent actor, require exactly `len(steps) * n_cell * n_prg` decisions; missing diagnostics are an error. These fields then flow into the existing training row/CSV merge without a second writer.

- [ ] **Step 5: Add deterministic evaluator collection and warmup-excluded failure sums**

Add the same CLI option, validation, and `collect_diagnostics` pass-through to `_select_joint_action`. Reject enabling it for a non-independent actor.

For `step_idx >= stats_warmup_steps` only:

- append the trace diagnostic to a measured-window list;
- call `extract_failed_transmission_bytes(next_info, expected_n_ue=n_active_ue)`;
- add scheduled, UE-goodput, and failed bytes to `float64` accumulators;
- continue collecting the existing TB tx/error counts unchanged.

After the loop, require:

```python
expected_decisions = stats_count * graph.snapshot.static_meta.n_cell * graph.snapshot.static_meta.n_prg
actual_decisions = int(eval_diag.valid_ue_count.numel())
if actual_decisions != expected_decisions:
    raise ValueError(...)
```

Merge deterministic diagnostic keys into `eval_summary.json` as `diagnostic_<key>`, and add:

```text
independent_policy_diagnostics_enabled
stats_sum_scheduled_bytes
stats_sum_ue_goodput_bytes
stats_sum_failed_bytes
stats_mean_scheduled_mbps
stats_mean_ue_goodput_mbps
stats_failed_mbps
```

Use:

```python
stats_mean_scheduled_mbps = scheduled_bytes / stats_count * 0.016
stats_mean_ue_goodput_mbps = goodput_bytes / stats_count * 0.016
stats_failed_mbps = failed_bytes / stats_count * 0.016
```

Fail if `stats_mean_ue_goodput_mbps` and the existing `stats_mean_goodput_mbps` differ beyond a tight numerical tolerance (`max(1e-6, 1e-9 * abs(existing))`). This proves both sources use the same measured window.

- [ ] **Step 6: Test end-to-end argument plumbing without running the simulator**

Extend `test_independent_online_path.py` to assert:

- `_sample_joint_action(..., independent_policy_diagnostics=True)` returns a trace with 9 diagnostic decisions for the three-PRG fixture;
- seeded action arrays, trace choices, logp, entropy, and global RNG state match flag-off results;
- `summarize_independent_policy_diagnostics(..., classification_prefix="shadow_")` emits the expected prefixed keys;
- replay still ignores diagnostic probabilities and reproduces the submitted Type-0 action.

- [ ] **Step 7: Run focused and evaluator-adjacent suites**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_reward.py \
    training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py
'
```

Expected: all pass.

- [ ] **Step 8: Commit observation-path aggregation**

Run on the host:

```bash
git add training/stageb_hgraph/independent_blank_diagnostics.py training/stageb_hgraph/online_ppo.py training/stageb_hgraph/eval_checkpoint_online.py training/stageb_hgraph/tests/test_independent_blank_diagnostics.py training/stageb_hgraph/tests/test_independent_online_path.py
git commit -m "feat: report blank mass and failed bytes"
```

---

## Task 4: Add deterministic B-subject selection and the diagnostic-only runner phase

**Files:**

- Modify: `training/stageb_hgraph/independent_blank_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_blank_diagnostics.py`
- Modify: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh:12-32,203-239,368-410,449-460`

- [ ] **Step 1: Write failing B fallback and hypothesis-report tests**

Add two selection cases:

```python
def test_choose_b_subject_prefers_formal_selection() -> None:
    selection = {"selected_iteration": 12, "candidates": {}}
    assert choose_b_diagnostic_iteration(selection) == 12

def test_choose_b_subject_falls_back_to_best_valid_goodput() -> None:
    selection = {
        "selected_iteration": None,
        "candidates": {
            "4": {"present": True, "metrics": {
                "goodput_mbps": 1500.0,
                "action_replay_mismatch_count": 0.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
            "8": {"present": True, "metrics": {
                "goodput_mbps": 1520.0,
                "action_replay_mismatch_count": 0.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
            "12": {"present": True, "metrics": {
                "goodput_mbps": 1600.0,
                "action_replay_mismatch_count": 1.0,
                "illegal_action_count": 0.0,
                "independent_deletion_equal": True,
            }},
        },
    }
    assert choose_b_diagnostic_iteration(selection) == 8
```

Add tests that reject missing/no-valid B candidates and break equal-goodput ties by lower fixed-cadence iteration.

Add a report test with A4/A8/B synthetic summaries asserting `H_frag` for A8 is:

- `supported=True` only when A8 learned count is nonzero and fragmentation share is strictly greater than `0.5`;
- `supported=False` at exactly `0.5` or below;
- `supported=None` and `reason="no learned Blank decisions"` when learned count is zero.

- [ ] **Step 2: Run focused tests and confirm RED**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q training/stageb_hgraph/tests/test_independent_blank_diagnostics.py
'
```

Expected: failure because B selection/report functions do not exist.

- [ ] **Step 3: Implement only the deterministic subject selection and report builders**

Add:

```python
def choose_b_diagnostic_iteration(selection: Mapping[str, object]) -> int: ...

def build_blank_diagnostic_report(
    *,
    summaries: Mapping[str, Mapping[str, object]],
    b_iteration: int,
    b_formal_selection: Mapping[str, object],
) -> dict[str, object]: ...
```

Selection uses the formal `selected_iteration` when non-null. Otherwise it scans fixed-cadence candidates and selects maximum formal `goodput_mbps` only among entries with:

```text
present == true
action_replay_mismatch_count == 0
illegal_action_count == 0
independent_deletion_equal == true
```

Ties choose the lower iteration. This fallback chooses a diagnostic subject only. The report builder derives `formal_gate_promoted` by comparing `b_formal_selection["selected_iteration"]` with `b_iteration`; it must be false when formal selection was null.

The report builder must fail closed unless labels are exactly `A4`, `A8`, and `B`, and every summary has:

- deterministic `decode_mode=argmax`;
- independent actor with `requires_pfq_features=false`;
- `independent_deletion_equal=true`;
- enabled diagnostics;
- `stats_count=512`, TTI 1001--1512, and 26,112 decisions;
- zero replay mismatch and illegal action sums;
- identical nonempty simulator binary hashes;
- finite Blank distribution fields, exact scheduled/goodput/failed identity, and measured TB counts.

Report per subject: Blank composition, probability/margin quantiles, scheduled/goodput/failed Mbps, TB tx/error/BLER, checkpoint hash, and B iteration. Evaluate primary `H_frag` on A8 because it is the near-goodput checkpoint whose Blank rate prompted this diagnostic; retain A4 as the known severe sample-to-greedy positive control.

The single next-experiment recommendation is deterministic, with forced-Blank dominance checked first:

- if forced Blank is the largest A8 Blank component: inspect candidate validity and sequential demand estimation before any policy change;
- otherwise, if A8 fragmentation share among learned Blank is `> 0.5`: test a minimal hierarchical `Serve/Blank -> UE` action head with matched PPO log-probability;
- otherwise: retain the flat categorical and test learned conservatism evidence (reward/BC calibration) without forced fill.

- [ ] **Step 4: Add a diagnostic-only runner phase**

Extend accepted phases and dispatch with:

```text
blank-diagnostics-seed41
```

Extend `evaluate_checkpoint` with a seventh optional positional argument, `diagnostics=${7:-0}`, and pass:

```bash
--independent-policy-diagnostics "${diagnostics}"
```

Existing callers therefore remain unchanged with diagnostics disabled.

Implement `run_blank_diagnostics_seed41` to:

1. require the completed A and B selection artifacts and the already-built simulator;
2. call a small Python entry point in `independent_blank_diagnostics.py` to resolve the B iteration;
3. create `blank_diagnostics_seed41/run_manifest.json` with current source revision, simulator hash, seed 41, warmup 1000, measure 512, labels/checkpoint paths/hashes, and the original formal selection paths;
4. call the existing evaluator for A4, A8, and B with seed 41, warmup 1000, measure 512, `normalize_gate=0`, and `diagnostics=1`;
5. invoke the report builder to write `blank_diagnostic_report.json` and a compact `blank_diagnostic_report.md` under the same directory.

Before the first launch, fail closed unless the current simulator SHA-256 equals `sim_binary_sha256` in the formal A4, formal A8, and chosen formal B `eval_summary.json`, and each checkpoint SHA-256 equals its corresponding formal `checkpoint_sha256`. Source revision is expected to differ because this phase adds observation code; simulator and checkpoint hashes must not.

Expose those two pure operations through an intentionally small CLI:

```bash
python -m training.stageb_hgraph.independent_blank_diagnostics select-b \
  --selection "${OUT_DIR}/pilot-B/checkpoint_selection.json"

python -m training.stageb_hgraph.independent_blank_diagnostics report \
  --a4 "${phase_dir}/A4/eval_summary.json" \
  --a8 "${phase_dir}/A8/eval_summary.json" \
  --b "${phase_dir}/B/eval_summary.json" \
  --b-iteration "${b_iteration}" \
  --b-selection "${OUT_DIR}/pilot-B/checkpoint_selection.json" \
  --json-out "${phase_dir}/blank_diagnostic_report.json" \
  --markdown-out "${phase_dir}/blank_diagnostic_report.md"
```

`select-b` prints only the integer iteration to stdout so Bash command substitution is unambiguous. `report` writes both outputs and exits nonzero on any validity failure.

The evaluation calls must remain:

```bash
evaluate_checkpoint "${a4_checkpoint}" 41 1000 512 "${phase_dir}/A4" 0 1
evaluate_checkpoint "${a8_checkpoint}" 41 1000 512 "${phase_dir}/A8" 0 1
evaluate_checkpoint "${b_checkpoint}" 41 1000 512 "${phase_dir}/B" 0 1
```

Do not add a new simulator launcher or copy checkpoints.

- [ ] **Step 5: Run focused tests plus Bash syntax validation**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q training/stageb_hgraph/tests/test_independent_blank_diagnostics.py &&
  bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
'
```

Expected: all pytest cases pass and Bash syntax exits `0`.

- [ ] **Step 6: Commit the runner and report path**

Run on the host:

```bash
git add training/stageb_hgraph/independent_blank_diagnostics.py training/stageb_hgraph/tests/test_independent_blank_diagnostics.py cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
git commit -m "feat: add seed41 blank diagnostic runner"
```

---

## Task 5: Verify behavior, execute the minimum experiment, and stop

**Files:**

- Verify: all files modified in Tasks 1--4
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/run_manifest.json`
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/A4/eval_summary.json`
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/A8/eval_summary.json`
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/B/eval_summary.json`
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/blank_diagnostic_report.json`
- Generate: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/blank_diagnostic_report.md`

- [ ] **Step 1: Run the proportional regression suite**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    training/stageb_hgraph/tests/test_independent_gate.py
'
```

Expected: all tests pass. This is the validity boundary for a diagnostic prototype; do not expand to an unrelated repository-wide suite.

- [ ] **Step 2: Run one short live smoke evaluation before the 512-TTI evidence runs**

Use A8 with 2 warmup and 4 measured TTIs in a disposable output directory, diagnostics enabled. Run this exact existing-path equivalent inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  set -euo pipefail
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor
  smoke_dir="output/pfq_independent_reuse/formal/blank_diagnostics_smoke"
  socket="/tmp/cumac_independent_blank_smoke_$$.sock"
  checkpoint="output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt"
  sim_bin="build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
  mkdir -p "${smoke_dir}"
  env CUMAC_ONLINE_BRIDGE=1 CUMAC_ONLINE_SOCKET="${socket}" CUMAC_ONLINE_PERSISTENT=1 \
      CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
      CUMAC_CELL_ASSOC_MODE=strongest CUMAC_UE_PLACEMENT_MODE=coop_boundary \
      CUMAC_UE_VORONOI_CLIP=1 CUMAC_BS_TX_PATTERN=omni \
      CUMAC_TOPOLOGY_SEED=41 CUMAC_TRAFFIC_SEED=41 \
      CUMAC_TRAFFIC_ARRIVAL_RATE=1.0 CUMAC_PACKET_TTL_TTI=0 CUMAC_PACKET_TTL_MS=200 \
      CUMAC_COMPACT_TTI_LOG=1 CUMAC_PROGRESS_TTI_INTERVAL=0 CUMAC_COMPARE_TTI_INTERVAL=0 \
      CUMAC_PACKET_DELAY_SAMPLES=1 \
      CUMAC_PACKET_OUTCOME_SAMPLES_CSV="${smoke_dir}/packet_outcome_samples.csv" \
      CUMAC_EXEC_MODE=both "${sim_bin}" -d 1 -b 2 -x 0 -f 0 -g 100 -r 3000 \
      > "${smoke_dir}/sim.log" 2>&1 &
  sim_pid=$!
  set +e
  python -m training.stageb_hgraph.eval_checkpoint_online \
      --checkpoint "${checkpoint}" --socket-path "${socket}" \
      --episode-horizon 6 --max-steps 6 --stats-warmup-steps 2 \
      --decode-mode argmax --seed 41 --sample-seed 41 --device auto \
      --print-every-steps 0 --save-steps 0 \
      --independent-policy-diagnostics 1 \
      --sim-bin "${sim_bin}" \
      --outcome-samples "${smoke_dir}/packet_outcome_samples.csv" \
      --out-dir "${smoke_dir}" > "${smoke_dir}/eval.log" 2>&1
  eval_rc=$?
  wait "${sim_pid}"
  sim_rc=$?
  set -e
  test "${eval_rc}" -eq 0
  test "${sim_rc}" -eq 0
'
```

Expected assertions from the smoke `eval_summary.json`:

```text
decode_mode == "argmax"
independent_policy_diagnostics_enabled == 1
diagnostic_decision_count == 4 * 51 == 204
stats_sum_action_replay_mismatch_count == 0
stats_sum_illegal_action_count == 0
stats_sum_failed_bytes == stats_sum_scheduled_bytes - stats_sum_ue_goodput_bytes
```

Delete no files; leave the disposable directory outside the formal diagnostic directory or name it `blank_diagnostics_smoke` so it cannot be mistaken for evidence.

- [ ] **Step 3: Execute the approved three-subject diagnostic**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  ./cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
    blank-diagnostics-seed41 \
    --build-dir build.reuse_actor \
    --out-dir output/pfq_independent_reuse/formal
'
```

Expected: three deterministic 1512-TTI evaluator runs complete and the JSON/Markdown reports are written. Do not run multi-seed or a fourth checkpoint.

- [ ] **Step 4: Independently verify evidence identities**

Run inside the container:

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python - <<"PY"
import json
from pathlib import Path

root = Path("output/pfq_independent_reuse/formal/blank_diagnostics_seed41")
rows = {label: json.loads((root / label / "eval_summary.json").read_text()) for label in ("A4", "A8", "B")}
hashes = {row["sim_binary_sha256"] for row in rows.values()}
assert len(hashes) == 1 and next(iter(hashes))
for label, row in rows.items():
    assert row["decode_mode"] == "argmax"
    assert row["requires_pfq_features"] is False
    assert row["independent_deletion_equal"] is True
    assert row["independent_policy_diagnostics_enabled"] == 1
    assert row["stats_count"] == 512
    assert row["measured_tti_first"] == 1001
    assert row["measured_tti_last"] == 1512
    assert row["diagnostic_decision_count"] == 512 * 51
    assert row["stats_sum_action_replay_mismatch_count"] == 0.0
    assert row["stats_sum_illegal_action_count"] == 0.0
    assert abs(row["stats_sum_failed_bytes"] - (row["stats_sum_scheduled_bytes"] - row["stats_sum_ue_goodput_bytes"])) <= 1e-6
print(json.dumps({label: {
    "blank": row["diagnostic_selected_blank_ratio"],
    "forced": row["diagnostic_forced_blank_ratio"],
    "fragmentation": row["diagnostic_fragmentation_blank_ratio"],
    "conservative": row["diagnostic_conservative_blank_ratio"],
    "fragmentation_among_learned": row["diagnostic_fragmentation_share_among_learned_blank"],
    "failed_mbps": row["stats_failed_mbps"],
    "tb_tx": row["measured_tb_tx_count"],
    "tb_err": row["measured_tb_err_count"],
} for label, row in rows.items()}, indent=2))
PY
'
```

Expected: all assertions pass and one compact three-row evidence object is printed.

- [ ] **Step 5: Inspect the report and apply the stop condition**

Read:

```bash
sed -n '1,240p' output/pfq_independent_reuse/formal/blank_diagnostics_seed41/blank_diagnostic_report.md
```

Report exactly:

- A4/A8/B selected, forced, learned, fragmentation, and conservative Blank composition;
- probability/margin distributions;
- scheduled, goodput, and failed Mbps plus TB tx/error/BLER;
- the formally selected or fallback B iteration and why;
- whether A8 supports `H_frag` under the strict `>50%` rule;
- the single next experiment selected by the predeclared rule.

Then stop. Do not implement a hierarchy, Blank bias, entropy schedule, forced fill, reward change, or new seed campaign.

- [ ] **Step 6: Run final Git and artifact hygiene checks**

Run on the host:

```bash
git status --short
git log -5 --oneline
```

Expected: only diagnostic experiment outputs and the user's pre-existing unrelated untracked artifacts remain uncommitted. Do not commit generated `output/` evidence unless the repository's existing convention explicitly tracks comparable formal result artifacts.

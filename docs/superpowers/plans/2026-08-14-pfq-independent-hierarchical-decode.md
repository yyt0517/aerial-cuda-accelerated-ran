> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-08-14-pfq-independent-hierarchical-decode.md)保留当时状态。

# PFQ-Independent Hierarchical Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a deterministic hierarchical `Serve/Blank -> UE` decoder removes A8 probability-fragmentation Blank decisions while preserving the frozen seed-41 boundary goodput of `1547.837 Mbps`.

**Architecture:** Keep the frozen PFQ-independent A8 actor, logits, mask construction, sequential demand update, Type-0 materializer, reward, checkpoint, and simulator unchanged. Add one independent-only evaluation mode that compares aggregate valid-UE service mass with Blank and, only when service wins strictly, selects the highest-logit valid UE. Extend the existing behavior-invariant diagnostics to compare this action with flat argmax, then run one isolated smoke and at most one `1000 + 512`-TTI treatment against the already-frozen A8 control.

**Tech Stack:** Python 3, PyTorch, pytest, Bash, the existing cuMAC simulator, Docker container `c_aerial_oai2`, and the existing PFQ-independent evaluation/reporting paths.

## Global Constraints

- This is a research-first mechanism experiment, not a new scheduler architecture. Implement only what is required to accept or reject `H_hier_decode`.
- Make source edits in the existing experiment worktree `/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor`, because it contains the ignored frozen checkpoints and formal outputs. Do not run the experiment from dirty `main`.
- Before editing, require the experiment worktree to have no tracked or staged changes, then fast-forward its branch from `main`. Its five untracked root artifacts are user-owned evidence; do not delete, restore, stage, or overwrite them.
- Run Python and simulator commands only after entering `c_aerial_oai2` and activating `/opt/nvidia/cuBB/.venv`. Host-side Git and `apply_patch` operations remain on the host.
- Reuse the frozen A8 checkpoint and control; do not retrain, fine-tune, rerun A4/B4/PFQ, add a seed, or sweep a threshold/bias.
- Existing `sample` and flat `argmax` behavior must remain bitwise unchanged. The new decode is deterministic and independent-head-only.
- Equality between aggregate Serve and Blank remains Blank. Invalid UE classes never enter the Serve aggregation. No safe-fill, repair, forced service, fallback, or new random draw is allowed.
- Preserve flat categorical log-probability and entropy semantics. Do not change PPO collection, replay, or loss behavior and do not run PPO.
- Treat `stats_sum_ue_goodput_bytes` as authoritative delivered bytes and require the exact measured-window identity `scheduled = goodput + failed`.
- Do not claim per-rescued-PRG byte causality. Report rescue counts and overall treatment deltas only.
- Stop immediately on an invalid smoke or formal run. If the valid formal goodput is below `1547.837 Mbps`, record a valid falsification and stop; do not tune.
- Commit after each source task. Live output remains ignored experiment evidence and is not committed.

---

### Task 0: Freeze the execution context and prove the baseline

**Files:**

- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/blank_diagnostics_seed41/A8/eval_summary.json`
- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/blank_diagnostics_seed41/run_manifest.json`
- Read only: `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt`
- Read only: `.worktrees/pfq-independent-reuse-actor/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection`
- Evidence only: `/tmp/pfq_hierarchical_root_before.sha256`

- [ ] **Step 1: Verify the experiment worktree is safe to advance**

Run on the host:

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$EXP" diff --quiet
git -C "$EXP" diff --cached --quiet
test -z "$(git -C "$EXP" ls-files -u)"
git -C "$EXP" status --short
```

Expected: the first three commands return `0`; status may show only the already-known untracked root artifacts and caches. If any tracked/staged/conflicted file appears, stop without merging or editing.

- [ ] **Step 2: Fast-forward the experiment branch to the approved design and plan**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$EXP" merge --ff-only main
git -C "$EXP" log -1 --oneline
git -C "$EXP" diff --quiet
git -C "$EXP" diff --cached --quiet
```

Expected: fast-forward succeeds and tracked/staged diffs remain empty. Do not apply or drop `stash@{0}`.

- [ ] **Step 3: Verify frozen identities and control invariants without rewriting evidence**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
sha256sum \
  "$EXP/output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt" \
  "$EXP/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor")
summary = json.loads((root / "output/pfq_independent_reuse/formal/blank_diagnostics_seed41/A8/eval_summary.json").read_text())
manifest = json.loads((root / "output/pfq_independent_reuse/formal/blank_diagnostics_seed41/run_manifest.json").read_text())
assert summary["checkpoint_sha256"] == "c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc"
assert summary["sim_binary_sha256"] == "3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c"
assert manifest["seed"] == 41
assert manifest["warmup_ttis"] == 1000
assert manifest["measure_ttis"] == 512
assert summary["stats_warmup_steps"] == 1000
assert summary["stats_count"] == 512
assert summary["measured_tti_first"] == 1001
assert summary["measured_tti_last"] == 1512
assert summary["diagnostic_decision_count"] == 26112.0
assert abs(summary["stats_mean_ue_goodput_mbps"] - 1547.837) <= 1e-6
assert summary["stats_sum_scheduled_bytes"] == summary["stats_sum_ue_goodput_bytes"] + summary["stats_sum_failed_bytes"]
print("frozen A8 control verified")
PY
```

Expected hashes: checkpoint `c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc`, simulator `3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c`; Python prints `frozen A8 control verified`.

- [ ] **Step 4: Snapshot protected root artifacts**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
cd "$EXP"
sha256sum output.txt output_short.txt packet_delay_samples.csv snr.txt ue_topology.csv \
  | tee /tmp/pfq_hierarchical_root_before.sha256
```

Expected hashes:

```text
489338736f2fc7fff488b91a02c3e6ab3b015e855541d230cfffeb9fee865d99  output.txt
1ce42f8bbbb4815307c0b59ccfd822278fd56c4fbf84adc9c100192815d09db6  output_short.txt
50e19e76179e6f5a38eac690980cfa2fd8767738bf5ab82713c44d627f54bbf0  packet_delay_samples.csv
2551411c05a411a7d5fc77bd07d1b238051c4f1055e7ec34803cb8dc9a90c4d8  snr.txt
8f20a1675e238196d7eddf91920acb6099649e44b59e5ef315cab3ff15ad47b5  ue_topology.csv
```

Compare the full hashes with the retained Task-5 evidence rather than accepting prefixes. If any differs, record it and stop before live execution.

- [ ] **Step 5: Run the existing focused baseline in the container**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    training/stageb_hgraph/tests/test_independent_gate.py
'
```

Expected: `88 passed`. Any failure invalidates the baseline and stops this plan.

---

### Task 1: Add the minimum hierarchical decoder without touching flat behavior

**Files:**

- Modify: `training/stageb_hgraph/independent_policy.py`
- Modify: `training/stageb_hgraph/tests/test_independent_policy.py`

- [ ] **Step 1: Write focused failing tests for the pure decision rule**

Add tests using literal logits and masks for these exact cases:

Create tests named `test_hierarchical_greedy_rescues_fragmentation_and_selects_best_valid_ue`, `test_hierarchical_greedy_keeps_conservative_blank_and_tie_blank`, `test_hierarchical_greedy_forces_blank_without_valid_ue`, and `test_hierarchical_greedy_excludes_invalid_high_logit`.

The first test must use a distribution equivalent to UE probabilities `0.30, 0.30` and Blank `0.40`: flat argmax is Blank, aggregate Serve is `0.60`, and hierarchical decode selects the best valid UE. The tie test must use aggregate Serve `0.50` and Blank `0.50` and assert Blank.

Add one policy-path test that calls `select_independent_action(actor, graph, packed, mode="hierarchical_greedy")` and asserts Type-0 preflight legality. Do not modify fixtures into a generalized decode framework.

- [ ] **Step 2: Write failing probability-identity tests**

For one finite masked distribution with at least two valid UEs, assert:

```python
flat_logp = torch.log_softmax(adjusted, dim=-1)
flat_prob = flat_logp.exp()
p_service = flat_prob[:, :blank].masked_fill(~valid_main, 0).sum(-1)
conditional = flat_prob[:, :blank] / p_service.unsqueeze(-1)

assert torch.allclose(
    p_service.log() + conditional.gather(1, selected_ue.unsqueeze(-1)).squeeze(-1).log(),
    flat_logp.gather(1, selected_ue.unsqueeze(-1)).squeeze(-1),
    atol=1e-6,
    rtol=0.0,
)
assert torch.allclose(
    hierarchical_group_entropy + p_service * conditional_entropy,
    -(flat_prob * flat_logp).sum(-1),
    atol=1e-6,
    rtol=0.0,
)
```

Also assert the returned trace still stores the gathered flat log-probability and flat entropy. This protects future matched-PPO semantics without adding a new head or training path.

- [ ] **Step 3: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_independent_policy.py \
    -k "hierarchical or factorization"
'
```

Expected: failures because the helper/mode does not exist or `hierarchical_greedy` is rejected. If the new tests pass before production changes, fix the tests.

- [ ] **Step 4: Implement one private selector and one new internal mode**

In `independent_policy.py`:

1. Extend the policy mode type to `Literal["sample", "greedy", "hierarchical_greedy"]`.
2. Add a small private pure helper with this contract:

```python
def _select_hierarchical_greedy(
    adjusted_logits: torch.Tensor,
    valid_main: torch.Tensor,
    blank_class_index: int,
) -> torch.Tensor:
    """Choose aggregate Serve strictly over Blank, then the best valid UE."""
```

3. Validate only the new path's shapes/index and finite relevant logits, failing closed on malformed/nonfinite input.
4. Compute service logits from `adjusted_logits[:, :blank_class_index]` after masking invalid UEs to `-torch.inf`.
5. Compute `serve_logit = torch.logsumexp(service_logits, dim=-1)`, `best_ue = service_logits.argmax(dim=-1)`, and `has_valid = valid_main.any(dim=-1)`.
6. Select a UE only for `has_valid & (serve_logit > adjusted_logits[:, blank_class_index])`; otherwise select Blank.
7. Insert only an `elif mode == "hierarchical_greedy"` branch after the existing sample branch. Leave stored-trace, teacher, sample, and greedy branches byte-for-byte unchanged where practical.
8. Keep the existing `logp_all.gather(1, selected.unsqueeze(-1)).squeeze(-1)`, flat entropy, sequential demand update, and Type-0 materializer unchanged.

- [ ] **Step 5: Prove GREEN and flat-path non-regression**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_independent_policy.py
'
```

Expected: all tests pass. Existing sample RNG-state, flat greedy choice, replay, legality, and diagnostic on/off invariance tests must remain green.

- [ ] **Step 6: Review and commit only Task 1**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$EXP" diff --check
git -C "$EXP" diff -- training/stageb_hgraph/independent_policy.py training/stageb_hgraph/tests/test_independent_policy.py
git -C "$EXP" add training/stageb_hgraph/independent_policy.py training/stageb_hgraph/tests/test_independent_policy.py
git -C "$EXP" commit -m "feat: add hierarchical independent decode"
```

Expected: one focused commit; no live output or user-owned artifact staged.

---

### Task 2: Make fragmentation rescue observable and route deterministic evaluation

**Files:**

- Modify: `training/stageb_hgraph/independent_blank_diagnostics.py`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Modify: `training/stageb_hgraph/tests/test_independent_blank_diagnostics.py`
- Modify: `training/stageb_hgraph/tests/test_independent_online_path.py`
- Modify only if needed for deletion-path coverage: `training/stageb_hgraph/tests/test_independent_pfq_deletion.py`

- [ ] **Step 1: Write failing diagnostic classification tests**

Extend literal diagnostic cases to assert the following per-decision tensors and summary counts:

```text
changed_from_greedy
fragmentation_rescued
selected_forced_blank
selected_fragmentation_blank
selected_conservative_blank
```

Required algebra:

```python
greedy_selected = adjusted_logits.argmax(dim=-1)
changed_from_greedy = selected != greedy_selected
fragmentation_rescued = fragmentation_blank & ~selected_blank
selected_forced_blank = forced_blank & selected_blank
selected_fragmentation_blank = fragmentation_blank & selected_blank
selected_conservative_blank = conservative_blank & selected_blank
```

For the `0.30, 0.30, 0.40` fragmentation case with a hierarchical UE selection, assert:

```text
counterfactual fragmentation count = 1
fragmentation rescue count = 1
selected fragmentation Blank count = 0
changed-from-greedy count = 1
```

For conservative, forced, and tie cases, assert no false rescue and the correct selected Blank subclass.

- [ ] **Step 2: Write failing evaluator-routing tests**

Add tests that require:

- parser accepts `--decode-mode hierarchical_argmax`;
- independent actor maps `argmax -> greedy` and `hierarchical_argmax -> hierarchical_greedy`;
- candidate/non-independent actors reject `hierarchical_argmax` before simulator startup;
- `sample` remains rejected for independent deterministic evaluation;
- PFQ deletion equality invokes the same requested internal mode for full and PFQ-deleted graphs;
- hierarchical selection still has zero replay mismatch and a legal Type-0 action.

Use monkeypatch/spies around `select_independent_action` for routing assertions; do not start the simulator in unit tests.

- [ ] **Step 3: Prove RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    -k "hierarchical or fragmentation_rescue or selected_fragmentation or deletion"
'
```

Expected: failures for missing fields, summary keys, CLI choice, or routing.

- [ ] **Step 4: Extend diagnostics with counts, not a second metric system**

Add the five tensors to `IndependentPolicyDiagnostics`. Populate them in `compute_independent_policy_diagnostics`; generic dataclass-field concatenation should then continue to work unchanged.

Extend `summarize_independent_policy_diagnostics` with these unprefixed keys, which the evaluator will expose with its existing `diagnostic_` wrapper:

```text
changed_from_greedy_count
changed_from_greedy_ratio
fragmentation_blank_count
fragmentation_rescue_count
fragmentation_rescue_ratio
selected_forced_blank_count
selected_forced_blank_ratio
selected_fragmentation_blank_count
selected_fragmentation_blank_ratio
selected_conservative_blank_count
selected_conservative_blank_ratio
```

Keep all existing diagnostic names and semantics. The `fragmentation_blank_count` remains the flat-argmax counterfactual even under hierarchical selection.

- [ ] **Step 5: Route the public evaluation mode and deletion check**

In `eval_checkpoint_online.py`:

1. Add `hierarchical_argmax` to `--decode-mode` choices.
2. For an independent actor, map `argmax` to `greedy` and `hierarchical_argmax` to `hierarchical_greedy`; reject every other decode mode.
3. Reject `hierarchical_argmax` for non-independent actors before entering their decode path.
4. Pass the selected internal mode into `_independent_deletion_equal` so both full and PFQ-deleted graphs exercise the treatment path.
5. Keep checkpoint loading, graph validation, replay, action materialization, summary aggregation, and diagnostic on/off behavior unchanged.

- [ ] **Step 6: Prove GREEN**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py
'
```

Expected: all selected tests pass, including flat action/RNG invariance and both deletion modes.

- [ ] **Step 7: Review and commit only Task 2**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$EXP" diff --check
git -C "$EXP" add \
  training/stageb_hgraph/independent_blank_diagnostics.py \
  training/stageb_hgraph/eval_checkpoint_online.py \
  training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
  training/stageb_hgraph/tests/test_independent_online_path.py \
  training/stageb_hgraph/tests/test_independent_pfq_deletion.py
git -C "$EXP" commit -m "feat: diagnose hierarchical blank rescues"
```

Before committing, omit `test_independent_pfq_deletion.py` from `git add` if it was not actually changed.

---

### Task 3: Add one hypothesis gate and two explicit A8 runner phases

**Files:**

- Create: `training/stageb_hgraph/hierarchical_decode_gate.py`
- Create: `training/stageb_hgraph/tests/test_hierarchical_decode_gate.py`
- Modify: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh`

- [ ] **Step 1: Write literal RED tests for the scientific gate**

Use hand-built `dict` inputs rather than a production builder to test:

Create tests named `test_report_supports_valid_treatment_at_or_above_control_goodput`, `test_report_valid_but_falsified_below_control_goodput`, `test_report_rejects_identity_window_or_accounting_mismatch`, and `test_report_rejects_unrescued_or_selected_fragmentation_blank`.

The report function contract is:

The exact signature is `build_hierarchical_decode_report(control: Mapping[str, object], treatment: Mapping[str, object], *, control_manifest: Mapping[str, object], treatment_manifest: Mapping[str, object]) -> dict[str, object]`.

Tests must cover these hard gates:

- control decode `argmax`; treatment decode `hierarchical_argmax`;
- identical nonempty checkpoint and simulator SHA-256 values;
- both manifests identify scenario `coop_boundary-3c36ue-17prg-type0-svd-p3000-load1-ttl200-seed41`, seed `41`, warmup `1000`, measure `512`, and A8 iteration `8`;
- treatment `requires_pfq_features is False` and `independent_deletion_equal is True`;
- `completed_steps=1512`, `stats_warmup_steps=1000`, `stats_count=512`, TTIs `1001..1512`, and `diagnostic_decision_count=26112.0`;
- zero `stats_sum_action_replay_mismatch_count` and `stats_sum_illegal_action_count`;
- exact `stats_sum_scheduled_bytes == stats_sum_ue_goodput_bytes + stats_sum_failed_bytes`;
- `diagnostic_selected_fragmentation_blank_count == 0`;
- `diagnostic_fragmentation_rescue_count == diagnostic_fragmentation_blank_count`;
- frozen control `stats_mean_ue_goodput_mbps` is within `1e-6` of `1547.837`.

A hard-gate failure raises `ValueError` and produces no scientific verdict. Once valid, set `hypothesis_supported = treatment_goodput >= control_goodput`; a lower valid treatment is a falsification, not an exception.

- [ ] **Step 2: Prove gate RED**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_hierarchical_decode_gate.py
'
```

Expected: import/implementation failures.

- [ ] **Step 3: Implement the small report module and CLI**

Implement only:

```bash
python -m training.stageb_hgraph.hierarchical_decode_gate \
  --control CONTROL_SUMMARY \
  --treatment TREATMENT_SUMMARY \
  --control-manifest CONTROL_MANIFEST \
  --treatment-manifest TREATMENT_MANIFEST \
  --json-out REPORT_JSON \
  --markdown-out REPORT_MD
```

The JSON and Markdown must report:

- `valid` and `hypothesis_supported`;
- control/treatment/delta for authoritative goodput Mbps;
- scheduled Mbps, failed Mbps, measured TB tx/error/BLER;
- selected, forced, conservative, counterfactual fragmentation, rescue, and changed-action counts/ratios;
- expired packet/byte/system-time metrics, expiry-aware delay metrics, and UE minimum/P10/macro expiry-aware rates;
- checkpoint/simulator hashes, seed, window, and PFQ-independence checks;
- this limitation: rescued PRGs are counted, but aggregate TB bytes are not attributed to individual rescues.

Do not add plotting, schemas, reusable report frameworks, or a bias recommendation.

- [ ] **Step 4: Write RED runner checks**

Add a small text/CLI test in `test_hierarchical_decode_gate.py` that asserts the runner:

- supports the two explicit new phases `hierarchical-decode-smoke` and `hierarchical-decode-seed41` while retaining all existing phases;
- passes a decode argument through `evaluate_checkpoint` instead of hardcoding argmax;
- references only A8 iteration 8 for these phases;
- uses new isolated output directories;
- enables diagnostics;
- invokes the formal gate only after a successful formal evaluation.

Run the test and expect failure before editing the runner.

- [ ] **Step 5: Add the minimum runner wiring**

In `run_stageB_pfq_independent_reuse.sh`:

1. Add `local decode_mode="${8:-argmax}"` to `evaluate_checkpoint` and pass `--decode-mode "${decode_mode}"`.
2. Add `hierarchical-decode-smoke`:
   - frozen A8 iteration-8 checkpoint only;
   - seed/sample-seed `41`;
   - warmup `2`, measure `4`;
   - diagnostics `1`, decode `hierarchical_argmax`;
   - output `${OUT_DIR}/hierarchical_decode_smoke`;
   - inline fail-closed assertions for 4 measured TTIs, 204 decisions, zero illegal/replay mismatches, byte identity, zero selected fragmentation Blank, and rescue count equal to counterfactual fragmentation count.
3. Add `hierarchical-decode-seed41`:
   - frozen A8 iteration-8 checkpoint only;
   - frozen control summary `${OUT_DIR}/blank_diagnostics_seed41/A8/eval_summary.json` and its manifest;
   - seed/sample-seed `41`, warmup `1000`, measure `512`;
   - diagnostics `1`, decode `hierarchical_argmax`;
   - output `${OUT_DIR}/hierarchical_decode_seed41`;
   - a specific manifest containing phase, scenario, seed, window, A8 iteration/checkpoint SHA, and simulator SHA;
   - after successful evaluation, invoke `hierarchical_decode_gate` to produce `hierarchical_decode_report.json` and `.md`.
4. Preserve existing phase behavior; calls that omit the eighth argument must still use flat `argmax`.
5. Do not add a loop, seed list, threshold option, automatic fallback, or resume across mismatched manifests.

- [ ] **Step 6: Prove GREEN and syntax validity**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q training/stageb_hgraph/tests/test_hierarchical_decode_gate.py &&
  bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh &&
  python -m py_compile training/stageb_hgraph/hierarchical_decode_gate.py
'
```

Expected: all tests pass; both syntax checks return `0`.

- [ ] **Step 7: Review and commit only Task 3**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
git -C "$EXP" diff --check
git -C "$EXP" add \
  training/stageb_hgraph/hierarchical_decode_gate.py \
  training/stageb_hgraph/tests/test_hierarchical_decode_gate.py \
  cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
git -C "$EXP" commit -m "test: gate hierarchical decode experiment"
```

---

### Task 4: Run the proportional regression and one 2+4 smoke

**Files:**

- Read: all source files changed in Tasks 1-3
- Create ignored evidence: `output/pfq_independent_reuse/formal/hierarchical_decode_smoke/`
- Compare: `/tmp/pfq_hierarchical_root_before.sha256`

- [ ] **Step 1: Run the complete relevant regression once**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python -m pytest -q \
    training/stageb_hgraph/tests/test_independent_blank_diagnostics.py \
    training/stageb_hgraph/tests/test_independent_policy.py \
    training/stageb_hgraph/tests/test_independent_online_path.py \
    training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
    training/stageb_hgraph/tests/test_independent_gate.py \
    training/stageb_hgraph/tests/test_hierarchical_decode_gate.py &&
  python -m py_compile \
    training/stageb_hgraph/independent_policy.py \
    training/stageb_hgraph/independent_blank_diagnostics.py \
    training/stageb_hgraph/eval_checkpoint_online.py \
    training/stageb_hgraph/hierarchical_decode_gate.py &&
  bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
'
```

Expected: all selected tests and syntax checks pass. This is the proportional suite; do not expand to the whole repository unless an observed failure requires it.

- [ ] **Step 2: Confirm the smoke destination is new**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
test ! -e "$EXP/output/pfq_independent_reuse/formal/hierarchical_decode_smoke"
```

Expected: return `0`. If it exists, do not delete or overwrite it; stop and report the evidence-directory collision.

- [ ] **Step 3: Run exactly one smoke**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  bash cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
    hierarchical-decode-smoke \
    --build-dir build.reuse_actor \
    --out-dir output/pfq_independent_reuse/formal
'
```

Expected: evaluator and simulator return `0`; the phase's fail-closed smoke assertions pass. Do not launch the formal phase if any assertion fails or `eval_summary.json` is absent.

- [ ] **Step 4: Independently inspect the smoke invariants**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python - <<PY
import json
from pathlib import Path
p = Path("output/pfq_independent_reuse/formal/hierarchical_decode_smoke/eval_summary.json")
s = json.loads(p.read_text())
assert s["decode_mode"] == "hierarchical_argmax"
assert s["stats_count"] == 4
assert s["diagnostic_decision_count"] == 204.0
assert s["stats_sum_action_replay_mismatch_count"] == 0.0
assert s["stats_sum_illegal_action_count"] == 0.0
assert s["independent_deletion_equal"] is True
assert s["requires_pfq_features"] is False
assert s["stats_sum_scheduled_bytes"] == s["stats_sum_ue_goodput_bytes"] + s["stats_sum_failed_bytes"]
assert s["diagnostic_selected_fragmentation_blank_count"] == 0.0
assert s["diagnostic_fragmentation_rescue_count"] == s["diagnostic_fragmentation_blank_count"]
print("hierarchical smoke valid")
PY
'
```

Expected: `hierarchical smoke valid`.

- [ ] **Step 5: Prove root-artifact isolation**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
cd "$EXP"
sha256sum -c /tmp/pfq_hierarchical_root_before.sha256
git status --short
```

Expected: all five hashes report `OK`; no tracked source diff remains after the three commits. If hashes changed, stop and report experimental contamination.

- [ ] **Step 6: Apply the smoke stop rule**

If the smoke is valid, record its exact summary path and proceed to Task 5. If invalid, stop with the failing invariant and saved logs; do not patch and rerun in the same evidence round without a new diagnosis and approval.

---

### Task 5: Run one formal A8 treatment, decide the hypothesis, and stop

**Files:**

- Read control: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/A8/eval_summary.json`
- Read control manifest: `output/pfq_independent_reuse/formal/blank_diagnostics_seed41/run_manifest.json`
- Create ignored evidence: `output/pfq_independent_reuse/formal/hierarchical_decode_seed41/`
- Read generated: `hierarchical_decode_report.json`
- Read generated: `hierarchical_decode_report.md`

- [ ] **Step 1: Confirm the formal destination is new and frozen inputs still match**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
test ! -e "$EXP/output/pfq_independent_reuse/formal/hierarchical_decode_seed41"
sha256sum \
  "$EXP/output/pfq_independent_reuse/formal/pilot-A/candidate_checkpoints/ppo_actor_iter_0008.pt" \
  "$EXP/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
```

Expected: destination absent; hashes exactly match Task 0. Never delete an existing destination to make this check pass.

- [ ] **Step 2: Run exactly one formal treatment**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  bash cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
    hierarchical-decode-seed41 \
    --build-dir build.reuse_actor \
    --out-dir output/pfq_independent_reuse/formal
'
```

Expected: only A8 hierarchical evaluation runs for seed 41, with `1000` warmup and `512` measured TTIs; runner and gate return `0`. Do not rerun on failure.

- [ ] **Step 3: Independently validate the generated verdict**

```bash
docker exec c_aerial_oai2 bash -lc '
  cd /opt/nvidia/cuBB && source .venv/bin/activate &&
  cd .worktrees/pfq-independent-reuse-actor &&
  python - <<PY
import json
from pathlib import Path
root = Path("output/pfq_independent_reuse/formal/hierarchical_decode_seed41")
r = json.loads((root / "hierarchical_decode_report.json").read_text())
assert r["valid"] is True
assert r["control_goodput_mbps"] == 1547.837
assert r["treatment_selected_fragmentation_blank_count"] == 0.0
assert r["treatment_fragmentation_rescue_count"] == r["treatment_counterfactual_fragmentation_blank_count"]
expected = r["treatment_goodput_mbps"] >= r["control_goodput_mbps"]
assert r["hypothesis_supported"] is expected
print(json.dumps({
    "supported": r["hypothesis_supported"],
    "control_goodput_mbps": r["control_goodput_mbps"],
    "treatment_goodput_mbps": r["treatment_goodput_mbps"],
    "delta_goodput_mbps": r["delta_goodput_mbps"],
}, sort_keys=True))
PY
'
```

Expected: validity assertions pass and one verdict line is printed. The Boolean identity comparison is intentional because the report values are JSON booleans.

- [ ] **Step 4: Recheck isolation and repository hygiene**

```bash
EXP=/home/oai2/aerial-cuda-accelerated-ran/.worktrees/pfq-independent-reuse-actor
cd "$EXP"
sha256sum -c /tmp/pfq_hierarchical_root_before.sha256
git diff --check
git diff --quiet
git diff --cached --quiet
git status --short
```

Expected: protected hashes remain `OK`; tracked/staged trees are clean; ignored evidence exists only under the new output directories.

- [ ] **Step 5: Report the research result and stop**

Report exactly:

- what changed: deterministic grouped decode only;
- what was tested: frozen A8, cooperative boundary, seed 41, `1000 + 512` TTIs;
- whether all validity gates passed;
- control/treatment/delta goodput;
- counterfactual fragmentation, rescue, selected fragmentation, forced/conservative Blank, scheduled/failed Mbps, BLER, expiry-aware delay, and UE-rate metrics;
- whether `H_hier_decode` is supported or falsified;
- remaining uncertainty: no multi-seed or retraining evidence and no per-rescue byte attribution.

Then stop. Do not start bias tuning, reward changes, a trainable two-head actor, multi-seed evaluation, refactoring, or production hardening without a new user-approved research question.

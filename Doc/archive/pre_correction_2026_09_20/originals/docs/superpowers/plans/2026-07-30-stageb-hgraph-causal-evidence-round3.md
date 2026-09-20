# Stage-B HGraph Causal Evidence Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit inherited commit `08412ea` and finish direct-only policy
gradient evidence, Python preflight availability, and exact Type-0 replay.

**Architecture:** Keep `evaluate_causal_gate(mapping)` unchanged. Direct PPO
emits a policy-loss-only candidate gradient; rollout collection emits real
per-step Python diagnostic availability; the CLI accepts only the resulting
real history fields. Exact replay remains at the `env.step` seam.

**Tech Stack:** Python 3.10, PyTorch autograd, pytest, CSV/JSON gate CLI, CUDA
online smoke through the existing cuMAC container/build.

## Global Constraints

- Preserve `08412ea`; no checkout, reset, amend, or history rewrite.
- Preserve legacy warning behavior and avoid legacy gradient-metric overhead.
- Run only one fresh 2x64 seed-41 CUDA/direct smoke after source/tests commit.
- Archive old smoke artifacts before overwrite; do not commit generated files.

---

### Task 1: Direct-only candidate policy gradient

**Files:**
- Modify: `training/stageb_hgraph/online_ppo.py`
- Modify: `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`

**Interfaces:**
- Consumes: `policy_loss`, `actor.named_parameters()`,
  `candidate_policy_mode`.
- Produces: direct-only `candidate_policy_grad_norm: float`.

- [ ] Add a regression that runs direct PPO with zero advantages and nonzero
  entropy coefficient, asserting `candidate_policy_grad_norm == 0.0` while
  `candidate_head_grad_norm > 0.0`.
- [ ] Add a legacy regression that fails if `torch.autograd.grad` is called and
  asserts `candidate_policy_grad_norm` is absent.
- [ ] Run both tests against inherited `08412ea` and record expected failures:
  actor-loss isolation is not yet covered; legacy calls autograd and emits the
  field.
- [ ] Restrict candidate parameter selection to explicit
  `action_head.candidate_*` trainable parameters, call `torch.autograd.grad`
  only in direct mode, and conditionally add the metric only to the direct
  summary.
- [ ] Run the focused gradient tests and record GREEN.

### Task 2: Python preflight availability schema

**Files:**
- Modify: `training/stageb_hgraph/online_ppo.py`
- Modify: `training/stageb_hgraph/causal_gate.py`
- Modify: `training/stageb_hgraph/tests/test_causal_gate.py`
- Modify: `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`

**Interfaces:**
- Produces/consumes:
  `rollout_python_preflight_diagnostics_available_mean`.

- [ ] Change fixture/tests first from the untrusted unsuffixed field to the
  `_mean` field and assert a two-step `[1.0, 0.0]` rollout reports `0.5`.
- [ ] Run the availability cases and record RED because inherited production
  emits/consumes the unsuffixed all-steps boolean.
- [ ] Emit `np.mean([step.python_preflight_diagnostics_available])`, retain
  direct exception propagation and legacy warnings, and require every history
  row's `_mean` value to equal exactly `1.0`.
- [ ] Run availability and full gate tests and record GREEN.

### Task 3: Exact replay audit and mutation proof

**Files:**
- Audit: `training/stageb_hgraph/online_ppo.py`
- Audit: `training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py`

**Interfaces:**
- Consumes the exact arrays immediately before `runner.env.step`.
- Produces `action_replay_mismatch_count`.

- [ ] Confirm fresh `materialize_candidate_choices_to_type0` output is compared
  by shape and element content for both Type-0 arrays.
- [ ] Run slot-permutation and shape-only regressions.
- [ ] Temporarily restore semantic-only replay and confirm those regressions
  fail; restore exact replay without altering history.

### Task 4: Verification, commit, and smoke

**Files:**
- Update:
  `.superpowers/sdd/2026-07-30-stageb-hgraph-causal-action-alignment/task-6-report.md`
- Generate only:
  `training/stageb_hgraph/runs/causal_direct_smoke/*`

**Interfaces:**
- Produces a committed source/test correction and uncommitted final evidence.

- [ ] Run both focused files, the full four-file Phase-1 suite, `py_compile`,
  `bash -n`, and `git diff --check`.
- [ ] Commit audited source/tests separately.
- [ ] Archive existing `causal_direct_smoke` under an explicit timestamped
  sibling name.
- [ ] Run exactly one 2x64 seed-41 CUDA/direct smoke in `c_aerial_oai2` with
  the current worktree binary and `--build-method skip`.
- [ ] Regenerate the gate, inspect both raw CSV rows for the new gradient and
  `_mean` availability fields, compute report SHA-256, restore
  `parameters.h`, and confirm no process remains.
- [ ] Update the Task-6 report with decisions A/B/C, RED/GREEN/mutation
  evidence, commit, smoke evidence, report hash, and final status.

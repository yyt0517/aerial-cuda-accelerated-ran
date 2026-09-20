> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/specs/2026-07-30-stageb-hgraph-causal-evidence-round3-design.md)保留当时状态。

# Stage-B HGraph Causal Evidence Round 3 Design

## Goal

Close three evidence gaps in the direct Stage-B PPO causal gate without
changing legacy behavior or the direct `evaluate_causal_gate(mapping)` API.

## Candidate policy gradient

Direct PPO updates emit `candidate_policy_grad_norm`. The value is the
pre-clipping L2 norm of gradients from `policy_loss` alone to every trainable
candidate-specific action-head parameter. Entropy and auxiliary losses are
excluded. The additional autograd traversal and metric exist only when
`candidate_policy_mode == "direct"`; legacy update rows omit the field.

The gate consumes only this emitted field. `candidate_head_grad_norm` remains a
broader optimizer-gradient diagnostic and cannot satisfy the causal gate.

## Python preflight availability

Every rollout step records whether all five Python Type-0 preflight diagnostics
were computed. Direct-mode diagnostic exceptions propagate before environment
submission. Legacy mode preserves its warning-and-continue behavior and records
the failed step as unavailable.

Each history row emits the actual mean of those per-step zero/one values as
`rollout_python_preflight_diagnostics_available_mean`. The gate requires the
field on every row and requires every value to equal exactly `1.0`, in addition
to requiring all five preflight reject means to total zero.

## Exact Type-0 replay

For direct mode, exact replay means shape-and-content equality between the
arrays about to be submitted to `env.step` and a fresh deterministic
`materialize_candidate_choices_to_type0` result built from the carried packed
candidates and sampled trace.

The mismatch count is the number of unequal array elements across
`action_ue_select` and `action_prg_alloc`. A shape mismatch contributes a
nonzero mismatch even when flattened values match. Same-cell slot permutations
are therefore mismatches despite preserving per-PRG UE semantics, because they
do not reproduce deterministic first-occurrence Type-0 materialization.

## Failure behavior and tests

Missing/nonfinite gradient or availability history remains fail-closed.
`evaluate_causal_gate(mapping)` is unchanged. Regression tests cover positive
direct policy gradient, zero policy gradient with entropy-only candidate-head
updates, legacy omission/no extra autograd traversal, direct preflight
propagation, legacy availability reporting, exact slot permutation mismatch,
and exact shape mismatch. Mutation checks replace `policy_loss` with
`actor_loss`, swallow the direct preflight exception, and restore semantic-only
replay; each relevant regression must fail.

After source verification and commit, a single fresh two-iteration,
64-step, seed-41 CUDA/direct smoke regenerates the final history and gate.
Existing artifacts are archived under an explicit timestamped name before
overwrite.

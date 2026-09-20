# Seed45 Dimensionless Service Features and CTBS-Legal Blank Guard Design

> Status: completed historical experiment protocol. Its retained result and the current feature semantics are consolidated in [`Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md) and the [DPA-v1 main design](./2026-08-13-pfq-independent-reuse-actor-design.md).

## Goal

Test the smallest remaining mechanism hypothesis behind the seed45 failure:
whether a scale-stable per-UE representation can turn already observable
deadline pressure into learned service priority. Only if feature-only PPO still
permits a backlogged UE to receive zero scheduled dequeue for one full TTL is a
deterministic CTBS-legal blank guard enabled.

## Fixed scope

- Initialize from `deadline_tail_rate_alignment_v1` absolute-best iter21.
- Keep the 16-dimensional UE observation and the current actor architecture.
- Keep deadline-tail-rate reward coefficients, candidates, 17-step PRG order,
  Type-0 mapping, CTBS-v1 accounting, and all PHY settings unchanged.
- Use seed45 as the sole development stress case.
- Do not scan feature scales, reward coefficients, or guard thresholds.
- Do not use UE identity or layout position as a priority input.
- Do not run seed48 in this experiment.

## Dimensionless feature contract

The existing UE feature slots 12--15 are replaced without changing tensor
shape. Slots 0--11 retain their existing meanings.

For UE `i`, the post-action scheduled-dequeue EWMA is causal:

```text
S_hat_i(t+1) = 0.95 * S_hat_i(t) + 0.05 * scheduled_dequeue_bytes_i(t)
```

At the next pre-action snapshot, if `Q_i == 0`, slots 12--15 are all zero.
Otherwise:

```text
slack_tti = max(TTL_slack_tti, 1)
drain_pressure = Q_i / (S_hat_i * slack_tti + 1 byte)
f12 = log1p(drain_pressure)
f13 = clip(S_hat_i * slack_tti / max(Q_i, 1 byte), 0, 1)
f14 = clip(age_since_positive_scheduled_dequeue_tti / 400, 0, 1)
f15 = 0
```

Feature 12 is `log1p_drain_pressure`, feature 13 is
`scheduled_dequeue_adequacy`, feature 14 is
`ttl_normalized_service_age`, and feature 15 is `reserved_zero`.

Service age is reset only by positive scheduled dequeue, never by successful
bytes. The current TTI result is used only to construct the next snapshot.
Thus PHY success/failure does not change queue-service history semantics.

## Checkpoint compatibility

The tensor shape remains 16, so iter21 loads without a shape adapter. All
weights are retained and PPO may update the complete actor. No projection is
reinitialized. This intentionally tests whether the existing policy can adapt
to a better representation without adding a network path.

The old formal results remain tied to their recorded simulator binary hash.
They are not rerun through the new feature semantics and are not claimed to be
binary-identical.

## Feature-only experiment

Run a fixed short seed45 adaptation:

- 8 PPO iterations;
- 512 measured rollout steps per iteration;
- 1000 warmup steps per iteration;
- deadline-tail-rate reward unchanged;
- last checkpoint selected in advance; no best-iteration cherry-pick.

Evaluate the fixed last checkpoint with 1000 warmup TTIs followed by 1000
measured TTIs. Preserve per-UE scheduled-dequeue traces, allocation, outcome,
expiry, goodput, BLER, replay, legality, and CTBS parity.

Feature-only starvation is present when any UE with positive backlog satisfies
either condition in the measured window:

1. total scheduled dequeue is zero; or
2. consecutive positive-backlog TTIs with zero scheduled dequeue reaches the
   400-TTI packet TTL horizon.

If neither condition occurs, stop without implementing the guard. Expiry,
goodput, BLER, outcome delay, and per-UE rates are still reported separately.

## Conditional CTBS-legal blank guard

The guard is implemented only after feature-only starvation is observed. It is
an evaluation/deployment safety layer, not a PPO training intervention.

For each cell and TTI, the guard may replace at most one actor-selected blank.
An eligible UE must:

- have positive demand;
- be a currently valid candidate at that exact sequential PRG step;
- be evaluated under CTBS-v1 pre-action headroom state;
- have `drain_pressure > 1`;
- have `scheduled_dequeue_adequacy < 1`;
- have `ttl_normalized_service_age >= 0.25`.

Among eligible candidates, select lexicographically by greatest normalized
service age and then greatest log drain pressure. UE ID is used only as a
deterministic final tie-break, not as a priority feature. A positive scheduled
dequeue on the guarded TTI resets service age on the next snapshot.

The guard never replaces a nonblank actor choice in v1. Therefore it tests the
known seed45 over-blank failure with minimal policy interference. If no blank
is available, or no CTBS-valid urgent candidate exists, it does nothing.

The guarded rollout uses the same feature-only checkpoint and no further
training. Replay mismatch, illegal action, candidate truncation, mask
reactivation, and CTBS MCS/layer/TBS mismatch must all remain zero.

## Evidence and stop conditions

The experiment stops after one of these outcomes:

- feature-only removes starvation: do not build or run the guard;
- feature-only starves but guarded evaluation removes it: report the resource
  and business-metric cost of the forced blank replacements;
- guarded evaluation still starves: reject blank-only guard v1 and record
  whether the blocker was no blank, no CTBS-valid candidate, or insufficient
  one-PRG-per-cell intervention.

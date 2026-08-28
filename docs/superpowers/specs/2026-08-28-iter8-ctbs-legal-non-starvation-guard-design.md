# Iter8 CTBS-Legal Non-Starvation Guard V1 Design

## Research question

Can a deterministic, blank-only, CTBS-legal safety layer prevent the frozen
iter8 actor from sacrificing a deadline-stressed UE on seed51, while preserving
the actor's goodput, BLER, resource-efficiency, sequential-decision, and Type-0
legality properties?

This is a guard-mechanism experiment, not a new reward, training, or model
experiment. It is intentionally limited to the smallest intervention supported
by the seed51 failure evidence.

## Prior evidence and hypothesis

The frozen iter8 dimensionless-feature actor passed seed48 but failed untouched
seed51. On seed51, goodput, BLER, outcome-delay mean/p90, and UE p10 rate all
improved relative to PFQ, while byte-weighted expiry increased from `4.0893%`
to `4.8509%`.

The failure was concentrated in UE4 and UE6:

| UE | policy | PRGs | scheduled dequeue | expired bytes | maximum positive-backlog zero-dequeue streak |
|---:|---|---:|---:|---:|---:|
| 4 | PFQ | 3385 | 6.239 MB | 5.968 MB | 19 TTI |
| 4 | actor | 165 | 0.311 MB | 11.896 MB | 307 TTI |
| 6 | PFQ | 3820 | 7.313 MB | 4.531 MB | 7 TTI |
| 6 | actor | 1280 | 2.898 MB | 9.060 MB | 272 TTI |

The hypothesis is that the actor already exposes enough causal pressure state,
but needs a narrow deployment invariant: once a UE has been inadequately served
for one quarter of its TTL and cannot drain before expiry, an otherwise blank
CTBS-legal PRG should provide positive marginal scheduled capacity.

## Frozen artifacts and algorithm settings

The following values remain unchanged:

- checkpoint:
  `output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt`
- checkpoint SHA-256:
  `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`
- actor-state SHA-256:
  `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`
- simulator SHA-256:
  `4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`
- UE observation dimension `16` and all feature definitions;
- feature slots 12--14: `log1p(drain pressure)`, scheduled-dequeue
  adequacy, and TTL-normalized service age;
- deadline-tail-rate reward and every coefficient;
- actor architecture and weights;
- candidate construction and complete runtime candidate width;
- fixed local PRG order `0..16`;
- argmax decoding;
- CTBS-v1 pre-action handshake and cumulative-TBS calculation;
- Type-0 materialization;
- PHY, topology, traffic, packet size, TTL, and evaluation horizons;
- no slot repair, safe fill, blank budget, logit boost, or other decoder
  intervention.

No training, adaptation, coefficient scan, threshold scan, or checkpoint
selection is permitted in this protocol.

## Guard configuration

Guard V1 has one enabled configuration and one disabled configuration. There
are no user-tunable research coefficients.

```text
enabled = true or false
minimum_log1p_drain_pressure = log(2)
maximum_scheduled_dequeue_adequacy = 1.0
minimum_ttl_normalized_service_age = 0.25
maximum_overrides_per_cell_per_tti = 1
override_nonblank = false
require_positive_ctbs_capacity_delta = true
```

The pressure comparison is strict: `log1p(drain pressure) > log(2)`, which is
equivalent to `drain pressure > 1`. Adequacy is also strict:
`scheduled_dequeue_adequacy < 1`. Service age is inclusive:
`TTL-normalized service age >= 0.25`, corresponding to `100 TTI / 50 ms` under
the frozen `400 TTI / 200 ms` TTL.

Waiting for a full `400 TTI` service-age threshold is forbidden in V1 because
seed51 UE4 and UE6 already accumulated severe expiry at 307 and 272 TTIs.

## Exact sequential insertion point

The guard operates only in the independent actor path. For each local PRG in
the existing `0..16` loop:

1. score all three cells from the existing common pre-update state;
2. let the actor produce its unmodified argmax choice for each cell;
3. for an actor-selected blank, evaluate Guard V1 eligibility using the same
   `valid_main` mask and the same pre-update CTBS state;
4. replace at most one blank per cell per TTI;
5. update remaining demand, allocated PRGs, predicted service, and exact CTBS
   state using the final executed choices;
6. continue to the next local PRG without changing the fixed order.

The guard is not a post-Type-0 repair. Placing it before the existing CTBS state
update ensures a guarded selection affects downstream headroom and validity in
the same causal manner as an actor selection.

## Candidate eligibility and CTBS legality

For a cell whose actor choice is blank and whose per-TTI guard budget is unused,
a candidate is eligible only when all conditions hold at that exact sequential
step:

- it is present in the packed candidate row;
- it is true in the current `valid_main` mask;
- it has positive remaining demand;
- its `log1p_drain_pressure` is strictly greater than `log(2)`;
- its `scheduled_dequeue_adequacy` is strictly less than `1`;
- its `ttl_normalized_service_age` is at least `0.25`;
- applying the current local PRG to that UE's current `CtbsHeadroomState` with
  the frozen per-PRG layer cap produces a strictly larger cumulative TBS.

The positive marginal capacity rule is exact:

```text
candidate_state = current_state.with_selected_prg(local_prg, layer_cap)
candidate_state.cumulative_tbs_bytes > current_state.cumulative_tbs_bytes
```

Guard enablement without a CTBS pre-action tuple, a 15-or-more-dimensional UE
feature tensor, or the exact feature names at slots 12--14 is a fatal
configuration error. The implementation must fail closed instead of silently
falling back to estimated service.

## Deterministic choice

Among eligible candidates within the current cell and local PRG, select the
lexicographic maximum of:

1. TTL-normalized service age;
2. `log1p(drain pressure)`;
3. negative UE ID.

Thus greater age wins, then greater pressure, then the numerically smallest UE
ID. Adequacy is an eligibility condition, not a ranking coefficient. Candidate
position and PHY quality are not additional tie-breaks. CTBS positivity is a
legality condition, not a utility score.

## Action and replay semantics

The trace must distinguish:

- `actor_choice`: the unmodified actor argmax class;
- `choice`: the final executed class after the optional guard;
- `guard_override`: a Boolean per PRG token.

The existing `choice` field remains the executed action used by Type-0
materialization and action replay. The actor logits are not modified. Guarded
execution must not be used as a PPO training trajectory or imitation target in
this protocol.

When Guard V1 is disabled, `actor_choice == choice`, every guard diagnostic is
zero, and choices plus Type-0 actions must be bitwise identical to the current
unguarded implementation.

## Required diagnostics

Every guarded evaluation records:

- total overrides;
- per-cell and per-UE override counts;
- actor-choice and executed-choice vectors;
- override local PRG, candidate class, and UE ID;
- pressure, adequacy, and normalized service age at each override;
- CTBS cumulative capacity before and after each override;
- nonblank override count;
- maximum overrides observed per cell per TTI;
- ineligible/no-action classifications:
  `actor_nonblank`, `cell_budget_used`, `no_pressure_candidate`, and
  `no_positive_ctbs_headroom`.

The following invariants are Gate conditions:

- nonblank override count is zero;
- maximum overrides per cell per TTI is no more than one;
- every override satisfies all three feature thresholds;
- every override selected a current `valid_main` candidate;
- every override has a strictly positive CTBS capacity delta;
- executed action replay mismatch, illegal action, and candidate truncation
  counts are zero;
- CTBS MCS, layer, and TBS mismatch counts are zero;
- CTBS mask reactivation remains zero;
- reference-label deletion equality remains true.

## Test-first implementation requirements

Before production code is changed, focused tests must demonstrate these
behaviors and must be observed failing for the intended missing-guard reason:

1. disabled Guard V1 preserves choices, log-probability inputs, and Type-0
   action bitwise;
2. an eligible blank is replaced by the unique urgent CTBS-positive candidate;
3. a nonblank actor choice is never replaced;
4. pressure, adequacy, and age threshold boundaries use the specified strict or
   inclusive comparisons;
5. zero or negative marginal CTBS capacity rejects a candidate;
6. at most one override occurs per cell per TTI;
7. lexicographic ranking uses age, pressure, then smallest UE ID;
8. the guarded selection updates sequential CTBS state before the next PRG;
9. materialized action passes Type-0 preflight and replay is exact;
10. guard enablement fails closed when CTBS inputs or feature contract are
    missing;
11. evaluator output records the frozen configuration and all guard counters.

Existing independent-actor, candidate-width, CTBS, replay, formal-Gate, and
feature-contract tests must remain green.

## Phase A: seed51 minimum falsifiable evaluation

Phase A is the only automatically authorized experiment under this protocol.
It uses:

- fixed iter8 checkpoint;
- topology seed `51` and traffic seed `51`;
- `1000 TTI` warmup;
- `4000 TTI` measured, TTI `1001..5000`;
- packet TTL `400 TTI / 200 ms`;
- argmax decode and CTBS-v1;
- Guard V1 enabled with the single frozen configuration;
- complete offered-packet, packet-outcome, service-balance, and guard traces.

The existing frozen seed51 PFQ and unguarded actor artifacts are the paired
references. A new PFQ run is unnecessary if simulator/checkpoint/feature hashes
remain frozen and the guarded actor's canonical offered-packet hash exactly
matches the existing PFQ hash. Any hash mismatch invalidates the comparison and
requires a fresh paired PFQ run with identical arguments.

## Phase A Gates

### Provenance and mechanism

- all frozen checkpoint, actor-state, simulator, and feature hashes match;
- all guard invariants and structural/CTBS Gates pass;
- the canonical offered-packet stream equals the frozen seed51 PFQ stream;
- guard override count is positive, proving the treatment executed.

### Existing business Gate

- actor goodput is at least `99%` of PFQ;
- actor BLER is no more than PFQ plus `0.002` absolute;
- actor expiry is no more than PFQ plus `0.0005` absolute.

### Target-UE mechanism Gate

Relative to the frozen unguarded seed51 actor, both UE4 and UE6 must satisfy:

- scheduled dequeue bytes strictly increase;
- expiry-aware rate strictly increases;
- expired bytes strictly decrease;
- maximum positive-backlog zero-dequeue streak strictly decreases.

All backlogged UEs must retain positive total scheduled dequeue, a positive
expiry-aware rate, and a maximum zero-dequeue streak below the `400 TTI` TTL.
Outcome-delay mean/p50/p90/p95/max, UE macro/p10/minimum expiry-aware rate,
PRG utilization, blank ratio, scheduled bytes, failed bytes, and per-UE expiry
are recorded regardless of Gate outcome.

Phase A passes only if provenance/mechanism, the existing business Gate, and
the target-UE mechanism Gate all pass.

## Stop and follow-up rules

- Do not tune thresholds or add another guard rule after observing seed51.
- Do not train or adapt the actor.
- Do not replace nonblank choices.
- Do not run seed48 or additional seeds automatically.
- If Phase A passes, stop and request separate authorization for a seed48
  non-degeneration evaluation.
- If Phase A fails, classify the blocker using the guard trace:
  insufficient blank opportunities, no positive CTBS headroom, one-PRG budget
  insufficient, or business-metric cost.
- A Phase A failure rejects blank-only Guard V1. Nonblank preemption, multiple
  overrides per cell, logit bias, or reward/model changes require a new design
  and protocol.

## Output contract

All Phase A artifacts are written under a new root:

`output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/`

The root contains:

- a frozen manifest;
- guarded actor summary and normalized Gate metrics;
- packet outcomes and service-balance trace;
- guard decision trace and guard audit report;
- paired offered-stream comparison;
- combined final report with the exact stop classification.

The previous unguarded seed48/seed51 artifacts remain immutable and are
referenced by path and SHA-256 rather than copied or rewritten.

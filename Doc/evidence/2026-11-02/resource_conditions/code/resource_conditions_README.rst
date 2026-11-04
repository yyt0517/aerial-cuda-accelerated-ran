Resource/layer and activity conditional ablations (2026-11-02)
============================================================

Three diagnostic conditions: per-UE whole-TB count+exact minimum layer;
per-cell/per-PRG activity bits; both together. Teacher UE-to-PRG positions,
expected service values, J and edit paths never enter the solver. Same
deterministic bounded physical decoder also has a no-condition control.
Ordinary incremental212 and full DPLS are external controls. No new model,
CTBS/headroom masks, PPO, native trajectory or final seeds311--315.

Quota decoding uses a square linear assignment: one cap==L witness token
per allocated UE, count-1 cap>=L tokens, then legal blanks. Joint conditions
restrict rows to active PRGs and require resource conservation. A matching
is exact for feasibility and its LINEAR surrogate only, not nonlinear PHY J.
Two teacher assignments with the same compressed conditions are indistinguish-
able to the solver. Impossible conditions fail explicitly; no minimum-layer
relaxation or copied teacher identity positions. Legality can imply unique
positions in special cases; that is inherent in the supplied constraints.

Sixteen fixed global seed variants,4-beam archive, local pair swaps/3-cycles,
3-UE alternatives when quotas free, periodic global redecoding and top8 exact
scores/root. Max212 complete exact scores (initial PFQ included),56 feature
roots/rounds. Feasible incumbent maximizes canonical deficit-weighted J under
own raw-PFQ90% floor. No feasible conditional candidate means unknown value
and explicitly flagged raw-PFQ fallback, not a zero score or success.

The separate diagnostic C++ extension includes the unchanged canonical PHY
source. A causal MMSE gamma cache prepares U*P*2^(C-1) unique solves per
decision inside timing, counted in requests/misses. Root-anchored smooth
reliability proxy uses that bank without candidate native BLER/exact J;
proxy schedules and UE rows are separately counted. Equal212 score caps are
not equal physical/cache/proxy work or equal wall time. New pure physics
bank/decoder benefits cannot be called AI contribution.

Fixed engineering pilot: first/last training state41--43,2 teacher+8 solver
runs, no tuning. Validation44--45:12 historical states; freeze conditional
quality and execution-cost candidates before development46--48:9 historical
PFQ states. No new blind-test claim. Fresh full teacher preparation is timed
and charged separately. Conditional solver time excludes teacher preparation
and cannot be advertised as online AI latency. Three rotating serial repeats,
per-state medians then seed-equal means; full input/PFQ decode through PHY/
bank/matching/proxy/exact scoring/final score/cleanup inside the timer.

Raw floor and canonical BLER remain. Scientific conditional gates: every
J>=99.5% full, no newly unallocated UE vs full, positive-reference UE P5 and
bottom20% expected service>=98%, execution mean<=80% full. A conditional pass
shows headroom for a future causal predictor, not learned/deployable success.
Finite decoder failure is not an infeasibility theorem or a complete bound.

Build once, outside measured decisions, then run with single BLAS/OMP/MKL
threads and new output directory::

  .venv/bin/python -m training.stageb_lns.resource_conditions
  .venv/bin/python -m unittest training.stageb_lns.test_resource_conditions -v
  .venv/bin/python -m training.stageb_lns.resource_condition_campaign --output deliverables/resource_conditions_20261102_v1

Frozen sources/binaries, condition checks, actual work, teachers, all negative
outcomes and independent NumPy final audits are retained. Default algorithm
registry and canonical physics source are never modified by this experiment.

Equivalent cost remeasurement
----------------------------

V1 Python candidate enumeration and row-wise deduplication are retained in
``deliverables/resource_conditions_20261102_v1/source``. After V1 quality and
profiles were read, V2 replaces only ordered enumeration with a stateless
compiled pool and first-occurrence dedup with byte keys. Dynamic utilities
still use identical NumPy arithmetic. Candidate order and complete decisions
are checked, including all21 final actions AND values against V1. Search,
conditions, seeds, budget, beam, proxy and global reseeds stay fixed. This is
engineering remeasurement on exposed states, not new quality confirmation.

Build the additional pool once and use a fresh directory::

  .venv/bin/python -m training.stageb_lns.resource_conditions_fast
  .venv/bin/python -m unittest training.stageb_lns.test_resource_fast -v
  .venv/bin/python -m training.stageb_lns.resource_condition_fast_campaign --output deliverables/resource_conditions_20261102_v2

The original default Python backend remains available. V2 introduces explicit
backend injection arguments, so V1 reproduction uses its frozen source files;
both executed source manifests and both versions' full outputs are retained.

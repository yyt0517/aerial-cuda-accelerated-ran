Whole-TB coordinated candidates and bounded continuation (2026-11-02)
==================================================================

Small mechanism probe, before any new model training. No CTBS, PPO, headroom
mask, registry update, native trajectory or final seeds 311--315. Legacy saved
states are adapted to causal raw PHY; own unpruned PFQ90% dequeue floor stays.

Nine frozen arms: full; frozen trained top4/212; shared-proxy single/joint and
single/joint+continuation at212; single/joint and joint+continuation at424.
The larger cap is a paid control, not an equal-time comparison. Descriptors
are capped at56 roots, sweeps at5. All new methods first execute a common
4-single/PRG sweep. They complete a second sweep before speculative work.
Later joint batches reserve2 single slots and2 macro slots when available.

Macro actions edit2--3 slots: same-PRG joint UE/activity intentions, cross-PRG
move/swap/compensation and optional victim change. Proxy uses exact full-TB
allocation count/min-layer/dequeue plus frozen current reliability and a
cheap interference activity estimate. It does not evaluate candidate BLER or
MMSE. Final canonical PHY checks include all affected UEs/resources.

Continuation expands at most4 causal PRG domains, width2, horizon2. Every
intermediate node is a complete schedule and all uncached exact evaluations
count against the same cap. Below-floor/non-improving intermediate nodes may
expand; only feasible >1B final improvements execute. Truncated/missing
future gains remain unknown, and observed gains are not optimal values.
Complete-action PHY memoization is local to one immutable physical decision.
Logical descriptors are keyed by complete incumbent; no cross-decision cache.
Single/joint controls compute the same pool and proxy families to isolate
which candidate rows consume the exact quota; this is not optimized runtime.

Validation44--45, 12 historical states; freeze quality-best and within-cost
selection before reading development46--48,9 historical states. Three serial
rotating repeats, per-state median then seed-equal means. All states are
previously exposed development, not a new blind test. Gates unchanged:
each J>=99.5% full, no newly unallocated UE, effective-UE P5 and bottom20%
expected service>=98% full, complete mean time<=80% full. These are
instantaneous saved-state service diagnostics, not packet delay/fairness.

Complete timer includes numerical input/PFQ decode, fresh PHY/cache/pools,
proxy/ranking, exact speculative nodes, final scoring and cleanup. Excludes
archive/model/library IO and first immutable TBS setup. Independent NumPy
audits occur after all timed runs. No offline teacher is used online.

Run with single BLAS/OMP/MKL threads::

  .venv/bin/python -m unittest training.stageb_lns.test_tb_macro -v
  .venv/bin/python -m training.stageb_lns.tb_macro_campaign --output deliverables/tb_macro_continuation_20261102_v1

Protocol, sources, checksums, measured work, continuation events, actions,
per-state metrics, validation selection and all negative outcomes are saved.
No test is permitted to alter a previously frozen output directory.

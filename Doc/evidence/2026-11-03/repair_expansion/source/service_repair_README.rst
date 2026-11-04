Causal service sets and finite-budget terminal repair (2026-11-02)
================================================================

Eleven online arms: full, frozen legacy AI, incremental unprotected212,
main180 only, all-PFQ lock control, heuristic/learned subset-lock+repair32,
heuristic/size-matched/learned unlocked repair32 and learned unlocked repair64.
All new paths retain the incremental engine, original single/joint pools,
canonical BLER/deficit J, raw PFQ90% dequeue floor,212 total exact scores and56
feature roots. No CTBS/headroom masks, PPO, native rollouts or final311--315.

40-dimensional causal per-UE features read verified initial PFQ, current
geometry/queue/MCS/weights and cell-relative ranks only. Two sigmoid heads
predict full terminal allocation/effective-service membership. Full terminal
sets are separately paid offline labels, never inputs to online proposal.
Training18 historical states41--43; validation12 states44--45 calibrates
checkpoint and thresholds, then freezes arm selection before development9
PFQ states46--48. Historical exposure is retained, not a new blind test.

One60epoch AdamW round,0/10/30/60 checkpoints, nine threshold pairs. Target
set gate: every state recall>=98% for both sets, mean allocated size<=75%
eligible. Otherwise fixed recall/size/precision loss picks a diagnostic model.
Pure heuristic K calibration uses the same objective. Size-matched control
uses learned per-cell cardinalities but causal heuristic identities, including
model cost; it isolates identity choice, not cardinality learning.

Unlocked repair32 paths share exactly the main180 prefix. Optional subset
locking prevents dropping currently allocated proposed members in the main
stage; it does not lock all PFQ UEs or impose their initial P5. Terminal repair
gets32/64 NEW exact scores inside total212. At most4 fair full-PRG passes,
one candidate per PRG per pass; cache hits remain separately counted. Reset
root-local rejection at phase boundary because acceptance objective changes.
No exact cache crosses physical decisions. Up to56 descriptor roots may stop
repair before physical cap; pool stalls/limits are not optimality certificates.

Repair ranks missing proposed allocations, missing proposed effective service,
then proposed-set P5 and weak20% in1B bins, then J. All actions must satisfy
the raw floor and J>=99.5% of FIXED main terminal J; this prevents compounding
repair losses, but does not establish99.5% of full. Empty/insufficient repair
domains retain incumbent. Final gates remain unchanged against full DPLS.

Two teacher-set conditional paths are run once on validation only, in an
explicit diagnostic API; formal online APIs reject teacher masks. Their times
exclude separately recorded full-teacher preparation and are not deployable
cost claims, nor a complete candidate-space upper bound. They never change
model/arm selection. Terminal model labels and conditional physics are charged.

Three rotating serial formal repeats, per-state medians then seed-equal
means. Complete timers include input/PFQ decode, fresh PHY, proposal features/
network, descriptors/pools, exact scoring, repair and cleanup. Compilation,
archive/model/library IO and first immutable TBS setup are excluded. Labels,
training, conditional diagnostics and independent audits are separate.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1::

  .venv/bin/python -m unittest training.stageb_lns.test_service_repair -v
  .venv/bin/python -m training.stageb_lns.service_repair_campaign --output deliverables/service_repair_20261102_v1

Final quality/service/cost gates and learning contribution remain required;
count recall, compressed sets, saved-state J and instantaneous service alone
do not prove packet goodput, latency, fairness, generalization or deployment.

Structured joint actions and continuation-value pilot
===================================================

This independent J-only experiment preserves the frozen 79-input cached actor,
the active registry, the original PHY, and the own-draft 90% dequeue floor.
It implements three separable changes, with an additional combined control:

* ``structured_repair.py`` enumerates editable activity patterns, exhaustively
  solves each cell's small assignment subproblem, retains (J,dequeue) Pareto
  frontiers, and merges them against the global floor. Complete TBs, including
  PRGs outside the block, are scored. This decomposition depends on the current
  activity-only interference model; it is not valid for arbitrary PHY models.
* ``continuation_repair.py`` compares likelihood, exact current-completion J,
  and finite 1--2-step continuation ranking. Every online PHY query uses the
  ordinary solve callback and counts against the total candidate cap.
* An additional 95-input, 32/16-hidden-unit value head uses the original 79
  features plus sixteen causal suffix/TB/floor descriptors. The original actor
  weights remain frozen. The reachable-dequeue descriptor means assigning all
  eligible suffix positions to that UE, not an upper bound on every completion.
  The head learns the best feasible J *observed* by the declared finite beam,
  relative to the verified root. It does not predict a globally optimal suffix,
  a process advantage, actual packet delay, or a multi-TTI return.

The recorder observes the first retained student parent at every depth in the
first four visited blocks. The student decision finishes before the teacher
queries an independent PHY context. Missing feasible continuation values and
truncated queries are masked, never treated as zero. A second correction round
collects prefixes visited by the selected first-round value policy; both rounds
are aggregated for the second head. This is prefix-level dataset aggregation,
not a native multi-TTI DAgger campaign. Original states remain historical.

The initial pilot trains on seeds 41--43 and selects on 44--45, using current
and PFQ source snapshots at TTIs 4001/8001/18000. Seeds 46--48 are read only
after model selection, at the historical 27-state evaluation used previously.
All these seeds have prior project exposure. Seeds 311--315 remain unread.
There is no claimed scene generalization or native packet improvement.

Each learner gets a hard 4096 actual scored-schedule cap, including a shared
causal-floor preparation evaluation. Physical reference greedy/4/16 controls
have a one-million cap and are explicitly stronger-work references. The table
reports actual candidates, affected UEs, MMSE work, and complete serial timing;
equal candidate caps are not equal FLOPs or equal wall time. The combined arm
runs learned proposals followed by at most four exact block repairs, sharing
the same global cap. Pure structure attempts at most eight blocks. All arms
retain the same acceptance thresholds and verified incumbent protections.

The offline scientific gate is predeclared: every historical evaluation state
has J >= 99.5% of the full physical4 reference, and seed-equal mean decision
cost is no higher than cached79. Engineering checks do not imply this gate.
Existing payload/TTL/mean/P95/UE-P5 and degraded-UE native gates still apply
before any deployment change. No final test or native campaign starts here.

Run from the repository root into a new directory::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_structured_continuation training.stageb_lns.test_cached_repair training.stageb_lns.test_physics training.stageb_lns.test_repair -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.structured_campaign --output deliverables/my_structured_continuation_v1 --epochs 20 --repeats 3

The script freezes protocol and source hashes before collection. Existing
artifacts are never overwritten. Python 3.10+, NumPy, Torch, pandas and the
existing compiled ``_physics.so`` are required. GPU/native simulation is not.
Serial timing includes snapshot conversion, fresh PHY/graph/cache construction,
the shared floor probe, feature/network work, proposals, verification, and
cleanup. Model loading, archive IO and immutable TBS table setup are excluded.
Teacher labeling and independent audits are outside the inference measurement
and separately counted. This is not production scheduling latency.

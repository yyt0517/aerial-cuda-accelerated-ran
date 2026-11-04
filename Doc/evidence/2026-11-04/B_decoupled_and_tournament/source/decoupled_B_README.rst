Decoupled method B and separately gated overhead engineering
===========================================================

On November4 the user explicitly removed the failed40% deployment wall gate
as a prerequisite for B's scientific comparison. The old failed result and
the40% deployment target remain unchanged. Only B's entry condition changes:
the exact solver must still pass its correctness gate. The frozen grid,
seeds41--43, causal weights, FIFO, initialization, metric windows and quality
rules remain unchanged. All15 scientific arms use the same original
``_bounded.so``; the engineering library below is never substituted mid-run.

Run the frozen13 new trajectories, reusing the two already completed exact
k0 controls at seeds41/42. Every new trajectory has4000 PFQ warmup +4000
settling +4000 measured +400 followup TTIs. This is161200 newly paid native
steps; reused24800 steps are not charged again. Source/runtime hashes and
the common warmup/traffic/cohort/FIFO/native-PHY checks are mandatory::

    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.ttl_decoupled_campaign --output deliverables/B_ttl_NEW --workers 4 --cpu 252
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_ttl_frontier --source deliverables/B_ttl_NEW --output deliverables/B_frontier_NEW

The analyzer pays for52 independent full-integer weighted oracle replays.
All five coefficients0/.25/.5/1/2 and all seeds are retained. Mean seed payload,
pooled mature-packet expiry/successful P95 and maximum(seed,UE) expiry define
the four-objective frontier. Historical quality and the stronger B no-worse
rule are both reported. A coefficient scan is not a fitted state-dependent
policy and does not establish generalization; no final confirmation seeds or
ranking labels are used. Parallel trajectory wall times are not speed data.

Zero-PHY activity-mask audit
----------------------------

``gamma`` strips the victim's serving-cell bit. At C=3, each(UE,PRG) needs
at most4 distinct cache keys, or2448 per36UE/17PRG TTI. Existing aggregate
miss counters bound tuple coverage, not individual per-pair key frequencies.
The original kernel already uses fixed4x4 Eigen matrices. No matrix batching
or Woodbury numerical change is made in this experiment::

    .venv/bin/python -m training.stageb_lns.analyze_activity_mmse --output deliverables/activity_coverage_NEW

Separate exact engineering library
---------------------------------

``_tournament.so`` includes the original physics arithmetic. Unknown bounds
use persistent local per-PRG tournaments and a global tree with one leaf per
PRG. Changed-PRG generation and shared-UE dependencies update dirty leaves.
Since the dequeue floor is global, unchanged-support entries crossing its
feasibility cut must also update. Cached exact maximum seeding still scans
cheap metadata each round; the implementation does not claim to eliminate
all full metadata traversals.

The optional direct gamma array copies results computed by the original
``Context::gamma``; every miss executes the same Eigen expression, and repeat
requests only skip associative lookup. C>8 falls back to the original cache.
Native input preparation moves MCS reconstruction, TBS selection, legal
domains and PFQ slot mapping into C++; NumPy normalization and covariance/
signal construction are retained to preserve arithmetic bits. Context itself
was already native. The arm breakdown measures actual savings instead of
assuming the entire old Python-adapter cost disappears.

Use a fresh directory; audit may run separately from B, but fair timing is
serial after our native campaign completes. Original12 plus240 load states
receive byte-pattern checks at every root, UE field, bound, cached delta,
status, selected key and final value against the frozen heap. The previous
independent full-oracle gates supply transitive equivalence; synthetic tests
also compare directly with the independent full oracle. All counters match::

    .venv/bin/python -m training.stageb_lns.build_tournament
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_bounded_dpls training.stageb_lns.test_certified_bounds training.stageb_lns.test_tournament
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.audit_tournament --output deliverables/tournament_NEW --phase audit --cpu 251
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.audit_tournament --output deliverables/tournament_NEW --phase timing --cpu 255 --repeats 3

The five timing arms are native full, original heap, tournament, tournament
with direct gamma, and native input+tournament+direct gamma. Every timed
decision includes fresh adapter/PFQ/context/search/close; observers, audit,
history and profiling are off. Warm immutable tables/imports only, rotate
serial order, retain all interruptions. Saved-state P95/max do not replace
the original continuous measured-TTI tails. Separate profiles/work ledgers
charge additional calls; no new default or closed-loop engineering claim.

Step 0 and the causal anytime DPLS physical control
=================================================

This implements the first measurement step of the November 3 scheduling
proposal. It freezes service-PPO training and leaves the active algorithm
registry alone. The initial protocol is **no CTBS, causal deficit weights,
canonical native BLER**, matching the recent budget-search research. It is not
the equal-weight PPO protocol or the legacy-BLER default's old timing result.

What already existed
--------------------

``physics_kernel.cpp`` already recomputes only affected UEs and caches MMSE by
UE, PRG and interfering-cell activity. A change in TB minimum layer or resource
count recomputes that UE's complete TB, including other PRGs. MCS is fixed during
one decision, reconstructed from fresh OLLA/link inputs each TTI. Caches and PHY
values do not cross TTI boundaries. Consequently incremental scoring is a
shared existing optimization, not a new speedup introduced by this experiment.

Controls and budgets
--------------------

``cold_full`` is the unchanged full-neighborhood control. ``cold_n1``,
``cold_n2``, ``cold_n3`` and ``cold_n5`` stop after those numbers of scans.
``warm_full`` and ``warm_nN`` first repair the **same arm's previous executed
action** against current queue, association and PRG availability. They evaluate
it with current PHY and choose it only if its J is higher than current raw PFQ
and it satisfies the current raw-PFQ ceil(90%) dequeue floor. Bootstrap uses the
previous PFQ warmup action. No current-TTI full-teacher decision is queried.

``cold_b106`` / ``warm_b106`` (likewise 160/212 or another positive integer)
cap all complete candidate evaluations, including PFQ and warm initialization.
A partial final scan uses original candidate order, accepts only a verified
improvement and never reports convergence. With a one-evaluation budget warm
initialization is skipped. These controls supply the previously missing
equal-PHY-work comparison. Counts of affected UEs and MMSE requests/misses are
also reported: complete-candidate count is not an equal-wall-time guarantee.

The original neighborhood includes a same-PRG source mute plus a neighboring
cell's UE change. Cross-PRG joint macros and a lazy gain heap are not added in
this stage. Previously scored gains cannot simply be reused: TB layer/count
coupling and activity changes can invalidate them outside the changed PRG.

Tracing and timing
------------------

Each measured decision can save all candidate exact J and delta J, floor
eligibility, accepted index, root and compressed one-PRG assignments in an NPZ.
Offsets identify each iteration. The final action/PHY values are included;
unscored candidates have NaN delta, never a fabricated label. All copying is
charged to ``observer_ms`` inside the decision; compression and writes are
reported separately in ``trace_write_ms``. Per-round JSON includes the maximum
eligible delta, number of positive candidates and cumulative physics counters.

Overlap is between consecutive executed actions, not sparse archived snapshots:
exact slot agreement; nonblank assignment agreement divided by the union of
active slots; activity-only agreement. The second metric prevents idle slots
from masking changed UE identities. Results are descriptive, correlated TTI
measurements, not independent wireless repetitions.

``e2e_ms`` covers pre-action metadata extraction, fresh MCS/TBS/current PFQ,
fresh PHY context, generation, scoring, reductions and destruction. Kernel,
legality/input validation, output-copy and observer times are measured.
``other_ms`` includes NumPy, allocation, Python and wrapper overhead; it is
**not an exclusive Python CPU profile**. Immutable table/library warmup is
excluded. Native logging runs on a shared host; use alternating, trace-free
replay for cost diagnostics. Replay uses the recorded source arm's predecessor
and explicitly records that provenance. It does not prove another warm arm's
closed-loop quality. Native arms always use their own executed predecessor.

Run inside the existing CUDA container from ``/opt/nvidia/cuBB``::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_anytime_dpls training.stageb_lns.test_ranked_dpls training.stageb_lns.test_physics training.stageb_lns.test_bounded_probes training.stageb_lns.test_edit_ranker training.stageb_lns.test_no_ctbs_inputs -q

    # Full step-0 capture: each seed has 2 s PFQ + 2 s settling + 2 s measurement.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.anytime_dpls_campaign --output deliverables/anytime_dpls_capture_NEW --seeds 41,42,43 --arms cold_full

    # Development physical frontier: same closed-loop protocol in every arm.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.anytime_dpls_campaign --output deliverables/anytime_dpls_frontier_NEW --seeds 44,45 --arms cold_full,cold_n1,cold_n2,cold_n3,cold_n5,warm_n1,warm_n2,warm_n3,warm_n5,warm_full,cold_b106,warm_b106,cold_b160,warm_b160,cold_b212,warm_b212 --trace-every 0

    # Source-arm predecessor replay; not another arm's own trajectory.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.anytime_dpls_campaign --output deliverables/anytime_dpls_replay_NEW --replay deliverables/anytime_dpls_capture_NEW --arms cold_full,cold_n1,cold_n2,cold_n3,cold_n5,warm_n1,warm_n2,warm_n3,warm_n5,warm_full,cold_b106,warm_b106,cold_b160,warm_b160,cold_b212,warm_b212 --repeats 3

All output directories must be new. Protocol, source copies, hashes, raw traces,
captured causal inputs, decisions and native checks are retained. Short runs
with fewer than 400 followup TTIs deliberately leave packet metrics unevaluated.

Gates before learning
---------------------

The native smoke only verifies execution and logging. A requires paired native
development quality gates and mean PHY evaluations <=40% of frozen same-protocol
full DPLS before becoming a new physical control. Final combined targets are
mean/P95 PHY counts <=25%/40% and mean wall time <=40%. Quality requirements are
paired payload mean >=99.7% (every seed >=99.4%); expiry <=full+0.10 percentage
points; mean/P95 successful delay <=full+0.5/+1 ms; weak-UE payload P5 >=97%;
worst-UE expiry <=max(full worst+2 percentage points,5%); no new persistently
unserved UEs. These gates are not automatically certified by this capture CLI.

Next, E must beat the best fixed-round/fixed-delta heuristic by >=20% rounds at
>=97% gain retention, including the worst 5% of states. F must explicitly use
the own executed t-1 solution's conditions, compared with teacher/no-condition
arms; teacher agreement is a separately charged diagnostic. B starts with five
scalar urgency values, with training seeds41--43 only. C/D wait for an A/E
quality or cost gap. No further PPO continuation, repair-guard confidence sweep,
or model expansion is part of this stage. Estimated 3--8x cost reductions remain
hypotheses until native paired quality and trace-free timing establish them.

Persistent integer DPLS: exact cache and guarded cap4
===================================================

This is a separate physical research entrypoint. It does not change the active
algorithm registry, train a model, or consume current teacher terminal actions.
``solve_persistent(problem, raw_pfq, PersistentConfig('cached'))`` returns an
action, per-UE PHY values and a work record. Callers must check ``converged``.
Modes are ``full`` (refresh every edit), ``cached``, ``cap4``, ``guarded4``.
Full and cache share the same persistent sparse edit-table architecture.

An edit stores PRG, canonical overwrite key, affected-UE dependency mask,
sparse exact PHY payload, integer delta-J and delta-dequeue. Only changed PRGs
are regenerated. Other entries are invalidated by the affected-UE union.
Unchanged rows are never copied into a complete per-candidate schedule or
reduced across all UEs online. Metadata selection still scans all candidates.
The audit observer deliberately constructs dense values; keep it off for timing.
The cache is per decision, never reused across different TTIs/PHY contexts.

The new objective is the int64 sum of per-UE nearest-even contributions
``nearbyint((expected_bytes * causal_weight) * 1e6)``. Strict thresholds are
1e6 during rounds 1/2, 16e6 later. The true maximum anchors a one-unit tie band;
the smallest canonical edit key wins. The common floor is ceil90% of raw PFQ
dequeue. This defines a new numerical baseline; finite agreement with the old
floating terminal is not a universal floating-objective equivalence claim.
Integer queue/objective overflow, mask and key bounds are checked. This version
requires UE+PRG <=63 and int64-safe keys and is not a general large-topology
implementation. The kernel rejects a non-nearest-even rounding environment.

``_persistent.so`` has its own Context allocation and reuses unchanged
``physics_kernel.cpp``. It never passes a Context across the two libraries.
The original ``_physics.so`` is not rebuilt. Linux g++, Eigen3 and the existing
Python virtual environment are required::

    .venv/bin/python -m training.stageb_lns.build_persistent
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_persistent_dpls training.stageb_lns.test_physics training.stageb_lns.test_cached_dpls > /tmp/persistent_reconstruction_20261103_tests.txt 2>&1

The guarded batch preserves the best single move, packs at most four edits
with disjoint PRGs/affected UEs, and checks the common floor after every edit.
It falls back to that best single if the batch newly deallocates a UE or reduces
any causal-deficit-top20% positive-queue UE's expected service by >1e-6 B,
relative to the single counterfactual. Guard inputs are already scored sparse
payloads: no additional PHY. This is a one-decision guard, not a long-term TTL
or fairness guarantee.

Reproduce the paid calibration and integer gate
-----------------------------------------------

Use fresh output directories. The paths below refer to retained local inputs.
Cold/warm and randomized full solves require fresh PHY work; this is not a
zero-cost log calculation. The source shuffle is a canonical-enumeration
negative control. The six ensemble members have only 1--3 distinct service
patterns per state; report the deduplicated diagnostics, not raw72/72 LOO::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.service_gate_calibration --source deliverables/anytime_dpls_20261103_smoke_v3 --reference deliverables/canonical_cached_dpls_gate_20261103_v1 --output deliverables/full_service_calibration_NEW
    .venv/bin/python -m training.stageb_lns.calibrated_envelope --source deliverables/full_service_calibration_NEW --output deliverables/full_service_envelope_NEW
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.persistent_campaign --source deliverables/anytime_dpls_20261103_smoke_v3 --reference deliverables/canonical_cached_dpls_gate_20261103_v1 --envelope deliverables/full_service_envelope_NEW --output deliverables/persistent_integer_gate_NEW

The v2 envelope is a descriptive componentwise full min/max range with fixed
quality tolerances, frozen before evaluating the new guarded candidate. It
does not estimate a probability distribution. J and causal weak-service
definitions, deduplication and noninformative P5 are recorded in its protocol.
The old single-reference gate remains reported beside it. All12 envelope rows
must pass before the native development entry is enabled.

The independent integer reference materializes full plans and independently
rounds/sums all UE contributions. Every full/cache round checks all candidate
PHY fields, integer deltas, floor, root and selected keys against that reference.
Batch roots and terminal values have separate native/NumPy audits with costs.

Fair wall time and native development
------------------------------------

The benchmark is serial, CPU-pinned and rotates arm order. Logging, history,
observers, profiling and auditing are off inside the timed decisions. All timed
outputs are checked afterward. Report both old Python full and same-architecture
native full controls; compilation/representation savings are not cache-only
or learned-policy savings::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.benchmark_persistent --source deliverables/anytime_dpls_20261103_smoke_v3 --gate deliverables/persistent_integer_gate_NEW --output deliverables/persistent_wall_NEW --cpu 255 --repeats 5

The native runner requires container ``c_aerial_oai2``, repository mounted at
``/opt/nvidia/cuBB``, the frozen simulator/worktree and existing PFQ reference
trajectories used by ``stageb_ppo.native``. The normal mode is two exposed
development seeds44/45, two policies, 4000 common-PFQ warmup +4000 settling
+4000 measured +400 followup TTIs each. It is a conditional development test,
not final blind confirmation. ``--smoke`` is only16 TTIs per policy::

    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.persistent_native --gate deliverables/persistent_integer_gate_NEW --output deliverables/persistent_native_smoke_NEW --workers 2 --smoke
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.persistent_native --gate deliverables/persistent_integer_gate_NEW --output deliverables/persistent_native_NEW --workers 4
    .venv/bin/python -m training.stageb_lns.summarize_persistent_native --source deliverables/persistent_native_NEW --output deliverables/persistent_native_analysis_NEW

Native trajectories may run concurrently; their elapsed times are not used
for fair scheduler speed comparisons. Native actions, PHY closure, queue/FIFO,
offered inputs, common PFQ prefixes and mature arrival cohorts are validated.
Long-term quality and PHY cost gates are reported separately.

Recorded November 3 outcome
--------------------------

Twelve correlated seed41 inputs passed the new integer gate: 378 rounds and
624670 candidates for each of full/cache. Serial native full/cache wall means
were 23.48/10.95 ms (cache 46.62% of native full); old floating Python full at
181.72 ms is a separate engineering reference. Guarded cap4 was 11.02 ms and
passed 12/12 empirical envelopes. Its native seed44 UE24 expiry increased
12.0554->14.8253%,
failing the frozen +2 percentage point guard. Seed45 passed. No promotion or
registry change; M and learning remain paused and 161700 old labels unchanged.

The report/archive generator only reads the completed fixed campaign paths,
validates hashes, creates CSV/figures and archives evidence; it runs no PHY.
The archive includes its exact source, inputs, protocols, cost ledgers, packet
cohorts and hashes. Recreating it requires the retained deliverable paths::

    .venv/bin/python -m training.stageb_lns.report_persistent_reconstruction --output Doc/evidence/NEW/persistent_cache_reconstruction --tests /tmp/persistent_reconstruction_20261103_tests.txt

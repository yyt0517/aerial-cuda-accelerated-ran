Canonical DPLS with exact cross-round dependency caching
=======================================================

``cached_dpls.solve_canonical`` is a separate versioned research entrypoint.
``CanonicalDplsConfig('full')`` and ``CanonicalDplsConfig('cached')`` share a
canonical tie rule. Historical ``solve_ranked`` and the active registry remain
unchanged. This is cross-round caching, distinct from the existing C++
affected-UE evaluation inside one candidate. No cache survives a TTI.

The default epsilon is 1e-6 bytes. Among currently floor-feasible edits whose
gain strictly exceeds the original 1/16-byte acceptance threshold, find the
true maximum, then choose the smallest canonical operation key within epsilon
of that maximum. Keys order PRG followed by cell overwrite targets (unchanged
-2, blank -1, then UE). Max anchoring avoids a nontransitive pairwise epsilon
comparator. The acceptance threshold is not softened by epsilon.

After one move, candidates on its PRG or sharing an affected UE are dirty.
Activity changes include all active neighboring UEs on that PRG. New operation
keys are also dirty. The cache retains exact affected-UE PHY values. It copies
current-root values for other UEs and recomputes complete candidate J and D in
the same array layout as full mode. Old global scores, rounded deltas and old
eligibility flags are never reused.

Tests include every candidate's exact PHY equality, warm starts, empty
neighborhoods, common-floor packing and permutation-invariant canonical ties::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_cached_dpls training.stageb_lns.test_dpls_joint_cost training.stageb_lns.test_dpls_move_analysis training.stageb_lns.test_ranked_dpls training.stageb_lns.test_physics -v

First two diagnostics: zero new PHY
----------------------------------

The union analysis applies each logged-root pack, regenerates the logical
post-pack neighborhood, and counts new keys plus the dependency union. Packs
use the original acceptance threshold and preserve the original first move.
Only the twelve saved-input states enforce their exact shared dequeue floor;
other packs are explicitly structural projections.

The local model is ``N_first + sum(F_union/k)``, with the first complete scan
retained. It recovers single-move work when k=1. It is not a batch rollout,
performance bound or speed measurement; cache and batch savings are not
multiplied. Within-round and across-TTI historical exact-delta rankings are
reported separately. Warm_n3's last scored neighborhood predates its final
accepted move and is not labelled as executed-terminal history::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_dpls_joint_cost --source deliverables/anytime_dpls_20261103_smoke_v3 --output deliverables/dpls_joint_cache_NEW

New baseline and engineering equivalence: paid PHY
-------------------------------------------------

The canonical baseline must be freshly scored. This gate executes one new full
and one cached converged decision per saved input, using the same source-arm
predecessor and current PHY initialization. It compares all candidate fields,
scores, floor checks, roots and selections bit for bit every round. A separate
NumPy terminal audit is charged. Raw times include storing/checking full PHY
arrays and cannot establish a fair wall-time speedup::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.cached_dpls_gate --source deliverables/anytime_dpls_20261103_smoke_v3 --output deliverables/canonical_cached_dpls_NEW

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.report_dpls_joint_cache --offline deliverables/dpls_joint_cache_NEW --gate deliverables/canonical_cached_dpls_NEW --offline-evidence Doc/evidence/NEW/dpls_joint_cache_diagnostics --gate-evidence Doc/evidence/NEW/canonical_cached_dpls

Every output directory must be new. The November 3 gate passed on twelve
correlated saved inputs, 378 rounds and 624670 complete candidate PHY audits.
The full/cached decisions paid 624690/183079 complete candidate evaluations;
both incurred 22203 native MMSE misses. Fewer candidate evaluations are not
the same as proportionally fewer MMSE inversions or faster wall time.

Next implement shared-floor multiple-move acceptance as a separate trajectory
and compare it against this frozen canonical control. Preserve the exact cache
gate before changing trajectory semantics. Old-delta budgets remain offline
recall controls; no quality, long-window or new-seed gate has passed yet.

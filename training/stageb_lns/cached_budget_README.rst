Measured exact cache, shared-floor cap4 and dirty M16/32/64
=========================================================

This stage measures CPU wall time and actually executes the cap4 and dirty
budget trajectories on the same twelve captured inputs. It uses no CTBS,
causal deficit weights, canonical native BLER and canonical floating ties.
It runs no new native packet simulation and changes no active registry.

``CanonicalDplsConfig(diagnostics=False)`` disables per-round timers, history
and PHY profiling without changing decisions or work. Primary timing includes
fresh MCS/TBS/PFQ reconstruction, PHY setup, search and close. Inputs, immutable
tables and libraries load outside timing; independent checks run afterwards.
Each arm is repeated five times in one serial process, pinned to CPU255 with
one BLAS thread, in alternating/rotating order. Profile shares come from
separate diagnostic decisions, never the primary timing interval.

The cap4 path retains the canonical best edit, adds disjoint PRG/affected-UE
edits while guarding the running aggregate dequeue floor, then pays for one
joint PHY check when more than one move is accepted. Its next cache update
invalidates the union of all accepted dependencies.

The dirty path always scores the first full neighborhood and all novel edits.
Current-valid clean values are exact. It scores at most M dirty-known edits
ranked by their last exact delta; entries not scored remain invalid across
rounds. No online full-teacher output or current eligibility oracle is used.
When the shortlist finds no qualifying improvement, it scores the remaining
dirty complement before claiming the original local stopping condition.

Regret queries happen after each entire decision, in a separate PHY, and only
fill missing dirty labels plus their audit root. They cannot change the policy
or its cache/counters. All audit work is separately charged. ``gain_retention``
is selected delta / true maximum; absolute regret is also retained. For cap4,
this statistic refers to its first move, not the summed batch gain.

Observer arrays are current only where ``valid`` is true; invalid gains are
NaN, and their old labels may be several rounds old. The exact-cache control
also saves eight-feature dirty labels using work it already performs. Labels
on a budget policy's unscored, policy-induced states require paid queries.

Run from the repository root, always using fresh output directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_budgeted_cached_dpls training.stageb_lns.test_cached_dpls training.stageb_lns.test_dpls_joint_cost training.stageb_lns.test_dpls_move_analysis training.stageb_lns.test_ranked_dpls training.stageb_lns.test_physics -v

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.benchmark_cached_dpls --source deliverables/anytime_dpls_20261103_smoke_v3 --reference deliverables/canonical_cached_dpls_gate_20261103_v1 --output deliverables/cached_wall_NEW

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.cached_budget_campaign --source deliverables/anytime_dpls_20261103_smoke_v3 --reference deliverables/canonical_cached_dpls_gate_20261103_v1 --output deliverables/cached_budget_quality_NEW

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.benchmark_cached_frontier --source deliverables/anytime_dpls_20261103_smoke_v3 --reference deliverables/canonical_cached_dpls_gate_20261103_v1 --quality deliverables/cached_budget_quality_NEW --output deliverables/cached_frontier_wall_NEW

The report generator ``report_cached_budget`` accepts ``--initial-wall``,
``--quality``, ``--wall``, ``--previous-wall``, ``--reference``, ``--output``
and optionally ``--tests``. The preserved November 3 previous-wall run records
the unnecessary complete epsilon-band ranking. The current limited ranking
returns exactly the same first M entries and all timed states match the frozen
quality actions, PHY, round counts and candidate counts.

The report and curve are under
``Doc/evidence/2026-11-03/cached_cost_budget/``. Exact cached wall is 67.31% of
full; cap4 is 33.82% but passes only4/12 snapshot J/service gates. Dirty M16/32/64
pass1/4/1 and retain only71.55/73.47/75.38% of the best per-round gain on average.
Thus no advanced arm is promoted to native closed-loop evaluation.

The 161700 labels are from twelve correlated captured input rows, one exposed
seed, and shared warmup histories. They are not independent scenes. The
archived schema explicitly corrects the v1 generator's input-count field name.
No model is trained here. Integer contribution sums were checked only as an
offline sensitivity calculation: no choices changed on378 exact-cache roots,
but this is not an implemented integer incremental scorer or a new target's
equivalence gate. Such a version still needs its own rounding/overflow contract
and newly scored baseline before any cost or equivalence claim.

AI acceleration of DPLS without CTBS
===================================

This independent first experiment removes CTBS from data histories, input
construction, teacher labels, all controls, and inference. Raw PFQ allocation
is decoded without pruning resources whose capacity exceeds queue demand.
The shared floor is ceil(0.90 * this state's raw PFQ physical dequeue).
Original causal deficit weights are retained. Only physical association,
positive queue, and PRG availability restrict candidates; UE/cell headroom or
remaining-demand masks are not used. Full histories are CTBS-free deficit
search and PFQ; evaluation histories are exclusively PFQ-only.

Old archives store pre-action link-adaptation diagnostics in a field named
``ctbs``. Archive unpacking copies WB SINR, CPU OLLA, MCS configuration, layer
caps and association into ``raw_phy`` and removes that field before decisions.
No CTBS algorithm or reconstructed-MCS method runs. Numeric link adaptation
is ordinary native MCS reconstruction, included inside each decision timer.
Actual chosen actions, future arrivals and post-action outcomes are discarded.

Three controls share the exact original one-PRG edit neighborhood and the same
canonical native-BLER physical kernel:

* ``full`` evaluates the complete neighborhood before accepting one edit.
* ``heuristic`` ranks cheap whole-TB resource/interference proxies, scores the
  first K logically feasible candidates, and scans the unscored remainder if
  no qualifying improvement exists.
* ``ai`` uses a trained 48-input 32/16-hidden-unit residual MLP over that same
  heuristic, with identical K, feature cache, precise verification and fallback.

Every iteration has a fixed incumbent and accepts at most one edit. Ties use
the original candidate index, independent of scoring order. Thresholds are
>1 byte for iterations one/two and >16 thereafter; at least three iterations
are checked. Already evaluated shortlist candidates are not rescored during
fallback. The stopping iteration always covers the full original neighborhood.
This proves the declared local stopping condition, not the same optimum or
packet behavior as full DPLS. The 256-iteration cap is a quality failure.

Heuristic/AI share the compiled ``edit_features.cpp`` implementation. Its 48
features and exact logical dequeue screen are independently tested against the
NumPy reference and physical kernel. Static geometry is initialized once per
decision; changed incumbent state is refreshed. No candidate MMSE, BLER lookup,
or physical scoring is hidden in feature construction. Shared-library loading
and compilation happen outside timing; context creation/destruction are inside.

Training and selection use seeds 41--43 / 44--45, respectively, from CTBS-free
deficit-search and PFQ-only sources. Each has six saved TTIs, with correlated
adjacent samples and common warmup states explicitly retained as correlated
observations. Full-teacher exact gains train the first model. A second round
labels the selected student's actual visited roots in an independent PHY;
these labels never change the executing student's decisions or physics counts.
The model learns a residual ranking over the physical prior. Epoch zero is an
explicit zero-residual heuristic fallback, and cannot count as learned benefit.

Twenty epochs per round use sampled lists containing exact gain leaders,
heuristic leaders and uniform candidates, with listwise gain-weighted ranking
plus Huber calibration. Epochs 0/5/10/20 and K=64/128 are selected only on
validation: all states J>=99.5% of full and convergence first, mean shortfall
second, complete seed-equal decision time third. This fixed protocol is written
before collecting outcomes. All full labels and work counts remain archived.

After selection, historical PFQ-only seeds46--48 at TTIs 2001/2501/3001/3501/
4001/4501/5000 provide 21 development states. These are not new blind tests.
Three serial alternating timing repeats are reduced to per-state medians,
then seed-equal summaries. Timers include raw input/MCS/PFQ reconstruction,
fresh physical and feature contexts, full candidate construction, feature and
network work, sorting, all exact evaluations, fallback, final scan and cleanup.
Archive unpacking, weights/library loading and first immutable TBS table setup
are excluded. Independent NumPy audits run after all timed decisions.

The offline gates are every state's J>=99.5% of full, convergence with full final
scan, and mean complete cost<=80% of full. P95 is also reported. Independent AI
contribution additionally requires nonzero learned weights and either less time
than a quality-passing same-K heuristic or passing quality where that heuristic
fails. Exact physical counts alone do not establish speedup. Raw canonical
full-search reference may differ from the older frozen legacy BLER scorer.

Run in a fresh output directory, with one CPU/BLAS thread::

    .venv/bin/python -m training.stageb_lns.build_edit_ranker
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_no_ctbs_inputs training.stageb_lns.test_edit_ranker training.stageb_lns.test_ranked_dpls training.stageb_lns.test_physics -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.ranked_dpls_campaign --output deliverables/my_no_ctbs_ranked_dpls_v1 --epochs 20 --repeats 3
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.report_ranked_dpls --source deliverables/my_no_ctbs_ranked_dpls_v1 --destination Doc/evidence/my_no_ctbs_ranked_dpls_v1

Original registry and historical artifacts are not overwritten by this pilot.
No final seeds311--315 or new native trajectories are read/run. Offline success
would still require full-packet, TTL, average/P95 delay and weak-UE native checks
before changing the effective scheduler. Saved-state CPU time is not deployed
scheduling latency or an air-interface packet-delay measurement.

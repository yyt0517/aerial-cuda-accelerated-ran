Single trajectory: exact top2/top4 and root-local failure deduplication
====================================================================

The campaign first compares a fixed scalar ranker, the same physical heuristic,
top2/top4, and repeat-shortlist ablations. Every new arm shares a decision-local
cache of candidate enumeration, features and ranking. The cache is rebuilt
after an accepted edit; exact candidates failing the >1B improvement threshold
are skipped only at that same root when deduplication is enabled. No state,
features or negative result survives a new CSI/decision. Repeat controls retain
the same feature cache but deliberately rescore the failed shortlist, isolating
the engineering mechanism from learning. The previous uncached top1 trajectory
is kept as a separately named historical-architecture control.

All methods retain raw PFQ, causal deficit weights and its 90% logical dequeue
floor, without CTBS/headroom pruning, spatial feature bank, PPO or full search
inside the trajectory. New arms share max56 feature roots/max106 actual PHY
scores. Actual work, complete time and scoring-buffer costs are measured.
Sparse precise scores use the original neighborhood shape/row positions to
preserve numerical tie handling. Full DPLS remains the canonical independent
quality control with its original1/16B stopping thresholds; the scalar paths
retain their declared >1B acceptance rule.

Training/selection/development seeds are41--43 /44--45 /46--48, with the same
previously exposed fixed histories as the preceding campaigns. Phase A uses
only the12 validation states. K is selected by complete quality, positive-UE
service coverage and cost, then fixed before one bounded20-epoch own-state
training round. This diagnostic round is run after engineering checks even if
the final quality gate is not yet passed. Epoch0 can preserve the old weights.

After student decisions finish, up to8 actual visited states/decision are
sampled. A state comprises the original saved state, current action root and
failed-before mask. These correlated rejection states are not new physical
scenarios. An independent teacher scores complete original neighborhoods;
training availability excludes indices already rejected by the student.
The prior53 anchor groups are retained. Normalization and initial scalar model
are frozen; fine tuning uses the already documented fresh AdamW protocol.
No validation/development data trains the model. Final weights are frozen
before reading the nine development states in this campaign; their historical
exposure remains explicit. A second K is a transfer control if not selected.

The quality gate is every state J>=99.5% full. Coverage now explicitly uses
no newly unallocated UE relative to full (COUNT>0). P5 and bottom20% service
use reference expected-service>1B UEs and must each retain at least98%. This
avoids the earlier all-UE zero-P5
degeneracy. Cost is seed-equal median complete time<=80% full. These saved-state
criteria do not establish packet throughput, TTL, delay, generalization or
production deadlines. Final seeds311--315 and active registry remain untouched.

Run in fresh directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_shortlist_trajectory training.stageb_lns.test_own_trajectory training.stageb_lns.test_bounded_probes training.stageb_lns.test_no_ctbs_inputs training.stageb_lns.test_ranked_dpls -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.shortlist_campaign --output deliverables/new_shortlist_campaign

Full models, labels, physical arrays and original source snapshots remain in
deliverables. Small evidence includes frozen protocols, selection, actual-state
provenance, teacher work, all negative results and independent final PHY audits.

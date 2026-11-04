Frozen-model PHY-budget quality/cost curves
==========================================

This campaign changes only max physical scores106/160/212 and topK2/4. The
original scalar checkpoint, last selected own-state checkpoint, solver code,
max56 feature roots, failed-candidate semantics, raw PFQ90% causal floor,
deficit weights and canonical native-BLER scoring are frozen. A physical
heuristic uses identical feature, root-cache, rejection and verification code.
There is no training, CTBS/headroom pruning, PPO or new native trajectory.

The18 points plus full DPLS each receive three serial alternating repeats on
12 validation states (seeds44--45, two histories) and nine previously exposed
PFQ development states (seeds46--48). Validation selection is committed before
reading development outcomes in this run. Both a quality ceiling point and
the best point within80% of full's complete mean time are frozen; they are
not silently interchanged. All points remain reported. No final seeds311--315
are read, and the active scheduler registry is unchanged.

The final partial shortlist may differ with budget and change the subsequent
path. Per-state J monotonicity is not assumed, enforced or reconstructed as an
envelope. Physical-budget and feature-root stops, actual work, accepted edits,
negative budget increments and UE service changes are separately recorded.

Complete time covers numeric input/PFQ decoding, fresh physical and feature
contexts, candidates/features/ranking, verification, canonical score-buffer
aggregation and cleanup. Only archive/model/library IO and first immutable
TBS setup are excluded. All terminal NumPy audits follow timing.

Quality requires every state J>=99.5% full. Service coverage requires no newly
unallocated COUNT>0 UE relative to full, plus98% of reference P5 and bottom20%
service on UEs with full expected bytes>1. Cost requires seed-equal per-state
median complete mean time<=80% full. J and instantaneous service are not
packet throughput, TTL, long-run fairness or delay measures.

Run in fresh directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_budget_curve training.stageb_lns.test_shortlist_trajectory training.stageb_lns.test_own_trajectory training.stageb_lns.test_bounded_probes training.stageb_lns.test_no_ctbs_inputs training.stageb_lns.test_ranked_dpls -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.budget_curve --output deliverables/new_budget_curve

Models, physical arrays and the original full source snapshot stay in
deliverables. Small evidence contains the complete curves, paired budget
increments, cap-stop diagnostics, frozen selections, audits and source hashes.
Plots use actual measured points and show all negative results. No saved-state
success alone promotes the model or establishes deployment deadlines.

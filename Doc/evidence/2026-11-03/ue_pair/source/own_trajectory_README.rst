Shallow/deep scalar repair and own-state teacher correction
=========================================================

This independent campaign compares the previously frozen scalar generator's
7-wide/8-step beam and 1-wide/32-step trajectory before new training. Both
share upper limits of 56 feature roots and 106 physical candidate scores;
actual work and complete serial time are reported, not claimed equal. A
single path evaluates one top-ranked edit each step. Failed edits retain the
parent, so 32 attempts do not mean 32 accepted edits. Repeated roots and
stagnation are explicitly counted. The shallow implementation reproduces the
previous generator's expansion, scoring, distinct-action retention and tie rule.

Fresh numeric state conversion, PFQ decode, physical/feature context creation,
candidate enumeration, every feature pass, network/sort, exact verification
and cleanup are timed. No complete search or post-decision teacher is available
inside the generator. The original unpruned PFQ is retained, as are causal
deficit weights and its 90% dequeue floor. There is no CTBS/headroom pruning,
spatial feature bank, proactive dropping, PPO or native simulation.

Phase A compares both geometries and matching physical-heuristic controls on
seeds44--45 only. Geometry and checkpoint selection use complete validation
outputs: quality/coverage/cost joint success first, then J shortfall, coverage
shortfall, and mean relative J. The deep geometry wins exact ties. Coverage
requires no newly unserved UE relative to full DPLS and at least 98% of its
instantaneous expected-service P5 in every state. Weak-UE expected service and
the number of UE service drops>2% are also reported; they are not packet or
long-run throughput measurements.

Training uses seeds41--43 and two CTBS-free source histories at three fixed
TTIs each. After each completed student decision, unique actually visited
roots are sampled evenly (at most eight/state). An independent PHY labels the
full original one-PRG neighborhood; no teacher changes the executing action,
student counters or caches. Two fixed rounds add these roots to the previous
53 anchor groups, deduplicating identical state/root pairs. Each round starts
from the selected weights and frozen original normalization, with a declared
fresh AdamW optimizer. Epochs0/5/10/20 are selected only on full validation
generator outputs. Labels are immediate exact whole-TB gains: neither suffix
optimality nor cross-TTI value is claimed. Repeated stalled roots are not new
data. Selection can keep the unmodified model if training does not improve.

Only after final selection are nine previously exposed PFQ development states
from seeds46--48 read in this campaign. Full DPLS and both geometries' heuristic,
initial and trained generators receive three serial alternating repeats. The
first32 accepted full-DPLS edits provide a separate diagnostic reference,
without being called a depth32 global upper bound or free online capability.

Quality still requires every state J>=99.5% full, complete time<=80% full, and
the declared coverage gate. Learning additionally requires changed weights,
positive gains in every seed and at least six wins among nine states against
the same-geometry initial model. Test seeds311--315 and the active registry
remain untouched. Development history stays disclosed; these are not blind
tests, native packet results or real-time deployment measurements.

Run in fresh output directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_own_trajectory training.stageb_lns.test_bounded_probes training.stageb_lns.test_no_ctbs_inputs training.stageb_lns.test_ranked_dpls -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.own_trajectory_campaign --output deliverables/new_own_trajectory

Models, full labels, original source snapshots and action/value arrays remain
in deliverables. Small evidence includes the frozen protocol, geometry/model
selection, per-state quality/time/coverage, teacher work, training histories,
independent audits and the result-dependent next-path decision.

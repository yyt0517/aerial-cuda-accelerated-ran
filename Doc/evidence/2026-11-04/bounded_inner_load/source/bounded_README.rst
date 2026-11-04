Exact capacity-bound inner solver and causal TTL coefficient frontier
===================================================================

``solve_bounded`` is a separate physical research entrypoint. It retains the
frozen persistent_q1e6_v1 objective, initialization, common raw-PFQ90% floor,
strict1/16-byte thresholds, and max-anchored one-unit canonical tie. It uses
the basic BLER>=0 capacity bound only. The interference-monotonicity tightening
is not enabled. Existing persistent/PHY source and binaries are unchanged.

An affected UE's count/min-layer/TBS/dequeue can be found without MMSE. The
same positive floating multiplications and nearest-even quantization give
the integer upper contribution. Each root keeps per-UE minimum layers with
each PRG excluded, so a sparse edit computes its bound over affected UEs only.
Edits and bounds persist across rounds; changed PRGs are regenerated, and
shared-UE exact payloads/bounds are invalidated. Never-scored edits remain
unscored across unrelated root changes. No candidate layout or whole-UE J
reduction is materialized online.

The heap is rebuilt from CURRENT bounds each round, ordered by descending UB
then ascending canonical key. It has no stale entries across rounds. Current
valid exact cache hits seed the maximum. Continue while an unknown eligible
entry has UB>=m, or UB=m-1 with key below the known winner. Entries with
UB<=threshold cannot qualify. There is no budget stop before certification.

The ``certificate_sets`` audit uses the paid independent full oracle. Its
required NEW evaluations exclude current exact cache hits. Accepted rounds
include unknown eligible UB>=true maximum, UB=maximum-1 with key below the
final winner, and the unknown winner itself when needed in the one-unit band.
Confirmation includes unknown eligible UB>threshold. This is the required set
for the frozen maximum-based certificate, not an information-theoretic lower
bound over every conceivable key-aware proof.

Build and run the focused tests::

    .venv/bin/python -m training.stageb_lns.build_bounded
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_bounded_dpls training.stageb_lns.test_certified_bounds > /tmp/bounded_20261104_tests.txt 2>&1

Paid saved-state gate and fair wall time
--------------------------------------

Use fresh output directories. The gate pays for an independent Python full
integer oracle and audits every bound, exact cached delta, root, logical D,
choice and terminal. The timer runs full/cached/bounded serially with rotated
order and no observers/log/history/profile/auditing inside timed decisions.
Immutable imports/library/table loading is outside timing; causal input/PFQ,
Context creation/search/destruction are inside::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.bounded_campaign --output deliverables/bounded_gate_NEW --cpu 255 --repeats 5

November4 gate: all12 inputs,378 rounds,624670 candidate bounds/choices passed.
Actual new evaluations7363 / required7363 =1.000. Including20 initializations,
7383 candidate evaluations replaced624690 full and183079 cached evaluations.
Serial wall means were23.46/10.97/4.46 ms for native full/cache/bounded.
These are saved-state shared-host measurements, not a deployment deadline.
No ranking model is fitted; the old161700 labels remain untouched.

The subsequent completed load check found that saved-state means were not
representative of the whole trajectory. Across240 new saved inputs, same-input
bounded/full wall ratios were35.94%,43.49%,48.57% at load1/1.4/1.6. All exact
and candidate-work gates passed, but the two heavy wall ratios exceeded40%.
The formal B coefficient scan was therefore NOT launched. Its code and frozen
grid below remain reviewable; no frontier or learned weights are claimed.
All24000 measured native TTIs retain their P95 and maxima in the evidence.

Native load checks and conditional method B
-----------------------------------------

``RateNativeEnv`` copies the frozen NativeEnv constructor solely to expose
the arrival-rate parameter. Its step/refresh/FIFO/action/PHY validation uses
the unchanged adapter. The simulator and libcumac binaries are hash-verified.
No CTBS action clipping or current DPLS teacher is invoked. Legacy diagnostic
fields remain transport metadata. The current algorithm registry is unchanged.

Load multiples are relative to0.8: x1.4=1.12 and x1.6=1.28. The stress campaign
is SERIAL, pinned to one CPU, two exposed training seeds41/42 at all3 rates.
Each trajectory has4000 common raw-PFQ warmup +4000 settling +4000 measurement
+400 mature-cohort followup TTIs. Every100 measured TTIs saves a causal state;
all measured decisions retain wall/process time, candidate/full-neighborhood
counts and UE/MMSE counters. P95 and max retain interruptions/outliers.
Separate saved-input full/cache/bounded replays provide fair paired speed
controls and a newly paid full equivalence oracle at every saved state::

    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.bounded_native --mode smoke --gate deliverables/bounded_gate_NEW --output deliverables/bounded_smoke_NEW --workers 1 --cpu 255
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.bounded_native --mode stress --gate deliverables/bounded_gate_NEW --output deliverables/bounded_stress_NEW --workers 1 --cpu 255
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_bounded_stress --source deliverables/bounded_stress_NEW --output deliverables/bounded_stress_analysis_NEW --cpu 255 --repeats 3

The inherited cost entry is per-load candidate work mean<=25%, P95<=40% of
the same full neighborhood and serial wall mean<=40% of native full. Density
growth versus baseline and native per-TTI P95/max are reported, not hidden by
these absolute gates. No additional hard realtime deadline is invented.

Only after this entry passes, method B scans the frozen coefficients
0,.25,.5,1,2 on training/development41--43 at rate0.8. The fixed-weight control
uses the SAME exact solver and original deficit weights, with no second
normalization at coefficient0. Positive coefficients multiply those weights
by1+k*(1-HOL_slack/400), then normalize mean1. Current HOL slack is reconstructed
from a read-only view of the causal FIFO; calling before() a second time would
incorrectly reset expiry reward bookkeeping, so the feature never does that.
Expired heads are skipped, and new arrivals receive full400TTI slack.

All five points and seeds are retained. Existing stress k0 seeds41/42 are
reused exactly, leaving13 new native runs. Each uses the same12400TTI protocol.
Grid trajectories may run in parallel for quality; their elapsed policy time
is not used for fair speed claims. No gradient training or sorting model is
started. The scan is a first objective-weight experiment, not learned-policy
generalization. Mature-packet cohorts, offered traffic and common PFQ prefixes
are checked, then four weighted saved states per new point receive a paid
independent integer full oracle audit::

    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.bounded_native --mode ttl --gate deliverables/bounded_gate_NEW --stress deliverables/bounded_stress_analysis_NEW --stress-source deliverables/bounded_stress_NEW --output deliverables/bounded_ttl_NEW --workers 4 --cpu 252
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_ttl_frontier --source deliverables/bounded_ttl_NEW --output deliverables/bounded_ttl_analysis_NEW

The frontier maximizes complete payload and minimizes mature expiry,
successful-packet P95 and worst(seed,UE) expiry. Report every seed separately;
aggregate payload is the mean across seeds, expiry/P95 pool mature packets,
and worst expiry is the maximum across all seed/UE pairs. Historical quality
gates and the stronger B rule(payload/expiry no worse, P95<=baseline+1ms) are
reported separately. No threshold/grid is adjusted after outcomes, no final
confirmation seeds are used, and the cap4 seed44 failure remains unchanged.

RSRP ordered ablation and search-cost diagnosis
==============================================

Scope and protocol
------------------

This extends training/stageb_rsrp without editing its frozen implementation.
Three cells, 36 UEs, 17 PRGs, 4T4R; Boundary08 and Uniform, seeds 41/42/43.
Each trajectory uses 4000 common PFQ warmup + 4000 settling + 20000 measured
+ 400 followup TTIs (0.5 ms/TTI). DPLS uses deficit weights and DA-PLS uses
the existing M-LWDF-PF weights. Native simulator, original channel execution,
traffic, raw PFQ, MCS/layers/OLLA, BLER curves, CRC draws, search thresholds,
capacity bounds and dequeue floor are unchanged. HARQ remains disabled.

Six stages, applied only to neighbor covariance in candidate evaluation:

* exact: instantaneous H H^H (reuse 12 validated original trajectories).
* spatial: trace(H H^H)/4 I_4 separately for every UE, PRG and neighbor.
  This keeps each matrix trace but removes off-diagonal structure AND unequal
  receive-direction eigenvalues; it is not merely zeroing off-diagonals.
* frequency: frequency-average the preceding covariance, retaining all TX
  ports' instantaneous power. This requires oracle H and is an ablation only.
* pilot: current wideband power from TX reference port 0 instead of all ports.
  This extra bridge prevents single-port sampling error being mislabeled as
  temporal filtering. It matches the original proxy's measurement definition.
* filtered: causal EMA of pilot power, tau 200 TTIs = 100 ms, immediately
  accessible each TTI. No sample/hold or transport staleness in this stage.
* reported: same EMA, reports every 20 TTIs = 10 ms, one-TTI transport delay
  = 0.5 ms (reuse 12 validated original RSRP trajectories). The last contrast
  combines periodic hold and transport delay, not just the 0.5 ms delay.

Scalar stages use 4 beta I_4, preserving the original serving H and precoder.
The proxy uses simulator-scaled linear power, not dBm or a standardized real
RSRP report. Noise, quantization and beam/calibration mismatch are absent.
Information quantities are controlled here; no deployable scalar-specialized
MMSE kernel is added. The spatial/frequency controls are deliberately oracles.

The 48 new closed-loop runs plus 24 reused endpoints form 72 trajectories.
Adjacent contrasts are causal effects conditional on this ordering, not
order-independent main effects. Differences may be negative: smoothing can
help or interact with earlier approximation, so do not force a loss budget.

Measurement and validation
--------------------------

Packet metrics use the measurement arrival cohort, followed until all outcomes
are known. Throughput counts only fully successful packet payload; MAC byte
goodput is separate. P95/P99 use successful packets, and all-outcome P95,
expiry and other nondelivery are also reported. Three seeds receive equal
weight. Paired nominal 95% t intervals are exploratory, without multiplicity
correction. A nonsignificant difference is not evidence of equivalence.

Every original snapshot is checked unchanged before execution. Native checks,
three saved-state independent NumPy PHY audits per trajectory, original
channel/traffic/topology/warmup pairing, full channel/report stream hashes
where recorded, and source/binary hashes protect experiment comparability.
The reported endpoint reproduces prior short-run native results bit for bit.
Four unit checks and 24 short native trajectories precede the long campaign.

Search mechanism
----------------

For an edit's affected UE set A, let Q(x) be round-to-even at 1e-6 weighted
byte resolution, d be dequeue bytes and e = d*(1-BLER) expected bytes:

  UB = sum_A [Q(w*d_candidate) - Q(w*e_root)]
  gain = sum_A [Q(w*e_candidate) - Q(w*e_root)]
  UB - gain = sum_A [Q(w*d_candidate) - Q(w*e_candidate)]

Thus bound slack is quantized weighted predicted failure bytes. The replay
independently enumerates every candidate in EVERY visited round for every model, checks
this identity against the actual solver, verifies canonical winners, and
checks that the frozen certificate's required fresh set equals the paid set.
It is not an information-theoretic lower bound for all conceivable searches.
Identical first-round PFQ roots, neighborhoods, weights and floor eligibility isolate the
PHY-induced bound effect from subsequent changed actions and queues.

Full search counters satisfy disjoint accounting on every measured TTI:

  neighborhood occurrences = paid + known exact reuse + floor-pruned
                           + threshold-pruned + certificate-heap-pruned
  candidate evaluations = paid + initialization evaluations
  heap pushes = paid + certificate-heap-pruned

Mean paid evaluations = mean rounds * neighborhood occurrences/round
                       * paid fraction.

Ratios of these three factors multiply exactly to the paid-evaluation ratio.
This is descriptive accounting, not three independent causal interventions.
Changed accepted edits invalidate overlapping UE/PRG dependencies and can
increase rescoring. The unchanged matrix kernel and activity-mask cache still
apply to scalar/isotropic models; fewer input scalars alone does not imply
less runtime. Measured results determine which effects dominate.

Timing
------

After the quality campaign finishes, replay uses 72 states balanced across
both original/RSRP endpoints, two algorithms, two scenes and three seeds.
All six models receive identical recorded queues/weights/native link state,
and reports from the same checked exogenous channel stream. Each model is
timed six times in cyclic order on one pinned CPU, occupying every order
position once, with observer/profile off and fresh contexts. Separate profiling
and complete enumeration of all visited neighborhoods are not included in latency. Scope includes reconstruction,
PFQ decode, covariance conversion and solver; excludes GPU PFQ, bridge, HOL
scan, report emulation and transport. Concurrent online times are descriptive.
Profiling is diagnostic wall-clock attribution, not a separate speed claim.
Intermediate policies' 144 additional saved states also receive six repeated
serial timings under their own stage, reported separately as own-trajectory
latency. Together with endpoint own timings this covers all 216 saved states.
Inherited sample times 8001/18001/28000 correspond to report ages 20/20/19
TTIs, near the stale end of the report cycle. Same-state timing/mechanisms
describe these sparse states, not an all-phase real-time guarantee. Closed-
loop quality and online search counters cover every measurement TTI.

Additional user-requested diagnostics
------------------------------------

ue_diagnostics.py reports all 36 UEs in all 72 trajectories: full-window TB
CRC rates, native predicted BLER, MCS, successful TB payload, packet expiry,
queue and tail delay. OLLA is reconstructed with float32 arithmetic from
initial zero delta, retained last CRC, and update ONLY on scheduled TTIs;
all 216 saved pre-decision vectors must match exactly. Positive delta is
subtracted from native SINR. Full OLLA arrays and percentiles are retained.

Signed SINR error (estimate minus original) and variance are reported per UE
with explicit sample counts for selected and raw-PFQ actions. These errors
use three saved states per trajectory, not an unrecorded full-time SINR
series. Native BLER/OLLA/MCS/expiry and candidate counts use all measured TTIs.
MCS is unchanged by model substitution within the same state; closed-loop
scheduling can indirectly change native OLLA/MCS through actual CRC history.
With HARQ off, failed bytes are dequeued; failed bytes are not retransmitted.

The complete per-TTI candidate file covers 1,440,000 measurement decisions.
Generated edit entries (including regenerations), neighborhood occurrences
across visited rounds, fresh scores and floor-eligible counts are distinct.
The saved-state full-round oracle separately establishes the current frozen
certificate's required fresh evaluations, paid/required ratio, and bound
violations. 0/0 ratios are undefined. The bound certifies the same approximate
objective as fine scoring, not the native true-channel outcome objective.

native_timing_probe.py makes a separate timing-only binary from the frozen
source/object/link manifest. Host clocks bracket existing synchronized
channel generation and SVD, without additional synchronization. After the
main replay, eight sequential short probes must reproduce uninstrumented
smoke actions, all native metrics and channel/report hashes bit for bit.
Channel/SVD/PFQ wall times, scheduler process CPU and bridge round-trip wall
time are separate scopes. Main quality trajectories never use this binary.
The main matched-state CPU benchmark performs no channel generation at all.

Commands
--------

Use new output directories; never overwrite archived evidence. The existing
GPU container and MPS service are used as in the preceding study::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_rsrp_ablation.test_model -v
    # inside c_aerial_oai2, /opt/nvidia/cuBB, with existing MPS variables:
    .venv/bin/python -u -m training.stageb_rsrp_ablation.campaign --output deliverables/NEW/smoke_v1 --workers 24 --smoke
    # On the host, audit short runs before starting the long container campaign:
    .venv/bin/python -m training.stageb_rsrp_ablation.validate_smoke --root deliverables/NEW
    # Back inside the container:
    .venv/bin/python -u -m training.stageb_rsrp_ablation.campaign --output deliverables/NEW/campaign --workers 32
    # --smoke in a separate directory runs 24 short trajectories.

After completion, on the host with BLAS/OMP threads set to one::

    .venv/bin/python -u -m training.stageb_rsrp_ablation.analyze --source deliverables/NEW/campaign --output deliverables/NEW/analysis --workers 4
    .venv/bin/python -u -m training.stageb_rsrp_ablation.replay --source deliverables/NEW/campaign --output deliverables/NEW/replay --repeats 6
    .venv/bin/python -m training.stageb_rsrp_ablation.channel_diagnostics --root deliverables/NEW

Then, inside the GPU container with the original MPS variables, build and run
the separate timing probes (the build refuses host paths)::

    .venv/bin/python -u -m training.stageb_rsrp_ablation.native_timing_probe --root deliverables/NEW --build --run

Finally, on the host, generate per-UE diagnostics and the report::

    MPLCONFIGDIR=/tmp/rsrp_ablation_mpl .venv/bin/python -m training.stageb_rsrp_ablation.report --root deliverables/NEW

For unattended local sequencing, postprocess.py waits for the quality
campaign, runs analysis/replay/channel diagnostics/report; launch the container
probe separately with --build --run --wait-for-replay. The report waits for
the independent probe validation. Preflight and short-run endpoint checks
should be completed before the long campaign as in this workspace study.

campaign.py freezes itself, model.py, and all original policy/native sources.
Do not edit frozen files during a campaign. Analysis/report changes cannot
change decisions and are separately archived and hashed with the results.

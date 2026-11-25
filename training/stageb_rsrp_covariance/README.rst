RSRP plus periodically updated normalized spatial covariance
===========================================================

Experiment
----------

Three cells, 36 UEs, 17 PRGs, 4T4R; Boundary08 and Uniform; DPLS and
DA-PLS; seeds 41/42/43. There are 120 new trajectories and 24 verified reused
exact/scalar endpoints. Each trajectory retains 4000 common PFQ warmup,
4000 settling, 20000 measured, and 400 followup TTIs, at 0.5 ms per TTI.
Original native execution channel, MCS/layers/OLLA, traffic, PFQ draft,
search, dequeue floor and HARQ-off behavior are unchanged.

The scalar power beta remains the previous single-reference-port RSRP proxy:
100 ms causal linear-power EMA, 10 ms reporting period, 0.5 ms transport.
Only the neighbor interference model changes:

  C = H H^H
  R = 4 C / trace(C), so trace(R) = 4
  C_hat = 4 beta_RSRP R_reported

For ``prg``, R is distinct for each UE/neighbor/PRG. For ``wideband``, first
average C over all PRGs, then normalize, and reuse that one matrix on every
PRG. Both use all TX reference ports for ideal spatial measurement. Serving
H and precoder are always original. Scalar RSRP corresponds to R = I.

Periods are 1/4/20/100/200 TTIs = 0.5/2/10/50/100 ms. Every spatial report
has one-TTI delay. Spatial covariance has no additional temporal smoothing;
this is slow sampling/holding of normalized instantaneous covariance, not a
long-term covariance estimator. Wider periods can retain an obsolete spatial
direction; the existing Rayleigh generator does not guarantee that spatial
structure changes slowly. The 0.5 ms point also has 0.5 ms transport delay.

The measurement emulator accesses H, but the scheduler interface consumes
only serving CSI, scalar RSRP, and normalized neighbor reports. Each 4x4
Hermitian trace-four matrix is packed into three real diagonal entries plus
six complex off-diagonals: exactly 15 independent real parameters. Reports
are decoded into a PSD factor only on arrival and cached between updates.
The existing H-based PHY interface is retained, including its dense MMSE
kernel. CPU results include this compatibility adapter, not the hypothetical
cost of a new direct-covariance API or a scalar-specialized kernel.

Feedback accounting
-------------------

Count only neighbor-link payload (72 links), excluding unchanged serving CSI,
headers, IDs, coding, and reference-signal airtime. A normalized matrix costs
15 real parameters; RSRP costs one real parameter every 10 ms. Per-PRG shape
reports multiply matrix count by 17. The original PHY needs a full Hermitian
covariance (16 reals) per neighbor/PRG/TTI; the current raw complex-H interface
actually carries 32 reals, so both baselines are provided.

16-bit and 32-bit payload rates are hypothetical encoding budgets. Performance
uses unquantized float64 packets. The float64 transport-equivalent payload is
also reported; no quantization noise or measurement error is simulated.

Validation and measurement
--------------------------

Five focused unit checks cover normalization/PSD/packing, unchanged serving
CSI, independence from current neighbor H at the scheduler interface, delay
and cache reuse, singular covariance, and bounded versus independent full
search. All 48 short native cases and all eight archived endpoint cases are
validated before the full sweep. Exact/scalar endpoints reproduce all native
metrics, actions and input hashes bit for bit.

Every execution step checks original H unchanged. Three standard states per
trajectory get independent NumPy PHY audits against native execution; topology,
traffic, common warmup and available full channel/report streams are paired.
All saved covariance packets match an independent witness trajectory, and
complete covariance hash streams match between objectives with the same seed.
All measured candidate counters satisfy the existing exact partition identity.

Packet metrics use the arrival cohort followed to terminal outcomes. Throughput
counts fully delivered packet payload; successful-byte MAC goodput is separate.
P95/P99 use successful packets; expiry and other nondelivery are separate. The
200 ms TTL can saturate successful-packet tails while expiry keeps increasing.
Recovery is the improvement over scalar RSRP divided by the original-to-scalar
gap, using equal-seed means. It is not clipped to 0--100%; zero gaps are
undefined. Paired nominal 95% t intervals use n=3, without multiplicity correction.

New trajectories also record selected-action SINR/BLER predictions throughout
the policy window, enabling full-window per-UE errors against native execution.
Reused endpoints lack these historical predictions; common-state comparisons
cover all models, and the report distinguishes the two scopes.

CPU benchmark
-------------

After all native runs finish, time all 12 models on 72 common states from the
prg_t1 trajectories: six stratified, independently dithered phases per seed,
scene and objective. This avoids locking all samples to the stale edge of
every reporting cycle. The states still come from one policy's queue history;
own-policy timings on all 432 ordinary saved states are reported separately.

Pin one CPU and perform 12 cyclic repetitions, so each model occupies every
position once. Each evaluation creates a fresh PHY/search context. Alternate
cache-hit and report-refresh order. Report cached and refreshed decision CPU
separately, plus the amortized estimate

  cached * (1 - 1/K) + refreshed / K.

This is a controlled sampled-state cost estimate, not a full-window online
latency or p95 guarantee. Input reconstruction, fixed recorded weights, PFQ
decode, shape adapter and search are included. Native simulator, channel
generation, GPU PFQ, HOL computation, measurement emulator and transport are
excluded. Concurrent online times are descriptive only. Model acquisition
cost is separate; the all-period witness emulator is never charged to the
timed scheduler.

Untimed independent enumeration checks every visited round for one diagnostic
phase per witness and the whole first neighborhood at other phases. Audits
verify integer bound/gain/slack, winner and required-versus-paid score sets.
No observer or profiler runs in the timing pass. Normalized shape error and
direction similarity are measured against original, unmodified H, separately
from the timing pass.

Each model's chosen action is also evaluated, outside timing, by the NumPy
PHY on the original channel with identical recorded weights and native MCS/
layers. Report its true weighted expected-byte utility relative to the
exact-policy action, and prediction error for its own selected utility.
The exact-policy action is a local-search reference, not a global optimum;
these are one-step expectations, not closed-loop throughput. The additional
diagnostic is checked against archived native dequeue/BLER expectations.

Commands
--------

Use a new output directory and single-thread BLAS/OMP. In the existing GPU
container c_aerial_oai2 at /opt/nvidia/cuBB, retain the prior MPS environment::

    .venv/bin/python -m unittest training.stageb_rsrp_covariance.test_model -v
    .venv/bin/python -u -m training.stageb_rsrp_covariance.campaign --output deliverables/NEW/smoke_v1 --workers 32 --smoke

On the host, validate the short runs::

    .venv/bin/python -m training.stageb_rsrp_covariance.validate_smoke --root deliverables/NEW
    .venv/bin/python -m training.stageb_rsrp_covariance.replay --source deliverables/NEW/smoke_v1 --output deliverables/NEW/replay_smoke --repeats 2

Then in the GPU container::

    .venv/bin/python -u -m training.stageb_rsrp_covariance.campaign --output deliverables/NEW/campaign --workers 48

After native completion, on the host::

    .venv/bin/python -m training.stageb_rsrp_covariance.analyze --source deliverables/NEW/campaign --output deliverables/NEW/analysis --workers 4
    .venv/bin/python -m training.stageb_rsrp_covariance.replay --source deliverables/NEW/campaign --output deliverables/NEW/replay --repeats 12
    MPLCONFIGDIR=/tmp/rsrp_covariance_mpl .venv/bin/python -m training.stageb_rsrp_covariance.report --root deliverables/NEW

The postprocess module sequences these host operations after campaign completion.
The report bundle includes every trajectory's compressed per-TTI search log
under search_traces, with hashes and measurement-window bounds in index.csv.
Filter to those bounds to reproduce measured candidate/score summaries;
concurrent online CPU/wall times are descriptive, not the serial benchmark.
Do not edit frozen model/campaign or inherited decision sources during a run.
The current formal campaign uses 48 workers after a memory-headroom check.
An incomplete initial 32-worker attempt is retained separately, with the
resource-only restart documented in concurrency_adjustment.json. Those
unfinished traces do not enter the quality results or serial CPU benchmark.

Neighbor-RSRP approximation study
================================

This experiment changes only the neighbor-interference counterfactual model
inside DPLS/DA-PLS. The native simulator, channel, traffic, receiver, CRC draws,
queue transitions, OLLA, MCS, layer caps and raw PFQ initialization are retained.
No prior implementation or archived result is overwritten.

Frozen protocol
---------------

Three cells, 36 UEs, 17 PRGs, Boundary/rate0.8 and Uniform/rate1.0;
seeds 41, 42 and 43; 4000 common PFQ warmup + 4000 settling + 20000 measurement
+ 400 followup TTIs, 0.5 ms per TTI. Four arms: original deficit DPLS,
original M-LWDF-PF DA-PLS, and one RSRP approximation for each weight.
The exact originals reuse the validated 2026-11-04 trajectories only after
checking simulator, solver and policy source hashes. New output directories
reject overwrites. The three seeds are an exploratory paired study, not an
unseen-data test or proof of universal performance equivalence.

Channel audit
-------------

channel_audit.py traces the actual frozen executable and its archived source:
large-scale path loss + shadowing set amplitude; the Rayleigh kernel updates
complex H using rho=0.9938 and rhoPrime=0.0786; genFastFadingGpu receives the
continuous time index t; the pre-action exporter reads estH_fr_actUe; the
physical evaluator constructs H H^H from the exported H. It also checks saved
channels across PRGs and separated times. This is instantaneous small-scale
plus large-scale H, not a large-scale-only scalar. The legacy rho comment
mentions 1 ms; the experiment actually uses 0.5 ms and inherits those constants.

RSRP approximation
------------------

There is no standardized neighbor RSRP reporting interface in this simulator.
RsrpMeasurements emulates an ideal reference measurement using TX port 0:
linear |H|^2 averaged across the 17 PRGs and four receive branches. It applies
a causal EMA with tau=200 TTIs (100 ms), reports every 20 TTIs (10 ms), and
delivers a report after one TTI (0.5 ms). The initial report is obtained during
common warmup. No observation after the decision is used.

The output is one nonnegative scalar beta[u,cell], in the simulator's scaled
power units. It is not dBm, not a direct large-scale gain oracle, and not a
standards-compliant RSRP report. Equal reference/data per-port power and known
calibration are assumed; measurement noise, quantization, SSB/data beam
mismatch and actual air-interface overhead are not simulated.

The scheduler gets only serving CSI and these neighbor scalars. For each
neighbor it uses C_hat = 4 beta I_4 on all PRGs. The factor four accounts for
the four transmit ports in the existing activity-only H H^H interference model.
A synthetic sqrt(4 beta) I_4 channel factor makes the unchanged physical
evaluator consume exactly this covariance. It is never written to the native
snapshot. Serving CSI, SVD precoder, measured native link state, raw PFQ and
layer/MCS selection remain unchanged. Consequently this is an ablation of
fine-grained neighbor interference evaluation, not a complete limited-feedback
deployment or removal of all instantaneous information from scheduling.

The same capacity-bound search and matrix kernel are used in both versions.
No scalar-specialized MMSE implementation is introduced. Cost changes may
therefore arise from different search trajectories; reduced measurement input
does not imply a proportional reduction in runtime.

Validation and outputs
----------------------

test_model checks causal delivery/filtering, phase/nonpilot-port information
isolation, exact isotropic covariance, unchanged original physical evaluation,
and approximate bounded-search equality to independent full enumeration.
Eight short native runs check closed-loop execution before the long study.
Saved states are audited using independent NumPy PHY for both predicted and
actual SINR/BLER. Raw H is checked unchanged before every native step.

Statistics use whole successful packets from the measurement arrival cohort,
followed until every packet terminates. MAC goodput, successful-packet P95/P99,
all-outcome P95 (including expired and CRC-damaged packets), expiry and CRC
nondelivery are kept distinct. First aggregate within a seed, then give seeds
equal weight; paired differences include nominal 95% t intervals.

Concurrent online wall times are descriptive. timing.py runs only after the
quality campaign completes, pins one CPU, rotates exact/approximate timing
order and recreates fresh contexts. Both modes use the same saved state and
weights; saved states from both trajectories are represented equally. Report
measurement emulation is timed separately online and is not air-interface
overhead. Serial timing excludes GPU PFQ, bridge and HOL scanning.

Commands
--------

Host checks::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_rsrp.test_model -v
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m training.stageb_rsrp.channel_audit --output deliverables/NEW/channel_audit.json

Run in the existing GPU container, with its existing MPS service::

    CUDA_DEVICE_MAX_CONNECTIONS=2 CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_rsrp.campaign --output deliverables/NEW/campaign --workers 12

Use --smoke in a separate directory for the eight short validation runs.
Never edit model.py, campaign.py or their frozen dependencies during a run.

After the campaign completes, on the host::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_rsrp.analyze --source deliverables/NEW/campaign --output deliverables/NEW/analysis --workers 3
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_rsrp.timing --source deliverables/NEW/campaign --output deliverables/NEW/timing --repeats 5
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m training.stageb_rsrp.diagnostics --root deliverables/NEW
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m training.stageb_rsrp.report --root deliverables/NEW

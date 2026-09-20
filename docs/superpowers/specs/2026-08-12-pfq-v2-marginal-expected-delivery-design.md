> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/specs/2026-08-12-pfq-v2-marginal-expected-delivery-design.md)保留当时状态。

# PFQ-v2 Marginal Expected Delivery Design

## 1. Goal

Introduce an independent PFQ-v2 scheduler that combines delivery-aware ranking
with a set-aware demand cap. The experiment asks whether ranking each candidate
PRG by its marginal expected useful delivery improves the existing PFQ baseline
under the current Stage-B boundary scenario.

The existing PFQ remains unchanged and is rerun from the same new binary as
PFQ-v2. PFQ-v2 is promoted only when paired seed 41--50 results improve mean
goodput without regressing the agreed reliability, tail, weak-UE, worst-seed,
or runtime safeguards.

## 2. Current behavior and motivation

The current Type-0 PFQ implementation ranks a UE/PRG candidate using an
instantaneous Shannon-like reachable rate, PF average-rate normalization, a
queue-size multiplier, a rate-derived byte estimate, a demand cap, and an
intra-TTI repeated-grant decay. It then visits PRGs in native order and greedily
commits the best eligible UE.

That estimate does not model whether the resulting TB is likely to decode, and
the current demand cap adds independent per-PRG byte estimates rather than
recomputing the service capacity of the complete PRG allocation set. Therefore
two grants with similar reachable rate can receive similar credit even when
their MCS, layer count, BLER, final TBS, and useful delivery differ materially.

Existing 10-seed artifacts are context only. They cannot be mixed with the new
experiment because adding predictor inputs and diagnostics changes the binary.
Both PFQ-v1 and PFQ-v2 must be rebuilt and rerun together.

## 3. Scope

The first implementation and promotion gate cover only:

- 3 coordinated cells and 36 active UEs;
- 17 PRGs with 16 PRBs per PRG;
- Type-0 allocation in native PRG order `0..16`;
- `coop_boundary` placement and strongest-cell association;
- SVD precoding and the existing MCS/BLER/TBS tables;
- 3000-byte packets, 200 ms TTL, and 4000 TTIs;
- paired topology and traffic seeds 41--50.

PFQ-v2 is exposed as a new baseline selection, `-b 4`. Existing selections
`-b 0`, `-b 1`, `-b 2`, and `-b 3` retain their current behavior. PFQ-v2 uses
the same UE pool, channel inputs, PF average-rate state, queue weight, beta,
association constraints, and intra-TTI decay as PFQ-v1.

The first version does not:

- add a HOL packet-completion bonus;
- reorder PRGs or search over PRG permutations;
- change actual layer selection, MCS selection, OLLA, HARQ, buffer debit,
  packet expiry, or traffic generation;
- add a full PHY counterfactual simulation per candidate;
- claim support for the 7-cell scenario, other PRG sizes, Type-1 allocation,
  CQI-based MCS selection, or non-SVD precoding;
- modify or discard the user's existing uncommitted actual-interference
  diagnostics.

These exclusions isolate the effect of marginal expected delivery and the
refined demand cap.

## 4. Alternatives considered

### 4.1 Per-PRG success multiplier

Multiply the existing per-PRG rate by a success probability estimated for that
PRG alone. This is inexpensive but ignores that MCS, layers, BLER, and TBS are
selected for the final allocation set. It cannot correctly represent a PRG
that changes the reliability of an already assembled TB.

### 4.2 Allocation-set-aware marginal expected delivery

Maintain a small state for every UE and evaluate the change caused by adding a
candidate PRG to its current set. This captures non-additive TBS quantization,
layer reduction, demand saturation, and reliability changes while retaining an
`O(PU)` greedy scheduler. This is the selected design.

### 4.3 Full counterfactual PHY evaluation

Rerun layer, MCS, interference, and BLER processing for every candidate set.
This is closest to the simulated PHY but turns the scheduler into a nested
simulation, materially increases runtime, and entangles PFQ-v2 with future
cross-cell decisions. It is out of scope for the first experiment.

## 5. Architecture

PFQ-v2 has three bounded components.

### 5.1 `ExpectedDeliveryPredictor`

A side-effect-free C++ predictor accepts immutable channel/link inputs, a UE's
current allocation state, and one candidate PRG. It returns:

- predicted layer count;
- predicted MCS;
- predicted BLER and success probability;
- set-level raw service capacity;
- set-level expected useful delivery;
- marginal expected useful delivery;
- a validity/error code.

The CPU scheduler and the CUDA scheduler's existing host postprocessing call
the same implementation. The predictor does not update real OLLA, PF average
rates, HARQ, allocation arrays, buffers, or packet state.

For the CUDA path, the scheduler exposes the required candidate features to
host postprocessing through persistent host-mirrored buffers. Post-EQ SINR and
singular-value/layer-support data are transferred in one batched step at the
existing postprocessing synchronization boundary; PFQ-v2 must not introduce a
per-candidate device synchronization or allocation. The additional transfer is
part of the runtime gate.

### 5.2 `PfqV2UeState`

For each UE and TTI, the scheduler keeps:

- number of committed PRGs;
- per-layer linear post-EQ SINR sums;
- minimum supported layer count across committed PRGs;
- current raw service capacity `C(S)`;
- current predicted success probability `p(S)`;
- current expected useful delivery `D(S)`.

Candidate evaluation copies or derives a tentative state and cannot mutate the
committed state. Only the winning UE's `commit` operation updates its state.

### 5.3 `SuccessfulDeliveryShadowLedger`

An observation-only ledger receives the same packet arrivals as the existing
traffic model. It consumes bytes only for successful TBs and records successful
bytes, completed packets, and completion latency. A failed TB does not consume
the shadow queue.

This ledger never feeds scheduler inputs and never modifies existing buffer,
HARQ, expiry, packet-delay, or goodput accounting. Its packet completion is an
ordered successful-service diagnostic under the observed grant stream, not a
claim that the current simulator implements reliable retransmission.

The same ledger runs for the newly built PFQ-v1 and PFQ-v2 arms. This makes its
delivery results paired and keeps its instrumentation overhead common to both
algorithms; it is not used only on the candidate scheduler.

## 6. Predictor definition

Let `S` be the set of PRGs already committed to UE `u`, let `g` be the current
candidate PRG, and let `Q_u` be the MAC buffer demand snapshotted at the start
of the current TTI. For an empty set, raw capacity and expected delivery are
both zero; the first candidate's layer count is its own supported layer count.

### 6.1 Layer prediction

The supported layer count for each UE/PRG is precomputed from the existing
singular-value threshold rule. The tentative set-level layer count is:

\[
L_u(S \cup \{g\}) = \min_{h \in S \cup \{g\}} L_u(h).
\]

Per-layer SINR sums are retained even when the current set uses more layers, so
adding a lower-rank PRG can recompute the set using the reduced layer count in
constant time.

### 6.2 MCS prediction and OLLA isolation

MCS prediction uses the current wideband SINR and the same configured MCS LUT
as the real selector. PFQ-v2 maintains a host-side OLLA shadow initialized with
the same per-UE target and step values as the actual selector. At each TTI
boundary, the shadow applies the same deterministic transition from the
previous TB outcome, producing an immutable snapshot for all candidate
evaluations in that TTI.

The shadow does not write the actual CPU or GPU OLLA state. Fixed-seed parity
tests compare the predicted final MCS against the MCS selected by the real
pipeline; mismatch is a predictor defect, not a reason to retune OLLA.

### 6.3 BLER prediction

For the tentative allocation set, PFQ-v2 averages the available post-EQ SINR
contributions over all selected PRGs and predicted layers in linear scale,
converts the result to dB, applies the same lower bound as the current BLER
path, and interpolates the configured per-MCS BLER table. The success
probability is:

\[
p_u(S) = 1 - \widehat{\mathrm{BLER}}_u(S).
\]

This is a scheduler-side estimate. It reuses available post-EQ inputs and does
not recompute the realized cross-cell interference matrix for every
counterfactual allocation. Calibration against actual TB outcomes is therefore
required and is reported separately.

### 6.4 Raw service capacity

The predictor calls the same TBS rule using:

- `|S| * 16` allocated PRBs in the canonical scenario;
- the configured PDSCH data-symbol count;
- predicted layer count;
- code rate and QAM order for the predicted MCS.

`C_u(S)` uses the numeric service-unit convention currently passed to the MAC
buffer-debit path. The current simulator has historical ambiguity between the
nominal TBS bit result and fields named as bytes. PFQ-v2 does not silently fix
that issue in only one scheduler because doing so would change the experiment's
buffer and packet semantics. The implementation must name the compatibility
unit explicitly and also log nominal physical TBS bytes (`TBS / 8`) as a
separate diagnostic.

### 6.5 Expected and marginal useful delivery

Expected useful delivery is:

\[
D_u(S) = p_u(S) \cdot \min(C_u(S), Q_u).
\]

The value used to assess candidate `g` is:

\[
\Delta D_{u,g} = D_u(S \cup \{g\}) - D_u(S).
\]

This definition gives no credit beyond current demand and permits negative
marginal value when an additional PRG lowers the predicted success probability
enough to reduce total expected delivery.

## 7. PFQ-v2 scoring and allocation

Define the candidate's effective instantaneous service rate as:

\[
R^{\mathrm{eff}}_{u,g} = \frac{\max(\Delta D_{u,g}, 0)}{T_{\mathrm{slot}}}.
\]

The candidate metric is:

\[
M_{u,g} =
\frac{(R^{\mathrm{eff}}_{u,g})^\beta}{\overline{R}_u}
\cdot W_{\mathrm{queue}}(Q_u)
\cdot W_{\mathrm{decay}}(n_u).
\]

`W_queue`, `beta`, the PF average rate, and the existing intra-TTI decay remain
unchanged from PFQ-v1. The compatibility service-to-rate scale is common to all
candidates and follows the current numeric convention; this experiment does
not redefine historical average-rate units.

For each cell, the scheduler visits PRGs in existing order. It evaluates every
associated and otherwise eligible UE, chooses the largest strictly positive
metric, preserves the current deterministic UE-index tie behavior, commits the
winner, and continues. If no candidate has a positive valid metric, the PRG is
left blank.

The nonpositive threshold is a fixed relative epsilon:

\[
\epsilon_u = \max(10^{-3}, 10^{-6} \cdot \max(Q_u, 1)).
\]

A candidate participates only when `Delta D > epsilon_u`.

## 8. Dual-ledger demand cap

PFQ-v2 keeps two distinct values:

- the raw-capacity ledger `C_u(S)`, used for the hard demand boundary;
- the expected-delivery ledger `D_u(S)`, used for ranking.

The first version uses the current buffer `Q_u` as its PFQ-v2 hard demand and
does not add PFQ-v1's one-packet cap slack. The rules are:

1. If `C_u(S) >= Q_u`, UE `u` receives no more PRGs in the current TTI.
2. If `C_u(S) < Q_u`, one candidate may cross the demand boundary because a PRG
   cannot be split.
3. The useful part of that crossing candidate is limited by `min(C, Q)` in
   `D(S)`.
4. Even below the raw cap, a candidate with `Delta D <= epsilon_u` is rejected.

This avoids additive byte-credit drift, prevents grants beyond the first
capacity crossing, and separates service feasibility from expected value.

## 9. Data flow and caching

At the start of each TTI:

1. Validate PFQ-v2 mode and required inputs before committing any PRG.
2. Advance the host OLLA shadow once using the previous TB outcome.
3. Precompute UE/PRG eligibility, supported layers, and per-layer post-EQ SINR
   contributions.
4. Initialize one empty `PfqV2UeState` per UE.

For each PRG:

1. Form a tentative state for every eligible UE.
2. Perform constant-size layer aggregation, MCS lookup, BLER interpolation,
   and TBS calculation.
3. Compute `Delta D` and the PFQ-v2 metric.
4. Select the winner and commit only that UE's tentative state.

With the antenna/layer limit fixed by the 4T4R path, precomputation is
`O(PUL)` and greedy evaluation is `O(PU)`, where `L` is a small constant. The
design explicitly rejects a naive `O(P^2 U)` recomputation of every allocation
set.

Detailed per-candidate logging is disabled by default. Aggregate counters and
final per-TB predictor records are sufficient for validation without making
I/O the dominant runtime cost.

## 10. Error handling and fallback

Eligibility filtering happens before the predictor and is not counted as a
predictor error.

For an otherwise eligible candidate, nonfinite SINR, invalid layer count,
invalid MCS/LUT index, or nonfinite capacity/delivery marks only that candidate
invalid. It is excluded and increments a reason-specific counter.

If required set-level inputs are absent, or PFQ-v2 is invoked under an
unsupported allocation, precoding, or MCS mode, validation fails before the
first PRG. That complete TTI falls back to the original PFQ algorithm and
increments `predictor_fallback_tti_count`. PFQ-v2 never partially allocates a
TTI and then falls back.

The canonical promotion run requires both eligible-candidate predictor errors
and TTI fallbacks to be zero. Any fallback makes that seed's result invalid for
promotion, so compatibility fallback cannot hide an implementation defect.

Debug builds may assert after recording the reason. Normal experiment builds
remain deterministic and emit the failure counters in their result manifest.

## 11. Diagnostics and accounting

Existing goodput, BLER, expiry, packet delay, UE-throughput, fairness, PRG
utilization, buffer, and HARQ behavior remain unchanged. They are the
same-semantics guardrails for PFQ-v1 versus PFQ-v2.

PFQ-v2 adds aggregate scheduler diagnostics for:

- predicted raw capacity and expected delivery;
- final predicted MCS, layer count, BLER, and success probability;
- raw-cap stops and nonpositive-marginal stops;
- blank PRGs;
- invalid candidates by reason and fallback TTIs;
- nominal physical TBS bytes in addition to compatibility service units.

For every scheduled final TB, predictor calibration records pair:

- predicted success probability and actual success indicator;
- predicted and actual MCS;
- predicted and actual layer count;
- predicted and actual TBS values.

The aggregate output includes observed success rate, mean predicted success
rate, ten equal-width `[0, 1]` calibration buckets, and Brier score:

\[
\mathrm{Brier} = \frac{1}{N}\sum_i (p_i-y_i)^2.
\]

The observation-only shadow ledger consumes
`min(actual_final_TBS / 8, shadow_backlog_bytes)` on a successful TB and zero
on a failed TB. It never consumes the predictor's tentative TBS. It reports
PHY-successful bytes, shadow completed-packet count, shadow pending bytes, and
successful-completion latency distribution. Because the real buffer retains
its existing debit behavior, these shadow metrics are diagnostic and are not
substituted for the legacy promotion guardrails.

Scheduler latency p50/p95/p99 and total process wall time are recorded. The
existing actual-interference diagnostics remain intact and may be correlated
with prediction error, but they are not inputs to candidate scoring.

## 12. Testing strategy

### 12.1 Predictor unit tests

Deterministic tests cover:

- empty allocation state;
- a normal positive marginal candidate;
- a candidate that crosses demand;
- zero demand and an already saturated raw ledger;
- a low-quality PRG that creates nonpositive marginal delivery;
- a PRG that reduces the set's layer count;
- MCS and BLER LUT boundary interpolation;
- TBS quantization boundaries;
- NaN, invalid layer, and invalid LUT inputs;
- fixed epsilon behavior;
- OLLA shadow transition equivalence without mutation of real OLLA.

Tests are written before each predictor behavior and use a pure helper so they
do not require a stochastic end-to-end run.

### 12.2 Scheduler regression and parity

A short fixed-seed run verifies:

- baseline modes `-b 0..3` preserve their allocation and KPI behavior;
- PFQ-v2 CPU and CUDA host-postprocess produce the same Type-0 allocation;
- tie breaking and native PRG order are unchanged;
- cap-only and full delivery-aware modes differ only in the success model;
- fallback occurs only for deliberately unsupported/missing inputs;
- final predicted MCS and layer selections match the actual pipeline for at
  least 99.9% of scheduled new-transmission TBs before performance evaluation
  begins.

The 99.9% threshold allows only diagnostic edge cases; systematic mismatch
blocks the multiseed experiment. Calibration and Brier score are reported but
are not tuned against the promotion seeds.

### 12.3 Three-seed screening and ablation

Seeds 41--43 run three arms:

1. PFQ-v1;
2. PFQ-v2-cap-only, implemented by fixing success probability to one while
   retaining set-aware TBS and the dual-ledger cap;
3. full PFQ-v2 with BLER-derived success probability.

Cap-only is an experiment flag under `-b 4`, not another public baseline. This
screen identifies whether a result comes from the refined cap or the delivery
model. It is for defect detection and attribution, not parameter search. An
obvious correctness, KPI, or runtime failure stops the full run.

The reproducible interface is
`CUMAC_PFQ_V2_SUCCESS_MODEL=bler_lut` for full PFQ-v2 (and the `-b 4`
default), or `CUMAC_PFQ_V2_SUCCESS_MODEL=constant_one` for the cap-only arm.
Any other value is rejected at startup rather than silently mapped to a mode.

### 12.4 Ten-seed paired promotion run

The formal run rebuilds one binary and runs PFQ-v1 and full PFQ-v2 with the
same scenario, topology seed, traffic seed, and 4000-TTI duration for each seed
41--50. Old PFQ artifacts are not reused.

Every run records the Git commit, dirty-worktree marker or patch identity,
binary hash, predictor version, complete environment variables, algorithm
mode, topology seed, traffic seed, and timing configuration.

## 13. Promotion criteria

PFQ-v2 is promoted only when all conditions hold over the ten paired seeds:

1. Mean goodput is strictly greater than PFQ-v1.
2. Mean BLER is no greater than PFQ-v1.
3. Mean legacy expiry rate is no greater than PFQ-v1.
4. Mean legacy packet p95 latency is no greater than PFQ-v1.
5. Mean UE-goodput p5 is no less than PFQ-v1.
6. For every seed, PFQ-v2 goodput is at least 99% of paired PFQ-v1 goodput.
7. Eligible-candidate predictor-error and TTI-fallback counts are zero.

The comparison also reports paired per-seed deltas and a paired bootstrap
95% interval, using 10,000 paired resamples and analysis seed `20260812`, for
interpretation. Each mean guardrail is the arithmetic mean of the ten per-seed
scalar KPIs; packet p95 is therefore the mean of ten per-seed p95 values, not a
new percentile pooled across all packet samples. The interval and calibration
diagnostics do not replace the explicit promotion conditions.

If the formal run fails and the model is changed, the change receives a new
predictor version and the complete paired run is repeated. Results are not
repeatedly tuned on seeds 41--50 and then presented as the original test.

## 14. Runtime gate

The implementation target is incremental cached `O(PU)` host processing. A
2--5x increase in the PFQ host-postprocessing portion is acceptable only when
the end-to-end cost remains within the gate.

Performance measurement uses seed 41 with the identical canonical 4000-TTI
command, one untimed warm-up per algorithm, and three timed repetitions in
pair order PFQ-v1/PFQ-v2, PFQ-v2/PFQ-v1, then PFQ-v1/PFQ-v2. Both arms run the
shared successful-delivery ledger and the same aggregate diagnostic level; only
predictor-specific work differs. The wall-time increase is
`100 * (median_v2 / median_v1 - 1)`. The result also reports scheduler
p50/p95/p99:

- end-to-end increase at or below 10%: pass;
- above 10% and at or below 20%: optimize caching or batched LUT processing,
  then rerun the gate;
- above 20%: do not start the formal 10-seed promotion run.

Timing runs disable detailed candidate logging and use the same diagnostic
level for PFQ-v1 and PFQ-v2.

## 15. Interpretation boundaries

PFQ-v2 is still a native-order greedy scheduler. A positive result supports
the value of allocation-set-aware marginal expected delivery in the scoped
Stage-B scenario; it does not establish global optimality or generalization to
other topologies, loads, PRG sizes, channel models, or HARQ configurations.

A negative result remains useful. The cap-only arm, calibration records,
nonpositive-marginal counters, blank-PRG rate, and per-seed deltas distinguish
among an inaccurate success predictor, overaggressive demand saturation,
reliability/throughput trade-offs, and the limitations of native PRG order.

Packet completion-aware bonuses and PRG-order optimization are deliberately
left for later, separately gated experiments so their effects do not confound
PFQ-v2.

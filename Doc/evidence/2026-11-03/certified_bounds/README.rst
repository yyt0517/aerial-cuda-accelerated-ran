MMSE-free bounds and canonical lazy selection
===========================================

This stage performs an offline diagnostic on twelve already paid causal
states. It does not run PHY, MMSE, the native simulator or model fitting.
``analyze_certified_bounds`` blocks NativePhy/ReferencePhy/PersistentPhy
constructors/evaluators and NumPy solve/inverse. Existing exact per-UE root
values are inputs, not new labels. Lookup-only BLER branch probes are counted
as table checks, not candidate PHY evaluations.

``certified_bounds.logical_bounds`` recomputes count, minimum layer and TBS
for one-PRG edits, using fixed decision-time MCS. Positive weights and native
BLER in[0,1] imply expected bytes <=min(queue,TBS). The per-UE integer bound
uses the SAME nearest-even contribution definition as persistent_q1e6_v1:

    UB = sum_affected(round_even(weight * dequeue_cap * 1e6) - root_q)

Positive floating multiplication and rounding are monotone. No whole-J
floating re-reduction or ad hoc delta rounding is substituted. Unaffected UE
contributions cancel exactly. Logical dequeue gives an exact cheap floor test.
The dense arrays in this OFFLINE implementation are a diagnostic convenience;
the production persistent sparse table and ``_persistent.so`` are unchanged.

Interference monotonicity has two different proof obligations. In the current
activity-only model, R2=R1+Hc*Hc^H implies R2^-1<=R1^-1, so MMSE error covariance
increases and each post-MMSE SINR decreases when interference is added. The
victim's assignment/layer/MCS/signal must remain fixed. All28 native BLER
lookup functions are nonincreasing on their reachable branches. MCS3's SNR
AXIS is out of order; sorting it would change the function. The actual scan
still has nonpositive slopes/jumps. Existing root transitions provide43
monotone victim comparisons with zero observed violation. This does not give
a rigorous finite-error bound for Eigen LDLT/inverse. The add-only-interference
tightening is therefore marked THEORY ONLY. The capacity bound does not need
MMSE monotonicity and is the proposed first finite-kernel certificate.

The max-anchored one-integer-unit tie needs a key-aware stop rule. Let m be
the maximum currently exact qualifying gain and k the smallest known key in
[m-1,m]. An unresolved eligible candidate can still matter if its upper bound
is >=m, or equals m-1 with a smaller key. Other candidates whose UB<=threshold
are rejected for free. If no exact candidate exceeds the threshold, all
remaining upper bounds must be <=threshold to certify convergence. Example:
known(key20,gain100), unseen(key10,UB99) cannot stop at100>=99+1; key10 may win.
The ``lazy_canonical`` function is a correctness reference. It accepts arbitrary
finite priorities, but only valid current bounds and current exact cache entries
enter its certificate. Stale delta labels are priorities only. It is not an
integrated heap implementation or a benchmarked online solver.

Reproduce the diagnostic and no-PHY tests, using fresh output directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_certified_bounds > /tmp/certified_bounds_20261103_tests.txt 2>&1
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.analyze_certified_bounds --output deliverables/certified_bounds_offline_NEW

Defaults read retained ``anytime_dpls_20261103_smoke_v3`` states,
``canonical_cached_dpls_gate_20261103_v1`` roots,
``persistent_integer_gate_20261103_v1`` choices and the old cached-budget
quantization diagnostic. Warm-init score gaps/floor checks and zero changed
quantized choices prove the stored roots also form the new integer path.
All378 rounds and624670 candidate bounds are covered. However, complete
candidate integer scores were NOT retained. Accepted gain b is known from
successive per-UE roots; the true maximum lies in[b,b+1]. Both endpoint counts
are reported. Confirmation uses UB>threshold, not an invented subthreshold max.

Offline result: among floor-feasible candidates, capacity-UB>=maximum occurs
for1.309--1.335% on first scans and3.800--3.828% in inner accepted rounds.
Confirmation still has6.344% with UB>threshold. There is no zero-fresh
confirmation certificate in the12 recorded terminal roots.

For UB-priority descending, at least one true-max candidate is evaluated before
any candidate with UB<max-1. Thus the count of all eligible UB>=b-1 (retaining
the strict acceptance threshold), plus confirmation UB>threshold, is a
conditional upper bound on evaluations even WITHOUT assuming any cross-round
cache reuse. The saved path yields23718 +20 initializations =23738,3.800% of
full and12.966% of the existing cache's candidate evaluations. This is a proof
about candidate counts given valid bounds/full domain/certified stopping, not
an executed lazy trajectory or wall-time promise. It is not a product of cache
and pruning percentages. A different priority such as stale delta or a model
does not inherit this UB-order work bound, but the same certificate preserves
the final canonical choice.

Next implementation must put cheap sparse bounds and heap generations into
a separately versioned native entrypoint, retaining every novel/dirty edit
until certified. Compare UB order and stale-delta order against the unchanged
integer full/cache baseline, pay for the complete equivalence oracle, and
measure serial wall time with audits/profiling off. Count UE evaluations,
MMSE requests/misses, bound updates and heap overhead separately; a candidate
count is not an MMSE inversion count. Only after this gate should the retained
161700 labels be considered for learned ordering. No budget stop may bypass
the certificate. Existing cap4 failures remain unchanged; future trajectory
approximations first need canonical/random-tie full native controls under a
separate frozen protocol.

The archive generator reads the fixed completed v4 diagnostic, checks old
labels/default registry hashes, copies compact logs and writes a report::

    .venv/bin/python -m training.stageb_lns.report_certified_bounds --output Doc/evidence/NEW/certified_bounds

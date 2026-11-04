Cross-load fixed weights, literature forms and conditional regime imitation
=========================================================================

This is a NEW November4 protocol. It does not append coefficients to or
rewrite the previous0/.25/.5/1/2 sweep. Grid0/1/2/4/8 times loads1/1.4/1.6
times development seeds41--43 gives45 points. Thirteen identical old points
are hash-verified/reused;32 new fixed points and18 literature-form points
are run. Every new point uses4000 common PFQ warmup,4000 settling,4000 measured
TTIs and400 mature-cohort followup, or12400 native steps. Fifty new runs cost
620000 steps. Existing161700 ranking labels are never used.

All scientific arms use the SAME original exact ``_bounded.so``. No SoA,
tournament, new numeric kernel, approximate scoring or different floor is
substituted. The original canonical objective thresholds and bit contracts
remain. Coefficients are frozen before outcomes. Parallel policy wall times
are not speed evidence, and no default/final confirmation changes occur.

Baselines and literature attribution
-----------------------------------

The linear urgency term is a delay-sensitive weighting rule, not a claimed
new learning contribution. Read-only FIFO features preserve the original
expiry accounting; no extra before() call or future packet outcome is read.

Andrews et al., Providing Quality of Service over a Shared Wireless Link,
IEEE Communications Magazine39(2),150--154,2001, defines M-LWDF and describes
using delay target and loss tolerance to set its weights. Author-hosted PDF:
https://9p.io/who/stolyar/varchannel_commag.pdf .

Biernacki and Tutschku, Comparative Performance Study of LTE Downlink
Schedulers, Wireless Personal Communications74,585--599,2014(published2013),
describes the EXP/PF combination with average achieved throughput:
https://link.springer.com/article/10.1007/s11277-013-1308-4 .

The alpha-scaled HOL EXP/PF form is also explicitly given in the comparison
experiment by Ang et al., A performance analysis on packet scheduling
schemes based on an exponential rule for real-time traffic in LTE,2015:
https://link.springer.com/article/10.1186/s13638-015-0429-8 .

We adapt the RT weights to the common joint expected-byte local solver,
not reproduce either paper's RB scheduler, mixed-service controller or its
optimality guarantee. All flows here share400TTI TTL and PLR target.01.
Let age=max(1,400-HOLslack), a=-ln(.01)/400, PF=1/max(EMA,1e-9).
M-LWDF-PF weights are PF*a*age. EXP-PF weights are
PF*exp((a*age-x)/(1+sqrt(x))), x=active mean(a*age). Normalize to mean1.
One-slot age floor keeps fresh/empty queues finite; empty queues have no
positive service term. The EMA remains .995/.005 successful service bytes.
These literature arms use an uncapped PF denominator, whereas the linear
grid keeps the original clipped deficit base. Their difference therefore
does not isolate only the urgency shape. No paper parameters are fitted.

Predeclared analysis and conditional fit
--------------------------------------

Report the full four-objective Pareto set per load. For a single oracle
comparison retain fixed coefficients with mean payload>=99.7% the best
mean at that load, then lexicographically minimize worst(seed,UE) expiry,
pooled expiry, successful P95, negative payload, coefficient. Report the
train-only41/42 oracle separately from the descriptive three-seed oracle.

A universal near-optimal fixed coefficient must satisfy the protocol's
per-coordinate tolerances at every load; if no point can satisfy competing
coordinate minima, say so instead of calling an unachievable utopia optimal.
Residual high-load losses alone do not establish that an adaptive policy
could remove them. Different oracle labels allow the small development fit;
constant labels provide no useful classifier target and no fit is launched.
A fixed coefficient satisfying the frozen near-optimal tolerances at every
load also blocks fitting, even if the literal oracle argmin differs slightly.

Only after the45-point grid completes, fit the predeclared depth2 tree,
min_samples_leaf256, on settlingTTIs4001--8000 of NEW fixed-policy runs
from41/42. Balance loads. Inputs are queue mean/P95, active fraction,
HOL urgency mean/P95 and mean successful-service EMA, normalized as specified
in code. No rate/seed/oracle/future outcome input is allowed. Labels are the
train-only perload fixed oracle. These are regime labels, not per-TTI
counterfactual optimal-action labels. No reward or model retuning follows.
Closed-loop comparison to fixedk2 and train-only oracle decides usefulness.
Seed43 is excluded from labels/fit but remains an already exposed development
seed, so no final generalization claim follows even from success.

Run in fresh directories::

    .venv/bin/python -m pip install --no-deps -r training/stageb_lns/crossload_requirements.txt
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.crossload_campaign --smoke --output deliverables/crossload_smoke_NEW --workers 3 --cpu 253
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.crossload_campaign --output deliverables/crossload_NEW --workers 12 --cpu 244
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_crossload --source deliverables/crossload_NEW --output deliverables/grid_analysis_NEW --grid-only
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.crossload_learning --source deliverables/crossload_NEW --grid deliverables/grid_analysis_NEW --output deliverables/adaptive_NEW --workers 4 --cpu 236
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_crossload --source deliverables/crossload_NEW --output deliverables/analysis_NEW --cpu 243
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_crossload_learning --source deliverables/crossload_NEW --learning deliverables/adaptive_NEW --output deliverables/adaptive_analysis_NEW --cpu 242

The full analyzer checks common offered traffic, PFQ prefixes, input hashes,
native/FIFO execution and cohort membership. Four weighted saved states per
new point pay an independent full-integer oracle:200 states for50 new arms,
plus36 states if the nine adaptive runs are launched. Reused audits are not
charged again. Payload averages seeds; expiry and successful P95 pool mature
packets; worst expiry maximizes over(seed,UE). Keep every point and seed.

Last SoA engineering attempt
---------------------------

Separate AVX512 SoA code computes all4 masks per(UE,PRG), eight independent
keys per vector, with pivoted LDLT, triangular solves and cofactor inverse.
The SIMD and original-Eigen translation units compile separately; both
disable FMA contraction/fast math, and the original reference retains its
original architecture flags. No Woodbury or reciprocal approximation.

The explicit formulas still need a bitwise gate against Eigen's internal
packet/reduction ordering. On the completed252-state test the prototype
failed this gate, despite zero FMA instructions. It never enters a solver
or the scientific campaign. This is a failed implementation attempt, not
a proof that every possible SoA implementation is inexact or unhelpful.
Per user instruction this ends the40% engineering effort; retain the exact
34%/41%/46% saved-state cost curve and the old measured-TTI tails. That curve
uses the previously saved k0 deficit-weight inputs and same-architecture
native full denominator; it is not a cost guarantee for every new weight arm. No new
deployment speed or quality claim follows from a non-equivalent prototype::

    .venv/bin/python -m training.stageb_lns.build_gamma_soa
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.audit_gamma_soa --output deliverables/gamma_gate_NEW --cpu 243

Focused behavioral checks::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_bounded_dpls training.stageb_lns.test_certified_bounds training.stageb_lns.test_tournament training.stageb_lns.test_crossload_weights

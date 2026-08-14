# PFQ-Independent Rescue-Cohort Diagnostic Design

**Date:** 2026-08-14
**Mode:** Research-first, behavior-invariant diagnostic
**Status:** Approved in conversation; pending written-spec review

## Research question

Why did deterministic hierarchical decoding eliminate fragmentation Blank
decisions and improve delay and per-UE rates, yet reduce goodput for the frozen
A8 PFQ-independent checkpoint in the cooperative boundary seed-41 scenario?

The completed formal comparison reported:

- flat A8 goodput: `1547.837 Mbps`;
- hierarchical goodput: `1534.51175 Mbps`;
- goodput delta: `-13.32525 Mbps`;
- scheduled-capacity delta: `-9.89875 Mbps`;
- failed-transmission delta: `+3.42650 Mbps`;
- BLER: `0.10611020 -> 0.11047333`;
- rescued counterfactual fragmentation decisions: `4728 / 4728`;
- selected fragmentation Blank decisions after hierarchical decode: `0`;
- expiry-aware delay: `2.86602 -> 1.16668 ms`;
- UE minimum expiry-aware rate: `1.61367 -> 3.59939 Mbps`.

The result supports probability fragmentation as a real mechanism but falsifies
decode-only hierarchy as a goodput-preserving deployment change for this frozen
checkpoint and scenario. The next smallest question is whether rescued service
has lower expected delivery value than ordinary service.

## Core hypothesis

`H_rescue_quality`: UE-by-TTI observations planned for at least one rescued PRG
have lower expected successful service value than UE-by-TTI observations planned
for non-Blank service with no rescued PRGs.

The loss can arise through three distinguishable paths:

1. **service-value path:** fewer predicted or actual scheduled bytes per PRG;
2. **reliability path:** a larger failed-byte fraction or TB error rate;
3. **acceptance path:** a smaller planned-to-accepted-to-actual PRG retention
   ratio.

The first two paths are primary. The acceptance path is a fail-closed alternative
that redirects investigation to the Type-0/PDSCH acceptance chain rather than to
a new policy rule.

This is a descriptive mechanism test on one checkpoint, scenario, and seed. It
does not identify the causal contribution of an individual PRG and does not
establish multi-seed generalization.

## Evidence boundary and unit of analysis

The simulator exposes transmission outcomes per UE per TTI:

- actual PRG count;
- scheduled and goodput bytes;
- TB transmission and error counts;
- accepted pre-PDSCH PRG count.

It does not expose a PHY outcome for each individual scheduled PRG. A UE can
receive both rescued and ordinary PRGs in the same TTI, so assigning the UE's
aggregate TB result to one rescued PRG would be false precision.

The authoritative unit of analysis is therefore **UE x TTI**. The cohort
universe contains every UE with at least one planned non-Blank PRG, including a
UE whose plan is later entirely rejected:

- `rescue_exposed`: a UE planned for at least one rescued PRG in that TTI;
- `ordinary_service`: a UE planned for non-Blank PRGs but zero rescued PRGs;
- `rescue_share`: rescued planned PRGs divided by all planned PRGs for the UE.

Per-decision values may be summarized within a UE x TTI observation, but
post-decode outcomes must remain at UE x TTI resolution. Results must use
association language, not per-PRG causal attribution.

## Approaches considered

### Existing-output-only analysis

Rejected as insufficient. Existing summaries contain overall rescue counts and
aggregate transmission metrics but not the rescued UE identity, decision-time
SINR/risk, remaining demand, or within-TTI rescue exposure.

### Policy-side diagnostic joined to existing UE outcomes

Selected. Reuse the existing independent-policy and evaluator paths. Record only
decision-time values already available in the graph and join them after the
environment step to the existing per-UE outcome arrays. This is the shortest
path to falsifiable evidence and requires no simulator or strategy change.

### Per-PRG simulator/PHY tracing

Deferred. It would give finer attribution but requires a larger C++/protocol
change and is justified only if the UE x TTI result is inconclusive.

## Minimum diagnostic delta

Do not change actor parameters, graph contents, logits, masks, decode decisions,
sequential demand updates, reward, simulator, or random-number consumption.

For each selected non-Blank decision, collect the selected UE and:

- counterfactual flat-fragmentation and rescued indicators;
- local PRG index, grouped as early `0..5`, middle `6..11`, or late `12..16`;
- estimated candidate service bytes from the existing
  `estimate_candidate_service_bytes` calculation;
- base UE-PRG edge values: quality dB, quality gap dB, and PRG risk;
- UE `tb_err_last`;
- remaining-demand ratio immediately before the decision;
- `p_blank`, aggregate `p_service`, and `p_best_ue`;
- the existing read-only success proxy:

```text
sigmoid((quality_db
         - 0.35 * quality_gap_db
         - 7.5 * prg_risk
         - 6.0 * ue_tb_err_last) / 4.0)
```

Aggregate those values by selected UE within the TTI. After the simulator step,
join the aggregate to:

- planned PRG count;
- accepted pre-PDSCH PRG count;
- actual PRG count;
- scheduled, goodput, and failed bytes;
- TB transmission and error counts.

Only the formal measured window contributes to the report. Warmup data may drive
state but must not enter cohort statistics.

## Primary metrics

For each cohort and for the difference between cohorts, report:

- predicted service bytes per planned PRG;
- actual scheduled bytes per actual PRG;
- actual goodput bytes per actual PRG;
- failed bytes divided by scheduled bytes;
- TB errors divided by TB transmissions;
- planned-to-accepted and accepted-to-actual PRG retention;
- quality, quality gap, PRG risk, UE historical TB error, success proxy, and
  remaining-demand ratio;
- UE x TTI count, TTI coverage, PRG count, byte sums, and TB count denominators.

All ratios must be derived from summed numerators and denominators, not averages
of per-row ratios. Empty denominators must be reported explicitly rather than
silently converted into evidence.

To reduce obvious allocation-size confounding, repeat service and reliability
outcomes for the positive actual-PRG-count strata `1..2`, `3..5`, and `>=6`.
Keep `actual_prg_count == 0` as a separate acceptance outcome; do not invent
per-PRG service ratios for it. Also report per-TTI cohort differences and the
fraction of populated TTIs whose difference has the same direction as the
overall result. These are effect-size and consistency diagnostics, not
independent-sample significance claims.

## Demand-depletion auxiliary analysis

For each cell and TTI, record whether selected forced or conservative Blank
decisions occur after the cell's first rescue, plus the rescue local-PRG phase
and remaining-demand ratio. Compare these counts with cells that have not yet
rescued a PRG.

This analysis is descriptive only. A later Blank may share state with earlier
actions, but the available data cannot identify one rescue as its cause. This
auxiliary result cannot by itself select a policy change.

## Validity checks

The diagnostic is valid only if all of the following hold:

1. diagnostics off/on produce identical choices, Type-0 actions, log-probability,
   and entropy for sample, flat greedy, and hierarchical greedy paths;
2. PFQ-field deletion equality remains exact;
3. selected UE identities and planned counts reconstruct the materialized Type-0
   action;
4. per-UE planned, accepted, and actual counts are non-negative and have the
   expected shape;
5. scheduled and goodput byte arrays are finite and non-negative;
6. exact measured-window identity holds:
   `failed_bytes = scheduled_bytes - goodput_bytes`;
7. checkpoint, simulator, scenario, seeds, traffic/TTL settings, warmup, and
   measured window match the frozen formal treatment;
8. illegal-action and replay-mismatch counts are zero;
9. protected repository-root simulator artifacts remain byte-identical.

A diagnostic join or denominator inconsistency is a validity failure, not a
zero-valued result.

## Minimum experiment

### Focused local checks

Add only enough synthetic coverage to demonstrate:

- rescue and ordinary selected decisions aggregate to the correct UE x TTI;
- mixed rescued/ordinary PRGs remain one rescue-exposed UE x TTI and are not
  falsely attributed per PRG;
- the outcome join and summed-ratio calculations are exact;
- the zero-actual-PRG acceptance bucket is retained, and the three positive
  actual-PRG-count strata have disjoint, exhaustive membership among UEs with
  actual service;
- diagnostics do not change policy behavior.

### Live smoke

Run the frozen A8 checkpoint in the cooperative boundary seed-41 scenario with
`hierarchical_argmax`, `2` warmup TTIs, and `4` measured TTIs in an isolated
directory. Require evaluator and simulator success, expected decision count,
non-empty cohort output when observed, exact byte accounting, zero illegal and
replay mismatches, and unchanged protected-root hashes.

### Formal diagnostic

If the smoke passes, run exactly one treatment:

- checkpoint: frozen A8, SHA-256
  `c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc`;
- simulator SHA-256:
  `3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c`;
- scenario: cooperative boundary;
- topology and traffic seed: `41`;
- decode: `hierarchical_argmax`;
- warmup: `1000` TTIs;
- measured window: `512` TTIs (`1001..1512`);
- expected policy decisions: `26112`.

Do not rerun flat A8, A4, B4, PFQ, another checkpoint, or another seed. The
formal treatment is rerun only because the earlier output lacks the required
decision-to-UE diagnostic join.

## Interpretation and next-action rule

Classify the result using the following predeclared order:

1. **Reliability mechanism supported:** rescue exposure has a worse success-risk
   profile and worse observed failed-byte or TB-error ratio, with the same
   direction in at least two populated PRG-count strata. The only next candidate
   experiment is reliability-weighted aggregate Serve mass with the actor
   frozen.
2. **Service-value mechanism supported:** reliability is comparable, but rescue
   exposure has lower predicted and actual scheduled bytes per PRG, again with
   consistent populated strata. The only next candidate experiment is
   service-value-weighted aggregate Serve mass with the actor frozen.
3. **Acceptance mechanism supported:** planned-to-accepted or
   accepted-to-actual retention is the dominant degradation. Stop policy changes
   and inspect the Type-0/PDSCH acceptance path.
4. **Inconclusive:** directions conflict, denominators are inadequate, or the
   primary aggregate effect disappears within allocation-size strata. Stop and
   consider per-PRG simulator tracing under a separate design.

No branch is treated as multi-seed evidence. Delay and UE-rate improvements are
reported as context but cannot override a goodput loss.

## Stop conditions

- Stop immediately on any validity failure; do not interpret cohort metrics.
- Stop after the one formal diagnostic produces a trustworthy classification.
- Do not automatically implement the indicated weighting, change reward, train,
  rerun another seed, or add simulator tracing.
- A subsequent intervention requires a separate hypothesis and approval.

## Explicit non-goals

This experiment does not:

- alter hierarchical decisions or deploy the decode-only hierarchy;
- add candidate-risk features to the independent actor or checkpoint;
- use PFQ fields, PFQ residuals, or same-stage PFQ in the online reward;
- change reward composition or make delay/UE rate primary objectives;
- train or fine-tune the actor;
- modify C++ simulator/protocol outputs;
- claim individual rescued-PRG causality;
- run multiple seeds, checkpoints, baselines, thresholds, or weight sweeps;
- introduce a generalized telemetry framework or production configuration.

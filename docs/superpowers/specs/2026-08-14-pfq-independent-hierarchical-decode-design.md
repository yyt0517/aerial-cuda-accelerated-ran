# PFQ-Independent Hierarchical Decode Experiment Design

**Date:** 2026-08-14
**Mode:** Research-first, single-mechanism experiment
**Status:** Approved in conversation; pending written-spec review

## Research question

Can the existing A8 PFQ-independent actor recover probability-mass fragmentation
without losing goodput if deterministic decoding first chooses between aggregate
Serve mass and Blank, then chooses the best valid UE?

The completed seed-41 boundary diagnostic found:

- A8 goodput: `1547.837 Mbps`;
- selected Blank ratio: `0.1623391544`;
- forced Blank ratio: `0.0010723039`;
- learned Blank count: `4211`;
- fragmentation share among learned Blank: `0.9959629542` (`4194 / 4211`);
- checkpoint SHA-256:
  `c3f2399f94e4dc8930f9b17313acff0727d3a8bdc916ba3cc31e88665a1d92cc`;
- simulator SHA-256:
  `3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c`.

This supports a narrow causal hypothesis: A8 often chooses Blank because the UE
service probability is split across many flat classes, not because aggregate
Blank mass is dominant.

## Hypothesis and falsification rule

`H_hier_decode`: replacing only deterministic flat argmax with grouped
Serve/Blank argmax will eliminate actual fragmentation Blank decisions while
preserving or improving the frozen A8 goodput.

The hypothesis is supported only if the formal treatment satisfies all hard
gates and measured goodput is at least `1547.837 Mbps`. It is falsified for this
checkpoint/scenario if goodput is lower, even if Blank frequency decreases.

This is a mechanism test on one checkpoint and one seed. It is not evidence of
multi-seed generalization or a production-ready action-head architecture.

## Minimal intervention

Add one deterministic decode mode, `hierarchical_argmax`, to the existing
PFQ-independent policy path. Reuse the frozen A8 actor, encoder, logits,
candidate construction, masks, checkpoint, reward, and simulator unchanged.

For each reuse-group row with adjusted logits `z` and valid UE set `V`:

```text
if V is empty:
    select Blank
else:
    serve_logit = logsumexp(z[u] for u in V)
    blank_logit = z[Blank]
    if serve_logit > blank_logit:
        select argmax(z[u] for u in V)
    else:
        select Blank
```

Equality remains conservative Blank, matching the diagnostic definition
`p_blank >= p_service`. Invalid UE logits must never contribute to
`serve_logit`. The selected choice continues through the existing Type-0
materializer and sequential remaining-demand update.

The existing `argmax` and `sample` modes remain unchanged. The new mode is valid
only for the independent action head and fails closed for incompatible actors or
malformed/nonfinite distributions. It must not introduce repair, safe-fill,
fallback actions, new random draws, or trainable parameters.

## Probability and PPO identity

This experiment performs deterministic evaluation only, but the grouped
factorization must retain the original flat probability semantics:

```text
P(Serve) = sum(P_flat(u) for u in V)
P(u | Serve) = P_flat(u) / P(Serve)
P(Serve) * P(u | Serve) = P_flat(u)
```

Equivalently, the joint hierarchical UE log-probability equals the existing flat
UE log-probability. The flat entropy also equals the hierarchical joint entropy
`H(Serve/Blank) + P(Serve) * H(UE | Serve)`. Tests must establish these identities
without changing the training path. No PPO training is run in this experiment.

## Diagnostic evidence

For each decision, retain the authoritative flat argmax as a counterfactual and
record:

- flat-greedy Blank;
- hierarchical selected Blank;
- changed-action indicator;
- fragmentation-rescue indicator: flat fragmentation Blank followed by a
  hierarchical UE selection;
- selected forced, conservative, and fragmentation Blank indicators.

The formal run must demonstrate that selected fragmentation Blank count is zero.
Counterfactual flat-fragmentation and rescue counts must agree for every valid
fragmentation opportunity.

Scheduled, goodput, failed-transmission bytes, TB error counts, packet expiry,
expiry-aware delay, and per-UE expiry-aware rates remain measured over the same
formal window. Bytes cannot be attributed exactly to an individual rescued PRG
because transport-block accounting is aggregated after the full scheduling
decision. Therefore the experiment reports rescue counts plus overall treatment
byte deltas; it must not label aggregate byte changes as per-rescue causal bytes.

## Minimum experiment

### Unit and path checks

Add only tests needed to protect the conclusion:

1. fragmentation case: flat argmax is Blank, aggregate Serve wins, best valid UE
   is selected;
2. conservative case: aggregate Blank wins and remains Blank;
3. no-valid-UE case: Blank is forced;
4. tie case: Blank is retained;
5. invalid UE classes do not enter `logsumexp`;
6. flat `argmax` and `sample` paths are bitwise unchanged;
7. hierarchical UE joint log-probability and entropy equal the flat categorical
   values within the existing numerical tolerance;
8. replay, Type-0 legality, sequential valid masks, and PFQ-deletion equality
   remain valid.

### Live smoke

Run A8 with boundary topology/traffic seed `41`, deterministic
`hierarchical_argmax`, `2` warmup TTIs, and `4` measured TTIs in an isolated
evaluation directory. Require:

- evaluator and simulator return zero;
- `4 * 51 = 204` decisions;
- zero illegal actions and replay mismatches;
- exact `failed_bytes = scheduled_bytes - goodput_bytes`;
- no selected fragmentation Blank;
- simulator side outputs remain inside the evaluation directory;
- protected repository-root artifact hashes do not change.

### Formal treatment

If the smoke passes, run exactly one treatment:

- subject: frozen A8 checkpoint;
- scenario: cooperative boundary;
- topology and traffic seed: `41`;
- decode: `hierarchical_argmax`;
- warmup: `1000` TTIs;
- measured window: `512` TTIs (`1001..1512`);
- expected decisions: `512 * 51 = 26112`.

Reuse the existing deterministic A8 flat-argmax result as the control only after
verifying checkpoint SHA, simulator SHA, seeds, traffic/TTL settings, warmup, and
measurement window are identical. Do not rerun A4, B4, PFQ, or another seed.

## Decision gates

Hard validity gates:

- checkpoint and simulator identities match the frozen control;
- `requires_pfq_features == false`;
- PFQ-field deletion equality is true;
- zero illegal actions and replay mismatches;
- exactly `512` measured TTIs and `26112` decisions;
- byte accounting identity holds;
- selected fragmentation Blank count is zero;
- each counterfactual fragmentation opportunity is rescued.

Primary performance gate:

```text
hierarchical_goodput_mbps >= 1547.837
```

Secondary diagnostics are reported but are not reward terms or pass/fail gates:

- selected/forced/conservative Blank ratios;
- scheduled and failed-transmission Mbps;
- TB transmissions, errors, and BLER;
- expired packets/bytes and expiry-aware delay;
- UE minimum, P10, macro, and distributional expiry-aware rate metrics.

## Stop conditions

- If a validity gate fails, stop and debug the experiment; do not interpret KPI
  changes.
- If goodput is below `1547.837 Mbps`, reject decode-only hierarchy for A8 seed 41
  and inspect whether extra service raised failed transmissions or BLER. Do not
  automatically tune a Blank bias, reward, or model.
- If goodput meets the hard floor, report the mechanism as supported for this one
  scenario and stop. Multi-seed evaluation and a trainable two-head actor require
  separate approval.

## Explicit non-goals

This experiment does not:

- change or add reward terms;
- compare against same-stage PFQ in the reward;
- retrain or fine-tune A8;
- add a trainable Serve/Blank head;
- force-fill Blank decisions;
- add entropy schedules, Blank penalties, thresholds, or bias sweeps;
- run A4, B4, PFQ, multiple seeds, or hyperparameter searches;
- build a generalized decoding framework or production configuration system.

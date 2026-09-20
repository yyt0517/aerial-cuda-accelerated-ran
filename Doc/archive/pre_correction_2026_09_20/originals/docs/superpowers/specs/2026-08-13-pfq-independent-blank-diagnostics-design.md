# PFQ-Independent Blank-Mass Diagnostics Design

## 1. Goal

Add an observation-only diagnostic to determine why the PFQ-independent actor
selects Blank under deterministic deployment. The diagnostic must distinguish
three cases:

1. Blank is forced because no UE candidate remains valid.
2. Blank has greater probability mass than all UE actions combined, so the
   policy is genuinely conservative.
3. Blank is the largest single class, but the combined UE service probability
   is greater than Blank. This is categorical probability-mass fragmentation.

The diagnostic answers one narrow research question before any policy change:

> Is the deterministic Blank rate mainly caused by a learned preference for
> no transmission, or by converting a flat stochastic UE/Blank categorical
> into a single-class argmax?

The experiment remains in the canonical 3-cell `coop_boundary` scenario and
retains PFQ independence. Formal deployment remains deterministic.

## 2. Current evidence and causal boundary

The seed-41 formal window is TTI 1001 through 5000. PFQ and A8 both report zero
expired packets and zero expired bytes. Therefore A8's goodput gain cannot be
attributed to delivering packets that PFQ would have expired in this run.

The measured comparison is:

| Metric | PFQ | A8 |
|---|---:|---:|
| Goodput | 1537.906 Mbps | 1554.168 Mbps |
| TB transmissions | 97,368 | 85,651 |
| TB errors | 10,796 | 8,879 |
| TB BLER | 11.0878% | 10.3665% |
| Expired packets | 0 | 0 |

The full-run native logs show nearly equal scheduled throughput, 1729.852 Mbps
for PFQ and 1728.823 Mbps for A8, while the scheduled-minus-goodput gap is
199.676 Mbps for PFQ and 185.120 Mbps for A8. The current evidence therefore
attributes A8's goodput advantage primarily to fewer failed transmission bytes
and more efficient UE/PRG/reuse choices, not expiry avoidance.

A8 still has uneven UE service. In the formal outcome sample, 20.03% of packets
have system time above 5 ms and account for 61.22% of total packet system time.
There are no terminal outcomes above 21 ms, far below the 200 ms TTL. This is a
service-rhythm/fairness symptom, not evidence of near-expiry rescue.

## 3. Non-goals

This diagnostic does not:

- change actor logits, masks, action sampling, or greedy selection;
- add a Blank bias, Blank budget, forced fill, safe fill, or fallback;
- change PPO reward, entropy coefficient, optimizer, or checkpoint selection;
- add a `Serve/Blank -> UE` hierarchical policy;
- run a reward or threshold sweep;
- treat a TB error as a terminal packet expiry;
- claim that a one-step greedy shadow is a full counterfactual greedy rollout;
- alter the simulator binary or packet lifecycle.

Any policy improvement is a separate experiment selected only after this
diagnostic falsifies or supports the probability-fragmentation hypothesis.

## 4. Decision-level definitions

For one cell and local PRG, let `V` be the currently valid UE classes after the
existing association, schedulability, demand, and sequential headroom masks.
Let `b` be the always-legal Blank class. Probabilities use the exact adjusted
logits already used by the policy:

```text
p = softmax(masked_logits)
p_blank = p[b]
p_service = sum(p[u] for u in V)
p_best_ue = max(p[u] for u in V), or 0 if V is empty
greedy_blank = argmax(masked_logits) == b
blank_margin = blank_logit - max(valid_ue_logits), when V is nonempty
```

The diagnostic classifications are:

```text
forced_blank = greedy_blank and |V| == 0
learned_blank = greedy_blank and |V| > 0
fragmentation_blank = learned_blank and p_service > p_blank
conservative_blank = learned_blank and p_blank >= p_service
```

Because `p_service = 1 - p_blank` when at least one UE is valid,
`fragmentation_blank` is equivalently a Blank argmax with `p_blank < 0.5`.
The explicit mass comparison is retained in output because it communicates
the scheduling interpretation directly.

For stochastic rollout, the existing sampled action remains authoritative.
The diagnostic also computes the argmax of that same decision distribution.
This is a one-step greedy shadow conditioned on the sampled trajectory's
current sequential state. It must not update a second greedy sequential state
or affect the environment action.

For deterministic evaluation, the selected action and greedy shadow are the
same action by definition.

When `V` is empty, the stored `blank_margin` tensor value is zero and a
separate validity mask excludes that row from margin distributions. It is not
interpreted as a zero-margin decision.

## 5. Recorded metrics

### 5.1 Per-decision tensors

The independent policy trace carries detached, observation-only values for
each of the 51 Cell-PRG decisions:

- valid UE count;
- `p_blank`;
- `p_service`;
- `p_best_ue`;
- `blank_margin`;
- sampled/selected Blank indicator;
- greedy-shadow Blank indicator;
- forced, learned, fragmentation, and conservative Blank indicators.

These tensors do not participate in autograd, PPO replay log-probability,
entropy, or action materialization. PPO update replay does not compare old
diagnostic probabilities with the updated policy: those probabilities are
expected to change after an optimizer step. Existing stored-choice replay and
action equality checks remain authoritative.

### 5.2 Rollout and evaluation aggregates

Training CSV and deterministic evaluation summary add:

- selected Blank ratio;
- greedy-shadow Blank ratio;
- forced Blank ratio;
- learned Blank ratio;
- fragmentation Blank ratio;
- conservative Blank ratio;
- fragmentation share among learned Blank decisions;
- mean valid UE count;
- mean and p10/p50/p90 for `p_blank`;
- mean and p10/p50/p90 for `p_service`;
- mean and p10/p50/p90 for `p_best_ue`;
- mean and p10/p50/p90 for `blank_margin`.

Ratios use all Cell-PRG decisions as the denominator unless the field name
explicitly says `among_learned_blank`. When there are no learned Blank
decisions, that conditional share is reported as zero together with a count of
zero; it is not NaN.

Blank classifications in stochastic rollout are explicitly prefixed
`shadow_` because they describe the one-step greedy shadow. Deterministic
evaluation may expose the same aggregates without ambiguity because selected
and shadow-greedy choices are identical. Margin aggregates also report their
valid-decision count and exclude forced-Blank rows.

### 5.3 Exact failed-transmission accounting

The deterministic evaluation summary also exposes the measured-window values:

```text
scheduled_bytes = sum(actual_ue_served_bytes)
goodput_bytes = sum(actual_ue_goodput_bytes)
failed_bytes = max(0, scheduled_bytes - goodput_bytes)
failed_mbps = mean_failed_bytes_per_tti * 0.016
```

The factor `0.016` converts bytes per 0.5 ms TTI to Mbps. These fields are
reported alongside the existing measured TB transmission count, TB error
count, and TB BLER. A failed byte/TB is a failed transmission attempt, not a
terminal packet outcome. Delivered and expired remain the only terminal packet
outcomes in the current simulator.

## 6. Data flow and behavior invariance

The implementation observes the existing policy distribution immediately
after masking and before selection:

```text
actor logits
  -> existing validity mask
  -> existing adjusted logits and softmax
  -> diagnostic reductions (detached)
  -> existing sample or argmax selection
  -> existing sequential-state update
  -> existing Type-0 materialization
```

No diagnostic value is read by the selection, sequential update, PPO loss,
critic, reward, or materializer. Computing `argmax` for the shadow must not
consume RNG state. Stochastic selection continues to make exactly one
`torch.multinomial` call per existing decision group.

Collection is controlled by one explicit independent-path diagnostic flag,
disabled by default and enabled by the diagnostic runner. A disabled run does
not allocate or serialize probability tensors. The flag changes observability
only; action invariance is tested across both settings.

The current shared formal A/B checkpoint evaluation must finish before the
policy or evaluator Python files are changed. This prevents later checkpoints
in the already-running sequence from loading a different implementation than
earlier checkpoints.

## 7. Validity checks

Only checks needed to trust the diagnosis are in scope.

### 7.1 Probability identities

- `p_blank`, `p_service`, and `p_best_ue` are finite and in `[0,1]`.
- `p_blank + p_service == 1` within numerical tolerance.
- `p_best_ue <= p_service`.
- `forced_blank` implies zero valid UE candidates and `p_blank == 1`.
- learned, fragmentation, and conservative Blank imply at least one valid UE.
- fragmentation and conservative Blank are mutually exclusive and partition
  learned Blank decisions.

### 7.2 Action invariance

For fixed graph inputs and actor weights:

- greedy choices and final Type-0 arrays are bitwise equal with diagnostic
  collection enabled and disabled;
- sampled choices and final Type-0 arrays are bitwise equal when both runs use
  generators initialized to the same state;
- rollout replay still reproduces the submitted Type-0 action exactly;
- diagnostic collection does not add RNG draws;
- policy log-probability and entropy are unchanged within exact floating-point
  equality where the existing implementation permits it.

### 7.3 Metric-window validity

- diagnostic and failed-byte aggregates honor the same warmup-excluded TTI
  interval as goodput and TB BLER;
- decision counts equal measured TTIs times 51 for complete runs;
- scheduled bytes are no less than goodput bytes up to numerical tolerance;
- `failed_bytes = scheduled_bytes - goodput_bytes` after nonnegative clipping;
- existing replay, legality, PFQ-deletion, and no-fallback gates remain zero or
  exact-true as applicable.

Non-finite probabilities, impossible classification identities, missing UE
served/goodput arrays, or mismatched decision counts fail the diagnostic run.
They must not silently produce zero-valued evidence.

## 8. Minimum experiment

After the current formal A/B sequence completes, run diagnostic-only
evaluations with:

- scenario: canonical 3-cell `coop_boundary`;
- topology and traffic seed: 41;
- deployment: deterministic greedy;
- warmup: 1000 TTIs;
- measured window: 512 TTIs;
- simulator binary: the same already-built corrected binary;
- actor inputs: PFQ fields deleted, as in the formal independent path.

Evaluate:

1. A4, the known severe sample-to-greedy failure, as a positive control;
2. A8, the near-goodput-aligned checkpoint under investigation;
3. the best completed B checkpoint selected from the unchanged formal results,
   to test whether BC's lower entropy reduces fragmentation.

The B checkpoint rule is deterministic: use the formal treatment selection
when it is non-null; otherwise choose the fixed-cadence B checkpoint with the
highest formal goodput among candidates that have zero replay mismatch, zero
illegal actions, and exact PFQ-deletion equality. This fallback selects a
diagnostic subject only and does not promote a gate-failing checkpoint.

The 512-TTI window yields 26,112 Cell-PRG decisions per checkpoint, enough to
estimate Blank composition without repeating the 4000-TTI promotion run. This
is a mechanism diagnostic, not a replacement formal KPI result.

Outputs are stored under a distinct diagnostic directory and must not replace
the original formal gate metrics or checkpoint-selection files.

## 9. Hypothesis test and stop condition

The primary hypothesis is:

```text
H_frag:
  most learned greedy Blank decisions in the problematic checkpoint are
  fragmentation_blank decisions, not conservative_blank decisions.
```

Interpretation:

- Support `H_frag` if more than 50% of learned Blank decisions satisfy
  `p_service > p_blank`, with nonzero learned Blank count. The next research
  candidate is a deterministic hierarchical `Serve/Blank -> UE` policy whose
  PPO log-probability follows the same hierarchy.
- Reject `H_frag` if at least 50% of learned Blank decisions have
  `p_blank >= p_service`. The next investigation stays with the flat policy and
  examines learned conservatism, reward/BC calibration, and service evidence.
- If forced Blank dominates, investigate candidate validity and sequential
  demand estimation before changing policy parameterization.

A4 is expected to show more fragmentation or learned Blank than A8 and the
best B checkpoint. This comparison is descriptive rather than an additional
promotion gate.

Stop after reporting:

- Blank composition for A4, A8, and the selected B checkpoint;
- probability and margin distributions;
- exact failed Mbps and TB failures;
- whether `H_frag` is supported;
- the single recommended next experiment.

Do not implement hierarchical actions, entropy annealing, Blank targets,
forced fill, reward changes, multi-seed campaigns, or deployment export in
this diagnostic phase.

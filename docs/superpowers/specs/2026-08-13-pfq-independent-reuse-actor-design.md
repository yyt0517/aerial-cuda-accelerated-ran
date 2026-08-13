# PFQ-Guided, PFQ-Independent Reuse Actor Design

## 1. Goal

Build and falsify a lightweight intelligent scheduler for the canonical
3-cell boundary scenario. PFQ may provide behavior-cloning labels during warm
start, but the converged policy must construct candidates, score actions, and
materialize the Type-0 schedule without reading PFQ data.

The first research question is deliberately narrow:

> Can a PFQ-free online actor, using the full associated-UE action space and
> explicit same-PRG cross-cell context, reach at least 99% of PFQ goodput on
> boundary seed 41 without violating the reliability and expiry gates?

The first experiment is a mechanism test, not a production scheduler. It stops
after the seed-41 question is answered. Seeds 41--43 are run only if the
seed-41 gate passes. Ten-seed validation, 7-cell scaling, deployment hardening,
and generalized architecture work are separate follow-ups.

## 2. Core hypotheses

The experiment separates three hypotheses that the previous PFQ-residual
stack could not isolate.

### H1: architecture feasibility

An HGraph encoder followed by explicit reuse-group attention and a sequential
headroom-aware UE scorer can reach 99% of PFQ goodput while choosing from every
legal associated UE, not only a PFQ proposal neighborhood.

### H2: PFQ is a training prior, not an online dependency

PFQ behavior cloning reduces the number of PPO interactions required to reach
the gate. A scratch policy and a PFQ-warm-start policy use the same actor and
the same PFQ-free online inputs. Their difference measures initialization and
sample efficiency, not action-space capacity.

### H3: secondary utility matters near the capacity ceiling

When aggregate goodput is close to capacity, delay-debt reduction and
per-UE expiry-aware successful rate provide useful gradients that aggregate
goodput alone does not. These terms are accepted only if they improve formal
secondary KPIs without sacrificing goodput or reliability.

## 3. Fixed experimental scope

The first experiment is fixed to:

- 3 cells;
- 36 UEs with fixed strongest-cell association;
- 17 local PRGs per cell and 16 PRBs per PRG;
- Type-0 allocation;
- `coop_boundary` UE placement;
- cell radius 500 m, corresponding to the canonical boundary geometry;
- SVD precoding;
- 3000-byte packets;
- one packet per TTI offered load;
- 200 ms packet TTL;
- topology and traffic seed 41 for training and the first gate;
- seeds 41--43 only after the seed-41 gate passes.

All PFQ and learned-policy comparisons must come from the same newly built
binary after the expiry-aware metric correction. Historical PFQ artifacts are
context only and cannot be used as the promotion reference.

### Out of scope

The first experiment does not:

- change cell association or handover;
- enumerate joint three-cell UE tuples;
- use a cross-cell autoregressive decoder;
- use PFQ score, rank, proposal, margin, residual action, or materializer state
  online;
- add STOP/BLANK/SWAP/FILL edits or a PFQ-relative action language;
- run counterfactual PFQ schedules on the learned policy's state;
- add safe-fill, blank repair, slot repair, or PFQ fallback;
- tune PRG granularity;
- run a reward-weight grid search;
- scale to 7 cells;
- build a complete production deployment path.

## 4. Architecture

The selected first-version architecture is:

```text
PFQ-free state
  -> HGraph encoder
  -> reuse-group attention
  -> sequential headroom-aware per-cell UE scorer
  -> deterministic Type-0 materializer
```

It reuses the existing Stage-B online bridge, feature snapshot, sparse graph
encoder, PPO rollout machinery, and KPI reporting wherever their semantics are
already suitable. The implementation adds one independent action path instead
of redesigning the existing PFQ-residual stack.

### 4.1 PFQ-free state

The online state is restricted to scheduling facts:

- queue: buffer bytes, HOL age, TTL slack, critical/near-expiry backlog;
- PHY: wideband and per-PRG post-EQ SINR, CQI, RI, MCS-relevant quality, recent
  TB outcome;
- history: average successful rate, recent service ratio, service gap, and
  successful-goodput deficit;
- topology: serving-cell association and cell adjacency;
- interference/reuse: same-local-PRG neighbor context and current interference
  proxies.

The actor input must not contain:

- PFQ score or transformed PFQ score;
- PFQ rank, second-best margin, or proposal membership;
- PFQ predicted or assigned bytes;
- PFQ residual headroom;
- teacher actions;
- PFQ residual traces or PFQ-relative utility.

Teacher actions may be present in a behavior-cloning data object, but they are
not part of the graph snapshot accepted by the actor.

### 4.2 Full candidate coverage

For every `(cell c, local PRG p)`, the candidate set is:

```text
all UEs associated with c that are schedulable and have positive demand
+ Blank
```

The canonical scenario is expected to have at most 12 associated UEs per cell,
so the complete set is small enough to score directly. A preflight check reads
the actual static association mask. Fixed-width packing uses that observed
maximum (expected 12) plus Blank and masks unused positions. A run fails before
training if the configured width is smaller than the observed maximum. There
is no PFQ Top-K and no learned or heuristic pre-ranking that can remove a legal
demand UE before the actor sees it.

The edge generator uses `candidate_mode=all_demand_ues` and disables the PFQ
score anchor. Existing non-PFQ UE-PRG physical, urgency, debt, and risk features
may be reused.

### 4.3 HGraph encoder

The existing entity types remain:

- Cell;
- UE;
- Cell-PRG.

The existing relation types remain:

- Cell -> Cell;
- Cell -> UE;
- Cell -> PRG;
- UE -> PRG;
- PRG -> PRG for the same local PRG across cells.

The first version keeps the existing structured template encoders and sparse
heterogeneous message passing. It does not add deeper graph stacks or a new
graph framework.

### 4.4 Reuse-group attention

After HGraph encoding, Cell-PRG embeddings are reshaped from `[C*P,H]` to
`[P,C,H]`. One lightweight attention layer processes the three cell tokens of
each local PRG:

```text
G_p = {(cell 0,p), (cell 1,p), (cell 2,p)}
```

The output for `(p,c)` therefore contains the other two cells' same-frequency
context before the UE scorer runs. The first version uses four attention heads
and a residual connection. It does not condition cell 1 on cell 0's sampled UE
or cell 2 on earlier sampled UEs. Cross-cell autoregression is considered only
if diagnostics show that per-cell choices are individually sound but their
same-PRG combinations remain the dominant failure mode.

### 4.5 Sequential headroom-aware actor

Reuse groups are processed in fixed native order `p=0..16`. At step `p`, each
cell scores every currently valid associated UE and Blank:

```text
z[p,c,u] = scorer(
    reuse_context[p,c],
    ue_embedding[u],
    ue_prg_edge[p,c,u],
    sequential_ue_state[u])
```

The sequential UE state contains only non-PFQ quantities:

- remaining demand bytes;
- PRGs already allocated in the current TTI;
- cumulative predicted service in the current TTI.

At TTI start, remaining demand equals the observed UE buffer. A candidate's
single-PRG service estimate is derived from its post-EQ physical-rate estimate
and slot duration, converted to bytes and clipped to at least one byte. After a
selection, that estimate is subtracted from remaining demand with a zero lower
bound. This estimator is independent of PFQ score and PFQ allocation state. No
allocation-set counterfactual PHY simulation is introduced in the first
experiment.

Association, schedulability, positive demand, and remaining headroom form the
hard mask. Blank is always legal. The three cells sample or greedily choose
from separate categoricals, but all three distributions use reuse-attended
context. After the three choices for `p`, their sequential states are updated
before `p+1`.

Rollout and PPO replay must execute the same sequence and masks. The stored
trace includes ordering, choices, and valid classes at each step. The submitted
Type-0 action must exactly match deterministic replay of that trace.

### 4.6 Deterministic Type-0 materialization

The actor directly selects UE or Blank per Cell-PRG. Materialization performs
only representation conversion:

1. collect the unique selected UEs in each cell;
2. map them deterministically to that cell's Type-0 slots in ascending UE-ID
   order;
3. point each nonblank PRG to the selected UE's slot;
4. assert association, slot capacity, PRG mask, and index legality.

There are at most 12 unique selected UEs and 12 cell-local slots in the fixed
scenario. Materialization must not change a chosen identity, add a fill, swap
UEs, enforce a second demand cap, or invoke PFQ. Invalid materialization is an
experiment failure, not a reason to fall back to PFQ.

## 5. PFQ guidance and independence contract

### 5.1 Behavior-cloning warm start

The native PFQ schedule is converted into direct per-Cell-PRG UE/Blank labels
over the full actor candidate set. Behavior cloning uses masked cross-entropy.
It does not regress PFQ scores or train a PFQ residual.

The fixed warm-start budget is reported separately from PPO environment
interactions. Scratch and warm-start policies receive the same measured PPO
interaction budget.

### 5.2 PPO phase

At PPO start:

- teacher loss is zero;
- teacher action fields are deleted from the actor input;
- no PFQ anchor is present;
- no KL or behavior regularizer references PFQ;
- no runtime fallback references PFQ.

The learned weights may encode behavior learned during cloning. This is the
intended meaning of PFQ-guided training. Independence refers to inputs and the
online action path, not to erasing knowledge acquired during warm start.

### 5.3 PFQ-deletion test

The formal independence test creates one observation with PFQ/teacher fields
and a second observation with those fields deleted. Graph tensors, logits, the
sequential trace under deterministic decoding, and final Type-0 actions must
be exactly equal. Random perturbation is not part of the test.

Checkpoint metadata records `requires_pfq_features=false`. Loading or exporting
an independent actor must reject a configuration that enables a PFQ score
anchor or PFQ-based candidate generation.

## 6. Expiry-aware packet accounting

### 6.1 Root cause in the current tracker

The shared traffic tracker currently updates packet system time only when
`RecordServedBytes()` completes a MAC packet. `AdvanceToTti()` removes an
expired packet and updates expired byte/packet counts, but does not record the
expired packet's residence time. Consequently, existing packet delay and
packet effective rate are conditional on MAC-completed packets.

That conditional metric is retained as a diagnostic. It is not sufficient as
the only QoS metric in a TTL-constrained experiment because a scheduler can
make the reported delay/rate look better by allowing difficult packets to
expire.

The correction applies in the common traffic tracker and therefore affects
PFQ, Scratch-PPO, and PFQ-BC->PPO identically. It must not be enabled only for
PFQ or only for the learned policy.

### 6.2 Expiry terminal time

When a packet expires, its terminal time is the TTL boundary:

```text
expiry_time_ms = arrival_time_ms + packet_ttl_tti * slot_duration_ms
expired_system_time_ms = normalize(expiry_time_ms - arrival_time_ms)
```

For the canonical experiment this is 200 ms. The tracker must not use the
`AdvanceToTti()` call's raw timestamp, which is one scheduling boundary after
the last eligible TTI in the current loop ordering.

An expired packet contributes:

- one expired terminal outcome;
- its TTL residence time;
- zero delivered packet bits;
- zero per-packet rate.

Partially dequeued but incomplete packets still count as expired packet
outcomes with zero completed-packet payload. Remaining expired bytes continue
to be reported separately.

### 6.3 Retained delivered-only metrics

The following semantics remain available and are explicitly labeled
delivered-only:

- delivered packet count;
- delivered packet system time;
- delivered-only delay distribution;
- delivered-only aggregate packet effective rate;
- delivered-only mean per-packet effective rate.

They answer: "Among packets that completed at the MAC queue, how long did
completion take?"

### 6.4 New outcome/expiry-aware metrics

The tracker adds cumulative, per-TTI, and per-UE quantities:

- expired packet count;
- expired packet system time;
- terminal outcome count = delivered + expired;
- terminal outcome system time = delivered time + expired time;
- outcome delay mean/p50/p90/p95/max over delivered and expired terminals;
- expiry-aware aggregate packet effective rate;
- expiry-aware mean per-packet effective rate.

Definitions:

```text
R_pkt_expiry_aware =
    delivered_packet_bits /
    (delivered_packet_system_time + expired_packet_system_time)

R_pkt_per_packet_mean_expiry_aware =
    sum(delivered_packet_individual_rates) /
    (delivered_packet_count + expired_packet_count)
```

Pending packets at the end of a finite run remain right-censored. They are
reported separately and are not silently converted into terminal samples.

Output names must distinguish the two meanings. Existing fields are not
silently redefined. New final log records use exactly
`TRAFFIC_PKT_OUTCOME_DELAY`, `TRAFFIC_PKT_RATE_EA`,
`UE_PKT_OUTCOME_DELAY`, and `UE_PKT_RATE_EA`. Parsers and summaries expose both
families.

### 6.5 PHY-success boundary

The current MAC packet tracker consumes scheduled/dequeued bytes before the
PHY-success accounting used by `actual_ue_goodput_bytes`. Adding expired time
fixes survivorship bias but does not turn MAC completion into an ACK-based
packet lifecycle.

Therefore:

- packet outcome metrics describe the current MAC tracker semantics;
- aggregate and per-UE successful utility use `actual_ue_goodput_bytes`;
- failed-byte penalties remain explicit;
- no claim is made that this change implements reliable retransmission or
  success-confirmed packet completion.

## 7. Reward design

The full candidate reward is:

```text
r_t =
    w_G * normalized_successful_goodput
  - w_F * normalized_failed_bytes
  - w_E * normalized_expired_bytes
  + w_D * delay_debt_reduction
  + w_U * expiry_aware_successful_UE_utility_delta
```

### 7.1 Goodput, failure, and expiry

Successful goodput, failed bytes, and expired bytes are normalized by offered
bytes per TTI. Raw PRG utilization, Blank count, PFQ-relative improvement, and
counterfactual utility are not reward terms.

### 7.2 Delay-debt shaping

The delay term is a dense state-cost reduction:

```text
Delta_D = D(s_t) - gamma * D(s_{t+1})
```

`D` is computed over all demand-active UEs from backlog, HOL, TTL slack, and
service gap, with emphasis on the highest-debt UEs. Raw mean delay of completed
packets is not directly penalized because it has completion-selection bias and
can punish the rescue of old packets.

### 7.3 Expiry-aware per-UE successful utility

For a trailing window `W=400` TTIs, matching one TTL horizon:

```text
R_EA[u] =
    actual successful goodput bits for u in W /
    (delivered packet system time for u in W
     + expired packet system time for u in W)

U_EA = mean over demand-active u of log(epsilon + R_EA[u])
Delta_U = gamma * U_EA(s_{t+1}) - U_EA(s_t)
```

An expiry adds TTL residence time and no successful bits, so it lowers the
utility. Successful service to a persistently low-rate UE has larger marginal
utility than the same successful bits given to an already high-rate UE.

The training and evaluation stream uses a 1000-TTI warmup before measured
steady-state windows. Missing per-UE expiry-aware diagnostics is a hard error
for treatments that enable this reward; the code must not silently substitute
delivered-only rate.

### 7.4 Reward scaling

`G`, `F`, and `E` are divided by offered bytes per TTI. `Delta_D` and `Delta_U`
are divided by their respective PFQ seed-41 steady-state 95th percentile
absolute magnitudes and clipped to `[-1,1]`; a zero percentile disables that
auxiliary term and fails the corresponding treatment preflight. The fixed
coefficients are:

```text
w_G = 1.00
w_F = 1.00
w_E = 2.00
w_D = 0.25
w_U = 0.25
```

The experiment does not perform a weight grid search. Scale values and PFQ raw
distributions are persisted in the run manifest.

## 8. Treatments and causal comparisons

The minimum matrix contains four treatments:

| ID | Initialization | Reward | Question |
|---|---|---|---|
| A | random | `G-F-E` | Can the independent architecture learn from scratch? |
| B | PFQ BC | `G-F-E` | Does PFQ warm start improve PPO sample efficiency? |
| C | same PFQ BC checkpoint | `G-F-E + Delta_D` | Does delay debt improve outcome latency? |
| D | same PFQ BC checkpoint | `G-F-E + Delta_D + Delta_U` | Does UE expiry-aware utility improve low-rate service? |

Treatments B--D start from the identical frozen BC checkpoint. All four use
the same actor, topology, traffic sequence, optimizer configuration, measured
PPO interaction budget, and deterministic evaluation path. BC interactions
and warmup interactions are reported separately.

PFQ is not another online reward term. A precomputed PFQ value subtracted from
per-TTI goodput is action-independent and supplies no new policy gradient.
Running a counterfactual PFQ on the learned policy's state would reintroduce the
heavy PFQ-relative training chain and would not represent a matched trajectory
after queue and average-rate states diverge.

PFQ-relative quantities are used only in checkpoint selection, final gates,
and the report.

## 9. Experiment sequence

### Phase 0: metric correction and new PFQ reference

1. Add expiry terminal-time accounting in the common traffic tracker.
2. Validate accounting identities with controlled packet traces.
3. Build one binary used by every arm.
4. Run PFQ seed 41 for 5000 TTIs.
5. Treat the first 1000 TTIs as warmup and use the remaining 4000 TTIs for the
   steady-state training scale and comparator.
6. Retain full-run native KPI as a diagnostic, but do not mix it with the
   warmup-excluded promotion comparison.

### Phase 1: action-path validity

Before training:

- every active associated demand UE appears exactly once in its Cell-PRG
  candidate row;
- no cross-cell UE appears;
- PFQ fields can be deleted with no graph/logit/action change;
- all actor actions materialize legally;
- sequential rollout and replay produce identical choices and masks;
- submitted Type-0 actions match materialized replay exactly.

### Phase 2: behavior cloning

Run 4000 PFQ teacher TTIs on seed 41 and save one BC checkpoint. The BC stage
passes when all labels are legal, no PFQ-selected UE is missing from the full
candidate set, and deterministic teacher imitation has no materialization
mismatch. Accuracy is reported, but the research claim does not require exact
PFQ reproduction at convergence.

### Phase 3: seed-41 PPO pilot

Run treatments A--D for 20 iterations of 512 measured PPO TTIs, or 10,240
measured PPO TTIs per treatment. Use the existing persistent online path, one
1000-TTI steady-state warmup before measured collection, and deterministic
argmax evaluation. Save checkpoints every four iterations, but select them
only through the PFQ-relative guarded score in Section 10. BC and warmup TTIs
are excluded from the matched PPO budget and reported separately.

If neither A nor B approaches the 99% PFQ goodput gate within the fixed budget,
stop. Do not tune reward auxiliaries or launch multi-seed experiments. That
result falsifies the current architecture/training path at the allotted budget.

### Phase 4: small generalization

Only the best gate-passing seed-41 treatment advances. Evaluate the fixed
checkpoint on seeds 41, 42, and 43 with paired new-binary PFQ runs, 1000 warmup
TTIs, and 4000 measured TTIs. Stop after reporting this result. A 10-seed run
requires a separate decision.

## 10. Checkpoint selection and gates

For a measured window `W`, the selection score is based on locked PFQ-relative
metrics:

```text
score_W =
    relative_goodput_delta
  + beta_U * relative_UE_expiry_aware_rate_delta
  - beta_D * max(0, outcome_delay - PFQ_outcome_delay)
```

This score selects among checkpoints that already satisfy hard reliability and
legality gates. It is not sent back as per-TTI reward.

### Seed-41 architecture gate

The actor passes only if:

- measured goodput is at least 99% of the new PFQ seed-41 goodput;
- TB BLER is no more than 0.2 percentage point above PFQ;
- expiry rate is no more than 0.05 percentage point above PFQ;
- action replay mismatch count is zero;
- illegal action count is zero;
- PFQ-deletion equality passes exactly.

### Reward-treatment preference

Among architecture-gate passers, an auxiliary reward treatment is preferred
only if:

- expiry-aware outcome delay p95 is no higher than PFQ;
- UE-macro expiry-aware successful rate is no lower than PFQ;
- UE p10 expiry-aware successful rate is no lower than PFQ;
- goodput and reliability remain inside the architecture gate.

If an auxiliary term changes training reward but not formal metrics, reject the
term. If C passes and D fails, retain delay debt without UE-rate utility. If B
matches C/D on formal metrics, retain the simpler `G-F-E` reward.

## 11. Validity checks

Only checks needed to trust the experiment are in scope.

### 11.1 Packet accounting

- With no expiry, delivered-only and expiry-aware aggregate rates are equal.
- One 3000-byte packet delivered in 0.5 ms produces 48 Mbps delivered-only
  packet rate.
- One 3000-byte packet expired at 200 ms adds 200 ms outcome time, zero bits,
  and a zero per-packet-rate sample.
- A mixed delivered/expired trace satisfies:

  ```text
  outcome_count = delivered_count + expired_count
  outcome_time = delivered_time + expired_time
  R_EA = delivered_bits / outcome_time
  ```

- Per-flow totals sum to cluster totals.
- CPU and GPU scheduler modes use the same host traffic accounting.

### 11.2 Action validity

- Full candidate coverage equals the active associated demand set.
- Blank is the only valid choice when a cell has no active demand UE.
- Headroom masks are monotone within a TTI once demand is exhausted.
- A selected UE never belongs to a different cell.
- Unique selected UE count never exceeds available cell-local slots.
- Materialization does not change selected UE identities.
- Rollout and PPO replay log-probabilities use identical masks and ordering.

### 11.3 Comparator validity

- PFQ and every treatment use the same source revision and binary fingerprint.
- Topology, traffic, association, PHY, TTL, warmup, and measured horizon match.
- PFQ scale statistics are frozen before learned-policy training.
- Scratch and warm-start PPO measured interaction budgets match.
- Historical pre-correction delay/rate numbers are never mixed with corrected
  metrics.

## 12. Failure behavior

The experiment fails closed when:

- post-EQ inputs required by the actor are missing;
- PFQ fields remain in the actor feature schema;
- a PFQ-selected BC label is absent from the full legal candidate set;
- expiry-aware diagnostics are unavailable for treatment D;
- materialization changes a choice or needs fallback;
- rollout replay differs from the submitted action;
- PFQ and treatment manifests do not match on the fixed scenario;
- a pre-correction baseline is supplied as the comparator.

None of these cases may silently fall back to PFQ, delivered-only rate, or a
heuristic candidate generator.

## 13. Stop conditions and interpretation

The research prototype is complete when one of the following occurs:

1. No independent treatment reaches the seed-41 architecture gate within the
   fixed budget. Report the falsification and the dominant observed failure.
2. At least one treatment reaches the gate. Report whether PFQ BC improved
   sample efficiency and whether delay/rate auxiliaries improved formal
   secondary metrics.
3. A seed-41 winner is evaluated on seeds 41--43. Report generalization and
   stop.

Do not automatically continue into autoregressive decoding, wider reward
sweeps, 10-seed campaigns, 7-cell scaling, ONNX/TensorRT deployment,
refactoring, or production hardening.

The final report must state:

- what code and metric semantics changed;
- which treatment and budget were tested;
- new-binary PFQ and learned-policy results;
- whether H1, H2, and H3 are supported;
- the main remaining uncertainty.

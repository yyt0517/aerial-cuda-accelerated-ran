# DPA-v1 Seed41--50 Historical Comparison Design

## Status and research question

This protocol is frozen before any new seed41--50 evaluation is run.

The current actor is named **DPA-v1 (Deadline-Pressure Actor v1)**. The
implementation provenance remains `feature-only iter8 checkpoint`; `iter8` is
not used as the reporting name.

The experiment asks one question: after replacing the raw service-severity
inputs with dimensionless deadline-pressure features, does the frozen DPA-v1
actor provide a better cross-layout throughput, reliability, delay, and
per-UE service trade-off than the historical PFQ and v9c evidence on seeds
41--50?

This is a historical comparison, not a new training experiment and not a
strict paired binary comparison. Its purpose is to quantify the current model
over the established 10-seed cohort before any further algorithm change.

## Frozen treatment

DPA-v1 is the actor stored at:

```text
output/pfq_independent_reuse/formal/
  deadline_dimensionless_features_seed45_v1/
  train_attempt2/ppo_actor_last.pt
```

Frozen identity:

- checkpoint file SHA-256:
  `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`;
- actor tensor-state SHA-256:
  `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`;
- current simulator binary SHA-256:
  `4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`;
- deterministic argmax decode;
- 16-dimensional UE observation with `log1p(drain_pressure)`,
  scheduled-dequeue adequacy, and TTL-normalized service age;
- CTBS-v1 accounting, 17-step PRG order, candidate construction, network,
  Type-0 mapping, PHY configuration, and reward unchanged;
- CTBS-legal non-starvation guard disabled;
- no further PPO updates, checkpoint selection, coefficient scan, decoder
  change, logit boost, or post-run model adjustment.

The scenario remains the canonical 3-cell, 36-UE, 17-PRG cooperative-boundary
case with Type-0 allocation, SVD precoding, 3000-byte packets, one packet per
UE per TTI, and a 400-TTI / 200-ms packet TTL. Only the topology seed changes
from 41 through 50.

## Comparison arms and provenance

Only DPA-v1 receives new rollouts.

### Historical v9c

Reuse the existing seed41--50 artifacts rooted at:

```text
/home/oai2/aerial-cuda-accelerated-ran/training/stageb_hgraph/runs/
  eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_
  s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_
  r500_t4000_w0/
```

Despite `threshold` in the directory and report label, each per-seed
`eval_summary.json` records that the actually loaded checkpoint was
`ppo_actor_best.pt`, whose SHA-256 is
`7e1243960798ba3be7287235538b318a42ad62004e5f5a83f5a115234499bfb3`.
It must not be silently replaced by `ppo_actor_threshold_best.pt`, whose
different SHA-256 is
`3aa7644596a3b8585dfcd9f3a7e6fd63a89942c7d8a612b8c330716dc1e76f05`.

The frozen historical aggregate manifest has SHA-256
`e1aba7d5b98d3169dbf2b1971cce9324bb74abae00226d3be3a399ca6788f6ee`.
The historical full-4000 PFQ/v9c comparison JSON has SHA-256
`6c72b39e8733bb2c2c5fa1d654e9fe9a0985184a5843b783069385824e43c03b`.

### Historical PFQ

Reuse the per-seed PFQ rows and aggregate already bound into the historical
comparison rooted at:

```text
/home/oai2/aerial-cuda-accelerated-ran/output/
  stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_
  legacyshadow_r500_t4000_w0_s41_s50/
```

Its aggregate manifest SHA-256 is
`0df7d8eb94c9c3a4ce40ed9fd73d903eeec32e5c6fa8a1c474a1fe7206bb6b47`,
and its mean comparison JSON SHA-256 is
`9c00fee4affc295762a16fe6616676a0cf270559d76e99dbef3385ffc45d04cb`.
Do not launch a fresh PFQ campaign. The preserved PFQ evidence supports
per-seed and aggregate full-4000 comparisons, but no equivalent per-TTI
post-1000 PFQ trace is available.

### Comparability contract

| Comparison | Window | Interpretation |
|---|---:|---|
| DPA-v1 vs historical v9c | full 4000 | Same seed cohort and horizon; historical binary comparison |
| DPA-v1 vs historical PFQ | full 4000 | Same seed cohort and horizon; historical binary comparison |
| DPA-v1 vs historical v9c | TTI 1001--4000 | Same 3000-TTI steady window; v9c comes from preserved 500-TTI block logs |
| DPA-v1 or v9c steady window vs PFQ | DPA/v9c 3000 vs PFQ full 4000 | Context only, not a paired comparison |

The historical PFQ/v9c executions used an older simulator binary and do not
preserve a complete offered-stream fingerprint. Therefore the report may say
that DPA-v1 is higher or lower on the recorded cohort, but must not claim
strict paired superiority or binary-identical replay.

## DPA-v1 execution protocol

Run exactly one 4000-TTI DPA-v1 rollout for each topology seed 41--50. There is
no separate warmup execution and no environment reset at TTI 1000.

Each physical rollout yields two reporting windows:

1. **full-4000:** TTI 1--4000;
2. **steady-3000:** TTI 1001--4000 after removing the first 1000 TTI.

The evaluator may use `stats_warmup_steps=1000` as an evaluation-only
accumulator filter because it does not affect observations, actions, queues,
or PHY state. Full-window metrics still come from the same uninterrupted
4000-TTI trace. Raw step, packet-outcome, offered-packet, and per-UE service
records must be retained so both windows can be independently reaggregated.
The evaluator `stats_*` values are a cross-check, not the sole source of the
post-1000 result.

Run seeds sequentially on the same GPU to avoid resource interference. Each
seed writes to a distinct immutable directory under:

```text
output/pfq_independent_reuse/formal/
  dpa_v1_seed41_50_historical_comparison_v1/
```

Before launch, write a campaign manifest containing the checkpoint, actor
state, simulator, relevant source/config, and historical artifact hashes.
Each per-seed directory must preserve the resolved command/configuration,
evaluator log and summary, step trace, packet outcomes, offered packets,
service-balance trace, topology, exit codes, and derived per-UE audit.

## Metric contract

Every metric must identify its arm, seed, window, unit, direction, source
artifact, and definition. Missing historical fields remain `N/A`; they must
not be reconstructed from a non-equivalent proxy.

### Throughput and reliability

- successful-byte goodput in Mbps;
- served traffic in Mbps where the historical definition is available;
- global TB error rate / BLER and success rate;
- expired bytes, expired packets, and expiry drop rate.

### Resource use

- PRG utilization, reuse, blank, and fill ratios;
- final blank count and available fill/repair counters;
- unique scheduled UE count where definitions match.

Low PRG use is not inherently a failure. It is interpreted together with
goodput, expiry, delay, and per-UE service rather than scored independently.

### UE fairness and service tail

- UE goodput/throughput mean, p5, p10, and Jain index where available;
- scheduled-ratio mean, p5, and Jain index where available;
- per-UE scheduled-dequeue bytes and expired bytes for DPA-v1;
- minimum served UE, zero-scheduled-dequeue UE count, and the maximum
  consecutive positive-backlog zero-dequeue streak for DPA-v1;
- expiry-aware per-UE rate macro mean, p10, and minimum for DPA-v1.

The DPA-v1-only service diagnostics are reported as mechanism evidence. They
are not assigned fabricated historical PFQ/v9c counterparts.

### Delay and packet service

- outcome delay including both delivered and expired packet outcomes:
  mean, p50, p90, p95, and maximum for DPA-v1;
- historical packet/queue delay mean and available p50/p90/p95 values;
- effective packet service rate, per-packet mean rate, and UE-macro packet
  rate where definitions are available;
- residual buffer ratio and backlog-free UE ratio where available.

Outcome delay and older served/queue-delay metrics remain separately labelled
because they have different populations and censoring semantics.

### Structural validity

- evaluator and simulator completion with the expected 4000 steps;
- full-window count 4000 and steady-window count 3000;
- zero illegal actions, replay mismatches, candidate truncation, and mask
  reactivation;
- zero CTBS MCS/layer/TBS parity errors;

### DPA-v1 service-health criteria

- no backlogged UE receives zero scheduled dequeue over the entire rollout;
- no positive-backlog zero-dequeue streak reaches the 400-TTI TTL horizon;
- every violation is reported with the affected seed and UE rather than being
  hidden by an aggregate mean.

These criteria judge the model, not the execution. A violation is a valid
negative experimental result and does not invalidate the seed trace.

## Aggregation and presentation

Produce one normalized row per arm, seed, and window for every metric that is
actually available. The report must include:

- a seed41--50 table for the headline metrics;
- mean, standard deviation, median, minimum, maximum, and worst-seed identity;
- per-seed DPA-v1 deltas and win counts against PFQ and v9c when both values
  exist under the same window definition;
- separate callouts for historical stress seeds 43, 45, and 48;
- the worst DPA-v1 topology and its dominant throughput, expiry, delay, or
  starvation mechanism;
- separate full-4000 and steady-3000 summaries rather than mixing horizons.

No new scalar leaderboard or tuned composite score is introduced. The result
is a predeclared scorecard: aggregate efficiency, expiry/tail risk, and per-UE
service robustness remain visible as distinct dimensions.

## Validity, failure handling, and stop rule

The campaign is internally valid only if all ten DPA-v1 seeds complete with
the frozen identity and structural checks above. A zero-service UE or a
400-TTI starvation streak is a valid negative model result, not an
infrastructure error and not a reason to suppress the remaining seeds.

Stop the campaign immediately on an execution, logging, identity, trace, or
structural-validity failure. Preserve the failed seed directory. A rerun is
allowed only after correcting an infrastructure or metric-collection defect;
the actor, features, reward, decoder, guard state, and scenario must remain
frozen, and the rerun reason must be recorded.

Once ten trustworthy DPA-v1 results and the historical comparison report have
been produced, stop. Do not automatically retrain, enable the guard, tune a
coefficient, select a new checkpoint, or expand the seed set. Those are
separate research decisions informed by this report.

## Required outputs

The versioned campaign root must contain at least:

- `campaign_manifest.json`;
- one immutable directory for each seed 41--50;
- `per_seed_metrics.csv` and `per_seed_metrics.json`;
- `aggregate_metrics.csv` and `aggregate_metrics.json`;
- `historical_comparability.json` recording all limitations and source hashes;
- `final_report.json` with validity, headline deltas, worst seeds, and the
  evidence-backed classification;
- a concise Markdown brief suitable for project reporting.

The final classification must describe the dominant evidence, for example
`aggregate_efficiency_improved_but_expiry_tail_failed` or
`cross_layout_service_improved_with_no_starvation`. It must be derived from the
frozen scorecard and may not conceal a failing seed behind an aggregate mean.

## Non-goals

- rerunning PFQ or v9c on the current binary;
- training or adapting DPA-v1 on seeds 41--50;
- enabling or evaluating the CTBS-legal non-starvation guard;
- changing reward coefficients, actor inputs, architecture, candidates, PRG
  order, decoder, or Type-0 mapping;
- claiming strict causal superiority from historical, non-binary-identical
  baselines;
- continuing automatically into algorithm optimization after the comparison.

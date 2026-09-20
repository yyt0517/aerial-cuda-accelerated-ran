> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/specs/2026-08-27-iter8-seed48-seed51-frozen-generalization-protocol.md)保留当时状态。

# Iter8 Seed48/Seed51 Frozen Layout-Generalization Protocol

> Status: completed historical experiment protocol. Seed48 passed; seed51 improved most KPIs but failed the expiry Gate. The retained result is consolidated in the [integrated report](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md).

## Research question

Does the fixed iter8 actor with dimensionless per-UE service features preserve
business performance and prevent complete per-UE starvation on the historical
seed48 stress layout and the untouched seed51 layout, without retraining,
retuning, or a non-starvation guard?

## Frozen artifacts

- Actor checkpoint:
  `output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt`
- Checkpoint file SHA-256:
  `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`
- Actor state SHA-256:
  `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`
- Simulator SHA-256:
  `4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`
- `DeadlineServiceFeatures.h` SHA-256:
  `4a353601a8378888d089e34d3f04225770ae757ac07d5bc4108533a1a384975a`
- `OnlineObservationExtrasBuilder.h` SHA-256:
  `ccf58c6c07e5ab65da373870d7d615b8048611508e4972910c4a4871744f876b`
- Python feature contract SHA-256:
  `e86976c6208ff6e23e350d1480beaac011d694870d4a68e4e6fb9f7077aa7f7f`

The UE feature dimension remains 16. Slots 12--15 remain:

1. `log1p(drain pressure)`;
2. scheduled-dequeue adequacy;
3. TTL-normalized service age;
4. reserved zero.

The scheduled-dequeue EWMA coefficient remains `0.05`; the TTL horizon remains
`400 TTI`. Reward, actor network, state normalization, candidate construction,
17-step PRG order, decoder, Type-0 mapping, CTBS-v1 accounting, traffic load,
packet size, and PHY configuration are unchanged. The guard is disabled.

## Evaluation order and independence

Run seed48 first and seed51 second. For each seed, run a fresh paired PFQ
reference followed by the actor evaluation using the same topology seed and
traffic seed. Both seeds are evaluated even if seed48 fails because there is no
model update between them.

Seed51 remains untouched until its launch. No result from seed48 may change any
seed51 argument, threshold, checkpoint, feature, or decoder setting.

Each PFQ and actor run uses:

- `1000 TTI` warmup;
- `4000 TTI` measured, TTI `1001--5000`;
- packet TTL `400 TTI / 200 ms`;
- `argmax` actor decode;
- CTBS-v1 pre-action handshake;
- complete packet outcomes and service-balance traces;
- complete offered-packet stream for paired-input verification.

## Frozen Gates

### Provenance and structure

- checkpoint, actor-state, simulator, and feature-source hashes equal the
  frozen values;
- completed steps equal `5000`, with exactly `4000` measured TTIs;
- action replay mismatch and illegal action counts are zero;
- candidate coverage is complete and truncation is zero;
- CTBS MCS, layer, and TBS mismatches are zero;
- CTBS mask reactivation is zero;
- reference-label deletion equality is true;
- PFQ and actor offered-packet canonical hashes are identical.

### Paired business performance

Use the existing project architecture Gate without changing thresholds:

- actor goodput is at least `99%` of paired PFQ;
- actor BLER is no more than PFQ plus `0.002` absolute (`0.2 pp`);
- actor expiry rate is no more than PFQ plus `0.0005` absolute (`0.05 pp`).

Outcome-delay mean/p50/p90/p95/max, UE macro/p10/minimum expiry-aware rate,
PRG utilization, blank ratio, scheduled-dequeue rate, and failed-byte rate are
recorded even when they are not thresholded by the existing performance Gate.

### Per-UE non-starvation

For every UE with positive backlog in the measured interval:

- total scheduled dequeue must be positive;
- the maximum consecutive positive-backlog, zero-scheduled-dequeue streak must
  be less than `400 TTI`.

Per-UE scheduled bytes, actual PRGs, successful/failed bytes, delivered and
expired packets, outcome delay, and expiry-aware rate are recorded. A positive
UE minimum expiry-aware rate is required as a consistency check.

The overall seed Gate passes only when provenance/structure, paired business
performance, and per-UE non-starvation all pass.

## Stop and follow-up rules

- Do not train or tune after either seed.
- Do not enable the guard within this protocol.
- If either seed starves, finish the other frozen evaluation and classify the
  failure before proposing a separate guard protocol.
- If both seeds pass, classify the evidence as support for layout-generalized
  service protection on two held-out stress layouts, not as a universal proof.
- Do not run additional seeds automatically.

## Outputs

All outputs are written under:

`output/pfq_independent_reuse/formal/iter8_dimensionless_frozen_generalization_seed48_seed51_v1/`

Each seed contains fresh PFQ, actor, starvation Gate, paired Gate, and raw
trace artifacts. The root contains a frozen manifest and a final two-seed
report. Project conclusions are updated only after both seeds finish.

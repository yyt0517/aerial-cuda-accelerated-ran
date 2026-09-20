> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/specs/2026-09-01-dpa-v1-pfq-aligned-expiry-aware-comparison-design.md)保留当时状态。

# DPA-v1 / PFQ Aligned Expiry-Aware Comparison Design

> Status: completed protocol for the strict seeds 41--50 comparison. Final results are consolidated in the [DPA/PFQ/RRQ integrated report](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md); raw evidence remains under `output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/`.

## Research question

When PFQ and DPA-v1 are evaluated on the same simulator binary, seed, traffic
stream, 4000-TTI horizon, and post-hoc steady window, does DPA-v1 improve
aggregate efficiency and per-UE service without hiding expired packets from
delay or rate metrics?

## Frozen arms and cohort

- Arms: frozen DPA-v1 outputs and a fresh PFQ replay.
- Seeds: `41,42,43,44,45,46,47,48,49,50`.
- Full window: TTI `1--4000`.
- Steady window: TTI `1001--4000`; the first 1000 TTIs are removed after the
  same uninterrupted 4000-TTI rollout.
- Simulator: the exact binary recorded by the DPA-v1 campaign manifest.
- Scenario: 3 cells, 36 active UEs, 17 PRGs, `coop_boundary`, strongest-cell
  association, radius 500 m, 3000-byte packets, arrival rate 1.0 packet per UE
  per TTI, packet TTL 200 ms, topology seed equal to traffic seed.
- PFQ uses native teacher actions; DPA-v1 remains frozen with argmax, CTBS-v1,
  Guard OFF, and its existing checkpoint.
- No training, checkpoint selection, coefficient scan, guard, decoder, network,
  candidate, mapping, or PRG-order change is allowed.

## Alignment checks

For every seed, PFQ and DPA must have:

- the same simulator SHA-256;
- exactly 4000 decision/service TTIs;
- exactly 36 UE service rows per TTI;
- byte-identical offered-packet streams and topology dumps when both artifacts
  are available;
- the same inclusive full and steady window boundaries;
- zero illegal actions or replay mismatches on the DPA arm.

Any alignment or structural failure stops the campaign. A poor PFQ or DPA
metric is retained as a valid result.

## Canonical delay and rate semantics

All headline delay and packet-effective-rate metrics are expiry-aware.

- Outcome delay uses every terminal packet outcome in the selected window:
  both `delivered` and `expired`. Pending packets remain right-censored and are
  reported through terminal-outcome counts and end-of-window backlog, not
  assigned a fabricated delay.
- Per-UE effective service rate is

  `delivered_packet_bits / (delivered_system_time_ms + expired_system_time_ms)`.

  An expired packet contributes zero delivered bits and its full system time to
  the denominator.
- UE macro mean, p10, and minimum use the fixed set of all 36 active UEs. An
  active UE with no terminal outcome in the selected window contributes rate
  zero.
- Legacy delivered-only packet delay/rate values may be retained only under an
  explicit `served_only` diagnostic label. They are excluded from headline
  tables, Gates, win counts, and conclusions.
- Goodput remains successful PHY payload bytes per wall-clock time. Served rate
  remains scheduled MAC dequeue bytes per wall-clock time. These are separate
  efficiency metrics and are not renamed as packet-effective service rate.

## Required comparison

Produce paired full-4000 and steady-3000 results for:

- goodput, served rate, weighted TB BLER, expiry bytes/rate;
- outcome delay mean/p50/p90/p95/max;
- UE expiry-aware effective service rate mean/p10/min;
- UE goodput mean/p5/p10/min/Jain;
- PRG utilization, blank ratio, reuse ratio, and unique-UE count;
- end backlog, backlog-free UE ratio, zero-service UEs, and maximum backlogged
  zero-dequeue streak.

The final report must provide 10-seed mean/std/median/min/max, paired deltas and
win counts, followed by one table per seed. Seeds 45 and 48 must be called out
explicitly because they were historical stress layouts.

## Stop condition

The research task is complete when all ten PFQ runs pass alignment checks and a
machine-readable plus Markdown overall/per-seed comparison is generated. No
additional training or algorithm change follows automatically.

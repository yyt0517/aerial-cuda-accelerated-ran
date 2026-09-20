> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-08-27-iter8-seed48-seed51-frozen-generalization.md)保留当时状态。

# Iter8 Seed48/Seed51 Frozen Generalization Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the frozen iter8 dimensionless-service actor against fresh paired PFQ references on seed48 and untouched seed51 without any adaptation.

**Architecture:** Reuse the existing PFQ calibration, online checkpoint evaluator, CTBS formal Gate, and per-UE service-balance audit. Add only seed-aware audit identity, then run two independent `w1000/t4000` pairs and combine their immutable Gate reports.

**Tech Stack:** C++ Aerial simulator, Python/PyTorch online evaluator, JSON/CSV research artifacts, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-iter8-seed48-seed51-frozen-generalization-protocol.md`

## Global Constraints

- Work only in `.worktrees/pfq-independent-reuse-actor`.
- Enter the container with `./cuPHY-CP/container/attach_aerial.sh`, then activate `source .venv/bin/activate` before running Python.
- Preserve the frozen checkpoint, actor state, simulator, feature-source hashes, and all protocol thresholds.
- Run seed48 and seed51 even if one fails; never adapt between them.
- Do not enable or implement the non-starvation guard in this protocol.
- Preserve unrelated dirty-worktree changes and do not commit.

---

### Task 1: Seed-Aware Starvation Artifact

**Files:**
- Modify: `training/stageb_hgraph/tests/test_seed45_starvation_audit.py`
- Modify: `training/stageb_hgraph/seed45_starvation_audit.py`

**Interfaces:**
- Consumes: integer evaluation seed plus existing service trace, outcome CSV, and evaluator summary.
- Produces: a Gate JSON containing generic protocol ID and exact seed while preserving the frozen starvation calculation.

- [x] Add a failing test requiring generic protocol identity and seed48 identity.
- [x] Run `python -m pytest training/stageb_hgraph/tests/test_seed45_starvation_audit.py -q` and verify the expected failure.
- [x] Add the minimal `--seed` CLI field and generic protocol identity without changing starvation formulas.
- [x] Re-run the focused audit tests and the existing service-balance tests.

### Task 2: Frozen Manifest and Preflight

**Files:**
- Create: `output/pfq_independent_reuse/formal/iter8_dimensionless_frozen_generalization_seed48_seed51_v1/frozen_manifest.json`

**Interfaces:**
- Consumes: frozen hashes from the protocol.
- Produces: fail-closed manifest checked before every simulator launch.

- [x] Verify all six frozen artifact hashes and actor UE feature dimension 16.
- [x] Write the manifest with seeds `[48, 51]`, order, horizons, thresholds, and guard disabled.
- [x] Run focused feature/actor/audit tests and the standalone C++ feature formula test.

### Task 3: Seed48 Paired Formal Evaluation

**Files:**
- Create outputs under: `.../seed48/pfq/`
- Create outputs under: `.../seed48/actor/`
- Create: `.../seed48/gate_report.json`

**Interfaces:**
- Consumes: frozen manifest, simulator, and iter8 checkpoint.
- Produces: fresh PFQ and actor `w1000/t4000` summaries, traces, outcomes, and combined Gate.

- [x] Run `training.stageb_hgraph.pfq_reference_calibration` with topology/traffic seed48, `--formal`, and service-balance audit from TTI1001.
- [x] Re-verify frozen hashes, then run the online actor evaluator with seed48, argmax, CTBS-v1, PHY/TBS audit labels, offered packets, outcomes, and service trace.
- [x] Normalize the actor summary and run the seed-aware per-UE starvation audit.
- [x] Combine CTBS formal, offered-stream identity, UE minimum-rate, and starvation checks into `gate_report.json`.

### Task 4: Untouched Seed51 Paired Formal Evaluation

**Files:**
- Create outputs under: `.../seed51/pfq/`
- Create outputs under: `.../seed51/actor/`
- Create: `.../seed51/gate_report.json`

**Interfaces:**
- Consumes: the same frozen manifest, simulator, and checkpoint; no seed48-derived parameters.
- Produces: seed51 paired evidence and combined Gate.

- [x] Re-verify frozen hashes before revealing seed51 execution.
- [x] Run the fresh PFQ seed51 reference with the identical horizon and trace settings.
- [x] Run the frozen actor seed51 evaluation with the identical actor arguments.
- [x] Generate the seed-aware starvation artifact and combined Gate using the same code and thresholds as seed48.

### Task 5: Two-Seed Report and Documentation

**Files:**
- Create: `.../final_report.json`
- Create: `Doc/iter8_dimensionless_seed48_seed51_frozen_generalization_brief.md`
- Modify: `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`
- Modify: `Doc/README.md`

**Interfaces:**
- Consumes: immutable seed48 and seed51 Gate reports.
- Produces: two-seed classification, metric tables, per-UE failure evidence, and the next research decision.

- [x] Generate the final report without changing either seed result.
- [x] Record paired goodput/BLER/expiry, outcome delay/rate, resource use, and per-UE starvation evidence.
- [x] State whether the fixed representation is supported on both layouts and whether a separate guard experiment is warranted.
- [x] Run fresh tests, hash verification, JSON consistency checks, and `git diff --check` before completion.

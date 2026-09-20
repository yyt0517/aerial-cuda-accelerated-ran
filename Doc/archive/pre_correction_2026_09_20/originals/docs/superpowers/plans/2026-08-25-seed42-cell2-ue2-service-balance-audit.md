# Seed42 Cell2/UE2 Service-Balance Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while executing each task. This bounded research task is executed inline in the existing isolated worktree; no subagents or commits are used because the worktree already contains the preceding uncommitted experiment chain.

**Goal:** Determine whether frozen B4-CTBS has a stable, material UE2 queue-service deficit relative to a freshly paired PFQ trajectory on seed42.

**Architecture:** Add audit-only simulator CSV instrumentation for true traffic/queue/service state and a deterministic offered-packet stream. Analyze the two fresh arms offline in fixed 100-TTI windows, fail closed on pairing or conservation errors, and classify only the frozen mechanisms.

**Tech Stack:** C++17 simulator instrumentation, Python 3/NumPy offline audit, pytest, Bash experiment runner.

**Spec:** `docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md`

## Global Constraints

- Protocol ID: `seed42-cell2-ue2-service-balance-read-only-audit-v1`.
- B4 state-dict SHA256: `4ce1704303d9f38a6846511630d790a55532acbee777083330bd6445648b6571`.
- Freeze seed42, B4-CTBS-v1 handshake, checkpoint, reward, scorer, decoder, PRG order, and Type-0 mapping.
- Run one fresh PFQ arm and one fresh B4-CTBS arm with `1000 warmup + 4000 measured TTI`.
- Trace TTI 602--5000; analyze forty fixed 100-TTI windows over TTI 1001--5000.
- Material deficit means total CTBS/PFQ `<= 0.90` and CTBS is lower in at least 30/40 windows.
- PHY failure is reliability-only and never enters MAC queue or expiry conservation.
- Stop after this audit; do not train or run another seed/treatment.

---

### Task 1: Freeze Offline Audit Semantics

**Files:**
- Create: `training/stageb_hgraph/tests/test_service_balance_audit.py`
- Create: `training/stageb_hgraph/service_balance_audit.py`

**Interfaces:**
- Consumes: service-balance CSV, offered-packet CSV, packet-outcome CSV for PFQ and CTBS.
- Produces: `build_service_balance_report(...) -> dict[str, object]` and CLI-written `audit_report.json`.

- [ ] Write failing tests with hand-derived fixtures for offered-stream hashing, all three byte identities, fixed 100-TTI windows, the 0.90/30-of-40 boundary, and each permitted classification.
- [ ] Run the focused pytest file and verify failures are caused by the missing module/API.
- [ ] Implement the minimum loader, fail-closed validators, window reducer, and classifier.
- [ ] Re-run the focused tests to green.

### Task 2: Add Audit-Only Simulator Evidence

**Files:**
- Modify: `cuMAC/examples/trafficModel/trafficGenerator.hpp`
- Modify: `cuMAC/examples/trafficModel/trafficService.hpp`
- Modify: `cuMAC/examples/trafficModel/trafficPacketAccountingTest.cpp`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`

**Interfaces:**
- Consumes: existing traffic generator, queue, scheduler, TBS, MCS/layer, and TB-result state.
- Produces: `traffic_offered_packets.csv` and `service_balance_trace.csv` only when explicit audit environment variables are set.

- [ ] Add failing C++ tests proving offered packet identity/arrival/size capture without changing traffic totals.
- [ ] Build/run `trafficPacketAccountingTest` and verify the new assertions fail before implementation.
- [ ] Add the minimum offered-packet sample and per-UE TTI CSV instrumentation.
- [ ] Record true event order and the fields needed for exact queue/service identities; keep instrumentation disabled by default.
- [ ] Build/run the C++ test to green.

### Task 3: Wire the Single Paired Experiment

**Files:**
- Modify: `training/stageb_hgraph/pfq_reference_calibration.py`
- Modify: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh`
- Modify: `training/stageb_hgraph/tests/test_pfq_reference_calibration.py`
- Modify: `training/stageb_hgraph/tests/test_candidate_width_compatibility_runner.py`

**Interfaces:**
- Consumes: new simulator audit environment variables and frozen B4 checkpoint.
- Produces: a new versioned formal directory with `pfq/`, `actor/`, manifests, raw traces, and final audit report.

- [ ] Add failing CLI/environment and runner-phase tests.
- [ ] Run the focused tests and verify expected RED failures.
- [ ] Add audit CLI arguments/environment wiring and the single bounded runner phase.
- [ ] Re-run focused tests to green.

### Task 4: Verify and Execute Once

**Files:**
- Modify: `docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md`
- Create at runtime: `output/pfq_independent_reuse/formal/seed42_cell2_ue2_service_balance_read_only_audit_v1/`

**Interfaces:**
- Consumes: rebuilt simulator and frozen experiment artifacts.
- Produces: one authoritative paired audit result and no follow-up experiment.

- [ ] Run focused Python tests, relevant full Stage-B Python tests, the C++ accounting test, and the simulator build.
- [ ] Execute only `audit-seed42-cell2-ue2-service-balance` for 4000 measured TTIs.
- [ ] Verify hashes, trace coverage, conservation identities, structural gates, and report classification from fresh artifacts.
- [ ] Record the result and remaining uncertainty in the design document; stop.

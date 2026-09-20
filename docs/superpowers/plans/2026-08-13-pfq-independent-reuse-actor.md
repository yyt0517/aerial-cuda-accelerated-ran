> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-08-13-pfq-independent-reuse-actor.md)保留当时状态。

# PFQ-Independent Reuse Actor Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for serial execution, or `superpowers:subagent-driven-development` only when the user explicitly authorizes delegated/parallel work. Execute tasks in order, track the checkbox (`- [ ]`) steps, and stop at the research gates below.

**Goal:** Determine, with the smallest trustworthy boundary-scenario experiment, whether a PFQ-free HGraph reuse actor can reach at least 99% of a newly measured PFQ goodput while preserving reliability and expiry, and whether PFQ behavior cloning plus delay/UE-rate reward terms improve sample efficiency or secondary QoS.

**Architecture:** Keep the existing Stage-B observation, sparse HGraph encoder, bridge, Type-0 native action format, and PPO optimizer/rollout infrastructure. Add a separate `pfq_independent_reuse_v1` action path: all associated demand UEs plus Blank, one four-head same-local-PRG cross-cell attention layer, fixed `p=0..16` sequential selection with non-PFQ demand headroom, and an identity-preserving Type-0 materializer. Correct packet accounting first so expired packets terminate at their exact TTL boundary and contribute time but zero delivered bits to new expiry-aware metrics. PFQ is permitted only for a 4000-TTI BC warm start and as an offline comparator/calibration source.

**Tech Stack:** C++20/CUDA/CMake/CTest, Python 3.10, PyTorch, NumPy, pytest, Bash, existing cuMAC Stage-B runners and KPI parser.

**Authoritative design:** `docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md`

---

## Global Constraints

### Fixed canonical scenario

- 3 cells, 36 UEs, strongest-cell association, 12 expected associated UEs/cell;
- 17 local PRGs/cell, 16 PRBs/PRG, Type-0 allocation;
- `coop_boundary`, cell radius 500 m, SVD precoding;
- 3000-byte packets, one packet/UE/TTI, 200 ms TTL;
- seed 41 for implementation, PFQ calibration, BC, and the first gate;
- seeds 41--43 only after a seed-41 gate pass;
- one newly built binary for PFQ and every learned arm.

### Hypotheses and stop conditions

- **H1:** the independent actor reaches at least 99% of new-binary PFQ goodput without exceeding PFQ BLER by 0.2 percentage point or PFQ expiry by 0.05 percentage point.
- **H2:** the same actor initialized by PFQ BC learns faster than scratch within the same 10,240 measured PPO-TTI budget.
- **H3:** delay debt and expiry-aware successful UE utility improve their formal metrics without leaving the H1 gate.
- Run one 32-step online smoke before any long job.
- Run seed 41 before any multi-seed job.
- If neither scratch A nor BC B approaches the 99% goodput gate after the fixed pilot, stop; do not tune weights, add fallback, add autoregression, or launch seeds 42--43.
- If B matches C/D on formal metrics, keep the simpler `G-F-E` reward.
- After the best gate-passing checkpoint is paired against PFQ on seeds 41--43, report and stop.

### Dirty-worktree guard

The repository already contains user changes, including edits in:

- `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`;
- `cuMAC/examples/network.cu`;
- `cuMAC/examples/network.h`;
- documentation and untracked research artifacts.

Do not clean, reset, overwrite, or include unrelated changes. Before every commit:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

For `main.cpp`, use `git add -p` and stage only the expiry-aware bridge/logging hunks. Never stage `network.cu`, `network.h`, existing deliverables, caches, or unrelated documents.

Use a dedicated `build.reuse_actor/` directory. Do not reuse `build.x86_64/`, whose cached SDK paths may not match the current environment.

---

## Locked interfaces

### C++ packet outcome summaries

Add these structs in `cuMAC/examples/trafficModel/trafficGenerator.hpp` without changing the meaning of `PacketDelaySummary` or `PacketServiceRateSummary`:

```cpp
struct PacketOutcomeDelaySummary {
    unsigned long long delivered_packets = 0;
    unsigned long long expired_packets = 0;
    unsigned long long outcome_packets = 0;
    unsigned long long pending_packets = 0;
    double mean_delay_ms = 0.0;
    double p50_delay_ms = 0.0;
    double p90_delay_ms = 0.0;
    double p95_delay_ms = 0.0;
    double max_delay_ms = 0.0;
};

struct PacketExpiryAwareRateSummary {
    unsigned long long delivered_packets = 0;
    unsigned long long expired_packets = 0;
    unsigned long long pending_packets = 0;
    unsigned long long total_delivered_bits = 0;
    double delivered_packet_system_time_ms = 0.0;
    double expired_packet_system_time_ms = 0.0;
    double outcome_packet_system_time_ms = 0.0;
    double packet_expiry_aware_rate_mbps = 0.0;
    double packet_expiry_aware_rate_per_packet_mean_mbps = 0.0;
};

struct PacketOutcomeSample {
    uint32_t flow_id = 0;
    uint64_t packet_id = 0;
    int terminal_tti = -1;
    bool expired = false;
    double system_time_ms = 0.0;
};
```

Expose through `TrafficService`:

```cpp
void GetPacketOutcomeDelayStats(
    PacketOutcomeDelaySummary& total,
    std::vector<PacketOutcomeDelaySummary>& per_flow) const;

void GetPacketExpiryAwareRateStats(
    PacketExpiryAwareRateSummary& total,
    std::vector<PacketExpiryAwareRateSummary>& per_flow) const;

PacketExpiryAwareRateSummary GetLastPacketExpiryAwareRateStats() const;

void GetPacketOutcomeSamples(std::vector<PacketOutcomeSample>& samples) const;
```

The outcome sample stream is validation-only. It is needed to compute an exact warmup-excluded p95; it is not an actor input or reward term.

`PacketOutcomeSample.terminal_tti` is one-based: a delivery passed to `RecordMacServedBytes(served_bytes, current_tti=0)` records terminal TTI 1; an expiry records `arrival_tti + ttl_tti - 1`. This makes the formal measured interval exactly `1000 < terminal_tti <= 5000`.

### Online protocol additions

Bump both C++ and Python protocol versions from 11 to 12. Append six reward terms, preserving indices 0--16:

```text
17 packet_expired_count
18 packet_expired_system_time_ms
19 packet_outcome_count
20 packet_outcome_system_time_ms
21 packet_expiry_aware_rate_mbps
22 packet_expiry_aware_rate_per_packet_mean_mbps
```

Extend per-UE packet diagnostics with cumulative:

```text
actual_ue_packet_expired_packets
actual_ue_packet_expired_system_time_ms
actual_ue_packet_outcome_system_time_ms
actual_ue_packet_expiry_aware_rate_mbps
actual_ue_packet_expiry_aware_rate_per_packet_mean_mbps
```

Do not silently substitute delivered-only values when these fields are absent.

### New final log records

Keep the old delivered-only records and add exactly:

```text
TRAFFIC_PKT_OUTCOME_DELAY
TRAFFIC_PKT_RATE_EA
UE_PKT_OUTCOME_DELAY
UE_PKT_RATE_EA
```

Also emit `TRAFFIC_PKT_OUTCOME_SAMPLES_CSV path=<path> rows=<count>` when `CUMAC_PACKET_DELAY_SAMPLES=1`. CSV columns are:

```text
ue_id,packet_id,terminal_tti,outcome,system_time_ms
```

where `outcome` is `delivered` or `expired`.

### Independent actor and policy API

Create focused modules rather than adding PFQ-relative branches to the existing candidate decoder:

```text
# training/stageb_hgraph/independent_actor.py
IndependentActorConfig(
    cell_feat_dim: int = 5,
    ue_feat_dim: int = 16,
    prg_feat_dim: int = 12,
    hidden_dim: int = 128,
    message_layers: int = 2,
    up_edge_attr_dim: int = 5,
    attention_heads: int = 4,
    max_candidates: int = 12,
    dropout: float = 0.0,
)

PFQIndependentReuseActor.actor_family = "pfq_independent_reuse_v1"
PFQIndependentReuseActor.requires_pfq_features = False
PFQIndependentReuseActor.encode_graph(
    graph: SparseEntityGraph,
) -> dict[str, torch.Tensor]
PFQIndependentReuseActor.apply_reuse_attention(
    prg_embeddings: torch.Tensor,
    n_cell: int,
    n_prg: int,
) -> torch.Tensor                         # [P,C,H]
PFQIndependentReuseActor.score_reuse_group(
    encoded: dict[str, torch.Tensor],
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    local_prg: int,
    sequential_features: torch.Tensor,
) -> torch.Tensor                         # [C,K+1]
PFQIndependentReuseActor.model_config_dict(
) -> dict[str, int | float | str | bool]
```

`encode_graph()` reuses the existing template encoders and `HeteroMessageLayer` and returns raw entity embeddings plus `reuse_embeddings` produced by `apply_reuse_attention()`. `apply_reuse_attention()` reshapes PRG embeddings to `[P,C,H]`, applies `nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)`, and uses residual plus layer normalization. `score_reuse_group()` gathers every valid UE and UE-PRG edge for the three tokens at the requested local PRG, concatenates non-PFQ sequential features, and scores UE/Blank categoricals.

```python
# training/stageb_hgraph/independent_policy.py
@dataclass(frozen=True)
class IndependentPolicyTrace:
    choice: torch.Tensor             # [C*P], candidate class per graph token
    local_prg_order: torch.Tensor    # exactly arange(P)
    valid_at_step: torch.Tensor      # [P,C,K+1]
    logp: torch.Tensor
    entropy: torch.Tensor

resolve_full_candidate_width(graph: SparseEntityGraph) -> int
def estimate_candidate_service_bytes(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
) -> torch.Tensor                    # [C*P,K]
select_independent_action(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    mode: Literal["sample", "greedy"],
    generator: torch.Generator | None = None,
) -> tuple[IndependentPolicyTrace, DecodedType0Action]
replay_independent_trace(
    actor: PFQIndependentReuseActor,
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    trace: IndependentPolicyTrace,
) -> tuple[torch.Tensor, torch.Tensor, DecodedType0Action]
independent_teacher_targets_from_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    action_ue_select: torch.Tensor,
    action_prg_alloc: torch.Tensor,
    valid_at_step: torch.Tensor,
) -> torch.Tensor
materialize_independent_choices_to_type0(
    graph: SparseEntityGraph,
    packed: PackedPrgCandidates,
    choice: torch.Tensor,
) -> DecodedType0Action
```

The service estimator is fixed, not learned and not PFQ-derived:

```text
spectral_eff = sum_l log2(1 + post_eq_sinr_linear[l])
rate_bps = spectral_eff * (16 PRB * 12 subcarriers * 30 kHz) * (11 / 14)
service_bytes = max(1, rate_bps * 0.0005 / 8)
```

At every TTI, sequential features per UE are remaining-demand ratio, allocated-PRG ratio, and cumulative-predicted-service ratio. Process only `p=0..16`; score the three cells from the same pre-update state for that reuse group, then update selected UEs before `p+1`. Blank is always valid. Replay must use the stored order and mask exactly.

Use explicit denominators: remaining demand and cumulative predicted service divide by `max(initial_demand_bytes, 1)`, and allocated PRGs divide by 17. Clamp only these input ratios to `[0,1]`; retain the unrounded byte state for the hard headroom mask.

### Independent reward API

Create `training/stageb_hgraph/independent_reward.py`:

```python
@dataclass(frozen=True)
class IndependentRewardConfig:
    treatment: Literal["A", "B", "C", "D"]
    offered_bytes_per_tti: float = 108000.0
    slot_duration_s: float = 0.0005
    gamma: float = 0.99
    ue_window_steps: int = 400
    delay_scale_p95: float = 0.0
    ue_utility_scale_p95: float = 0.0
    goodput_weight: float = 1.0
    failed_weight: float = 1.0
    expired_weight: float = 2.0
    delay_weight: float = 0.25
    ue_utility_weight: float = 0.25

ExpiryAwareUeWindowTracker(
    n_ue: int,
    window_steps: int = 400,
)
ExpiryAwareUeWindowTracker.reset() -> None
ExpiryAwareUeWindowTracker.update(
    demand_mask: np.ndarray,
    successful_goodput_bytes: np.ndarray,
    cumulative_delivered_time_ms: np.ndarray,
    cumulative_expired_time_ms: np.ndarray,
) -> float

def compute_independent_reward(
    before_graph: SparseEntityGraph,
    after_graph: SparseEntityGraph,
    info: Mapping[str, object],
    tracker: ExpiryAwareUeWindowTracker,
    cfg: IndependentRewardConfig,
) -> dict[str, float]
```

Use:

```text
G = successful_goodput_bytes / offered_bytes
F = max(0, scheduled_bytes - successful_goodput_bytes) / offered_bytes
E = expired_bytes / offered_bytes
Delta_D = D(before) - gamma * D(after)
Delta_U = gamma * U(after_window) - U(before_window)
reward = G - F - 2E + 0.25*clip(Delta_D/scale_D) + 0.25*clip(Delta_U/scale_U)
```

Treatments A/B use only `G-F-2E`; C adds delay; D adds delay and UE utility. `D` reuses the existing non-PFQ queue-pressure potential over critical/near-TTL/age-weighted backlog and service gap. `U` is the demand-active UE mean of `log(epsilon + rate_mbps)`, where `rate_mbps = successful_goodput_bits / (outcome_system_time_ms * 1000)`, over the trailing 400 TTIs. Successful bits come from `actual_ue_goodput_bytes`; outcome time is the delta of cumulative delivered plus expired packet time. A demand-active UE with zero terminal outcome time in the window has rate zero, not infinity. Missing expiry-aware fields are a hard error for D.

---

### Task 1: Add exact expiry terminal accounting with focused C++ tests

**Files:**

- Modify: `cuMAC/examples/trafficModel/trafficGenerator.hpp`
- Modify: `cuMAC/examples/trafficModel/trafficService.hpp`
- Create: `cuMAC/examples/trafficModel/trafficPacketAccountingTest.cpp`
- Modify: `cuMAC/examples/trafficModel/CMakeLists.txt`

- [ ] **Step 1: Write the failing accounting test**

Construct `RadioResource` directly with one or two flows and controlled `FlowData.accepted_spans`. Add cases for:

1. one 3000-byte packet delivered at 0.5 ms: 24,000 bits, 48 Mbps, delivered-only equals expiry-aware;
2. one 3000-byte packet arriving at TTI 1 and expired with TTL 400 slots: exactly 200 ms outcome time, zero delivered bits, zero per-packet EA rate;
3. one delivered plus one expired outcome: count/time identities and aggregate EA rate;
4. per-flow counts, times, and bits sum to cluster totals;
5. a partially consumed but incomplete packet expires as one zero-completed-bit outcome while remaining expired bytes stay separately counted.

Register `trafficPacketAccountingTest` with CTest.

- [ ] **Step 2: Configure and prove the test fails before implementation**

```bash
cmake -S . -B build.reuse_actor \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build.reuse_actor --target trafficPacketAccountingTest -j8
```

Expected: compile/link failure because the outcome summary/getter API is not implemented yet.

- [ ] **Step 3: Implement terminal outcomes**

In `RadioResource`:

- retain all existing delivered-only vectors/counters;
- add per-flow and cluster expired packet time, outcome delays, and per-packet rate sums;
- reset the new per-TTI accumulator in `AdvanceToTti()` before the early return;
- on expiry use `arrival_time_ms + max_packet_age_tti * slot_duration_ms`, not the raw `current_tti` time, and record `terminal_tti = arrival_tti + max_packet_age_tti - 1` for the exact last eligible-slot boundary;
- mark the lifecycle record terminal/expired and add an outcome sample;
- on delivery add the same delivered time/bit contribution to both old delivered-only and new outcome summaries;
- on delivery store the one-based outcome sample terminal TTI as `current_tti + 1`;
- in `RecordServedBytes()`, append delivery contributions to the per-TTI EA accumulator without erasing expiries already observed by `AdvanceToTti()`;
- keep pending packets right-censored.

- [ ] **Step 4: Run the focused test**

```bash
cmake --build build.reuse_actor --target trafficPacketAccountingTest -j8
ctest --test-dir build.reuse_actor -R '^trafficPacketAccountingTest$' --output-on-failure
```

Expected: `100% tests passed` and all exact 0.5/200 ms identities hold.

- [ ] **Step 5: Commit only the tracker slice**

```bash
git add cuMAC/examples/trafficModel/trafficGenerator.hpp \
        cuMAC/examples/trafficModel/trafficService.hpp \
        cuMAC/examples/trafficModel/trafficPacketAccountingTest.cpp \
        cuMAC/examples/trafficModel/CMakeLists.txt
git diff --cached
git commit -m "feat: add expiry-aware packet outcome accounting"
```

### Task 2: Propagate expiry-aware metrics through logs, bridge, and KPI parsing

**Files:**

- Modify carefully: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineProtocol.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp`
- Modify: `training/stageb_hgraph/online_protocol.py`
- Modify: `training/stageb_hgraph/bridge_env.py`
- Modify: `training/stageb_hgraph/reward_utils.py`
- Modify: `cuMAC/scripts/summarize_stageA_kpi.py`
- Create: `training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py`
- Create: `cuMAC/scripts/tests/test_summarize_stageA_expiry_aware.py`

- [ ] **Step 1: Write failing Python tests**

Test that:

- protocol v12 decodes 23 reward terms without shifting indices 0--16;
- a payload with the expanded per-UE arrays is consumed exactly, with no trailing bytes;
- reward extraction returns all six appended terms;
- the four new final log records parse into cluster and per-UE fields;
- a sample CSV can be filtered by `terminal_tti > 1000` and produces the exact measured-window p95.

Run:

```bash
pytest -q training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py
python3 -m unittest cuMAC.scripts.tests.test_summarize_stageA_expiry_aware -v
```

Expected: failures for missing v12 fields/parsers.

- [ ] **Step 2: Extend C++ codec/protocol and Python decoder in append-only order**

- bump protocol version and reward-term count on both sides;
- append reward fields after old index 16;
- append per-UE packet arrays in the same order in `StepState`, reserve-size calculation, serializer, and Python parser;
- set the existing packet-diagnostic state flag only when every old and new per-UE array has exactly `nActiveUe` elements;
- add explicit `*_available` info and preserve dtypes.

- [ ] **Step 3: Add common final reporting and outcome CSV**

In `main.cpp`, call both old and new `TrafficService` getters for all scheduler modes. Emit delivered-only and new EA records side by side. Write outcome samples when packet samples are enabled. Do not relabel the old fields.

Because `main.cpp` is already dirty, inspect the file diff and later stage only these hunks.

- [ ] **Step 4: Extend KPI JSON without breaking old logs**

Add new parser functions and merge fields only when records are present. Expose at minimum:

- cluster outcome count/delay mean/p95;
- cluster EA aggregate and per-packet-mean rate;
- UE macro, p10, and minimum EA aggregate rate;
- UE macro outcome delay and p95 summary;
- outcome sample path/row count.

Old logs without EA records remain parseable, but the independent gate must reject missing new fields.

- [ ] **Step 5: Verify protocol, parser, focused C++, and simulator build**

```bash
pytest -q training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py
python3 -m unittest cuMAC.scripts.tests.test_summarize_stageA_expiry_aware -v
ctest --test-dir build.reuse_actor -R '^trafficPacketAccountingTest$' --output-on-failure
cmake --build build.reuse_actor --target multiCellSchedulerUeSelection -j8
```

Expected: all tests pass and the simulator links with protocol v12.

- [ ] **Step 6: Commit only the metric transport slice**

```bash
git add cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.h \
        cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp \
        cuMAC/examples/onlineTrainBridge/OnlineProtocol.h \
        cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.h \
        cuMAC/examples/onlineTrainBridge/OnlineBridgeServer.cpp \
        training/stageb_hgraph/online_protocol.py \
        training/stageb_hgraph/bridge_env.py \
        training/stageb_hgraph/reward_utils.py \
        cuMAC/scripts/summarize_stageA_kpi.py \
        training/stageb_hgraph/tests/test_online_protocol_expiry_aware.py \
        cuMAC/scripts/tests/test_summarize_stageA_expiry_aware.py
git add -p cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git diff --cached
git commit -m "feat: expose expiry-aware packet metrics"
```

### Task 3: Implement the PFQ reference/calibration path

**Files:**

- Create: `training/stageb_hgraph/independent_objectives.py`
- Create: `training/stageb_hgraph/pfq_reference_calibration.py`
- Create: `training/stageb_hgraph/tests/test_pfq_reference_calibration.py`
- Generated smoke only: `output/pfq_independent_reuse/calibration_smoke/`

- [ ] **Step 1: Test warmup exclusion and scale calculation**

Given a synthetic 5000-row trace, verify:

- rows 1--1000 never contribute to comparator means or p95 scales;
- rows 1001--5000 contribute exactly 4000 measured samples;
- p95 uses absolute raw `Delta_D`/`Delta_U`;
- a zero p95 is recorded and later fails only the corresponding auxiliary treatment;
- binary SHA256, source revision, scenario fields, and protocol version are mandatory manifest keys.
- an outcome exactly at `terminal_tti=1000` is excluded and one at `terminal_tti=1001` is included;
- the learned-policy side will use the same measured TTI boundary convention.

- [ ] **Step 2: Implement shared objective primitives and PFQ pass-through calibration**

Put `compute_delay_debt_potential`, `ExpiryAwareUeWindowTracker`, `compute_ue_expiry_aware_utility`, and p95 scale calculation in `independent_objectives.py`. Both calibration and the later learned-policy reward must import these exact functions; do not duplicate formulas.

Launch the online bridge with native queue-aware PFQ (`-b 2`) and send the provided teacher Type-0 action unchanged each step. Record 5000 TTIs, using the first 1000 only to fill state/history windows. Persist:

- `tti_trace.jsonl` with goodput, scheduled bytes, failed bytes, BLER, expired bytes, raw `Delta_D`, raw `Delta_U`, and per-UE EA diagnostics;
- `packet_outcome_samples.csv` for exact measured-window delay distribution;
- `reward_scales.json` containing seed-41 steady p95 absolute values;
- `reference_manifest.json` with canonical scenario and binary/source fingerprints;
- `reference_summary.json` with the formal 4000-TTI comparator plus full-run diagnostic fields.

This script must fail if teacher actions, post-EQ input, EA diagnostics, the requested measured-row count, or scenario fields are missing. Add `--formal`; formal mode rejects any horizon other than 1000 warmup plus 4000 measured TTIs.

Define the measured packet-outcome population as terminal events with `1000 < terminal_tti <= 5000`. Packets arriving during warmup but terminating after TTI 1000 therefore count, matching steady-state carry-over; pending packets at TTI 5000 remain right-censored. Use this same rule for PFQ and every treatment.

- [ ] **Step 3: Run only a 32-step calibration plumbing smoke**

```bash
python3 -m training.stageb_hgraph.pfq_reference_calibration \
  --sim-bin build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection \
  --topology-seed 41 \
  --traffic-seed 41 \
  --warmup-steps 16 \
  --measure-steps 16 \
  --out-dir output/pfq_independent_reuse/calibration_smoke
```

Expected: 32 bridge steps, exact manifest fields, non-empty EA diagnostics, and valid raw objective traces. This is plumbing evidence only. Defer the formal 5000-TTI PFQ reference until Task 12, after all implementation commits, so PFQ and treatments share the same source revision and binary fingerprint.

- [ ] **Step 4: Commit code/tests, not generated output**

```bash
git add training/stageb_hgraph/independent_objectives.py \
        training/stageb_hgraph/pfq_reference_calibration.py \
        training/stageb_hgraph/tests/test_pfq_reference_calibration.py
git diff --cached
git commit -m "feat: calibrate independent reward against PFQ"
```

### Task 4: Enforce full PFQ-free candidates and the deletion contract

**Files:**

- Modify: `training/stageb_hgraph/edge_generator.py`
- Modify: `training/stageb_hgraph/feature_snapshot.py`
- Create: `training/stageb_hgraph/independent_contract.py`
- Create: `training/stageb_hgraph/tests/test_independent_contract.py`

- [ ] **Step 1: Write failing candidate/deletion tests**

Cover:

- each `(cell,PRG)` row contains every and only schedulable, associated, positive-demand UE exactly once;
- Blank is the only class when no demand is active;
- the fixed width equals maximum static association count and raises if configured smaller;
- `include_pfq_score_anchor=False` produces no PFQ edge column;
- deleting `teacher_action_ue_select`, `teacher_action_prg_alloc`, and any future key whose normalized name contains `pfq` or `teacher` leaves actor-visible graph tensors bitwise equal;
- a contract validator rejects `topk_diversity`, PFQ anchors, PFQ residual configuration, or `requires_pfq_features=true`.

- [ ] **Step 2: Implement the narrow contract helper**

`independent_contract.py` owns:

```python
strip_pfq_fields(observation: Mapping[str, object]) -> dict[str, object]
validate_independent_graph(graph: SparseEntityGraph, *, max_candidates: int) -> None
validate_independent_checkpoint(checkpoint: Mapping[str, object]) -> None
assert_graph_tensor_equality(a: SparseEntityGraph, b: SparseEntityGraph) -> None
```

Do not change legacy default behavior globally. The independent runner explicitly constructs `EdgeGeneratorConfig(candidate_mode="all_demand_ues", include_pfq_score_anchor=False, include_candidate_risk_features=False)`.

- [ ] **Step 3: Verify**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_contract.py \
          training/stageb_hgraph/tests/test_feature_snapshot.py
```

Expected: exact tensor equality and complete candidate coverage.

- [ ] **Step 4: Commit**

```bash
git add training/stageb_hgraph/edge_generator.py \
        training/stageb_hgraph/feature_snapshot.py \
        training/stageb_hgraph/independent_contract.py \
        training/stageb_hgraph/tests/test_independent_contract.py
git diff --cached
git commit -m "feat: enforce PFQ-free full candidate contract"
```

### Task 5: Implement the lightweight reuse-attention actor

**Files:**

- Create: `training/stageb_hgraph/independent_actor.py`
- Create: `training/stageb_hgraph/tests/test_independent_actor.py`

- [ ] **Step 1: Write failing actor tests**

Assert:

- encoded Cell/UE/PRG/reuse shapes match 3/36/51 entities;
- attention input/output is `[17,3,H]`, has four heads, residual connection, and finite gradients;
- changing another cell's same-local-PRG embedding changes the target cell reuse context, while changing a different local PRG does not before HGraph recomputation;
- scoring returns `[3,K+1]`, masks no classes internally, and contains no module/parameter/config name with `pfq`, `proposal`, `residual_action`, or `fallback`;
- actor-only parameter count and serialized checkpoint bytes are reported as complexity diagnostics; no fixed “smaller than legacy” threshold is a validity or promotion gate because the legacy object contains unused multi-path heads;
- checkpoint metadata contains `actor_family=pfq_independent_reuse_v1` and `requires_pfq_features=false`.

- [ ] **Step 2: Implement only the selected architecture**

Reuse existing template encoders and `HeteroMessageLayer`; do not add a graph framework, autoregressive cross-cell decoder, PFQ head, success auxiliary head, slot head, repair head, or export path. Add one reuse attention block, one candidate edge projection, one UE scorer, one Blank scorer, and the minimal state pooling needed by the existing critic.

- [ ] **Step 3: Verify**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_actor.py
```

Expected: shape, locality, gradient, metadata, and complexity-report tests pass.

- [ ] **Step 4: Commit**

```bash
git add training/stageb_hgraph/independent_actor.py \
        training/stageb_hgraph/tests/test_independent_actor.py
git diff --cached
git commit -m "feat: add PFQ-independent reuse attention actor"
```

### Task 6: Add fixed-order headroom selection and identity-only materialization

**Files:**

- Create: `training/stageb_hgraph/independent_policy.py`
- Create: `training/stageb_hgraph/tests/test_independent_policy.py`
- Create: `training/stageb_hgraph/tests/test_independent_materializer.py`

- [ ] **Step 1: Write failing policy tests**

Cover:

- local PRGs are visited exactly `0..16`, with the three cells scored before that PRG's state update;
- remaining demand, PRG count, and predicted-service state update only for selected nonblank UEs;
- headroom validity is monotone after remaining demand reaches zero;
- service estimates follow the locked post-EQ formula and are at least one byte;
- sampling and greedy modes obey stored masks; replay returns the same choices, log-probability, entropy, and final action;
- a stored-mask or action mismatch raises instead of being repaired.

- [ ] **Step 2: Write failing materializer tests**

Assert that it:

- sorts unique selected UEs by ascending UE ID inside each cell;
- assigns each PRG to exactly the slot of its chosen UE;
- preserves Blank and every selected UE identity;
- raises on wrong-cell choice, missing slot, slot overflow, invalid PRG mask, or duplicate candidate row;
- reports zero legacy cap-drop, fallback, safe-fill, repair, and non-hard-override counts;
- passes `type0_action_preflight_reject_counts()` with all zeros.

- [ ] **Step 3: Implement the policy kernel and materializer**

Do not call `select_budgeted_candidate_actions`, `decode_candidate_actions_to_type0`, safe-fill, blank budget, slot repair, PFQ fallback, or any allocation-set counterfactual. The only post-choice work is ascending-ID representation conversion and assertions.

- [ ] **Step 4: Verify**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_policy.py \
          training/stageb_hgraph/tests/test_independent_materializer.py
```

Expected: exact rollout/replay/action equality and zero preflight rejects.

- [ ] **Step 5: Commit**

```bash
git add training/stageb_hgraph/independent_policy.py \
        training/stageb_hgraph/tests/test_independent_policy.py \
        training/stageb_hgraph/tests/test_independent_materializer.py
git diff --cached
git commit -m "feat: add sequential independent Type-0 policy"
```

### Task 7: Add direct PFQ behavior-cloning labels without actor leakage

**Files:**

- Modify: `training/stageb_hgraph/online_imitation.py`
- Create: `training/stageb_hgraph/tests/test_independent_bc.py`

- [ ] **Step 1: Test direct teacher conversion**

Build PFQ Type-0 actions with UE slots in a different order from candidate rows. Verify conversion follows the PRG's allocated slot to the UE ID and then finds that UE in the full row. Blank maps only to Blank. Missing PFQ UE, invalid slot, wrong-cell UE, or headroom-invalid label raises.

- [ ] **Step 2: Add an independent BC branch**

When `--action-head-mode independent`:

- construct `PFQIndependentReuseActor`;
- convert teacher actions to direct per-token classes;
- run the same fixed sequential masks used by rollout;
- optimize masked cross-entropy over 4000 teacher TTIs;
- save `hgraph_independent_bc_last.pt` and `hgraph_independent_bc_best.pt` with the independent contract metadata;
- never append teacher tensors to graph features;
- leave PPO out of this standalone BC process.

Keep legacy slot/candidate imitation paths unchanged.

- [ ] **Step 3: Run unit tests and a 32-teacher-TTI preflight**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_bc.py
python3 -m training.stageb_hgraph.online_imitation \
  --sim-bin build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection \
  --action-head-mode independent \
  --candidate-mode all_demand_ues \
  --include-pfq-score-anchor 0 \
  --episodes 1 \
  --max-steps 32 \
  --topology-seed 41 \
  --out-dir output/pfq_independent_reuse/smoke_bc
```

Expected: 32 legal labels, no missing candidate, no materialization mismatch, and a checkpoint marked PFQ-independent.

- [ ] **Step 4: Commit only the independent branches**

```bash
git add training/stageb_hgraph/online_imitation.py \
        training/stageb_hgraph/tests/test_independent_bc.py
git diff --cached
git commit -m "feat: warm start independent actor from PFQ labels"
```

### Task 8: Implement the four locked reward treatments

**Files:**

- Create: `training/stageb_hgraph/independent_reward.py`
- Modify: `training/stageb_hgraph/independent_objectives.py`
- Create: `training/stageb_hgraph/tests/test_independent_reward.py`
- Modify: `training/stageb_hgraph/online_ppo.py`

- [ ] **Step 1: Write failing reward tests**

Use controlled before/after graphs and cumulative per-UE counters to verify:

- 108,000 successful bytes and no failures/expiry gives base reward 1;
- scheduled minus goodput bytes is the failed-byte penalty;
- one expired 3000-byte packet adds `2 * 3000/108000` penalty;
- lower delay debt gives positive `Delta_D`;
- an expiry adds 200 ms and zero successful bits to its UE window, lowering `U`;
- serving a low-rate UE gives a larger log-utility increment than the same bits to a high-rate UE;
- A/B omit both auxiliaries, C includes only delay, D includes both;
- scales use frozen PFQ p95, clip to `[-1,1]`, and zero/missing scale fails the relevant treatment;
- delivered-only packet fields cannot satisfy treatment D's preflight.

- [ ] **Step 2: Implement the tracker and reward**

Reuse/re-export the Task-3 tracker and objective functions. Track cumulative-counter deltas defensively: a negative delta means environment reset and resets the window, not a negative outcome. Fill the 400-step window during the single 1000-step warmup. Log every raw, scaled, clipped, weighted, and total component so H3 can be evaluated without rerunning.

- [ ] **Step 3: Integrate only one new reward mode**

Add `--reward-mode pfq_independent` and `--independent-treatment {A,B,C,D}`. Load scales from the frozen seed-41 `reward_scales.json`; do not calculate or update scales from learned-policy data. Reject noncanonical offered load for this first experiment unless explicitly supplied and written to the manifest.

- [ ] **Step 4: Verify**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_reward.py \
          training/stageb_hgraph/tests/test_queue_pressure_reward.py
```

Expected: exact synthetic values and no regression in existing queue-pressure tests.

- [ ] **Step 5: Commit**

```bash
git add training/stageb_hgraph/independent_reward.py \
        training/stageb_hgraph/independent_objectives.py \
        training/stageb_hgraph/tests/test_independent_reward.py \
        training/stageb_hgraph/online_ppo.py
git diff --cached
git commit -m "feat: add expiry-aware independent scheduler reward"
```

### Task 9: Connect independent rollout/replay, checkpointing, and evaluation

**Files:**

- Modify: `training/stageb_hgraph/online_ppo.py`
- Modify: `training/stageb_hgraph/eval_checkpoint_online.py`
- Create: `training/stageb_hgraph/tests/test_independent_online_path.py`
- Create: `training/stageb_hgraph/tests/test_independent_pfq_deletion.py`

- [ ] **Step 1: Write failing integration tests**

Test with fixture graphs that:

- `_evaluate_graph_batch` uses the independent actor/action path and existing critic pooling;
- rollout stores `IndependentPolicyTrace` and the exact submitted arrays;
- PPO replay uses stored order/masks and produces zero action mismatch;
- policy gradients reach the reuse attention and UE scorer;
- no gradient/parameter belongs to PFQ, legacy slot head, safe-fill, repair, or teacher auxiliary loss;
- checkpoint loading rejects PFQ-required configs and accepts the independent family;
- one observation with teacher/PFQ fields and one with them deleted produce bitwise-equal graph tensors, logits, greedy trace, and final Type-0 arrays.

- [ ] **Step 2: Add narrow independent branches to PPO**

Branch at actor construction, action selection, replay log-probability, and checkpoint serialization. Do not route independent actions through legacy candidate repairs. Add `--rollout-warmup-once 1`; for this mode, apply 1000 warmup steps only before iteration 1 and zero before later persistent iterations. Preserve old behavior for legacy modes.

Independent PPO loads the frozen standalone BC checkpoint through `--init-checkpoint`; reject `--imitation-episodes > 0` and any nonzero teacher auxiliary coefficient for this actor family. A uses no init checkpoint.

For the locked pilot:

- 20 iterations;
- 512 measured steps/iteration;
- candidate checkpoint interval 4;
- same optimizer, PPO epochs, minibatch size, gamma, GAE, entropy/value coefficients for A--D;
- B--D load the exact same BC checkpoint hash;
- teacher auxiliary coefficient, PFQ residual initialization, PFQ anchor, candidate risk features, blank budget, safe-fill, slot repair, and all legacy auxiliary heads are zero/off.

- [ ] **Step 3: Extend deterministic evaluator**

Dispatch on checkpoint `actor_family`. For independent checkpoints, run greedy fixed-order selection/materialization, verify action replay every step, and emit legality, deletion-test, binary fingerprint, measured TTI range, and new EA metrics. Never infer an independent actor from old candidate metadata.

- [ ] **Step 4: Verify focused and existing causal tests**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_online_path.py \
          training/stageb_hgraph/tests/test_independent_pfq_deletion.py \
          training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py
```

Expected: exact replay/deletion equality and existing causal tests remain green.

- [ ] **Step 5: Commit**

```bash
git add training/stageb_hgraph/online_ppo.py \
        training/stageb_hgraph/eval_checkpoint_online.py \
        training/stageb_hgraph/tests/test_independent_online_path.py \
        training/stageb_hgraph/tests/test_independent_pfq_deletion.py
git diff --cached
git commit -m "feat: train and evaluate independent reuse actor"
```

### Task 10: Add one fixed experiment runner and guarded selector

**Files:**

- Create: `cuMAC/scripts/run_stageB_pfq_independent_reuse.sh`
- Create: `training/stageb_hgraph/independent_gate.py`
- Create: `training/stageb_hgraph/tests/test_independent_gate.py`

- [ ] **Step 1: Test manifests, checkpoint selection, and gates**

Synthetic PFQ/treatment manifests must prove:

- any binary/source/scenario mismatch fails before metric comparison;
- pre-correction or missing EA metrics fail;
- checkpoints at iterations 4/8/12/16/20 are scored only if replay/legality/reliability gates pass;
- selected score uses PFQ-relative goodput and UE EA rate and penalizes worse outcome delay;
- the seed-41 architecture gate uses 99% goodput, +0.2 pp BLER, +0.05 pp expiry, zero replay mismatch, zero illegal action, and exact deletion equality;
- C/D are preferred only when p95 outcome delay, UE macro EA rate, and UE p10 EA rate meet PFQ while the architecture gate remains satisfied;
- multi-seed launch is false unless seed 41 passes.
- the actor's outcome-delay p95 is computed from terminal samples in the same measured TTI interval as PFQ, never from full-run or delivered-only samples.

For H2, define time-to-gate at the four-iteration checkpoint cadence. B supports a BC sample-efficiency benefit when it first reaches the H1 gate in fewer measured PPO TTIs than A, or when B reaches it and A does not within budget. If both first pass at the same checkpoint, record no demonstrated benefit at the available 2048-TTI resolution.

- [ ] **Step 2: Implement a fixed-scope runner**

Support only these phases:

```text
build
pfq-reference
validity
bc
pilot-A
pilot-B
pilot-C
pilot-D
select-and-eval-seed41
paired-seeds-41-43
```

The script locks the canonical scenario, hashes the binary and BC checkpoint, enables outcome samples, saves all command lines/environment/manifests, and refuses an existing output directory unless `--resume` points to a compatible manifest. It does not expose reward weights, candidate pruning, fallback, autoregression, or topology expansion as options.

Treatments B--D must compare the SHA256 of their `--init-checkpoint` and fail unless identical. A has no init checkpoint. All use 20x512 measured PPO steps and one 1000-step warmup. Save checkpoint candidates every four iterations.

- [ ] **Step 3: Implement selection/report JSON**

`independent_gate.py` reads PFQ reference, PPO iteration metrics, outcome samples, deterministic evaluation, and manifests. It writes:

- `checkpoint_selection.json`;
- `seed41_gate.json`;
- `treatment_comparison.json`;
- after authorization by a seed-41 pass, `paired_seed_summary.json`.

Every decision contains the actual numerator, denominator, absolute delta, relative delta, threshold, and pass/fail reason.

- [ ] **Step 4: Verify**

```bash
pytest -q training/stageb_hgraph/tests/test_independent_gate.py
bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
```

- [ ] **Step 5: Commit**

```bash
git add cuMAC/scripts/run_stageB_pfq_independent_reuse.sh \
        training/stageb_hgraph/independent_gate.py \
        training/stageb_hgraph/tests/test_independent_gate.py
git diff --cached
git commit -m "feat: add independent scheduler research gate"
```

### Task 11: Run the minimum falsifiable smoke and validity gate

**Generated output only:** `output/pfq_independent_reuse/smoke/`

- [ ] **Step 1: Rebuild from the final implementation source**

```bash
cmake -S . -B build.reuse_actor \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build.reuse_actor \
  --target trafficPacketAccountingTest multiCellSchedulerUeSelection -j8
```

- [ ] **Step 2: Run all focused validity tests**

```bash
ctest --test-dir build.reuse_actor -R '^trafficPacketAccountingTest$' --output-on-failure
pytest -q training/stageb_hgraph/tests/test_independent_*.py
python3 -m unittest cuMAC.scripts.tests.test_summarize_stageA_expiry_aware -v
bash -n cuMAC/scripts/run_stageB_pfq_independent_reuse.sh
```

Expected: all pass.

- [ ] **Step 3: Run 32 online TTIs with greedy and sampled actions**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh validity \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse/smoke \
  --steps 32
```

Required result:

- complete full candidate rows;
- zero illegal/preflight/native rejects;
- zero rollout-replay-submitted mismatch;
- exact PFQ-deletion equality;
- finite logits, rewards, and gradients;
- non-empty EA diagnostics;
- no fallback/repair/safe-fill count.

If any condition fails, stop and debug it before BC or PPO. Do not reinterpret an invalid smoke as model underperformance.

### Task 12: Execute the approved seed-41 experiment, then conditionally seeds 41--43

**Generated output only:** `output/pfq_independent_reuse/`

- [ ] **Step 1: Confirm/freeze the PFQ seed-41 reference**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh pfq-reference \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Expected: 1000 warmup + 4000 measured PFQ TTIs from the final binary, exact EA sample p95, and frozen nonzero scales for enabled auxiliary treatments.

- [ ] **Step 2: Train one 4000-TTI PFQ BC checkpoint**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh bc \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Required: no missing/illegal labels, zero materialization mismatch, PFQ-independent metadata, and one checkpoint hash reused by B--D.

- [ ] **Step 3: Run A and B first**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh pilot-A \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh pilot-B \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Each gets exactly 10,240 measured PPO TTIs; B's 4000 BC TTIs and both arms' 1000 warmup TTIs are reported separately.

Evaluate the guarded checkpoint candidates. If neither A nor B approaches 99% PFQ goodput within the locked reliability/expiry envelope, stop and report H1 unsupported at this architecture/budget. Do not run C/D.

- [ ] **Step 4: Only if H1 remains plausible, run C and D**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh pilot-C \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh pilot-D \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Confirm B--D init-checkpoint hashes and all PPO budgets/configs match before comparing.

- [ ] **Step 5: Select and deterministically evaluate the seed-41 winner**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh select-and-eval-seed41 \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Required final seed-41 evidence:

- new-binary PFQ vs actor goodput, BLER, expiry;
- outcome delay p95 including expired TTL time;
- UE macro and p10 EA successful rate;
- zero illegal/replay mismatch;
- exact deletion equality;
- H1/H2/H3 decision with reasons.

- [ ] **Step 6: Only after seed-41 gate pass, run paired seeds 41--43**

```bash
cuMAC/scripts/run_stageB_pfq_independent_reuse.sh paired-seeds-41-43 \
  --build-dir build.reuse_actor \
  --out-dir output/pfq_independent_reuse
```

Each PFQ/actor pair uses 1000 warmup + 4000 measured TTIs, the same binary, the fixed winning checkpoint, and matched topology/traffic seed. Do not continue to 10 seeds, 7 cells, ONNX/TensorRT, architecture refactoring, or reward sweeps.

---

## Final research report checklist

Report only after the relevant stop condition is reached:

- exact packet-metric semantic change, including the remaining MAC-dequeue vs PHY-success limitation;
- binary/source/BC checkpoint fingerprints and fixed scenario;
- PFQ, A, B, C, D measured budgets and exclusions;
- new-binary PFQ and learned-policy goodput/BLER/expiry/outcome-delay/UE-EA-rate results;
- whether H1, H2, and H3 are supported;
- whether PFQ deletion equality, action replay, and legality passed;
- dominant failure mode if H1 is falsified;
- main remaining uncertainty after the seed-41 or three-seed stop.

Research completion does not imply production readiness.

# PFQ-v2 Marginal Expected Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PFQ-v2 as an independent `-b 4` baseline, then determine with paired experiments whether allocation-set marginal expected delivery improves PFQ-v1 without violating the approved reliability, tail, weak-UE, worst-seed, or runtime safeguards.

**Architecture:** Extract the existing PDSCH TBS calculation into a shared pure C++ primitive, build a side-effect-free allocation-set predictor and Type-0 greedy allocator around it, and call the same allocator from the CPU scheduler and the CUDA scheduler's existing host postprocessing boundary. Keep OLLA updates transactional, keep the legacy MAC/HARQ/buffer path unchanged, and add a separate successful-delivery shadow ledger plus machine-readable diagnostics. Extend the existing Stage-B runners and KPI parser so cap-only, full PFQ-v2, performance, and paired promotion decisions all use the same newly built binary.

**Tech Stack:** C++20, CUDA/CMake/CTest, Bash, Python 3 `unittest`, existing cuMAC MCS/BLER/TBS tables, existing Stage-B experiment/KPI scripts.

## Global Constraints

- Treat `docs/superpowers/specs/2026-08-12-pfq-v2-marginal-expected-delivery-design.md` as authoritative. Do not add packet-completion bonus, PRG reordering, Type-1 support, CQI-based MCS, or non-SVD support in this change.
- Preserve `-b 0`, `-b 1`, `-b 2`, and `-b 3` behavior. PFQ-v1 and PFQ-v2 must be run from the same newly built executable.
- Preserve the current numeric service-unit convention used by MAC buffer debit even though the existing path names it as bytes. Name it `service_units` in new code and separately report nominal physical bytes as `tbs_bits / 8`.
- Do not mutate actual OLLA, MCS, layer selection, HARQ, buffer debit, expiry, packet-delay, traffic generation, or current goodput accounting from the predictor.
- PFQ-v2 must validate all set-level inputs before PRG 0. Unsupported/missing inputs cause a whole-TTI PFQ-v1 fallback; no partially committed PFQ-v2 allocation may fall back.
- An otherwise eligible invalid candidate is excluded and counted by reason. The canonical run requires zero eligible-candidate errors and zero fallback TTIs.
- Keep Type-0 PRGs in native order and preserve the current deterministic lower-UE-index tie behavior.
- CUDA host postprocessing may use persistent mirrored storage and one batched transfer at the existing synchronization boundary. It must not allocate or synchronize per candidate.
- Preserve all unrelated dirty worktree changes, especially current actual-interference changes in `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`, `cuMAC/examples/network.cu`, and `cuMAC/examples/network.h`. Before every commit, stage only the files named by that task and inspect `git diff --cached`.
- Use `build.pfq_v2/`, not `build.x86_64/`; the latter is configured against `/opt/nvidia/cuBB`.
- The canonical experiment is fixed at 3 cells, 36 UEs, 17 PRGs, 16 PRBs/PRG, Type-0, `coop_boundary`, strongest association, SVD, 3000-byte packets, 200 ms TTL, 4000 TTIs, and paired seeds 41--50.

## Locked public interfaces

Implement these names once and use them consistently in CPU, CUDA-host, tests, and diagnostics:

```cpp
namespace cumac {

inline constexpr std::string_view kPfqV2PredictorVersion = "pfq-v2-med-v1";

uint32_t determinePdschTbsBits(
    int rb_size,
    int data_symbols,
    int layers,
    float code_rate,
    int qam_order);

enum class PfqV2SuccessModel : uint8_t {
    BlerLut = 0,
    ConstantOne = 1,
};

enum class PfqV2PredictError : uint8_t {
    None = 0,
    NonFiniteInput,
    InvalidLayerCount,
    InvalidMcsLut,
    InvalidTbsInput,
    InvalidShape,
    UnsupportedMode,
};

struct PfqV2Config {
    uint16_t prbs_per_prg = 16;
    uint8_t pdsch_data_symbols = 11;
    uint8_t max_layers = 4;
    uint8_t mcs_lut_type = 0;
    float mcs_sinr_cap_db = 100.0F;
    float singular_value_threshold = 0.1F;
    float slot_duration_sec = 0.0005F;
    float beta = 1.0F;
    float queue_buffer_coeff = 0.25F;
    float queue_buffer_scale_service_units = 3000.0F;
    float intra_tti_decay_coeff = 0.25F;
    float relative_epsilon = 1.0e-6F;
    float absolute_epsilon_service_units = 1.0e-3F;
    PfqV2SuccessModel success_model = PfqV2SuccessModel::BlerLut;
};

struct PfqV2UeState {
    uint16_t prg_count = 0;
    uint8_t layers = 0;
    std::array<double, 4> linear_sinr_sums {0.0, 0.0, 0.0, 0.0};
    float raw_capacity_service_units = 0.0F;
    float success_probability = 0.0F;
    float expected_delivery_service_units = 0.0F;
};

struct PfqV2CandidateInput {
    PfqV2UeState current;
    std::span<const float> post_eq_sinr_linear_by_layer;
    std::span<const float> singular_values;
    float wideband_sinr_linear = 0.0F;
    float olla_delta_db = 0.0F;
    float demand_service_units = 0.0F;
};

struct PfqV2CandidateResult {
    bool valid = false;
    PfqV2PredictError error = PfqV2PredictError::None;
    PfqV2UeState next_state;
    int16_t predicted_mcs = -1;
    uint32_t nominal_tbs_bits = 0;
    float predicted_bler = 1.0F;
    float marginal_expected_delivery_service_units = 0.0F;
};

float advancePfqV2OllaDelta(
    float current_delta,
    float delta_up,
    float delta_down,
    int8_t previous_tb_error) noexcept;

uint8_t pfqV2SupportedLayers(
    std::span<const float> singular_values,
    float threshold) noexcept;

PfqV2CandidateResult evaluatePfqV2Candidate(
    const PfqV2Config& config,
    const PfqV2CandidateInput& input) noexcept;

PfqV2SuccessModel parsePfqV2SuccessModel(std::string_view value);

struct PfqV2AllocationInput {
    std::span<const uint16_t> coordinated_cell_ids;
    std::span<const uint16_t> scheduled_slot_to_active_ue;
    std::span<const uint16_t> serving_cell_by_scheduled_slot;
    std::span<const uint32_t> demand_service_units_by_active_ue;
    std::span<const float> average_rate_by_scheduled_slot;
    std::span<const float> wideband_sinr_linear;
    std::span<const float> post_eq_sinr_linear;
    std::span<const float> singular_values;
    std::span<const float> tentative_olla_delta_db;
    uint16_t total_cell_count = 0;
    uint16_t active_ue_count = 0;
    uint16_t scheduled_ue_count = 0;
    uint16_t prg_count = 0;
    uint8_t max_layers = 0;
    PfqV2Config config;
};

struct PfqV2FinalPrediction {
    uint16_t scheduled_ue_index = 0;
    uint16_t active_ue_id = 0;
    int16_t predicted_mcs = -1;
    uint8_t predicted_layers = 0;
    uint16_t allocated_prgs = 0;
    uint32_t nominal_tbs_bits = 0;
    float raw_capacity_service_units = 0.0F;
    float expected_delivery_service_units = 0.0F;
    float predicted_bler = 1.0F;
    float predicted_success_probability = 0.0F;
};

struct PfqV2TtiDiagnostics {
    uint64_t cap_stop_count = 0;
    uint64_t nonpositive_stop_count = 0;
    uint64_t blank_prg_count = 0;
    std::array<uint64_t, 7> invalid_candidate_count_by_reason {};
    uint64_t fallback_tti_count = 0;
    std::vector<PfqV2FinalPrediction> final_predictions;
};

struct PfqV2AllocationOutput {
    bool valid = false;
    PfqV2PredictError error = PfqV2PredictError::None;
    std::vector<int16_t> allocation;
    std::vector<uint8_t> scheduled_active_ue;
    PfqV2TtiDiagnostics diagnostics;
};

PfqV2AllocationOutput allocatePfqV2Type0(
    const PfqV2AllocationInput& input);

class PfqV2OllaShadow {
public:
    void reset(std::span<const float> bler_targets, uint16_t active_ue_count);
    void beginTti(std::span<const int8_t> previous_tb_error_by_active_ue);
    std::span<const float> tentativeDeltaDb() const noexcept;
    void commitScheduled(std::span<const uint8_t> scheduled_active_ue);

private:
    std::vector<float> committed_delta_db_;
    std::vector<float> tentative_delta_db_;
    std::vector<float> delta_up_db_;
    std::vector<float> delta_down_db_;
};

}  // namespace cumac

struct SuccessfulDeliverySummary {
    unsigned long long successful_bytes = 0;
    unsigned long long completed_packets = 0;
    unsigned long long pending_packets = 0;
    unsigned long long pending_bytes = 0;
    double mean_delay_ms = 0.0;
    double p50_delay_ms = 0.0;
    double p90_delay_ms = 0.0;
    double p95_delay_ms = 0.0;
    double max_delay_ms = 0.0;
};

class SuccessfulDeliveryLedger {
public:
    explicit SuccessfulDeliveryLedger(size_t flow_count = 0);
    ~SuccessfulDeliveryLedger();
    SuccessfulDeliveryLedger(SuccessfulDeliveryLedger&&) noexcept;
    SuccessfulDeliveryLedger& operator=(SuccessfulDeliveryLedger&&) noexcept;
    SuccessfulDeliveryLedger(const SuccessfulDeliveryLedger&) = delete;
    SuccessfulDeliveryLedger& operator=(const SuccessfulDeliveryLedger&) = delete;
    void configure(size_t flow_count);
    void setSlotDurationMs(double slot_duration_ms);
    void enqueue(size_t flow_id, uint32_t accepted_bytes, int arrival_tti);
    void recordSuccessfulBytes(
        std::span<const unsigned long long> per_flow_bytes,
        int current_tti);
    SuccessfulDeliverySummary summary() const;
    std::vector<SuccessfulDeliverySummary> perFlowSummary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

Array layouts are part of the interface: `wideband_sinr_linear[active_ue * max_layers + layer]`, and `post_eq_sinr_linear`/`singular_values[((scheduled_slot * prg_count + prg) * max_layers) + layer]`. MCS parity deliberately reads the first wideband-SINR layer, matching `mcsSelectionLUTCpu.cpp`. `allocation[prg * total_cell_count + cell]` stores the scheduled-slot index used by the existing Type-0 allocation arrays, or `-1` for blank; `scheduled_slot_to_active_ue` is used only when active-UE channel, demand, OLLA, or diagnostic state is needed.

---

### Task 1: Create the focused C++ test target and extract the shared TBS primitive

**Files:**
- Create: `cuMAC/src/4T4R/pdschTbs.h`
- Create: `cuMAC/src/4T4R/pdschTbs.cpp`
- Create: `cuMAC/examples/pfqMarginalDeliveryTest/CMakeLists.txt`
- Create: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`
- Modify: `cuMAC/src/CMakeLists.txt`
- Modify: `cuMAC/examples/CMakeLists.txt`
- Modify: `cuMAC/CMakeLists.txt`
- Modify: `cuMAC/examples/network.cu`

**Interfaces:**
- `cumac::determinePdschTbsBits(...)` becomes the single TBS implementation.
- `network::determineTbsPdsch(...)` remains source-compatible and delegates to the shared function.
- `pfqMarginalDeliveryTest` is a standalone CTest target that throws on assertion failure so Release builds do not disable checks.

- [ ] **Step 1: Add the test harness and TBS regression tests before the implementation**

Create a minimal test runner with named cases and this exact regression table:

```cpp
struct TbsCase {
    int rb_size;
    int data_symbols;
    int layers;
    float code_rate;
    int qam_order;
    uint32_t expected_bits;
};

const std::array<TbsCase, 5> cases {{
    {16, 11, 1, 120.0F / 1024.0F, 2, 504},
    {16, 11, 2, 490.0F / 1024.0F, 4, 8064},
    {32, 11, 2, 490.0F / 1024.0F, 4, 16136},
    {272, 11, 4, 948.0F / 1024.0F, 8, 1048976},
    {1, 1, 1, 0.1F, 2, 24},
}};

for (const TbsCase& value : cases) {
    requireEqual(
        cumac::determinePdschTbsBits(
            value.rb_size,
            value.data_symbols,
            value.layers,
            value.code_rate,
            value.qam_order),
        value.expected_bits,
        "shared PDSCH TBS regression");
}
```

Create `pdschTbs.cpp` initially with only `#include "pdschTbs.h"`, so the test reaches the intended missing-symbol link failure. Use this CMake target and make `cumac_examples` depend on it:

```cmake
add_executable(pfqMarginalDeliveryTest main.cpp)
target_link_libraries(pfqMarginalDeliveryTest PRIVATE cumac)
enable_testing()
add_test(NAME pfqMarginalDeliveryTest COMMAND pfqMarginalDeliveryTest)
```

- [ ] **Step 2: Configure and prove the new test fails before the implementation exists**

Run:

```bash
cmake -S . -B build.pfq_v2 \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
```

Expected: configuration succeeds and the test target fails to link because `cumac::determinePdschTbsBits` has only been declared.

- [ ] **Step 3: Move the TBS body and keep the network wrapper**

Move the calculation currently in `network::determineTbsPdsch` into `pdschTbs.cpp`; do not change constants, branches, rounding, or the returned bit count. Replace the network body with:

```cpp
uint32_t network::determineTbsPdsch(
    int rb_size,
    int data_symbols,
    int layers,
    float code_rate,
    int qam_order)
{
    return cumac::determinePdschTbsBits(
        rb_size, data_symbols, layers, code_rate, qam_order);
}
```

- [ ] **Step 4: Build and run the focused test**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: build succeeds and CTest reports `100% tests passed`.

- [ ] **Step 5: Commit only the TBS/test-target slice**

Run:

```bash
git add cuMAC/src/4T4R/pdschTbs.h \
  cuMAC/src/4T4R/pdschTbs.cpp \
  cuMAC/src/CMakeLists.txt \
  cuMAC/examples/CMakeLists.txt \
  cuMAC/examples/pfqMarginalDeliveryTest/CMakeLists.txt \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp \
  cuMAC/CMakeLists.txt
git add -p cuMAC/examples/network.cu
git diff --cached --check
git diff --cached --stat
git diff --cached -- cuMAC/examples/network.cu
git commit -m "refactor: share PDSCH TBS calculation"
```

Expected: the commit contains no unrelated actual-interference hunks.

### Task 2: Implement predictor primitives and allocation-set marginal delivery

**Files:**
- Create: `cuMAC/src/4T4R/pfqMarginalDelivery.h`
- Create: `cuMAC/src/4T4R/pfqMarginalDelivery.cpp`
- Modify: `cuMAC/src/CMakeLists.txt`
- Modify: `cuMAC/src/cumac.h`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- Adds the locked predictor types and functions.
- Reuses existing MCS thresholds, `snrMcsArr`, `blerMscArr`, code-rate/QAM tables, and the shared TBS primitive.
- `ConstantOne` changes only success probability; it still performs the same layer, MCS, TBS, demand, and marginal calculations.

- [ ] **Step 1: Add failing primitive tests**

Add all locked predictor declarations to `pfqMarginalDelivery.h`, add `pfqMarginalDelivery.cpp` to `CUMAC_SOURCES`, and begin the translation unit with includes only. This makes the tests fail at link time for the intended missing implementations.

Add tests for these exact behaviors:

```cpp
const std::array<float, 4> singular_values {10.0F, 2.0F, 0.9F, 0.1F};
requireEqual(
    cumac::pfqV2SupportedLayers(singular_values, 0.1F),
    uint8_t{2},
    "singular-value threshold uses the highest supported prefix");

requireNear(
    cumac::advancePfqV2OllaDelta(0.0F, 1.0F, 0.1F, 1),
    1.0F,
    1.0e-7F,
    "OLLA NACK step");
requireNear(
    cumac::advancePfqV2OllaDelta(0.0F, 1.0F, 0.1F, 0),
    -0.1F,
    1.0e-7F,
    "OLLA ACK step");
requireNear(
    cumac::advancePfqV2OllaDelta(0.25F, 1.0F, 0.1F, -1),
    0.25F,
    1.0e-7F,
    "OLLA no-observation step");
```

Add one predictor fixture whose effective wideband SINR is 5.9 dB with zero OLLA delta and assert MCS 6, plus BLER interpolation boundary fixtures:

```cpp
requireEqual(result.predicted_mcs, int16_t{6}, "MCS LUT parity at 5.9 dB");
requireNear(mcs0_at_minus_4p75.predicted_bler, 0.17728F, 1.0e-5F, "MCS-0 BLER LUT sample");
requireNear(below_last.predicted_bler, 1.0F, 1.0e-7F, "BLER below LUT range");
requireNear(above_first.predicted_bler, 0.0F, 1.0e-7F, "BLER above LUT range");
```

- [ ] **Step 2: Run the focused test and observe failure**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
```

Expected: compilation or linkage fails because predictor symbols are absent.

- [ ] **Step 3: Implement the primitive parity rules**

Implement:

```cpp
float advancePfqV2OllaDelta(
    float current_delta,
    float delta_up,
    float delta_down,
    int8_t previous_tb_error) noexcept
{
    if (previous_tb_error == 1) {
        return current_delta + delta_up;
    }
    if (previous_tb_error == 0) {
        return current_delta - delta_down;
    }
    return current_delta;
}
```

For layers, copy the exact prefix rule from `multiCellLayerSelCpu.cpp`: layer `l + 1` is supported while `singular_values[l] >= singular_values[0] * threshold`. For MCS, use `10*log10(wideband_sinr_linear) - olla_delta_db`, apply the existing cap, and select from the configured LUT with the same comparison direction as `mcsSelectionLUTCpu.cpp`. For BLER, average committed-plus-candidate post-EQ SINR across selected PRGs and predicted layers in linear scale, convert to dB, clamp at `-4.75`, then interpolate the selected MCS row exactly as `network.cu` does.

- [ ] **Step 4: Add failing allocation-set tests**

Cover all nonlinear cases:

```cpp
requireEqual(first.next_state.prg_count, uint16_t{1}, "first grant commits one PRG");
requireEqual(second.next_state.prg_count, uint16_t{2}, "second grant commits set state");
requireTrue(second.nominal_tbs_bits >= first.nominal_tbs_bits, "set-level TBS is recomputed");
requireTrue(crossing.next_state.raw_capacity_service_units >= demand, "one PRG may cross demand");
requireNear(
    crossing.next_state.expected_delivery_service_units,
    crossing.next_state.success_probability * demand,
    1.0e-3F,
    "expected delivery is demand-clipped");
requireTrue(
    degrading.marginal_expected_delivery_service_units < 0.0F,
    "a reliability-degrading PRG can have negative marginal value");
requireEqual(layer_drop.next_state.layers, uint8_t{1}, "set layer count is the PRG minimum");
requireNear(constant_one.next_state.success_probability, 1.0F, 1.0e-7F, "cap-only ablation");
requireNear(zero_demand.marginal_expected_delivery_service_units, 0.0F, 1.0e-7F, "zero demand has no value");
requireNear(saturated.marginal_expected_delivery_service_units, 0.0F, 1.0e-7F, "saturated demand has no value");
requireTrue(!nan_input.valid, "nonfinite eligible input is invalid");
requireEqual(nan_input.error, cumac::PfqV2PredictError::NonFiniteInput, "invalid reason");
requireEqual(zero_layer.error, cumac::PfqV2PredictError::InvalidLayerCount, "layer error reason");
requireEqual(bad_lut.error, cumac::PfqV2PredictError::InvalidMcsLut, "LUT error reason");
```

- [ ] **Step 5: Implement `evaluatePfqV2Candidate`**

Use a tentative copy of `input.current`; never mutate the input. Retain all four layer sums even when the tentative minimum layer count decreases. Set `next.raw_capacity_service_units = static_cast<float>(result.nominal_tbs_bits)` to reproduce the numeric units consumed by the current MAC buffer path, while keeping `nominal_tbs_bits / 8` separate for physical-byte diagnostics. Compute:

```cpp
const float useful_capacity = std::min(
    next.raw_capacity_service_units,
    input.demand_service_units);
next.expected_delivery_service_units =
    next.success_probability * useful_capacity;
result.marginal_expected_delivery_service_units =
    next.expected_delivery_service_units -
    input.current.expected_delivery_service_units;
```

For an empty set, use zero raw/expected delivery and the candidate PRG's supported layer count. Return a reason-specific invalid result for nonfinite values, zero layers, unsupported LUT, invalid TBS arguments, or a span shorter than `config.max_layers`.

- [ ] **Step 6: Run predictor tests**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: every primitive, BLER, cap, negative-marginal, layer-drop, ablation, and invalid-input case passes.

- [ ] **Step 7: Commit predictor code and tests**

Run:

```bash
git add cuMAC/src/4T4R/pfqMarginalDelivery.h \
  cuMAC/src/4T4R/pfqMarginalDelivery.cpp \
  cuMAC/src/CMakeLists.txt \
  cuMAC/src/cumac.h \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add PFQ-v2 expected delivery predictor"
```

### Task 3: Implement transactional OLLA shadow and the shared Type-0 allocator

**Files:**
- Modify: `cuMAC/src/4T4R/pfqMarginalDelivery.h`
- Modify: `cuMAC/src/4T4R/pfqMarginalDelivery.cpp`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- `PfqV2OllaShadow` exposes immutable tentative deltas for one TTI and commits only scheduled UEs.
- `allocatePfqV2Type0` is the only PFQ-v2 greedy allocation implementation used by CPU and CUDA-host paths.
- Shape/mode validation happens before `allocation` can contain a committed PRG.

- [ ] **Step 1: Add failing OLLA transaction tests**

Use three UEs with BLER targets `0.1`, prior TB outcomes `{1, 0, -1}`, and scheduled mask `{1, 0, 1}`:

```cpp
const std::array<float, 3> bler_targets {0.1F, 0.1F, 0.1F};
const std::array<int8_t, 3> previous_errors {1, 0, -1};
const std::array<uint8_t, 3> scheduled_mask {1, 0, 1};
const std::array<int8_t, 3> no_observations {-1, -1, -1};
shadow.reset(bler_targets, 3);
shadow.beginTti(previous_errors);
requireNear(shadow.tentativeDeltaDb()[0], 1.0F, 1.0e-7F, "tentative NACK delta");
requireNear(shadow.tentativeDeltaDb()[1], -0.1F, 1.0e-7F, "tentative ACK delta");
shadow.commitScheduled(scheduled_mask);
shadow.beginTti(no_observations);
requireNear(shadow.tentativeDeltaDb()[0], 1.0F, 1.0e-7F, "scheduled UE committed");
requireNear(shadow.tentativeDeltaDb()[1], 0.0F, 1.0e-7F, "unscheduled UE rolled back");
```

- [ ] **Step 2: Add failing allocator tests**

Create compact 2-UE/2-PRG fixtures and assert:

```cpp
requireEqual(output.allocation, std::vector<int16_t>({0, 1}), "deterministic PRG winners");
requireEqual(tie_output.allocation[0], int16_t{0}, "tie selects lower global UE index");
requireEqual(cross_output.diagnostics.cap_stop_count, uint64_t{1}, "no grant after first cap crossing");
requireEqual(negative_output.allocation[1], int16_t{-1}, "nonpositive marginal leaves PRG blank");
requireEqual(negative_output.diagnostics.blank_prg_count, uint64_t{1}, "blank PRG counted");
requireEqual(at_epsilon_output.allocation[0], int16_t{-1}, "marginal equal to epsilon is rejected");
requireTrue(above_epsilon_output.allocation[0] >= 0, "marginal strictly above epsilon participates");
requireTrue(!bad_shape.valid, "invalid shape rejects complete TTI");
requireEqual(bad_shape.error, cumac::PfqV2PredictError::InvalidShape, "shape error reason");
requireTrue(
    std::all_of(bad_shape.allocation.begin(), bad_shape.allocation.end(), [](int16_t ue) { return ue == -1; }),
    "invalid shape commits no PRG");
```

- [ ] **Step 3: Run tests and observe failure**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
```

Expected: new OLLA/allocator tests fail because the transaction and allocator are not implemented.

- [ ] **Step 4: Implement validation and OLLA transaction semantics**

`reset` sets committed deltas to zero, `delta_up` to `1.0F`, and `delta_down` to each UE's BLER target, matching the existing selector. `beginTti` derives tentative deltas from committed values without modifying committed storage. `commitScheduled` copies tentative to committed only where the mask is `1`.

Reject before the PRG loop unless all dimensions, spans, finite configuration values, cell IDs, active/global mappings, and serving cells are valid. Also reject unsupported allocation, SVD, or MCS mode at the scheduler integration layer before calling the allocator.

- [ ] **Step 5: Implement the native-order greedy allocator**

For each coordinated cell and PRG, evaluate associated active UEs whose demand is nonzero and whose committed raw capacity is still below demand. Use:

```cpp
const float epsilon = std::max(
    input.config.absolute_epsilon_service_units,
    input.config.relative_epsilon *
        std::max(demand_service_units, 1.0F));
const float positive_marginal =
    candidate.marginal_expected_delivery_service_units;
if (!(positive_marginal > epsilon)) {
    ++output.diagnostics.nonpositive_stop_count;
    continue;
}
const float effective_rate =
    positive_marginal / input.config.slot_duration_sec;
const float queue_weight = 1.0F +
    input.config.queue_buffer_coeff *
        std::log2(1.0F + demand_service_units /
                  input.config.queue_buffer_scale_service_units);
const float decay_weight = 1.0F /
    (1.0F + input.config.intra_tti_decay_coeff * state.prg_count);
const float metric =
    std::pow(effective_rate, input.config.beta) /
    std::max(average_rate, 1.0e-6F) *
    queue_weight * decay_weight;
```

Keep the current comparison semantics so equal metrics retain the lower UE index. Commit only the winning candidate state. Mark the UE capped only after a winning state reaches or crosses demand. Populate one `PfqV2FinalPrediction` per scheduled final TB after the entire PRG loop.

- [ ] **Step 6: Run the complete focused suite**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: OLLA commit/rollback, winner, tie, demand crossing, blank, error-counter, and precommit validation tests pass.

- [ ] **Step 7: Commit the shared allocator**

Run:

```bash
git add cuMAC/src/4T4R/pfqMarginalDelivery.h \
  cuMAC/src/4T4R/pfqMarginalDelivery.cpp \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add shared PFQ-v2 Type-0 allocator"
```

### Task 4: Add PFQ-v2 configuration, CLI mode, and strict preflight validation

**Files:**
- Modify: `cuMAC/src/api.h`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Modify: `cuMAC/examples/network.cu`
- Modify: `cuMAC/examples/network.h`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- `-b 4` selects PFQ-v2; `-b 0..3` retain their labels and behavior.
- `CUMAC_PFQ_V2_SUCCESS_MODEL=bler_lut` is default; `constant_one` is the only ablation value; any other value exits nonzero before simulation.
- New `cumacCellGrpPrms` fields carry the locked PFQ-v2 configuration to CPU/GPU schedulers.

- [ ] **Step 1: Add a pure success-model parser test**

Expose `parsePfqV2SuccessModel(std::string_view)` from the shared PFQ-v2 module and test:

```cpp
requireEqual(
    cumac::parsePfqV2SuccessModel("bler_lut"),
    cumac::PfqV2SuccessModel::BlerLut,
    "default success model");
requireEqual(
    cumac::parsePfqV2SuccessModel("constant_one"),
    cumac::PfqV2SuccessModel::ConstantOne,
    "cap-only success model");
requireThrows(
    [] { static_cast<void>(cumac::parsePfqV2SuccessModel("unknown")); },
    "unsupported success model");
```

- [ ] **Step 2: Run the focused test and observe failure**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
```

Expected: parser tests fail until the strict parser exists.

- [ ] **Step 3: Add API fields with inert defaults**

Add these to `cumacCellGrpPrms` adjacent to the existing PFQ fields:

```cpp
uint8_t pfqV2Enabled = 0;
uint8_t pfqV2SuccessModel = 0;
uint16_t pfqV2PrbsPerPrg = 0;
uint8_t pfqV2PdschDataSymbols = 0;
```

Do not reinterpret existing PFQ fields. PFQ-v2 reuses `betaCoeff`, queue coefficient/scale, intra-TTI decay, slot duration, MCS LUT type/cap, singular-value threshold, and BLER targets. For `-b 4`, initialize the PFQ-v1 fallback fields exactly as `-b 2`, including its existing cap slack; the normal v2 allocator ignores that slack and applies its no-slack dual-ledger rule.

- [ ] **Step 4: Add `-b 4` and fail-fast environment parsing**

Add `kBaselinePfQueueAwareV2 = 4`, extend usage and label functions, and set `pfqV2Enabled` only for `-b 4`. Parse:

```cpp
const char* pfq_v2_model_env = std::getenv("CUMAC_PFQ_V2_SUCCESS_MODEL");
const std::string_view pfq_v2_model =
    pfq_v2_model_env == nullptr ? "bler_lut" : pfq_v2_model_env;
const auto success_model = cumac::parsePfqV2SuccessModel(pfq_v2_model);
```

Parse the success-model environment only when `-b 4` is selected. Before TTI 0, preflight DL, Type-0, SVD, 3 coordinated cells, 36 active UEs, 17 PRGs, 16 PRBs/PRG, SINR-based MCS, valid MCS LUT, and non-null BLER/SINR/singular-value inputs. An invalid environment model exits immediately; an unsupported scheduler shape/mode records `UnsupportedMode` and makes every affected TTI take the approved whole-TTI PFQ-v1 fallback. Emit one `PFQ_V2_CONFIG` line with algorithm, model, and `predictor_version=pfq-v2-med-v1`.

- [ ] **Step 5: Build and exercise CLI validation without a full simulation**

Run:

```bash
cmake --build build.pfq_v2 --target multiCellSchedulerUeSelection -j8
set +e
CUMAC_PFQ_V2_SUCCESS_MODEL=invalid \
  build.pfq_v2/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -b 4 \
  > /tmp/pfq_v2_invalid_model.log 2>&1
status=$?
set -e
test "$status" -ne 0
rg -n 'unsupported.*CUMAC_PFQ_V2_SUCCESS_MODEL|invalid' /tmp/pfq_v2_invalid_model.log
```

Expected: the binary exits nonzero before TTI 0 and identifies the invalid environment value.

- [ ] **Step 6: Run a legacy CLI smoke**

Run:

```bash
build.pfq_v2/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -h \
  > /tmp/pfq_v2_help.log 2>&1
rg -n -- '-b.*0.*1.*2.*3.*4' /tmp/pfq_v2_help.log
```

Expected: help lists all five modes and returns zero.

- [ ] **Step 7: Commit configuration and CLI changes**

Run:

```bash
git add cuMAC/src/api.h \
  cuMAC/src/4T4R/pfqMarginalDelivery.h \
  cuMAC/src/4T4R/pfqMarginalDelivery.cpp \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git add -p cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git add -p cuMAC/examples/network.cu
git add -p cuMAC/examples/network.h
git diff --cached --check
git diff --cached --stat
git diff --cached -- cuMAC/examples/multiCellSchedulerUeSelection/main.cpp \
  cuMAC/examples/network.cu cuMAC/examples/network.h
git commit -m "feat: expose PFQ-v2 baseline configuration"
```

Expected: inspect the staged versions of the three already-dirty example files and confirm only minimal PFQ-v2 hunks are included.

### Task 5: Integrate the shared allocator into the CPU Type-0 SVD path

**Files:**
- Modify: `cuMAC/src/4T4R/multiCellSchedulerCpu.h`
- Modify: `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
- Modify: `cuMAC/examples/network.cu`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- `multiCellSchedulerCpu::run()` calls `allocatePfqV2Type0` only when `pfqV2Enabled == 1`.
- `getPfqV2Diagnostics()` returns the last completed TTI record without exposing mutable state.
- CPU setup supplies `wbSinr`, `sinVal`, `postEqSinr`, BLER targets, and prior TB outcomes while preserving PFQ-v1 paths byte-for-byte.

- [ ] **Step 1: Add an integration-shaped CPU fixture**

Extend the focused test with a canonical-shape synthetic input: 3 cells, 36 active/global UEs, 17 PRGs, 4 layers, one associated cell per UE, finite SINR, 3000 service units demand. Call the shared allocator twice with identical data and assert identical allocations, scheduled masks, final predictions, and counters.

Also build a PFQ-v1 sentinel allocation fixture and assert that toggling an independent `pfqV2Enabled` flag off selects the unchanged PFQ-v1 helper rather than the v2 allocator.

- [ ] **Step 2: Run the fixture before integration**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: pure canonical-shape determinism passes; the integration sentinel fails until CPU dispatch is wired.

- [ ] **Step 3: Extend the CPU dynamic descriptor and setup**

Add read-only fields for `wbSinr`, `blerTargetActUe`, `tbErrLastActUe`, MCS LUT/cap, PFQ-v2 mode/config, and PRBs/data symbols. Populate CPU `sinVal` for the SVD path instead of leaving it null. Store persistent `PfqV2OllaShadow` and `PfqV2TtiDiagnostics` members on the scheduler object.

At TTI setup:

1. map prior selected-slot TB outcomes back to active UE IDs;
2. call `beginTti` exactly once;
3. validate complete PFQ-v2 inputs;
4. on validation failure, execute the existing PFQ-v1 Type-0 function for the whole TTI and increment fallback once;
5. after either v2 allocation or fallback allocation, commit shadow OLLA only for actually scheduled active UEs.

- [ ] **Step 4: Replace only the PFQ-v2 branch with the shared allocator**

In `multiCellSchedulerCpu_svdMmse`, retain the existing post-EQ/SVD calculations. Build contiguous active-UE/PRG/layer views, call `allocatePfqV2Type0`, and copy its allocation into `allocSol`. Do not duplicate candidate loops in `multiCellSchedulerCpu.cpp`.

- [ ] **Step 5: Build and run focused plus legacy regression tests**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest multiCellSchedulerUeSelection cellAssociateTest -j8
ctest --test-dir build.pfq_v2 -R '^(pfqMarginalDeliveryTest|cellAssociateTest)$' --output-on-failure
```

Expected: focused tests pass; `cellAssociateTest` passes or, if unavailable on this GPU, its infrastructure failure is recorded separately and the targeted PFQ test remains green.

- [ ] **Step 6: Commit CPU integration**

Run:

```bash
git add cuMAC/src/4T4R/multiCellSchedulerCpu.h \
  cuMAC/src/4T4R/multiCellSchedulerCpu.cpp \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git add -p cuMAC/examples/network.cu
git diff --cached --check
git diff --cached --stat
git diff --cached -- cuMAC/examples/network.cu
git commit -m "feat: integrate PFQ-v2 CPU scheduler"
```

### Task 6: Integrate PFQ-v2 at the CUDA host-postprocessing boundary and prove parity

**Files:**
- Modify: `cuMAC/src/4T4R/multiCellScheduler.cuh`
- Modify: `cuMAC/src/4T4R/multiCellScheduler.cu`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- GPU scheduler owns persistent host mirrors for all PFQ-v2 inputs and returns `const PfqV2TtiDiagnostics& getPfqV2Diagnostics() const noexcept`.
- The existing GPU kernels still produce channel/PF inputs; the shared C++ allocator decides PFQ-v2 grants on host.
- CPU and GPU-host calls with identical input views must be bit-identical for allocation/MCS/layers/TBS and tolerance-identical for floating diagnostics.

- [ ] **Step 1: Add a CPU/GPU-host parity test fixture**

Feed one canonical synthetic TTI through two separately materialized input buffers and assert:

```cpp
requireEqual(gpu_host.allocation, cpu.allocation, "CPU/GPU-host allocation parity");
requireEqual(gpu_host.scheduled_active_ue, cpu.scheduled_active_ue, "scheduled mask parity");
requireEqual(
    gpu_host.diagnostics.final_predictions.size(),
    cpu.diagnostics.final_predictions.size(),
    "final prediction count parity");
for (size_t i = 0; i < cpu.diagnostics.final_predictions.size(); ++i) {
    requireEqual(gpu_host.diagnostics.final_predictions[i].predicted_mcs,
                 cpu.diagnostics.final_predictions[i].predicted_mcs,
                 "MCS parity");
    requireEqual(gpu_host.diagnostics.final_predictions[i].predicted_layers,
                 cpu.diagnostics.final_predictions[i].predicted_layers,
                 "layer parity");
    requireEqual(gpu_host.diagnostics.final_predictions[i].nominal_tbs_bits,
                 cpu.diagnostics.final_predictions[i].nominal_tbs_bits,
                 "TBS parity");
}
```

- [ ] **Step 2: Add persistent mirror members**

Add scheduler-owned vectors sized during setup/shape changes for:

```cpp
std::vector<float> pfq_v2_host_average_rate_;
std::vector<float> pfq_v2_host_wideband_sinr_;
std::vector<float> pfq_v2_host_post_eq_sinr_;
std::vector<float> pfq_v2_host_singular_values_;
std::vector<uint32_t> pfq_v2_host_buffer_size_;
std::vector<uint16_t> pfq_v2_host_mapping_;
std::vector<int8_t> pfq_v2_host_previous_tb_error_;
std::vector<float> pfq_v2_host_bler_target_;
```

Do not allocate these vectors inside the per-PRG loop or per candidate.

- [ ] **Step 3: Batch-copy at the existing synchronization boundary**

Extend the current host postprocessing transfer block to enqueue all required device-to-host copies on the existing stream, then synchronize once before calling `allocatePfqV2Type0`. Static or shape-stable BLER targets need copying only at initialization/shape change. There must be no `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaMalloc`, or vector resize in the candidate loop.

- [ ] **Step 4: Add GPU whole-TTI fallback and transactional OLLA**

Mirror CPU behavior: validate before PRG 0, run the untouched PFQ-v1 host postprocessor on whole-TTI failure, increment fallback once, derive the actual scheduled mask, and commit OLLA shadow only for those UEs.

- [ ] **Step 5: Run static and focused parity checks**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest multiCellSchedulerUeSelection -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
rg -n 'cuda(Stream|Device)Synchronize|cudaMalloc|resize\(' \
  cuMAC/src/4T4R/multiCellScheduler.cu
```

Expected: parity tests pass. Manual inspection of each matched synchronization/allocation confirms none is nested in the PFQ-v2 UE/PRG candidate loop.

- [ ] **Step 6: Run a short real CPU/GPU parity smoke**

Run the existing Stage-B runner with canonical topology, 10 TTIs, seed 41, `-b 4`, and CPU reference checking enabled by its normal mode:

```bash
cuMAC/scripts/run_stageB_main_experiment.sh \
  --build-dir build.pfq_v2 \
  --build-method cmake \
  --skip-compile-param-update 0 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --tti 10 \
  --cell-radius 500 \
  --cell-assoc-mode strongest \
  --ue-placement coop_boundary \
  --ue-voronoi-clip 1 \
  --topology-seed 41 \
  --traffic-seed 41 \
  --traffic-percent 100 \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --prbs-per-group 16 \
  --precoding svd \
  --fading-mode 0 \
  --cdl-profiles NA \
  --cdl-delay-spreads 0 \
  --packet-delay-samples 0 \
  --baseline-scheduler pfq_v2 \
  --compare-tti 1 \
  --exec-mode both \
  --tag pfq_v2_parity_seed41 \
  2>&1 | tee /tmp/pfq_v2_parity_seed41.log
```

Expected log assertions:

```bash
rg -n 'PFQ_V2_CONFIG.*success_model=bler_lut' /tmp/pfq_v2_parity_seed41.log
rg -n 'CPU and GPU scheduling solutions match|solution.*match' /tmp/pfq_v2_parity_seed41.log
! rg -n 'fallback_tti_count=[1-9][0-9]*|invalid_candidate_count=[1-9][0-9]*' /tmp/pfq_v2_parity_seed41.log
python3 - /tmp/pfq_v2_parity_seed41.log <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
line = next(value for value in text.splitlines() if value.startswith("PFQ_V2_DIAGNOSTICS "))
fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", line))
for name in ("mcs_match_ratio", "layer_match_ratio", "nominal_tbs_match_ratio"):
    assert float(fields[name]) >= 0.999, (name, fields[name])
PY
```

- [ ] **Step 7: Commit CUDA-host integration**

Run:

```bash
git add cuMAC/src/4T4R/multiCellScheduler.cuh \
  cuMAC/src/4T4R/multiCellScheduler.cu \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git diff --cached --check
git diff --cached --stat
git commit -m "feat: integrate PFQ-v2 CUDA host postprocessing"
```

### Task 7: Add the observation-only successful-delivery shadow ledger

**Files:**
- Create: `cuMAC/examples/trafficModel/successfulDeliveryLedger.hpp`
- Create: `cuMAC/examples/trafficModel/successfulDeliveryLedger.cpp`
- Modify: `cuMAC/examples/trafficModel/trafficGenerator.hpp`
- Modify: `cuMAC/examples/trafficModel/trafficService.hpp`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/CMakeLists.txt`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/CMakeLists.txt`
- Modify: `cuMAC/examples/pfqMarginalDeliveryTest/main.cpp`

**Interfaces:**
- `SuccessfulDeliveryLedger::enqueue(flow_id, accepted_bytes, arrival_tti)` observes the same accepted packet spans as `RadioResource`.
- `SuccessfulDeliveryLedger::recordSuccessfulBytes(per_flow_bytes, current_tti)` consumes FIFO bytes only for successful TB service.
- The ledger has no TTL expiration and never writes scheduler or legacy traffic state.

- [ ] **Step 1: Add failing ledger tests**

Add the locked ledger declarations and an include-only `successfulDeliveryLedger.cpp`, then link that source into `pfqMarginalDeliveryTest` before adding the cases below. The first run must fail on missing ledger symbols, not on missing headers or an omitted CMake source.

Test two flows with ordered packet spans:

```cpp
ledger.enqueue(0, 3000, 0);
ledger.enqueue(0, 3000, 2);
ledger.enqueue(1, 1000, 1);
const std::array<unsigned long long, 2> failed_service {0, 0};
const std::array<unsigned long long, 2> partial_service {2500, 1000};
const std::array<unsigned long long, 2> completing_service {3500, 0};
ledger.recordSuccessfulBytes(failed_service, 3);
requireEqual(ledger.summary().completed_packets, 0ULL, "failed TB consumes nothing");
ledger.recordSuccessfulBytes(partial_service, 4);
requireEqual(ledger.summary().completed_packets, 1ULL, "other flow completes one packet");
ledger.recordSuccessfulBytes(completing_service, 5);
requireEqual(ledger.summary().completed_packets, 3ULL, "partial plus next packet completion");
requireEqual(ledger.summary().successful_bytes, 7000ULL, "only available shadow backlog consumed");
requireEqual(ledger.summary().pending_packets, 0ULL, "all shadow packets complete");
requireEqual(ledger.summary().pending_bytes, 0ULL, "all shadow bytes complete");
```

Also assert per-flow counts and p50/p95 latency using the same sorted linear interpolation at position `(n - 1) * percentile / 100` as `RadioResource::percentile_ms`.

- [ ] **Step 2: Run the test and observe failure**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest -j8
```

Expected: ledger tests fail because the component is absent.

- [ ] **Step 3: Implement independent FIFO packet accounting**

Store `{remaining_bytes, original_bytes, arrival_tti}` per flow. On service, consume at most the current shadow backlog, complete packets in FIFO order, and record `max(1, current_tti - arrival_tti + 1) * slot_duration_ms`. Track both pending packet count and pending bytes. Do not remove packets due to the legacy TTL.

- [ ] **Step 4: Feed accepted spans and successful physical TBS bytes**

When `RadioResource::Enqueue` iterates `accepted_spans`, notify the ledger with the same flow ID, bytes, and arrival TTI. In `main.cpp`, after actual layer/MCS/PRG count and `tbErr` are known, compute:

```cpp
const uint32_t actual_tbs_bits = cumac::determinePdschTbsBits(
    allocated_prg_count * prbs_per_prg,
    pdsch_data_symbols,
    actual_layers,
    cumac::mcsTable_codeRate[actual_mcs] / 1024.0F,
    cumac::mcsTable_qamOrder[actual_mcs]);
const unsigned long long successful_physical_bytes =
    tb_error == 0 ? static_cast<unsigned long long>(actual_tbs_bits / 8U) : 0ULL;
```

Pass that value to `TrafficService::RecordPhySuccessfulBytes`. Run this for both PFQ-v1 and PFQ-v2; do not derive it from `servedBytesThisTti`.

- [ ] **Step 5: Build and run ledger tests**

Run:

```bash
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest multiCellSchedulerUeSelection -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: failed-TB, partial, multi-packet, cross-flow, percentile, and backlog clipping tests pass.

- [ ] **Step 6: Commit the shadow ledger**

Run:

```bash
git add cuMAC/examples/trafficModel/successfulDeliveryLedger.hpp \
  cuMAC/examples/trafficModel/successfulDeliveryLedger.cpp \
  cuMAC/examples/trafficModel/trafficGenerator.hpp \
  cuMAC/examples/trafficModel/trafficService.hpp \
  cuMAC/examples/multiCellSchedulerUeSelection/CMakeLists.txt \
  cuMAC/examples/pfqMarginalDeliveryTest/CMakeLists.txt \
  cuMAC/examples/pfqMarginalDeliveryTest/main.cpp
git add -p cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git diff --cached --check
git diff --cached --stat
git diff --cached -- cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git commit -m "feat: add successful delivery shadow ledger"
```

### Task 8: Add calibration, parity, stop-reason, and runtime diagnostics

**Files:**
- Modify: `cuMAC/src/4T4R/pfqMarginalDelivery.h`
- Modify: `cuMAC/src/4T4R/multiCellSchedulerCpu.h`
- Modify: `cuMAC/src/4T4R/multiCellScheduler.cuh`
- Modify: `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- Modify: `cuMAC/scripts/summarize_stageA_kpi.py`
- Create: `cuMAC/scripts/tests/test_summarize_stageA_kpi_pfq_v2.py`

**Interfaces:**
- Scheduler getters expose the last TTI's final predictions and counters read-only.
- Main pairs each scheduled final prediction with actual success/MCS/layers/TBS after PHY abstraction.
- KPI JSON adds top-level `pfq_v2`, `pfq_v2_calibration`, `phy_success_delivery`, and `scheduler_timing` objects without changing existing fields.

- [ ] **Step 1: Add failing parser tests with complete sample log lines**

Use a temporary log containing:

```text
PFQ_V2_DIAGNOSTICS success_model=bler_lut selected_tb_count=8 invalid_candidate_count=0 invalid_nonfinite_count=0 invalid_layer_count=0 invalid_mcs_lut_count=0 invalid_tbs_input_count=0 invalid_shape_count=0 unsupported_mode_count=0 fallback_tti_count=0 cap_stop_count=11 nonpositive_stop_count=3 blank_prg_count=2 mean_pred_success=0.920000 observed_success=0.875000 brier=0.081000 mcs_match_ratio=1.000000 layer_match_ratio=1.000000 nominal_tbs_match_ratio=1.000000 raw_capacity_service_units=24000.000000 expected_delivery_service_units=22080.000000 nominal_tbs_bytes=3000
PFQ_V2_CALIBRATION bin=9 lower=0.900000 upper=1.000000 count=8 mean_pred=0.920000 observed_success=0.875000
PHY_SUCCESS_DELIVERY successful_bytes=18000 completed_packets=6 pending_packets=2 pending_bytes=4800 mean_ms=3.000000 p50_ms=2.500000 p90_ms=5.000000 p95_ms=5.500000 max_ms=6.000000
SCHEDULER_TIMING algorithm=pfq_v2 samples=4000 p50_ms=0.120000 p95_ms=0.180000 p99_ms=0.220000 wall_ms=8123.000000
```

Assert exact JSON numeric values and that all pre-existing `global_kpi` fields from a representative legacy log remain unchanged.

- [ ] **Step 2: Run the parser test and observe failure**

Run:

```bash
python3 -m unittest -v cuMAC.scripts.tests.test_summarize_stageA_kpi_pfq_v2
```

Expected: assertions fail because the new records are not parsed.

- [ ] **Step 3: Aggregate predictor/actual calibration in main**

For each actual TB, find the prediction by active UE ID and accumulate:

```cpp
const double residual = predicted_success - actual_success;
brier_sum += residual * residual;
const int calibration_bin = std::min(
    9,
    std::max(0, static_cast<int>(predicted_success * 10.0)));
calibration[calibration_bin].predicted_sum += predicted_success;
calibration[calibration_bin].actual_sum += actual_success;
calibration[calibration_bin].count += 1;
```

Count exact equality for final MCS, layers, and nominal TBS. Accumulate invalid reasons and fallback TTIs across the run. Do not print per-candidate records by default.

- [ ] **Step 4: Add scheduler and process timing**

Measure scheduler time around the GPU scheduler call plus its required stream synchronization using `std::chrono::steady_clock`. Store one sample per executed TTI and report p50/p95/p99. Measure process wall time from immediately before initialization until after final KPI emission. Use the same timing instrumentation in PFQ-v1 and PFQ-v2.

- [ ] **Step 5: Emit stable machine-readable lines**

At end of run emit exactly one `PFQ_V2_DIAGNOSTICS` line for `-b 4`, including the aggregate invalid count and every reason-specific count shown in the fixture; ten calibration lines including zero-count bins; one `PHY_SUCCESS_DELIVERY` line for PFQ-v1 and PFQ-v2; and one `SCHEDULER_TIMING` line for all modes. Use the field names in the parser fixture. For zero denominators, emit `0.000000`, never NaN.

Extend `write_pass_key_log` in the Stage-B runner so compact output retains `PFQ_V2_CONFIG`, `PFQ_V2_DIAGNOSTICS`, `PFQ_V2_CALIBRATION`, `PHY_SUCCESS_DELIVERY`, and `SCHEDULER_TIMING`; otherwise `--compact-output 1` would delete the only copy before formal evaluation.

- [ ] **Step 6: Extend the KPI parser and schema descriptions**

Parse the new records into separate objects. Keep the legacy goodput/packet-delay definitions unchanged and describe the shadow completion metric explicitly as ordered successful-service accounting without TTL expiry.

- [ ] **Step 7: Run parser, focused, and build tests**

Run:

```bash
python3 -m unittest -v cuMAC.scripts.tests.test_summarize_stageA_kpi_pfq_v2
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_*.py' -v
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest multiCellSchedulerUeSelection -j8
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
```

Expected: all Python parser tests and C++ tests pass.

- [ ] **Step 8: Commit diagnostics and parsing**

Run:

```bash
git add cuMAC/src/4T4R/pfqMarginalDelivery.h \
  cuMAC/src/4T4R/multiCellSchedulerCpu.h \
  cuMAC/src/4T4R/multiCellScheduler.cuh \
  cuMAC/scripts/summarize_stageA_kpi.py \
  cuMAC/scripts/tests/test_summarize_stageA_kpi_pfq_v2.py
git add -p cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git diff --cached --check
git diff --cached --stat
git diff --cached -- cuMAC/examples/multiCellSchedulerUeSelection/main.cpp
git commit -m "feat: report PFQ-v2 delivery diagnostics"
```

### Task 9: Extend Stage-B runners for PFQ-v2 and build the promotion evaluator

**Files:**
- Modify: `cuMAC/scripts/run_stageB_main_experiment.sh`
- Modify: `cuMAC/scripts/run_stageB_rr_pf_compare.sh`
- Modify: `cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`
- Create: `cuMAC/scripts/evaluate_pfq_v2_promotion.py`
- Create: `cuMAC/scripts/tests/test_evaluate_pfq_v2_promotion.py`
- Create: `cuMAC/scripts/run_stageB_pfq_v2_experiment.sh`

**Interfaces:**
- Stage-B baseline name `pfq_v2` maps to `-b 4`.
- `--pfq-v2-success-model {bler_lut,constant_one}` exports the strict environment variable.
- The promotion evaluator consumes one row per paired seed/algorithm and emits JSON with every gate, paired deltas, bootstrap interval, and an overall decision.
- Every run records `git_commit`, `dirty_worktree`, SHA256 of the tracked diff when dirty, executable SHA256, `predictor_version`, baseline/model, topology seed, traffic seed, and complete frozen runtime arguments.

- [ ] **Step 1: Add failing promotion-evaluator tests**

Create fixtures for seeds 41--50 and test:

1. a full pass where PFQ-v2 mean goodput is strictly higher, mean BLER/expiry/p95 are no worse, mean UE-goodput p5 is no worse, every seed is at least 99% of PFQ-v1, and errors/fallbacks are zero;
2. equal mean goodput fails the strict improvement gate;
3. each guardrail fails independently;
4. a single seed at `98.99%` fails the worst-seed gate;
5. any invalid candidate or fallback fails validity;
6. two repeated invocations produce the same bootstrap interval with seed `20260812`.

Use assertions such as:

```python
self.assertTrue(result["gates"]["mean_goodput_strictly_higher"]["pass"])
self.assertFalse(equal_result["decision"]["promote"])
self.assertFalse(worst_seed_result["gates"]["each_seed_goodput_at_least_99pct"]["pass"])
self.assertEqual(result["bootstrap"]["resamples"], 10_000)
self.assertEqual(result["bootstrap"]["seed"], 20260812)
```

- [ ] **Step 2: Run tests and observe failure**

Run:

```bash
python3 -m unittest -v cuMAC.scripts.tests.test_evaluate_pfq_v2_promotion
```

Expected: import or assertions fail because the evaluator does not exist.

- [ ] **Step 3: Implement exact paired aggregation and gates**

Compute arithmetic means over the ten per-seed KPIs, never pool packets across seeds. Pair by seed and reject missing/duplicate arms. Required gates:

Use these exact fields from each `kpi_summary.json`:

```python
KPI_FIELDS = {
    "goodput": "global_kpi.cluster_goodput_mbps",
    "bler": "global_kpi.global_tb_bler",
    "legacy_expiry": "global_kpi.expiry_drop_rate",
    "legacy_packet_p95": "global_kpi.packet_delay_p95_ms",
    "ue_goodput_p5": "global_kpi.ue_goodput_p5_mbps",
    "invalid_candidate_count": "pfq_v2.invalid_candidate_count",
    "fallback_tti_count": "pfq_v2.fallback_tti_count",
}
```

The validity counts are required for PFQ-v2 rows and are treated as zero/not-applicable for PFQ-v1 rows. Missing formal KPI fields are errors rather than zeros.

```python
gates = {
    "mean_goodput_strictly_higher": mean_v2_goodput > mean_v1_goodput,
    "mean_bler_no_worse": mean_v2_bler <= mean_v1_bler,
    "mean_legacy_expiry_no_worse": mean_v2_expiry <= mean_v1_expiry,
    "mean_legacy_packet_p95_no_worse": mean_v2_p95 <= mean_v1_p95,
    "mean_ue_goodput_p5_no_worse": mean_v2_ue_p5 >= mean_v1_ue_p5,
    "each_seed_goodput_at_least_99pct": all(
        v2 >= 0.99 * v1 for v1, v2 in paired_goodput),
    "predictor_valid": invalid_candidate_count == 0 and fallback_tti_count == 0,
}
```

Use a paired bootstrap of the ten goodput deltas with 10,000 resamples and RNG seed 20260812. Report its 2.5th/97.5th percentiles as diagnostic only; do not add an unapproved significance gate.

- [ ] **Step 4: Extend baseline mapping and comparison scripts**

Accept `pfq_v2` in the main runner and map it to `-b 4`. Preserve current names for `pf`, `rr`, `pfq`, and `rrq`. In pair/multi-seed scripts, allow `pfq` as reference and `pfq_v2` as comparison. Pass the success-model option through unchanged.

Before each scenario starts, have the main runner write a machine-readable `run_metadata.json` and add its path plus the identity fields to the comparison manifest. Derive identity with non-mutating commands equivalent to:

```bash
git_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
dirty_worktree="$([[ -n "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=no)" ]] && echo true || echo false)"
tracked_diff_sha256="$(git -C "${ROOT_DIR}" diff --binary HEAD | sha256sum | awk '{print $1}')"
binary_sha256="$(sha256sum "${BIN}" | awk '{print $1}')"
```

Write the resolved predictor version parsed from `PFQ_V2_CONFIG`, or `none` for non-v2 modes. The formal manifest validator must require identical `git_commit`, tracked-diff hash, executable hash, and frozen canonical arguments across both arms of each seed.

- [ ] **Step 5: Add the dedicated orchestration script**

Implement phases:

```text
screen    seeds 41,42,43; arms pfq, pfq_v2/constant_one, pfq_v2/bler_lut
runtime   seed 41; one untimed warm-up per arm; three paired timed repetitions
formal    seeds 41..50; arms pfq and pfq_v2/bler_lut
evaluate  run KPI summarization and the promotion evaluator
```

Run every arm in an isolated subshell. Export `CUMAC_PFQ_V2_SUCCESS_MODEL` only for `pfq_v2`; explicitly unset it for PFQ-v1. Do not reuse warm-up KPI rows in the timed or formal manifests.

Use these fixed canonical arguments:

```bash
--topology-scenario 3cell
--total-ue-count 36
--tti 4000
--cell-radius 500
--cell-assoc-mode strongest
--ue-placement coop_boundary
--ue-voronoi-clip 1
--traffic-percent 100
--packet-size-bytes 3000
--traffic-arrival-rate 1.0
--packet-ttl-ms 200
--prbs-per-group 16
--precoding svd
--fading-mode 0
--cdl-profiles NA
--cdl-delay-spreads 0
--packet-delay-samples 1
--exec-mode gpu
```

The runtime execution order must be `v1/v2`, `v2/v1`, `v1/v2`. Compute increase from medians:

```text
100 * (median_wall_ms_v2 / median_wall_ms_v1 - 1)
```

Return `pass` at `<=10%`, `optimize_and_retest` at `>10% and <=20%`, and `block_formal_run` at `>20%`.

- [ ] **Step 6: Run shell syntax and Python unit tests**

Run:

```bash
bash -n cuMAC/scripts/run_stageB_main_experiment.sh
bash -n cuMAC/scripts/run_stageB_rr_pf_compare.sh
bash -n cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh
bash -n cuMAC/scripts/run_stageB_pfq_v2_experiment.sh
python3 -m unittest -v cuMAC.scripts.tests.test_evaluate_pfq_v2_promotion
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_*.py' -v
```

Expected: all shell syntax checks and Python tests pass.

- [ ] **Step 7: Commit experiment tooling**

Run:

```bash
git add cuMAC/scripts/run_stageB_main_experiment.sh \
  cuMAC/scripts/run_stageB_rr_pf_compare.sh \
  cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  cuMAC/scripts/evaluate_pfq_v2_promotion.py \
  cuMAC/scripts/tests/test_evaluate_pfq_v2_promotion.py \
  cuMAC/scripts/run_stageB_pfq_v2_experiment.sh
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add PFQ-v2 experiment and promotion tooling"
```

### Task 10: Run regression, screening, and correctness gates

**Files:**
- Create through runner: `output/pfq_v2_screen/`
- Verify: all implementation and test files from Tasks 1--9

**Interfaces:**
- Screening compares PFQ-v1, cap-only PFQ-v2, and full PFQ-v2 for paired seeds 41--43.
- Screening is diagnostic; it does not replace the formal 10-seed gate.

- [ ] **Step 1: Rebuild from the final source state**

Run:

```bash
cmake -S . -B build.pfq_v2 \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build.pfq_v2 --target pfqMarginalDeliveryTest multiCellSchedulerUeSelection -j8
```

Expected: both targets build without warnings introduced by new code.

- [ ] **Step 2: Run all focused automated checks**

Run:

```bash
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_*.py' -v
bash -n cuMAC/scripts/run_stageB_pfq_v2_experiment.sh
```

Expected: all checks pass.

- [ ] **Step 3: Prove legacy modes remain available from the same binary**

Run one TTI for each legacy baseline from the same configured binary:

```bash
build_method=cmake
for baseline in pf rr pfq rrq; do
  cuMAC/scripts/run_stageB_main_experiment.sh \
    --build-dir build.pfq_v2 \
    --build-method "${build_method}" \
    --topology-scenario 3cell \
    --total-ue-count 36 \
    --tti 1 \
    --cell-radius 500 \
    --cell-assoc-mode strongest \
    --ue-placement coop_boundary \
    --ue-voronoi-clip 1 \
    --topology-seed 41 \
    --traffic-seed 41 \
    --traffic-percent 100 \
    --packet-size-bytes 3000 \
    --traffic-arrival-rate 1.0 \
    --packet-ttl-ms 200 \
    --prbs-per-group 16 \
    --precoding svd \
    --fading-mode 0 \
    --cdl-profiles NA \
    --cdl-delay-spreads 0 \
    --packet-delay-samples 0 \
    --baseline-scheduler "${baseline}" \
    --exec-mode gpu \
    --tag "pfq_legacy_smoke_${baseline}" \
    2>&1 | tee "/tmp/pfq_legacy_smoke_${baseline}.log"
  build_method=skip
done
! rg -n 'PFQ_V2_CONFIG' /tmp/pfq_legacy_smoke_*.log
```

Expected: all four invocations exit zero, retain their original labels, and no legacy log contains `PFQ_V2_CONFIG`.

- [ ] **Step 4: Run three-seed screening**

Run:

```bash
cuMAC/scripts/run_stageB_pfq_v2_experiment.sh \
  --phase screen \
  --build-dir build.pfq_v2 \
  --output-dir output/pfq_v2_screen
```

Expected: nine completed runs: three PFQ-v1, three cap-only, and three full PFQ-v2, all paired by seed.

- [ ] **Step 5: Validate screening correctness before interpreting KPIs**

For all full PFQ-v2 seed logs assert zero hard errors, then parse every diagnostics line and require MCS/layer/TBS parity ratios of at least `0.999`:

```bash
! rg -n 'invalid_candidate_count=[1-9][0-9]*|fallback_tti_count=[1-9][0-9]*' \
  output/pfq_v2_screen
rg -n 'PFQ_V2_CALIBRATION bin=9|PHY_SUCCESS_DELIVERY|SCHEDULER_TIMING' \
  output/pfq_v2_screen
python3 - output/pfq_v2_screen <<'PY'
import pathlib
import re
import sys

lines = []
for path in pathlib.Path(sys.argv[1]).rglob("*.log"):
    lines.extend(
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("PFQ_V2_DIAGNOSTICS ")
    )
assert lines, "no PFQ-v2 diagnostics found"
for line in lines:
    fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", line))
    for name in ("mcs_match_ratio", "layer_match_ratio", "nominal_tbs_match_ratio"):
        assert float(fields[name]) >= 0.999, (name, fields[name], line)
PY
```

Expected: zero invalid/fallback counts, MCS/layer/TBS parity ratios at or above the approved 99.9% threshold, and complete calibration/shadow/timing records.

- [ ] **Step 6: Interpret the ablation before proceeding**

Generate a three-arm table with per-seed and mean goodput, BLER, expiry, legacy packet p95, UE-goodput p5, shadow completion p95, Brier score, blank/cap/nonpositive counts, and wall time. Confirm:

- cap-only isolates the refined demand cap because success probability is exactly 1;
- full PFQ-v2 adds BLER-aware ranking on top of the same cap;
- no arm changes legacy packet/buffer semantics;
- any improvement claim identifies whether it came from cap-only, delivery-awareness, or both.

- [ ] **Step 7: Stop for correctness defects, otherwise continue to runtime gate**

If predictor errors, fallback, or parity mismatch is nonzero, return to the smallest failing task and add a regression test before fixing. Do not start the formal run with a correctness defect.

### Task 11: Run the runtime gate, formal paired experiment, and write the decision report

**Files:**
- Create through runner: `output/pfq_v2_runtime/`
- Create through runner: `output/pfq_v2_formal/`
- Create: `Doc/pfq_v2_marginal_expected_delivery_evaluation.md`

**Interfaces:**
- Runtime gate controls whether the formal experiment may start.
- Formal results use seeds 41--50 and the same new binary for both arms.
- Final report states promotion/non-promotion from the locked gates and preserves the bootstrap interval as diagnostic evidence.

- [ ] **Step 1: Run the canonical runtime protocol**

Run:

```bash
cuMAC/scripts/run_stageB_pfq_v2_experiment.sh \
  --phase runtime \
  --build-dir build.pfq_v2 \
  --output-dir output/pfq_v2_runtime
```

Expected: one untimed warm-up per algorithm and three timed paired repetitions in the approved alternating sequence, all at seed 41/4000 TTIs.

- [ ] **Step 2: Apply the runtime decision**

Read the generated runtime JSON:

- `<=10%`: proceed;
- `>10% and <=20%`: profile persistent copies/host allocations, optimize without changing algorithm semantics, rerun focused/parity tests, then repeat the entire runtime protocol;
- `>20%`: stop and report the runtime blocker; do not launch the ten-seed formal run.

- [ ] **Step 3: Run the formal paired experiment only after runtime passes**

Run:

```bash
cuMAC/scripts/run_stageB_pfq_v2_experiment.sh \
  --phase formal \
  --build-dir build.pfq_v2 \
  --output-dir output/pfq_v2_formal
```

Expected: 20 completed runs, exactly one PFQ-v1 and one full PFQ-v2 arm for each seed 41--50.

- [ ] **Step 4: Validate manifests before evaluation**

Run checks that there are exactly ten unique seeds per arm, identical canonical arguments, identical executable checksum, no duplicate `(seed, algorithm)` rows, no failed runs, and no PFQ-v2 invalid/fallback counts.

Expected: validation exits zero. Any violation invalidates the formal comparison and must be corrected/rerun, not imputed.

- [ ] **Step 5: Evaluate promotion gates**

Run:

```bash
python3 cuMAC/scripts/evaluate_pfq_v2_promotion.py \
  --manifest output/pfq_v2_formal/manifest.csv \
  --output output/pfq_v2_formal/promotion_decision.json \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260812
```

Expected: JSON contains all seven approved KPI/validity gates, ten paired deltas, arithmetic means, and the paired bootstrap interval.

- [ ] **Step 6: Write the evidence-backed report**

Create `Doc/pfq_v2_marginal_expected_delivery_evaluation.md` with:

1. binary commit/checksum and canonical configuration;
2. implementation summary and explicit exclusions;
3. 3-seed ablation table;
4. runtime median and percentage increase;
5. 10-seed per-seed and arithmetic-mean KPI tables;
6. predictor calibration/Brier and MCS/layer/TBS parity;
7. all promotion gates with pass/fail reasons;
8. bootstrap interval labeled diagnostic;
9. final conclusion: promote, do not promote, or blocked by runtime/correctness.

Do not claim PFQ-v2 improves PFQ-v1 unless the strict mean-goodput gate and every guardrail pass.

- [ ] **Step 7: Run final verification before claiming completion**

Run:

```bash
ctest --test-dir build.pfq_v2 -R '^pfqMarginalDeliveryTest$' --output-on-failure
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_*.py' -v
bash -n cuMAC/scripts/run_stageB_main_experiment.sh
bash -n cuMAC/scripts/run_stageB_rr_pf_compare.sh
bash -n cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh
bash -n cuMAC/scripts/run_stageB_pfq_v2_experiment.sh
git diff --check
git status --short
```

Expected: focused C++ and all Python tests pass, shell syntax checks pass, no whitespace errors are introduced, and `git status` shows unrelated user changes still present and untouched.

- [ ] **Step 8: Commit the report without experiment artifacts**

Run:

```bash
git add Doc/pfq_v2_marginal_expected_delivery_evaluation.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: report PFQ-v2 paired evaluation"
```

Expected: generated `output/` data and build products are not committed.

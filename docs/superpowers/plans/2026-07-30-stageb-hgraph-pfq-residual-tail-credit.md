> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-07-30-stageb-hgraph-pfq-residual-tail-credit.md)保留当时状态。

# Stage-B HGraph PFQ Residual and Tail Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start direct candidate learning from PFQ ordering and provide dense, action-aligned credit for unfinished packets approaching their TTL.

**Architecture:** Reuse the existing PFQ candidate edge anchor as the fixed base logit and explicitly initialize the learned service-value head as a near-zero residual. Extend the UE observation from 12 to 16 features using HOL-derived critical/near-expiry backlog, age-weighted backlog, and service gap; compute a pure queue-pressure potential and add `before - gamma * after` shaping without embedding tail heuristics in the decoder.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, C++17, cuMAC online bridge, Bash.

## Global Constraints

- Phase 1 causal gate must pass before this plan starts.
- Keep one shared actor, 17 PRG × 16 PRB, current HGraph topology, hidden size 128, and two message layers.
- Use existing candidate PFQ anchor edge feature; do not copy PFQ final Type-0 actions.
- Fix PFQ anchor coefficient at 1.0 for the first residual experiment.
- Initialize learned candidate service-value logits to zero and use `candidate_blank_logit_bias=-4.0`.
- Append exactly four UE features, increasing the online UE feature dimension from 12 to 16.
- Critical and near-expiry thresholds are 20 ms and 60 ms.
- Queue pressure is a positive cost; shaping credit is `pressure_before - gamma * pressure_after`.
- Decoder tail/rescue/completion biases remain disabled; pressure influences policy through observation and reward.

---

## File Structure

- Create `training/stageb_hgraph/pfq_residual.py`: explicit residual-head initialization and checkpoint metadata.
- Create `training/stageb_hgraph/tests/test_pfq_residual.py`: anchor equality, zero initialization, checkpoint compatibility.
- Modify `training/stageb_hgraph/model.py`: expose separate base, residual, and combined candidate logits.
- Modify `training/stageb_hgraph/feature_snapshot.py`: name and validate four new UE features.
- Modify `cuMAC/examples/onlineTrainBridge/OnlineObservationTypes.h`: set extra/total UE feature dimensions to 8/16.
- Modify `cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`: compute the four new values and track last service.
- Modify `cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`: copy all eight UE extra features.
- Create `training/stageb_hgraph/tests/test_feature_snapshot.py`: dimension/name/value regression.
- Modify `training/stageb_hgraph/reward_utils.py`: pure queue-pressure potential and shaping function.
- Create `training/stageb_hgraph/tests/test_queue_pressure_reward.py`: monotonicity and sign.
- Modify `training/stageb_hgraph/online_ppo.py`: calculate before/after pressure from graph features and log terms.
- Modify `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`: add residual/tail preset.
- Modify `training/stageb_hgraph/README.md`: document feature schema and reward.

### Task 1: Make PFQ base and learned residual explicit

**Files:**
- Create: `training/stageb_hgraph/pfq_residual.py`
- Modify: `training/stageb_hgraph/model.py`
- Create: `training/stageb_hgraph/tests/test_pfq_residual.py`

**Interfaces:**
- Consumes: `StageBHGraphPolicy`, `candidate_pfq_anchor_logits`, candidate mask
- Produces:
  - `initialize_pfq_residual_head(actor: StageBHGraphPolicy) -> None`
  - `should_initialize_pfq_residual(checkpoint: Mapping[str, object] | None, enabled: bool) -> bool`
  - actor outputs `candidate_residual_logits`, `candidate_pfq_base_logits`, `candidate_prg_logits`

- [ ] **Step 1: Write the failing zero-residual test**

```python
def test_zero_initialized_residual_main_logits_equal_pfq_base():
    graph = make_candidate_graph(n_prg=2)
    actor = StageBHGraphPolicy(
        HGraphModelConfig(
            ue_feat_dim=12,
            action_head_mode="candidate",
            max_prg_candidates=2,
            candidate_success_logit_scale=0.0,
            candidate_blank_success_scale=0.0,
            candidate_pfq_anchor_logit_coef=1.0,
            candidate_blank_logit_bias=-4.0,
        )
    )
    initialize_pfq_residual_head(actor)
    out = actor(graph)
    valid = out["candidate_mask"]
    assert torch.allclose(
        out["candidate_residual_logits"][valid],
        torch.zeros_like(out["candidate_residual_logits"][valid]),
        atol=0.0,
    )
    assert torch.allclose(
        out["candidate_prg_logits"][:, :-1][valid],
        out["candidate_pfq_base_logits"][valid],
        atol=1e-6,
    )
    assert torch.allclose(
        out["candidate_prg_logits"][:, -1],
        torch.full_like(out["candidate_prg_logits"][:, -1], -4.0),
        atol=0.0,
    )
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_pfq_residual.py::test_zero_initialized_residual_main_logits_equal_pfq_base -v
```

Expected: FAIL because the initializer and explicit outputs are missing.

- [ ] **Step 3: Expose base/residual outputs**

In `JointType0ActionHead._candidate_logits_batched`, rename the local learned score to `candidate_residual_logits`, keep PFQ anchor as `candidate_pfq_base_logits`, and form:

```python
candidate_logits_main = candidate_residual_logits
if success_scale > 0.0:
    candidate_logits_main = candidate_logits_main + F.logsigmoid(
        candidate_success_logits * max(success_scale, 1.0e-3)
    )
candidate_logits_main = candidate_logits_main + candidate_pfq_base_logits
```

Return all three tensors while retaining `candidate_service_value_logits` and `candidate_pfq_anchor_logits` aliases for old checkpoints and analysis scripts.

- [ ] **Step 4: Implement explicit initialization**

```python
def initialize_pfq_residual_head(actor: StageBHGraphPolicy) -> None:
    pair = actor.action_head.candidate_pair_score
    blank = actor.action_head.candidate_null_head
    if (
        pair is None
        or blank is None
        or not isinstance(pair[-1], nn.Linear)
        or not isinstance(blank[-1], nn.Linear)
    ):
        raise ValueError("candidate residual head is unavailable")
    nn.init.zeros_(pair[-1].weight)
    nn.init.zeros_(pair[-1].bias)
    nn.init.zeros_(blank[-1].weight)
    nn.init.zeros_(blank[-1].bias)
```

The null head is the learnable residual around the fixed `-4.0` blank bias, so
zeroing it makes the initial blank logit deterministic. Do not reset HGraph
encoders, success head, or PFQ edge feature. Save checkpoint metadata:

```python
checkpoint["initialization"] = {
    "mode": "pfq_residual",
    "pfq_anchor_coef": 1.0,
    "candidate_blank_logit_bias": -4.0,
}
```

- [ ] **Step 5: Add compatibility tests**

Verify that a legacy actor config without the new output aliases loads
unchanged and that `build_model_from_config` still defaults missing fields
safely. Implement and test:

```python
def should_initialize_pfq_residual(
    checkpoint: Mapping[str, object] | None,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if checkpoint is None:
        return True
    initialization = checkpoint.get("initialization")
    if not isinstance(initialization, Mapping):
        return True
    return initialization.get("mode") != "pfq_residual"


def test_residual_initialization_is_applied_exactly_once():
    assert should_initialize_pfq_residual(None, enabled=True) is True
    assert should_initialize_pfq_residual({}, enabled=True) is True
    assert should_initialize_pfq_residual(
        {"initialization": {"mode": "legacy"}}, enabled=True
    ) is True
    assert should_initialize_pfq_residual(
        {"initialization": {"mode": "pfq_residual"}}, enabled=True
    ) is False
    assert should_initialize_pfq_residual(None, enabled=False) is False
```

- [ ] **Step 6: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_pfq_residual.py -v
git add training/stageb_hgraph/pfq_residual.py training/stageb_hgraph/model.py \
  training/stageb_hgraph/tests/test_pfq_residual.py
git commit -m "feat: initialize HGraph as PFQ residual policy"
```

Expected: tests pass and commit succeeds.

### Task 2: Extend UE observation to unfinished-packet pressure

**Files:**
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineObservationTypes.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h`
- Modify: `cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`
- Modify: `training/stageb_hgraph/feature_snapshot.py`
- Create: `training/stageb_hgraph/tests/test_feature_snapshot.py`

**Interfaces:**
- Consumes: `PacketHeadSummary.pending_bytes`, `hol_age_tti`, `ttl_slack_tti`, slot duration, observed per-TTI service
- Produces UE features at indices:
  - 12 `critical_ttl_backlog_bytes`
  - 13 `near_ttl_backlog_bytes`
  - 14 `age_weighted_backlog_bytes`
  - 15 `service_gap_tti`

- [ ] **Step 1: Write the failing Python schema test**

```python
def test_snapshot_exposes_sixteen_ue_features_in_stable_order():
    obs = make_candidate_observation(n_prg=2, ue_feat_dim=16)
    obs["obs_ue_features"][0, 12:16] = [3000.0, 6000.0, 1500.0, 7.0]
    snapshot = FeatureSnapshotBuilder().build(obs)
    assert snapshot.ue_features.dim == 16
    assert snapshot.ue_features.names[12:] == (
        "critical_ttl_backlog_bytes",
        "near_ttl_backlog_bytes",
        "age_weighted_backlog_bytes",
        "service_gap_tti",
    )
    assert snapshot.ue_features.values[0, 12:].tolist() == [3000.0, 6000.0, 1500.0, 7.0]
```

- [ ] **Step 2: Run the schema test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_feature_snapshot.py -v
```

Expected: FAIL because `FeatureSnapshotBuilder` truncates UE features to 12.

- [ ] **Step 3: Update the Python feature schema**

Add constants 12--15 and append the four exact names. Keep the first 12 indices unchanged so existing candidate feature offsets remain valid.

- [ ] **Step 4: Update the C++ feature layout**

Set:

```cpp
static constexpr uint32_t kUeExtraFeatDim = 8U;
static constexpr uint32_t kUeFeatDim = kUeBaseFeatDim + kUeExtraFeatDim; // 16
```

In `ObservationExtrasBuilder`, add `m_lastServedTti` initialized to `-1` per UE. During snapshot construction:

```cpp
const float pendingBytes = static_cast<float>(packetHead[ueIdx].pending_bytes);
const float ttlSlackMs =
    packetHead[ueIdx].ttl_slack_tti >= 0
        ? static_cast<float>(packetHead[ueIdx].ttl_slack_tti) * m_slotDurationMs
        : -1.0f;
const float holAgeMs =
    packetHead[ueIdx].hol_age_tti > 0
        ? static_cast<float>(packetHead[ueIdx].hol_age_tti) * m_slotDurationMs
        : 0.0f;
const float criticalBytes =
    (ttlSlackMs >= 0.0f && ttlSlackMs <= 20.0f) ? pendingBytes : 0.0f;
const float nearBytes =
    (ttlSlackMs >= 0.0f && ttlSlackMs <= 60.0f) ? pendingBytes : 0.0f;
const float ttlMs = static_cast<float>(m_packetTtlTti) * m_slotDurationMs;
const float ageWeight = ttlMs > 0.0f ? std::min(2.0f, holAgeMs / ttlMs) : 0.0f;
const float ageWeightedBytes = pendingBytes * ageWeight;
const int currentTti = trafSvc != nullptr ? trafSvc->GetCurrentTti() : 0;
const float serviceGapTti =
    m_lastServedTti[ueIdx] >= 0 ? static_cast<float>(currentTti - m_lastServedTti[ueIdx])
                                : static_cast<float>(std::max(0, currentTti));
```

Store these at UE extra offsets 4--7. Save `trafSvc->GetCurrentTti()` in `m_currentSnapshotTti` during `buildSnapshot`; in `observeTtiResult`, set `m_lastServedTti[ueIdx] = m_currentSnapshotTti` whenever the UE receives positive goodput bytes. Reset both trackers on builder initialization.

- [ ] **Step 5: Copy all eight extra features**

In `OnlineFeatureCodec.cpp`, replace the four explicit copies with a loop:

```cpp
for (uint32_t extraIdx = 0; extraIdx < ObservationFeatureLayout::kUeExtraFeatDim; ++extraIdx) {
    ueFeatures[base + ObservationFeatureLayout::kUeBaseFeatDim + extraIdx] =
        extras->ueExtraFeatures[extraBase + extraIdx];
}
```

- [ ] **Step 6: Run Python and C++ build checks**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_feature_snapshot.py -v
cmake --build build.$(uname -m) --target multiCellSchedulerUeSelection -j2
```

Expected: Python test passes and the C++ target builds.

- [ ] **Step 7: Commit**

```bash
git add cuMAC/examples/onlineTrainBridge/OnlineObservationTypes.h \
  cuMAC/examples/onlineTrainBridge/OnlineObservationExtrasBuilder.h \
  cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp \
  training/stageb_hgraph/feature_snapshot.py \
  training/stageb_hgraph/tests/test_feature_snapshot.py
git commit -m "feat: expose unfinished packet pressure to HGraph"
```

### Task 3: Implement queue-pressure potential shaping

**Files:**
- Modify: `training/stageb_hgraph/reward_utils.py`
- Create: `training/stageb_hgraph/tests/test_queue_pressure_reward.py`

**Interfaces:**
- Consumes: UE feature tensor `[U, >=16]`, optional active-demand mask, gamma
- Produces:
  - `compute_queue_pressure_potential(ue_features, demand_mask, topk=8) -> Tensor`
  - `compute_queue_pressure_shaping(before, after, gamma, weight) -> dict[str, float]`

- [ ] **Step 1: Write failing monotonicity/sign tests**

```python
def test_queue_pressure_increases_with_critical_backlog_and_service_gap():
    base = torch.zeros((2, 16))
    base[:, 0] = 3000.0
    high = base.clone()
    high[0, 12] = 6000.0
    high[0, 15] = 100.0
    mask = torch.tensor([True, True])
    assert compute_queue_pressure_potential(high, mask) > compute_queue_pressure_potential(base, mask)


def test_reducing_pressure_produces_positive_credit():
    shaped = compute_queue_pressure_shaping(
        before=1.0, after=0.5, gamma=0.99, weight=0.25
    )
    assert shaped["queue_pressure_delta"] == pytest.approx(0.505)
    assert shaped["queue_pressure_reward"] == pytest.approx(0.12625)
```

- [ ] **Step 2: Run the tests**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_queue_pressure_reward.py -v
```

Expected: FAIL because the functions are missing.

- [ ] **Step 3: Implement the pure potential**

For active-demand UEs:

```python
critical = torch.log1p(torch.clamp_min(ue_features[:, 12], 0.0) / 3000.0)
near = torch.log1p(torch.clamp_min(ue_features[:, 13], 0.0) / 3000.0)
aged = torch.log1p(torch.clamp_min(ue_features[:, 14], 0.0) / 3000.0)
gap = torch.clamp(torch.clamp_min(ue_features[:, 15], 0.0) / 100.0, max=4.0)
per_ue = 2.0 * critical + 1.0 * near + 0.5 * aged + 0.5 * gap
selected = per_ue[demand_mask]
potential = selected.topk(min(topk, selected.numel())).values.mean() if selected.numel() else per_ue.sum() * 0.0
```

Shaping uses:

```python
delta = float(before) - float(gamma) * float(after)
reward = float(weight) * delta
```

Return before, after, delta, weight, and reward for logging.

- [ ] **Step 4: Add edge-case tests**

Cover no active demand, negative input clamping, top-k smaller/larger than active count, and finite gradients through tensor potential.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_queue_pressure_reward.py -v
git add training/stageb_hgraph/reward_utils.py training/stageb_hgraph/tests/test_queue_pressure_reward.py
git commit -m "feat: add dense unfinished packet pressure credit"
```

### Task 4: Integrate residual initialization and pressure credit into online PPO

**Files:**
- Modify: `training/stageb_hgraph/online_ppo.py`
- Modify: `training/stageb_hgraph/tests/test_pfq_residual.py`
- Modify: `training/stageb_hgraph/tests/test_queue_pressure_reward.py`

**Interfaces:**
- Consumes: Phase 1 direct rollout, 16-dimensional UE features, queue-pressure functions
- Produces: checkpoint metadata and metrics `queue_pressure_before/after/delta/reward`

- [ ] **Step 1: Write the failing integration test**

Create pre/post fixture graphs where critical backlog falls from 6000 to 0 bytes. Invoke the extracted reward helper and assert:

```python
assert metrics["queue_pressure_before"] > metrics["queue_pressure_after"]
assert metrics["queue_pressure_reward"] > 0.0
assert reward_with_pressure > reward_without_pressure
```

- [ ] **Step 2: Run the integration test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_queue_pressure_reward.py -k integration -v
```

Expected: FAIL because online PPO does not call the new potential.

- [ ] **Step 3: Add CLI and initialization**

Add:

```python
p.add_argument("--pfq-residual-init", type=int, choices=[0, 1], default=0)
p.add_argument("--ue-feat-dim", type=int, default=12)
p.add_argument("--queue-pressure-weight", type=float, default=0.0)
p.add_argument("--queue-pressure-topk", type=int, default=8)
```

Require PFQ anchor coefficient 1.0, success-logit scale 0.0,
blank-success scale 0.0, and UE feature dimension 16 when
`pfq_residual_init=1`. Pass `ue_feat_dim=int(args.ue_feat_dim)` into
`HGraphModelConfig`.

Apply initialization exactly once:

- with no `--init-checkpoint`, initialize the new actor;
- with a causal/legacy initialization checkpoint whose
  `initialization.mode` is not `pfq_residual`, load compatible weights first
  and then reset only the candidate pair/null final layers;
- with a checkpoint already carrying `initialization.mode=pfq_residual`, do
  not reset learned residual weights.

Use `should_initialize_pfq_residual` for this branch and persist the
initialization metadata in every subsequent checkpoint so bucket macrocycles
cannot accidentally reinitialize the head.

- [ ] **Step 4: Add before/after shaping to rollout**

Compute pre-action pressure from the current graph, build the next graph from `next_obs`, compute post-action pressure, and add:

```python
pressure_terms = compute_queue_pressure_shaping(
    before=float(before_pressure),
    after=float(after_pressure),
    gamma=float(reward_cfg["gamma"]),
    weight=float(reward_cfg["queue_pressure_weight"]),
)
reward_result["reward_shaped"] += pressure_terms["queue_pressure_reward"]
reward_result.update(pressure_terms)
```

Cache the already-built next graph so the next loop iteration does not rebuild a semantically different snapshot.
Add `"gamma": float(args.gamma)`, `"queue_pressure_weight"`, and `"queue_pressure_topk"` to `reward_cfg`.

- [ ] **Step 5: Run unit regression**

```bash
.venv/bin/python -m pytest \
  training/stageb_hgraph/tests/test_pfq_residual.py \
  training/stageb_hgraph/tests/test_feature_snapshot.py \
  training/stageb_hgraph/tests/test_queue_pressure_reward.py \
  training/stageb_hgraph/tests/test_online_ppo_causal_alignment.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add training/stageb_hgraph/online_ppo.py \
  training/stageb_hgraph/tests/test_pfq_residual.py \
  training/stageb_hgraph/tests/test_queue_pressure_reward.py
git commit -m "feat: train direct HGraph with PFQ residual tail credit"
```

### Task 5: Add the residual-tail preset and online smoke

**Files:**
- Modify: `cuMAC/scripts/run_stageB_hgraph_online_ppo.sh`
- Modify: `training/stageb_hgraph/README.md`
- Produce: `training/stageb_hgraph/runs/causal_residual_smoke/`

**Interfaces:**
- Consumes: direct candidate preset
- Produces: `goodput_delay_rate_v21_causal_pfq_residual_tail_sample_fast`

- [ ] **Step 1: Add the preset**

Extend the v20 direct preset with:

```bash
PPO_PRESET="goodput_delay_rate_v21_causal_pfq_residual_tail_sample_fast"
CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
CANDIDATE_BLANK_LOGIT_BIAS=-4.0
CANDIDATE_SUCCESS_LOGIT_SCALE=0.0
CANDIDATE_BLANK_SUCCESS_SCALE=0.0
PFQ_RESIDUAL_INIT=1
QUEUE_PRESSURE_WEIGHT=0.25
QUEUE_PRESSURE_TOPK=8
UE_FEAT_DIM=16
```

Pass the three new Python flags and record the resolved feature dimension and initialization metadata.

- [ ] **Step 2: Validate shell syntax**

```bash
bash -n cuMAC/scripts/run_stageB_hgraph_online_ppo.sh
```

Expected: exit 0.

- [ ] **Step 3: Run the online bridge schema smoke**

```bash
./cuMAC/scripts/run_stageB_hgraph_online_ppo.sh \
  --ppo-preset goodput_delay_rate_v21_causal_pfq_residual_tail_sample_fast \
  --iterations 2 \
  --rollout-steps 64 \
  --episode-horizon 128 \
  --topology-seed 41 \
  --out-dir training/stageb_hgraph/runs/causal_residual_smoke
```

Expected:

- bridge reports `ue_feat_dim=16`;
- candidate gradient and entropy remain positive;
- non-hard override ratio stays below 0.05;
- all queue-pressure metrics are finite;
- no Python or native reject occurs.

- [ ] **Step 4: Inspect initial PFQ residual behavior**

Read the first rollout summary and require `candidate_residual_logits_abs_mean` near zero before the first optimizer update, while `candidate_pfq_base_logits_abs_mean` is positive. Archive these values in the smoke summary.

- [ ] **Step 5: Document and commit**

```bash
git add cuMAC/scripts/run_stageB_hgraph_online_ppo.sh training/stageb_hgraph/README.md
git commit -m "docs: add causal PFQ residual tail preset"
```

Do not commit generated run output.

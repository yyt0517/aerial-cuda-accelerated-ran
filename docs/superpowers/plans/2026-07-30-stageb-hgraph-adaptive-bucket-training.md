# Stage-B HGraph Adaptive Bucket Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the parity-passing PFQ-residual HGraph policy on G/U/T/R failure buckets while selecting checkpoints only by uniform paired seed41--50 performance and hard packet/worst-seed gates.

**Architecture:** Keep bucket identity outside the actor. A deterministic curriculum scheduler expands bucket weights into a seed schedule for each 25-iteration macrocycle; an orchestrator runs training, uniform validation, gate evaluation, weight updates, and rollback. Final promotion is based on full 4000-TTI paired results, never weighted training metrics.

**Tech Stack:** Python 3, pytest, NumPy, JSON/CSV, subprocess, existing Stage-B Bash runners, PyTorch checkpoints.

## Global Constraints

- Causal, PFQ-residual/tail, and deployment parity gates must all report `"passed": true`.
- Buckets are fixed initially: G={42,47}, U={45,48}, T={43}, R={41,44,46,49,50}.
- Initial weights are G25%, U30%, T25%, R20%.
- Floors are G20%, U20%, T15%, R15%; any bucket has a 40% ceiling.
- One macrocycle is exactly 25 PPO iterations.
- Bucket and seed identifiers are used only by the runner/orchestrator, never actor inputs or graph features.
- Every macrocycle ends with uniform seed41--50 paired evaluation.
- Final full-run and post-1000 mean goodput must exceed PFQ; worst-seed delta must be at least -1.0%.
- Pooled expiry must not exceed PFQ; pooled completion must not fall below PFQ; cross-seed mean p95 delay must not exceed PFQ.
- A U-bucket utilization increase without goodput/BLER improvement is not progress.
- A T-bucket completed-packet mean-delay improvement with worse expiry, pending, or p95 is not progress.
- Seeds51--60 are directional generalization only and do not choose the seed41--50 checkpoint.

---

## File Structure

- Create `training/stageb_hgraph/configs/bucket_seed41_50_v1.json`: immutable bucket, weights, floors, caps, macrocycle, and gate configuration.
- Create `training/stageb_hgraph/bucket_curriculum.py`: config validation, weighted seed schedules, severity-based weight updates.
- Create `training/stageb_hgraph/tests/test_bucket_curriculum.py`: determinism, proportions, floors/caps, failure weighting.
- Modify `training/stageb_hgraph/online_imitation.py`: optional explicit topology-seed override for each rollout.
- Modify `training/stageb_hgraph/online_ppo.py`: optional `--seed-schedule-file`, record bucket/seed per rollout.
- Create `training/stageb_hgraph/run_bucket_training.py`: macrocycle orchestration, subprocess manifests, resume/rollback.
- Create `training/stageb_hgraph/tests/test_run_bucket_training.py`: mocked command/resume/rollback tests.
- Create `cuMAC/scripts/check_stageB_hgraph_gates.py`: paired per-seed KPI computation and hard gate.
- Create `cuMAC/scripts/tests/test_check_stageB_hgraph_gates.py`: synthetic manifests covering every gate.
- Modify `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`: persist `n_prg` in every evaluation manifest row.
- Modify `training/stageb_hgraph/README.md`: curriculum, resume, final validation, and interpretation.

### Task 1: Freeze and validate the bucket configuration

**Files:**
- Create: `training/stageb_hgraph/configs/bucket_seed41_50_v1.json`
- Create: `training/stageb_hgraph/bucket_curriculum.py`
- Create: `training/stageb_hgraph/tests/test_bucket_curriculum.py`

**Interfaces:**
- Consumes: JSON configuration
- Produces:
  - `BucketConfig.load(path: Path) -> BucketConfig`
  - `BucketScheduler.make_macrocycle(iteration_offset: int, rng_seed: int) -> list[ScheduledSeed]`
  - `BucketScheduler.update(validation: Mapping[int, SeedDelta]) -> dict[str, float]`

- [ ] **Step 1: Write the immutable configuration**

```json
{
  "version": 1,
  "macrocycle_iterations": 25,
  "buckets": {
    "G": {"seeds": [42, 47], "weight": 0.25, "floor": 0.20},
    "U": {"seeds": [45, 48], "weight": 0.30, "floor": 0.20},
    "T": {"seeds": [43], "weight": 0.25, "floor": 0.15},
    "R": {"seeds": [41, 44, 46, 49, 50], "weight": 0.20, "floor": 0.15}
  },
  "max_bucket_weight": 0.40,
  "adaptation_rate": 0.10,
  "gates": {
    "warmup_tti": 1000,
    "worst_goodput_delta_pct": -1.0,
    "require_full_mean_goodput_positive": true,
    "require_post_warmup_mean_goodput_positive": true,
    "require_pooled_expiry_nonworse": true,
    "require_pooled_completion_nonworse": true,
    "require_mean_p95_delay_nonworse": true
  }
}
```

- [ ] **Step 2: Write failing validation and schedule tests**

```python
DEFAULT_BUCKET_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "bucket_seed41_50_v1.json"
)


def test_default_config_covers_each_seed_once():
    cfg = BucketConfig.load(DEFAULT_BUCKET_CONFIG)
    seeds = [seed for bucket in cfg.buckets.values() for seed in bucket.seeds]
    assert sorted(seeds) == list(range(41, 51))
    assert len(seeds) == len(set(seeds))


def test_macrocycle_is_deterministic_and_has_25_items():
    scheduler = BucketScheduler(BucketConfig.load(DEFAULT_BUCKET_CONFIG))
    first = scheduler.make_macrocycle(iteration_offset=0, rng_seed=123)
    second = scheduler.make_macrocycle(iteration_offset=0, rng_seed=123)
    assert first == second
    assert len(first) == 25
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_bucket_curriculum.py -v
```

Expected: FAIL because `bucket_curriculum.py` is missing.

- [ ] **Step 4: Implement strict config validation**

Reject:

- duplicate or missing seeds41--50;
- weights not summing to 1 within `1e-9`;
- floors summing above 1;
- weight below its floor or above 0.40;
- non-positive macrocycle length;
- unknown bucket names.

Define:

```python
@dataclass(frozen=True)
class ScheduledSeed:
    iteration: int
    seed: int
    bucket: str


@dataclass(frozen=True)
class SeedDelta:
    goodput_delta_pct: float
    expiry_delta: float
    completion_delta: float
    p95_delay_delta_ms: float
    bler_delta: float = 0.0
    utilization_delta: float = 0.0
    pending_packet_delta: float = 0.0
```

- [ ] **Step 5: Implement largest-remainder scheduling**

Multiply weights by 25, floor counts, assign remaining iterations to largest fractional remainders with bucket-name tie breaking, choose seeds round-robin within each bucket, then shuffle the 25 `ScheduledSeed` records using `random.Random(rng_seed + iteration_offset)`.

- [ ] **Step 6: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_bucket_curriculum.py -v
git add training/stageb_hgraph/configs/bucket_seed41_50_v1.json \
  training/stageb_hgraph/bucket_curriculum.py \
  training/stageb_hgraph/tests/test_bucket_curriculum.py
git commit -m "feat: define Stage-B seed bucket curriculum"
```

### Task 2: Implement bounded adaptive bucket weights

**Files:**
- Modify: `training/stageb_hgraph/bucket_curriculum.py`
- Modify: `training/stageb_hgraph/tests/test_bucket_curriculum.py`

**Interfaces:**
- Consumes: per-seed validation deltas
- Produces: new weights satisfying all floors and 0.40 cap

- [ ] **Step 1: Write failing severity tests**

```python
def default_scheduler() -> BucketScheduler:
    return BucketScheduler(BucketConfig.load(DEFAULT_BUCKET_CONFIG))


def passing_validation() -> dict[int, SeedDelta]:
    return {
        seed: SeedDelta(
            goodput_delta_pct=0.1,
            expiry_delta=0.0,
            completion_delta=0.0,
            p95_delay_delta_ms=0.0,
        )
        for seed in range(41, 51)
    }


def test_underfill_failures_raise_u_weight_without_breaking_bounds():
    scheduler = default_scheduler()
    validation = passing_validation()
    validation[45] = SeedDelta(-3.0, 0.0, 0.0, 0.0)
    validation[48] = SeedDelta(-2.0, 0.0, 0.0, 0.0)
    updated = scheduler.update(validation)
    assert updated["U"] > 0.30
    assert updated["G"] >= 0.20
    assert updated["T"] >= 0.15
    assert updated["R"] >= 0.15
    assert max(updated.values()) <= 0.40
    assert sum(updated.values()) == pytest.approx(1.0)


def test_tail_violation_dominates_small_goodput_gain():
    scheduler = default_scheduler()
    validation = passing_validation()
    validation[43] = SeedDelta(0.2, 0.001, -10.0, 8.0)
    assert scheduler.update(validation)["T"] > 0.25


def test_u_utilization_only_change_is_not_treated_as_progress():
    scheduler = default_scheduler()
    validation = passing_validation()
    validation[45] = SeedDelta(
        goodput_delta_pct=0.0,
        expiry_delta=0.0,
        completion_delta=0.0,
        p95_delay_delta_ms=0.0,
        bler_delta=0.01,
        utilization_delta=0.05,
    )
    assert scheduler.update(validation)["U"] > 0.30
```

- [ ] **Step 2: Run the focused tests**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_bucket_curriculum.py -k 'underfill or tail_violation' -v
```

Expected: FAIL because `update` is not implemented.

- [ ] **Step 3: Implement severity**

For each seed, compute this base severity:

```python
base_severity = (
    max(0.0, -delta.goodput_delta_pct) / 1.0
    + 2.0 * max(0.0, delta.expiry_delta) / 0.001
    + 1.0 * max(0.0, -delta.completion_delta) / 10.0
    + 1.0 * max(0.0, delta.p95_delay_delta_ms) / 5.0
)
```

Add a U mechanism penalty of
`1.0 + min(2.0, utilization_delta / 0.01) + min(2.0, bler_delta / 0.01)`
when utilization increases and either goodput is non-positive or BLER worsens.
For T, add `max(0.0, pending_packet_delta) / 10.0`. Bucket severity is
the mean of member-seed total severity.

If every bucket severity is zero, keep the current weights. Otherwise:

1. set `free_mass = 1.0 - sum(bucket.floor)`;
2. form `raw_target[b] = floor[b] + free_mass * severity[b] / sum(severity)`;
3. project `raw_target` to the floor/cap simplex;
4. form `proposed[b] = current[b] + adaptation_rate * (target[b] - current[b])`;
5. project `proposed` again and persist it.

The projection first clamps each bucket to `[floor, 0.40]`. If the sum is
below one, repeatedly divide the residual equally among buckets with remaining
headroom, applying each bucket's capped share simultaneously. If the sum is
above one, apply the same procedure to subtract equally from buckets above
their floors. Sort active bucket names before every redistribution round so
floating-point ties are deterministic.

- [ ] **Step 4: Add preservation pressure for G**

If either gain-anchor seed has negative goodput delta or violates packet gates, treat that regression as severity in G. Do not allow adaptive logic to starve G while fixing U/T.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_bucket_curriculum.py -v
git add training/stageb_hgraph/bucket_curriculum.py \
  training/stageb_hgraph/tests/test_bucket_curriculum.py
git commit -m "feat: adapt seed buckets within safety bounds"
```

### Task 3: Feed explicit seed schedules into online PPO

**Files:**
- Modify: `training/stageb_hgraph/online_imitation.py`
- Modify: `training/stageb_hgraph/online_ppo.py`
- Create: `training/stageb_hgraph/tests/test_seed_schedule.py`

**Interfaces:**
- Consumes: JSONL records `{"iteration": 1, "seed": 42, "bucket": "G"}`
- Produces rollout metadata `scheduled_seed`, `scheduled_bucket`

- [ ] **Step 1: Write the failing runner override test**

```python
def test_episode_seed_override_precedes_legacy_resolution():
    runner = object.__new__(OnlineEpisodeRunner)
    runner.resolved_topology_seed_mode = "fixed"
    runner.explicit_seed_list = []
    runner.args = SimpleNamespace(topology_seed=41)
    assert runner._episode_seed_for(episode_idx=3, topology_seed_override=48) == 48
    assert runner._episode_seed_for(episode_idx=3, topology_seed_override=None) == 41
```

Also test that no override preserves legacy fixed/sequential/list-cycle behavior.

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_seed_schedule.py -v
```

Expected: FAIL because `begin_rollout` does not accept the override.

- [ ] **Step 3: Add explicit runner override**

Add optional `topology_seed_override: int | None` to `_episode_seed_for`, `begin_rollout`, and `reset_episode`. When supplied, use it for both process topology seed and bridge reset seed; do not mutate the legacy resolver.

- [ ] **Step 4: Add schedule parsing to PPO**

Add:

```python
p.add_argument("--seed-schedule-file", default="")
```

Parse exactly one record per iteration, require contiguous 1-based iteration numbers, bucket in G/U/T/R, and no extra/missing records relative to `--iterations`. Pass the scheduled seed to `runner.begin_rollout`.

- [ ] **Step 5: Record metadata**

Add `scheduled_seed` and one-hot `scheduled_bucket_G/U/T/R` to iteration CSV, checkpoint args, and summary. Do not append these values to graph features.

- [ ] **Step 6: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_seed_schedule.py -v
git add training/stageb_hgraph/online_imitation.py training/stageb_hgraph/online_ppo.py \
  training/stageb_hgraph/tests/test_seed_schedule.py
git commit -m "feat: drive PPO with external seed schedules"
```

### Task 4: Implement paired KPI hard gates

**Files:**
- Create: `cuMAC/scripts/check_stageB_hgraph_gates.py`
- Create: `cuMAC/scripts/tests/test_check_stageB_hgraph_gates.py`

**Interfaces:**
- Consumes: PFQ and AI scenario manifests containing paired seed KPI/eval JSON paths
- Produces CSV per-seed deltas and JSON aggregate gate report; exit 0 only when all hard gates pass

- [ ] **Step 1: Write failing synthetic gate tests**

Create temporary PFQ/AI manifests for ten seeds. The test-local helper writes one `kpi_summary.json`, one `eval_summary.json`, and one manifest row per seed using this exact minimal schema:

```python
def write_paired_manifests(
    root: Path,
    ai_goodput_delta: float = 1.0,
    mutation: str = "",
) -> tuple[Path, Path]:
    manifests = []
    for label in ("pfq", "ai"):
        rows = []
        for seed in range(41, 51):
            goodput = 1500.0 + (ai_goodput_delta if label == "ai" else 0.0)
            global_kpi = {
                "cluster_goodput_mbps": goodput,
                "total_expired_packets": 0,
                "packet_delay_served_pkt_count": 4000,
                "packet_delay_pending_pkt_count": 0,
                "packet_delay_p95_ms": 2.0,
                "global_tb_bler": 0.11,
                "prg_utilization_ratio": 0.90,
            }
            traffic = {"generated_pkts": 4000}
            eval_summary = {
                "stats_mean_goodput_mbps": goodput,
                "stats_sum_packet_completed_count": 4000.0,
                "stats_mean_expiry_drop_rate": 0.0,
            }
            apply_named_mutation(label, seed, mutation, global_kpi, traffic, eval_summary)
            if label == "ai" and seed == 45 and mutation == "missing_generated_pkts":
                traffic.pop("generated_pkts")
            run = root / label / f"s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            kpi_path = run / "kpi_summary.json"
            eval_path = run / "eval_summary.json"
            kpi_path.write_text(
                json.dumps({
                    "tti_count": 4000,
                    "prg_utilization": {
                        "allocated_prg_count": 183600,
                        "total_prg_capacity": 204000,
                        "utilization_ratio": 0.90,
                    },
                    "global_kpi": global_kpi,
                    "traffic": traffic,
                }),
                encoding="utf-8",
            )
            eval_path.write_text(json.dumps(eval_summary), encoding="utf-8")
            rows.append({
                "scenario": "RAYLEIGH",
                "seed": seed,
                "n_prg": (
                    68
                    if label == "ai" and seed == 45 and mutation == "n_prg_mismatch"
                    else 17
                ),
                "kpi_summary_json": str(kpi_path),
                "eval_summary_json": str(eval_path),
            })
        if label == "ai" and mutation == "duplicate_seed":
            rows.append(dict(rows[-1]))
        manifest = root / f"{label}_manifest.csv"
        write_manifest_csv(manifest, rows)
        manifests.append(manifest)
    return manifests[0], manifests[1]
```

Define the two helpers in the same test file:

```python
def apply_named_mutation(label, seed, mutation, global_kpi, traffic, eval_summary):
    if label != "ai":
        return
    if mutation == "negative_full_mean":
        global_kpi["cluster_goodput_mbps"] -= 5.0
    elif mutation == "negative_post_mean":
        eval_summary["stats_mean_goodput_mbps"] -= 5.0
    elif mutation == "worst_below_minus_one" and seed == 45:
        global_kpi["cluster_goodput_mbps"] = 1480.0
    elif mutation == "expiry_worse":
        global_kpi["total_expired_packets"] = 10
    elif mutation == "completion_worse":
        global_kpi["packet_delay_served_pkt_count"] = 3990
    elif mutation == "p95_worse":
        global_kpi["packet_delay_p95_ms"] = 3.0


def write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "seed",
                "n_prg",
                "kpi_summary_json",
                "eval_summary_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
```

Then assert:

```python
def test_gate_passes_positive_means_and_packet_constraints(tmp_path):
    pfq, ai = write_paired_manifests(tmp_path, ai_goodput_delta=1.0)
    report = evaluate_manifests(pfq, ai, warmup_tti=1000, worst_goodput_delta_pct=-1.0)
    assert report["passed"] is True


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        ("negative_full_mean", "full_mean_goodput_positive"),
        ("negative_post_mean", "post_warmup_mean_goodput_positive"),
        ("worst_below_minus_one", "worst_goodput_delta"),
        ("expiry_worse", "pooled_expiry_nonworse"),
        ("completion_worse", "pooled_completion_nonworse"),
        ("p95_worse", "mean_p95_delay_nonworse"),
    ],
)
def test_each_hard_gate_fails_independently(tmp_path, mutation, failed_check):
    pfq, ai = write_paired_manifests(tmp_path, mutation=mutation)
    report = evaluate_manifests(
        pfq,
        ai,
        warmup_tti=1000,
        worst_goodput_delta_pct=-1.0,
    )
    assert report["passed"] is False
    assert report["checks"][failed_check] is False


@pytest.mark.parametrize(
    "mutation,error_pattern",
    [
        ("duplicate_seed", "duplicate seed"),
        ("n_prg_mismatch", "n_prg"),
        ("missing_generated_pkts", "generated_pkts"),
    ],
)
def test_invalid_paired_inputs_are_rejected(tmp_path, mutation, error_pattern):
    pfq, ai = write_paired_manifests(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match=error_pattern):
        evaluate_manifests(
            pfq,
            ai,
            warmup_tti=1000,
            worst_goodput_delta_pct=-1.0,
        )
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest cuMAC/scripts/tests/test_check_stageB_hgraph_gates.py -v
```

Expected: FAIL because the gate script is missing.

- [ ] **Step 3: Implement strict paired loading**

First modify `run_stageB_hgraph_checkpoint_multi_seed_compare.sh` to resolve
`prbs_per_group` from the forwarded `--prbs-per-group` argument, default it to
16, reject values that do not divide 272, compute `n_prg=272/prbs_per_group`,
and add `n_prg` to every manifest row.

Require identical seed sets, scenario, manifest `n_prg`, `tti_count`,
`traffic.generated_pkts`, `traffic.generated_bytes` when the field is present
in either member of a pair, and warm-up
window. Reject duplicate seeds and missing metrics. Load:

- full goodput: `global_kpi.cluster_goodput_mbps`;
- generated packets: `traffic.generated_pkts`;
- expired packets: `global_kpi.total_expired_packets`;
- completed packets: `global_kpi.packet_delay_served_pkt_count`;
- pending packets: `global_kpi.packet_delay_pending_pkt_count`;
- p95 delay: `global_kpi.packet_delay_p95_ms`;
- diagnostic BLER/utilization: `global_kpi.global_tb_bler`, `global_kpi.prg_utilization_ratio`;
- post-warmup goodput: `eval_summary.stats_mean_goodput_mbps`;
- post-warmup completion: `eval_summary.stats_sum_packet_completed_count`;
- post-warmup expiry diagnostic: `eval_summary.stats_mean_expiry_drop_rate`.

- [ ] **Step 4: Compute exact gates**

Report:

- per-seed absolute and percentage goodput delta;
- mean/std/min/max and paired bootstrap 95% interval using RNG seed 20260730;
- pooled expired/completed/generated-packet counts before computing rates;
- mean per-seed p95 delay delta;
- BLER and PRG utilization as diagnostic, not independent pass criteria.

The JSON has `checks`, `passed`, `per_seed`, `aggregate`, `input_hashes`, and `warmup_tti`.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest cuMAC/scripts/tests/test_check_stageB_hgraph_gates.py -v
git add cuMAC/scripts/check_stageB_hgraph_gates.py \
  cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  cuMAC/scripts/tests/test_check_stageB_hgraph_gates.py
git commit -m "feat: enforce paired HGraph packet and worst-seed gates"
```

### Task 5: Implement resumable macrocycle orchestration

**Files:**
- Create: `training/stageb_hgraph/run_bucket_training.py`
- Create: `training/stageb_hgraph/tests/test_run_bucket_training.py`

**Interfaces:**
- Consumes: bucket config, initial checkpoint, causal/deployment gate files, baseline PFQ manifest
- Produces:
  - `macrocycle_XX/seed_schedule.jsonl`
  - training command/result
  - uniform evaluation manifest
  - gate report
  - updated weights
  - `curriculum_state.json`

- [ ] **Step 1: Write failing command-generation test**

Mock subprocess and assert macrocycle 1 invokes:

1. online PPO with exactly 25 iterations and generated schedule;
2. checkpoint multi-seed compare with `--seed-list 41,42,43,44,45,46,47,48,49,50`;
3. the hard gate script with warm-up 1000 and worst delta -1.0.

- [ ] **Step 2: Write failing resume/rollback tests**

Assert:

- a completed macrocycle is not rerun;
- a partial macrocycle resumes from the last completed stage;
- two consecutive uniform mean/worst regressions select the last passing checkpoint and stop;
- a packet-gate failure marks the macrocycle rejected even if training reward rises.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_run_bucket_training.py -v
```

Expected: FAIL because the orchestrator is missing.

- [ ] **Step 4: Implement state machine**

Use stages:

```python
STAGES = (
    "schedule_written",
    "training_complete",
    "uniform_eval_complete",
    "gate_complete",
    "weights_updated",
)
```

Write state atomically with `Path.replace`, record command arrays rather than shell strings, and capture stdout/stderr paths and exit codes. Validate prerequisite gate JSON files before stage 1.

- [ ] **Step 5: Implement checkpoint selection**

Maintain:

- `last_checkpoint`;
- `last_passing_checkpoint`;
- `best_uniform_checkpoint`;
- previous two uniform mean/worst deltas.

Rank passing candidates lexicographically by:

1. gate pass;
2. full mean goodput delta;
3. post-warmup mean goodput delta;
4. worst-seed goodput delta;
5. lower pooled expiry;
6. lower mean p95 delay.

Define one macrocycle regression event as both its full-window mean goodput
delta and its worst-seed goodput delta being lower than the immediately
preceding completed macrocycle. Two consecutive regression events stop further
training, restore `last_passing_checkpoint`, and record
`stop_reason="two_consecutive_mean_and_worst_regressions"`. A hard packet-gate
failure never replaces `last_passing_checkpoint`.

- [ ] **Step 6: Run and commit**

```bash
.venv/bin/python -m pytest training/stageb_hgraph/tests/test_run_bucket_training.py -v
git add training/stageb_hgraph/run_bucket_training.py \
  training/stageb_hgraph/tests/test_run_bucket_training.py
git commit -m "feat: orchestrate gated HGraph bucket macrocycles"
```

### Task 6: Run fixed and adaptive curriculum

**Files:**
- Produce under: `output/stageB_hgraph_bucket_training_<timestamp>/`
- Modify: `training/stageb_hgraph/README.md`

**Interfaces:**
- Consumes: parity-passing checkpoint and PFQ seed41--50 manifest
- Produces: promoted checkpoint, all macrocycle manifests, final uniform and unseen-seed reports

- [ ] **Step 1: Run one fixed-weight macrocycle**

```bash
.venv/bin/python training/stageb_hgraph/run_bucket_training.py \
  --config training/stageb_hgraph/configs/bucket_seed41_50_v1.json \
  --init-checkpoint training/stageb_hgraph/runs/causal_residual/ppo_actor_best.pt \
  --pfq-manifest output/pfq_seed41_50/scenario_seed_manifest.csv \
  --causal-gate training/stageb_hgraph/runs/causal_direct_smoke/causal_gate.json \
  --deployment-gate training/stageb_hgraph/exports/action_parity_report.json \
  --macrocycles 1 \
  --adaptive 0 \
  --out output/stageB_hgraph_bucket_training_fixed
```

Expected: 25-iteration training, uniform ten-seed evaluation, and a gate report complete.

- [ ] **Step 2: Review mechanism metrics**

For G/U/T/R separately inspect goodput, BLER, PRG utilization, expiry, completion, p95 delay, candidate gradient, entropy, and override ratio. Reject:

- U improvement that only increases utilization while goodput or BLER worsens;
- T improvement limited to completed-packet mean delay while expiry/pending/p95 worsens;
- G regression hidden by U/T gains.

- [ ] **Step 3: Run adaptive macrocycles**

Resume the same output root with:

```bash
.venv/bin/python training/stageb_hgraph/run_bucket_training.py \
  --config training/stageb_hgraph/configs/bucket_seed41_50_v1.json \
  --resume output/stageB_hgraph_bucket_training_fixed/curriculum_state.json \
  --macrocycles 8 \
  --adaptive 1
```

Expected: each macrocycle weight vector respects all floors/caps and uniform evaluation runs every 25 iterations.

- [ ] **Step 4: Run the final 4000-TTI seed41--50 gate**

Use the best passing checkpoint:

```bash
./cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --checkpoint-path output/stageB_hgraph_bucket_training_fixed/best_uniform_checkpoint.pt \
  --model-label hgraph_causal_residual_bucket \
  --decode-mode argmax \
  --eval-episode-horizon 4000 \
  --eval-stats-warmup-steps 1000 \
  --baseline-mean-root output/pfq_seed41_50
```

Then run the gate script. Expected: all hard checks pass.

- [ ] **Step 5: Run unseen seeds51--60**

Run the same checkpoint and action operator on seeds51--60. Produce a separate directional report; do not modify curriculum weights or checkpoint ranking after seeing it.

- [ ] **Step 6: Document and commit**

Document config hash, checkpoint hash, architecture/deployment gate paths, fixed/adaptive weight history, final seed41--50 result, and unseen-seed result.

```bash
git add training/stageb_hgraph/README.md
git commit -m "docs: record adaptive HGraph bucket training workflow"
```

Do not commit generated output or checkpoints unless the repository's artifact policy explicitly requires them.

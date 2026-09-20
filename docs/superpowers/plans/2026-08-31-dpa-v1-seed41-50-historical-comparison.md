> 历史材料，2026-09-20 起不作为修正环境的性能证据或续跑许可。见[三项修正与新实验要求](../../../Doc/stageB_corrected_environment_v1.md)；[修正前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/plans/2026-08-31-dpa-v1-seed41-50-historical-comparison.md)保留当时状态。

# DPA-v1 Seed41--50 Historical Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one frozen 4000-TTI DPA-v1 rollout on each seed 41--50, derive full-4000 and post-1000/steady-3000 metrics, and compare them comprehensively with the preserved PFQ and v9c evidence.

**Architecture:** Reuse the existing online checkpoint evaluator, simulator traces, KPI summarizer, and per-UE service audit. Add one experiment-specific Python module that validates frozen provenance, aggregates DPA-v1 windows, reads historical PFQ/v9c artifacts, and writes the normalized comparison report; do not add a generalized experiment framework or modify scheduling behavior.

**Tech Stack:** C++/CUDA Aerial simulator, Python 3, PyTorch, NumPy, CSV/JSON, pytest, Markdown.

**Spec:** `docs/superpowers/specs/2026-08-31-dpa-v1-seed41-50-historical-comparison-design.md`

## Global Constraints

- Work only in `/opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor` after entering the container with `./cuPHY-CP/container/attach_aerial.sh` and activating `source .venv/bin/activate`.
- Report the model as **DPA-v1 (Deadline-Pressure Actor v1)**; use `feature-only iter8 checkpoint` only as provenance.
- Freeze seeds `41,42,43,44,45,46,47,48,49,50`, one uninterrupted 4000-TTI rollout per seed, argmax decode, CTBS-v1, and Guard OFF.
- Full window is TTI `1--4000`; steady window is terminal/service TTI `1001--4000`, exactly 3000 TTI.
- Preserve checkpoint file SHA-256 `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`, actor-state SHA-256 `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`, and simulator SHA-256 `4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`.
- Preserve feature-source hashes `DeadlineServiceFeatures.h=4a353601a8378888d089e34d3f04225770ae757ac07d5bc4108533a1a384975a`, `OnlineObservationExtrasBuilder.h=ccf58c6c07e5ab65da373870d7d615b8048611508e4972910c4a4871744f876b`, `feature_snapshot.py=e86976c6208ff6e23e350d1480beaac011d694870d4a68e4e6fb9f7077aa7f7f`, and starvation-audit hash `22b3ec97f0ef5737f518d7cd0b9766b42eb53e314136ef1120e383ac40a9a775`.
- Preserve the v9c checkpoint actually used by the historical evaluator, `ppo_actor_best.pt`, SHA-256 `7e1243960798ba3be7287235538b318a42ad62004e5f5a115234499bfb3`; never substitute the differently hashed threshold checkpoint.
- Reuse historical PFQ/v9c artifacts. Do not run PFQ, v9c, training, checkpoint selection, reward scans, guard evaluation, or any scheduler/model change.
- Resolve runtime candidate width from each topology (`--independent-runtime-candidate-width 0`). The fixed 17 is the PRG decision count, not a safe UE-candidate padding width.
- A zero-service UE or a 400-TTI service gap is a valid negative model result and does not stop later seeds. Stop only for provenance, execution, trace, or structural-invalidity failures.
- Run seeds sequentially on one GPU. Do not run simulator instances concurrently.
- Preserve all unrelated dirty-worktree changes. Stage only files newly created by this plan; do not stage existing dirty files as part of a commit.

## File Structure

- Create `training/stageb_hgraph/dpa_v1_historical_comparison.py`: frozen provenance, dual-window DPA aggregation, historical PFQ/v9c loaders, scorecard classification, CLI, and report writers.
- Create `training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py`: focused synthetic tests for window boundaries, expiry-aware outcome metrics, v9c block filtering, path remapping, structural checks, and classification.
- Reuse without modification `training/stageb_hgraph/eval_checkpoint_online.py`: run the frozen actor and emit per-step/full/steady evaluator statistics.
- Reuse without modification `training/stageb_hgraph/seed45_starvation_audit.py`: generate full and steady per-UE service-health artifacts despite its historical filename.
- Reuse without modification `cuMAC/scripts/summarize_stageA_kpi.py`: generate each DPA-v1 full-run `kpi_summary.json` from simulator outputs.
- Create outputs under `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/`.
- Create `Doc/dpa_v1_seed41_50_historical_comparison_brief.md`: reporting-ready result summary.
- Modify `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`: add the completed 10-seed evidence and next research decision.
- Modify `Doc/README.md`: link the new brief.

---

### Task 1: Dual-Window DPA Metric Aggregator

**Files:**
- Create: `training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py`
- Create: `training/stageb_hgraph/dpa_v1_historical_comparison.py`

**Interfaces:**
- Consumes: evaluator summary mapping, service-balance rows, packet-outcome rows, and `WindowSpec`.
- Produces: `summarize_dpa_window` returning `dict[str, object]` with exact window identity, system metrics, per-UE goodput/service statistics, outcome-delay distribution, expiry-aware rates, and starvation evidence.

- [ ] **Step 1: Write the failing dual-window aggregation test**

  Add this focused test and its row helper:

  ```python
  import math

  from training.stageb_hgraph.dpa_v1_historical_comparison import (
      WindowSpec,
      summarize_dpa_window,
  )


  def _service(tti, ue_id, backlog, scheduled, successful, expired=0, prgs=0):
      return {
          "tti": tti,
          "ue_id": ue_id,
          "pre_queue_bytes": backlog,
          "scheduled_dequeue_bytes": scheduled,
          "successful_bytes": successful,
          "failed_bytes": scheduled - successful,
          "expired_bytes": expired,
          "actual_prg_count": prgs,
      }


  def test_steady_window_includes_expired_outcomes_and_detects_zero_service():
      rows = [
          _service(1, 0, 3000, 3000, 3000, prgs=1),
          _service(1, 1, 3000, 0, 0),
          _service(2, 0, 3000, 3000, 3000, prgs=1),
          _service(2, 1, 3000, 0, 0),
          _service(3, 0, 3000, 0, 0),
          _service(3, 1, 6000, 0, 0, expired=3000),
          _service(4, 0, 3000, 3000, 3000, prgs=1),
          _service(4, 1, 6000, 0, 0),
      ]
      outcomes = [
          {"terminal_tti": 2, "ue_id": 0, "outcome": "delivered", "packet_size_bytes": 3000, "system_time_ms": 1.0},
          {"terminal_tti": 3, "ue_id": 0, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
          {"terminal_tti": 3, "ue_id": 1, "outcome": "expired", "packet_size_bytes": 3000, "system_time_ms": 200.0},
      ]
      summary = {
          "mean_goodput_mbps": 100.0,
          "mean_tb_err_rate": 0.20,
          "mean_expiry_drop_rate": 0.10,
          "mean_prg_utilization_ratio": 0.80,
          "mean_blank_ratio": 0.20,
          "stats_mean_goodput_mbps": 123.0,
          "stats_mean_tb_err_rate": 0.15,
          "stats_mean_expiry_drop_rate": 0.08,
          "stats_mean_prg_utilization_ratio": 0.75,
          "stats_mean_blank_ratio": 0.25,
      }

      result = summarize_dpa_window(
          eval_summary=summary,
          service_rows=rows,
          outcome_rows=outcomes,
          window=WindowSpec("steady_3000", 2, 4),
          slot_duration_s=0.0005,
      )

      assert result["tti_count"] == 3
      assert result["goodput_mbps"] == 123.0
      assert result["expired_bytes"] == 3000
      assert result["outcome_count"] == 3
      assert math.isclose(result["outcome_delay_mean_ms"], 401.0 / 3.0)
      assert result["outcome_delay_p50_ms"] == 200.0
      assert result["outcome_delay_p95_ms"] == 200.0
      assert result["outcome_delay_max_ms"] == 200.0
      assert result["zero_scheduled_dequeue_ue_ids"] == [1]
      assert result["max_backlogged_zero_dequeue_streak_tti"] == 3
      assert result["ue_goodput_min_mbps"] == 0.0
      assert result["ue_goodput_mean_mbps"] == 16.0
      assert result["ue_goodput_jain"] == 0.5
      assert result["ue_expiry_aware_rate_min_mbps"] == 0.0
  ```

- [ ] **Step 2: Run the focused test and verify the import failure**

  Run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py::test_steady_window_includes_expired_outcomes_and_detects_zero_service \
    -q
  ```

  Expected: FAIL because `dpa_v1_historical_comparison` does not exist.

- [ ] **Step 3: Implement the minimal window and aggregation functions**

  Define these public interfaces in the new module:

  ```python
  from __future__ import annotations

  from collections import defaultdict
  from dataclasses import dataclass
  from typing import Mapping, Sequence

  import numpy as np


  def _pct(values: Sequence[float], percentile: float) -> float:
      return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


  def _jain(values: Sequence[float]) -> float:
      total = float(sum(values))
      squares = float(sum(value * value for value in values))
      return 0.0 if squares <= 0.0 else total * total / (len(values) * squares)


  @dataclass(frozen=True)
  class WindowSpec:
      label: str
      first_tti: int
      last_tti: int

      @property
      def tti_count(self) -> int:
          return self.last_tti - self.first_tti + 1


  def summarize_dpa_window(
      *,
      eval_summary: Mapping[str, object],
      service_rows: Sequence[Mapping[str, object]],
      outcome_rows: Sequence[Mapping[str, object]],
      window: WindowSpec,
      slot_duration_s: float = 0.0005,
  ) -> dict[str, object]:
      if window.first_tti <= 0 or window.last_tti < window.first_tti:
          raise ValueError("invalid TTI window")
      if window.label not in {"full_4000", "steady_3000"}:
          raise ValueError(f"unsupported window label: {window.label}")
      prefix = "" if window.label == "full_4000" else "stats_"
      selected_service = [
          row for row in service_rows
          if window.first_tti <= int(row["tti"]) <= window.last_tti
      ]
      selected_outcomes = [
          row for row in outcome_rows
          if window.first_tti <= int(row["terminal_tti"]) <= window.last_tti
      ]
      by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
      for row in selected_service:
          by_ue[int(row["ue_id"])].append(row)
      if not by_ue:
          raise ValueError("service window is empty")
      expected_ttis = set(range(window.first_tti, window.last_tti + 1))
      for ue_id, rows in by_ue.items():
          observed = {int(row["tti"]) for row in rows}
          if observed != expected_ttis:
              raise ValueError(f"incomplete service coverage for UE {ue_id}")

      duration_s = window.tti_count * slot_duration_s
      ue_goodput: list[float] = []
      scheduled_ratio: list[float] = []
      positive_dequeue_ratio: list[float] = []
      zero_service_ids: list[int] = []
      max_streak = 0
      expired_bytes = 0
      for ue_id in sorted(by_ue):
          rows = sorted(by_ue[ue_id], key=lambda row: int(row["tti"]))
          successful = sum(int(row["successful_bytes"]) for row in rows)
          scheduled = sum(int(row["scheduled_dequeue_bytes"]) for row in rows)
          expired_bytes += sum(int(row["expired_bytes"]) for row in rows)
          ue_goodput.append(successful * 8.0 / duration_s / 1.0e6)
          scheduled_ratio.append(
              sum(int(row["actual_prg_count"]) > 0 for row in rows) / window.tti_count
          )
          positive_dequeue_ratio.append(
              sum(int(row["scheduled_dequeue_bytes"]) > 0 for row in rows)
              / window.tti_count
          )
          had_backlog = any(int(row["pre_queue_bytes"]) > 0 for row in rows)
          if had_backlog and scheduled == 0:
              zero_service_ids.append(ue_id)
          current_streak = 0
          for row in rows:
              if int(row["pre_queue_bytes"]) > 0 and int(row["scheduled_dequeue_bytes"]) == 0:
                  current_streak += 1
                  max_streak = max(max_streak, current_streak)
              else:
                  current_streak = 0

      delays = [float(row["system_time_ms"]) for row in selected_outcomes]
      outcomes_by_ue: dict[int, list[Mapping[str, object]]] = defaultdict(list)
      for row in selected_outcomes:
          outcomes_by_ue[int(row["ue_id"])].append(row)
      ue_rates: list[float] = []
      for rows in outcomes_by_ue.values():
          outcome_time_ms = sum(float(row["system_time_ms"]) for row in rows)
          delivered_bytes = sum(
              int(row["packet_size_bytes"])
              for row in rows
              if str(row["outcome"]) == "delivered"
          )
          if outcome_time_ms > 0.0:
              ue_rates.append(delivered_bytes * 8.0 / (outcome_time_ms * 1000.0))

      return {
          "window": window.label,
          "first_tti": window.first_tti,
          "last_tti": window.last_tti,
          "tti_count": window.tti_count,
          "goodput_mbps": float(eval_summary[f"{prefix}mean_goodput_mbps"]),
          "bler": float(eval_summary[f"{prefix}mean_tb_err_rate"]),
          "expiry_drop_rate": float(eval_summary[f"{prefix}mean_expiry_drop_rate"]),
          "prg_utilization_ratio": float(eval_summary[f"{prefix}mean_prg_utilization_ratio"]),
          "blank_ratio": float(eval_summary[f"{prefix}mean_blank_ratio"]),
          "expired_bytes": expired_bytes,
          "outcome_count": len(delays),
          "outcome_delay_mean_ms": float(np.mean(delays)) if delays else 0.0,
          "outcome_delay_p50_ms": _pct(delays, 50),
          "outcome_delay_p90_ms": _pct(delays, 90),
          "outcome_delay_p95_ms": _pct(delays, 95),
          "outcome_delay_max_ms": max(delays, default=0.0),
          "ue_goodput_mean_mbps": float(np.mean(ue_goodput)),
          "ue_goodput_p5_mbps": _pct(ue_goodput, 5),
          "ue_goodput_p10_mbps": _pct(ue_goodput, 10),
          "ue_goodput_min_mbps": min(ue_goodput),
          "ue_goodput_jain": _jain(ue_goodput),
          "scheduled_ratio_mean": float(np.mean(scheduled_ratio)),
          "scheduled_ratio_p5": _pct(scheduled_ratio, 5),
          "scheduled_ratio_jain": _jain(scheduled_ratio),
          "positive_dequeue_ratio_mean": float(np.mean(positive_dequeue_ratio)),
          "ue_expiry_aware_rate_mean_mbps": float(np.mean(ue_rates)) if ue_rates else 0.0,
          "ue_expiry_aware_rate_p10_mbps": _pct(ue_rates, 10),
          "ue_expiry_aware_rate_min_mbps": min(ue_rates, default=0.0),
          "zero_scheduled_dequeue_ue_ids": zero_service_ids,
          "max_backlogged_zero_dequeue_streak_tti": max_streak,
          "metric_units": {
              "goodput_mbps": "Mbps",
              "bler": "ratio",
              "expiry_drop_rate": "ratio",
              "outcome_delay_mean_ms": "ms",
              "ue_expiry_aware_rate_mean_mbps": "Mbps",
          },
      }
  ```

  The implementation must:

  - reject `first_tti <= 0`, `last_tti < first_tti`, and incomplete UE/TTI coverage;
  - filter service rows by `tti` and outcome rows by `terminal_tti`, both inclusively;
  - use `mean_*` evaluator keys for `full_4000` and `stats_mean_*` keys for `steady_3000`;
  - compute UE goodput from `successful_bytes * 8 / (tti_count * slot_duration_s) / 1e6`;
  - compute scheduled ratio from TTIs where `actual_prg_count > 0` and positive-dequeue ratio from TTIs where `scheduled_dequeue_bytes > 0`;
  - compute p5/p10/median/min/max with `numpy.percentile` and Jain as `(sum(x) ** 2) / (len(x) * sum(x*x))`, returning zero when the denominator is zero;
  - include delivered and expired outcomes in delay percentiles;
  - compute each UE expiry-aware rate as delivered packet bytes times eight divided by total delivered-plus-expired outcome time; a UE with outcome time but no delivered bytes contributes zero;
  - compute zero-total scheduled-dequeue UEs only among UEs that had positive backlog;
  - compute maximum consecutive positive-backlog zero-dequeue streak per UE and globally;
  - return explicit metric units and source semantics alongside the numeric mapping.

- [ ] **Step 4: Run the focused test and the existing starvation tests**

  Run:

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_seed45_starvation_audit.py \
    training/stageb_hgraph/tests/test_service_balance_audit.py \
    -q
  ```

  Expected: PASS.

- [ ] **Step 5: Commit only the new experiment module and test**

  ```bash
  git add \
    training/stageb_hgraph/dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py
  git diff --cached --name-only
  git commit -m "test: add DPA-v1 dual-window comparison metrics"
  ```

  Expected staged set: exactly the two files above.

---

### Task 2: Historical Loaders, Provenance, and Scorecard CLI

**Files:**
- Modify: `training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py`
- Modify: `training/stageb_hgraph/dpa_v1_historical_comparison.py`

**Interfaces:**
- Consumes: recorded paths rooted at `/opt/nvidia/cuBB`, PFQ per-seed comparison JSON, v9c per-seed evaluator/KPI artifacts, v9c 500-TTI logs, DPA per-seed directories, and frozen hashes.
- Produces: `preflight`, `extract-seed`, and `aggregate` CLI subcommands plus normalized CSV/JSON and deterministic evidence classification.

- [ ] **Step 1: Add failing tests for legacy paths and v9c steady blocks**

  Add:

  ```python
  from pathlib import Path

  from training.stageb_hgraph.dpa_v1_historical_comparison import (
      classify_scorecard,
      parse_v9c_steady_blocks,
      resolve_recorded_path,
  )


  def test_resolve_recorded_container_path_under_supplied_legacy_root(tmp_path):
      target = tmp_path / "output" / "run" / "kpi_summary.json"
      target.parent.mkdir(parents=True)
      target.write_text("{}\n", encoding="utf-8")
      assert resolve_recorded_path(
          "/opt/nvidia/cuBB/output/run/kpi_summary.json",
          legacy_root=tmp_path,
      ) == target


  def test_v9c_steady_parser_drops_the_first_two_500_tti_blocks(tmp_path):
      log = tmp_path / "checkpoint_eval.log"
      log.write_text(
          "\n".join(
              f"[hgraph-eval] tti={tti:04d} goodput={1000 + tti / 10:.4f} "
              f"tb_err=0.1000 blank=0.2000 fill=0.8000 uniq_ue=30.00"
              for tti in range(500, 4001, 500)
          ) + "\n",
          encoding="utf-8",
      )
      result = parse_v9c_steady_blocks(log, first_tti=1001, last_tti=4000)
      assert result["block_count"] == 6
      assert result["source_block_end_ttis"] == [1500, 2000, 2500, 3000, 3500, 4000]
      assert result["goodput_mbps"] == 1275.0
      assert result["tb_err_rate"] == 0.1
      assert result["prg_utilization_ratio"] == 0.8


  def test_classification_prioritizes_starvation_over_mean_goodput_gain():
      report = {
          "structural_valid": True,
          "dpa_full_mean": {"goodput_mbps": 1600.0, "bler": 0.10, "expiry_drop_rate": 0.0},
          "pfq_full_mean": {"goodput_mbps": 1500.0, "bler": 0.11, "expiry_drop_rate": 0.001},
          "v9c_full_mean": {"goodput_mbps": 1490.0},
          "dpa_service_health": {"zero_service_seed_ues": [{"seed": 45, "ue_id": 4}], "ttl_streak_seed_ues": []},
      }
      assert classify_scorecard(report) == "cross_layout_service_failed_starvation"
  ```

- [ ] **Step 2: Run the three new tests and verify failures**

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    -q
  ```

  Expected: FAIL because the historical loader and classification interfaces are absent.

- [ ] **Step 3: Implement exact historical parsing and frozen preflight**

  Add constants for all hashes and paths frozen in the spec. Implement:

  ```python
  import re
  import statistics
  from pathlib import Path


  V9C_BLOCK_RE = re.compile(
      r"tti=(?P<tti>\d+)\s+goodput=(?P<goodput>[0-9.]+)\s+"
      r"tb_err=(?P<tb_err>[0-9.]+)\s+blank=(?P<blank>[0-9.]+)\s+"
      r"fill=(?P<fill>[0-9.]+).*?uniq_ue=(?P<uniq_ue>[0-9.]+)"
  )


  def resolve_recorded_path(recorded: str, *, legacy_root: Path) -> Path:
      source = Path(recorded)
      if source.is_file():
          return source
      prefix = Path("/opt/nvidia/cuBB")
      try:
          relative = source.relative_to(prefix)
      except ValueError as exc:
          raise ValueError(f"unsupported historical path: {recorded}") from exc
      resolved = legacy_root / relative
      if not resolved.is_file():
          raise FileNotFoundError(resolved)
      return resolved


  def parse_v9c_steady_blocks(
      path: Path,
      *,
      first_tti: int = 1001,
      last_tti: int = 4000,
  ) -> dict[str, object]:
      rows: list[dict[str, float | int]] = []
      for line in path.read_text(encoding="utf-8").splitlines():
          match = V9C_BLOCK_RE.search(line)
          if match is None:
              continue
          row = {
              "tti": int(match.group("tti")),
              "goodput": float(match.group("goodput")),
              "tb_err": float(match.group("tb_err")),
              "blank": float(match.group("blank")),
              "fill": float(match.group("fill")),
              "uniq_ue": float(match.group("uniq_ue")),
          }
          if int(row["tti"]) - 499 >= first_tti and int(row["tti"]) <= last_tti:
              rows.append(row)
      expected_ends = [
          tti for tti in range(500, last_tti + 1, 500)
          if tti - 499 >= first_tti
      ]
      observed_ends = [int(row["tti"]) for row in rows]
      if observed_ends != expected_ends:
          raise ValueError(
              f"incomplete v9c steady blocks: observed={observed_ends} expected={expected_ends}"
          )
      return {
          "block_count": len(rows),
          "source_block_end_ttis": observed_ends,
          "goodput_mbps": float(statistics.fmean(float(row["goodput"]) for row in rows)),
          "tb_err_rate": float(statistics.fmean(float(row["tb_err"]) for row in rows)),
          "blank_ratio": float(statistics.fmean(float(row["blank"]) for row in rows)),
          "prg_utilization_ratio": float(statistics.fmean(float(row["fill"]) for row in rows)),
          "unique_ue_mean": float(statistics.fmean(float(row["uniq_ue"]) for row in rows)),
      }
  ```

  `parse_v9c_steady_blocks` must accept only complete 500-TTI blocks whose end
  TTI is at least `first_tti + 499` and at most `last_tti`. Require exactly the
  ends `[1500, 2000, 2500, 3000, 3500, 4000]`, then take the arithmetic mean
  because every block has equal length.

  Preflight must:

  - SHA-256 the checkpoint, simulator, v9c checkpoint, v9c aggregate manifest,
    v9c comparison JSON, PFQ aggregate manifest, and PFQ mean JSON;
  - SHA-256 the three dimensionless-feature sources and the frozen starvation
    audit, requiring the values listed under Global Constraints;
  - load the DPA checkpoint on CPU and hash actor tensors with
    `training.stageb_hgraph.independent_contract.state_dict_sha256`;
  - reject any mismatch before creating a seed directory;
  - write `campaign_manifest.json` with seeds, windows, model name, Guard OFF,
    all observed hashes, and the historical-comparison limitation.

- [ ] **Step 4: Implement seed extraction and aggregate report writers**

  The CLI must expose these exact forms:

  ```text
  python -m training.stageb_hgraph.dpa_v1_historical_comparison preflight --help
  python -m training.stageb_hgraph.dpa_v1_historical_comparison extract-seed --help
  python -m training.stageb_hgraph.dpa_v1_historical_comparison aggregate --help
  ```

  `extract-seed` must validate `completed_steps=4000`,
  `stats_warmup_steps=1000`, `stats_count=3000`, Guard OFF, checkpoint/binary/
  state hashes, argmax, CTBS-v1, no candidate truncation, no illegal action,
  no replay mismatch, no mask reactivation, and zero CTBS MCS/layer/TBS
  mismatches. It then writes `seed_metrics.json` containing both windows and
  returns zero even when the model-level starvation criteria fail.

  `aggregate` must write:

  ```text
  per_seed_metrics.csv
  per_seed_metrics.json
  aggregate_metrics.csv
  aggregate_metrics.json
  historical_comparability.json
  final_report.json
  ```

  Use normalized long rows with fields:

  ```text
  arm,seed,window,metric,value,unit,direction,source_artifact,comparison_scope
  ```

  PFQ full values come from each `rrq_vs_pfq_compare.json` entry's
  `rr_vs_pf_metrics[].pf`. v9c full values come from each source run's
  `kpi_summary.json` plus `eval_summary.json`. v9c steady values come only from
  the six preserved 500-TTI blocks. Missing historical fields are JSON `null`
  and CSV empty cells, never zero.

  Implement classification in this priority order:

  1. `invalid_experiment` when structural validity fails;
  2. `cross_layout_service_failed_starvation` when any DPA seed/UE has zero
     total scheduled dequeue while backlogged or a streak of at least 400 TTI;
  3. `cross_layout_service_improved_with_no_starvation` when DPA full mean
     goodput is at least both historical means, DPA mean BLER is no greater
     than PFQ, and DPA mean expiry is no greater than PFQ;
  4. `aggregate_efficiency_improved_but_expiry_tail_failed` when DPA full mean
     goodput is at least both historical means but expiry exceeds PFQ;
  5. `no_starvation_but_aggregate_efficiency_not_improved` otherwise.

  Set `strict_paired_superiority_claim_allowed=false` in every final report.

- [ ] **Step 5: Run all focused tests and validate the CLI help**

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_seed45_starvation_audit.py \
    training/stageb_hgraph/tests/test_service_balance_audit.py \
    -q
  python -m training.stageb_hgraph.dpa_v1_historical_comparison --help
  python -m training.stageb_hgraph.dpa_v1_historical_comparison preflight --help
  python -m training.stageb_hgraph.dpa_v1_historical_comparison extract-seed --help
  python -m training.stageb_hgraph.dpa_v1_historical_comparison aggregate --help
  ```

  Expected: all tests PASS and every help command exits zero.

- [ ] **Step 6: Commit the historical ingestion and report assembly delta**

  ```bash
  git add \
    training/stageb_hgraph/dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py
  git diff --cached --name-only
  git commit -m "feat: add frozen DPA-v1 historical scorecard"
  ```

  Expected staged set: exactly the two files above.

---

### Task 3: Frozen Manifest and Seed41 Measurement Smoke Test

**Files:**
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/campaign_manifest.json`
- Create outputs under: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/seed41/actor/`

**Interfaces:**
- Consumes: frozen DPA-v1 checkpoint, simulator, current source tree, and historical roots.
- Produces: the first full/steady DPA-v1 result and proof that the metric collection path is complete before spending the remaining nine rollouts.

- [ ] **Step 1: Enter the required execution environment**

  From the host repository root run:

  ```bash
  ./cuPHY-CP/container/attach_aerial.sh
  ```

  Then inside the container:

  ```bash
  cd /opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
  source .venv/bin/activate
  ```

- [ ] **Step 2: Run frozen preflight and create the campaign manifest**

  ```bash
  WORKTREE=/opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
  LEGACY_ROOT=/opt/nvidia/cuBB
  CAMPAIGN_ROOT="$WORKTREE/output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1"
  CHECKPOINT="$WORKTREE/output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt"
  SIM_BIN="$WORKTREE/build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
  V9C_ROOT="$LEGACY_ROOT/training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0"
  PFQ_ROOT="$LEGACY_ROOT/output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50"

  test ! -e "$CAMPAIGN_ROOT"
  mkdir -p "$CAMPAIGN_ROOT"
  python -m training.stageb_hgraph.dpa_v1_historical_comparison preflight \
    --worktree-root "$WORKTREE" \
    --legacy-root "$LEGACY_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --sim-bin "$SIM_BIN" \
    --v9c-root "$V9C_ROOT" \
    --pfq-root "$PFQ_ROOT" \
    --out "$CAMPAIGN_ROOT/campaign_manifest.json"
  python -m json.tool "$CAMPAIGN_ROOT/campaign_manifest.json" >/dev/null
  ```

  Expected: every frozen hash matches and the manifest lists exactly seeds
  41--50, full `1--4000`, steady `1001--4000`, and Guard OFF.

- [ ] **Step 3: Launch the single seed41 actor rollout**

  ```bash
  SEED=41
  ACTOR_DIR="$CAMPAIGN_ROOT/seed${SEED}/actor"
  SOCKET="/tmp/cumac_dpa_v1_seed${SEED}_4000.sock"
  test ! -e "$ACTOR_DIR"
  mkdir -p "$ACTOR_DIR"

  (
    cd "$ACTOR_DIR"
    env \
      CUMAC_ONLINE_BRIDGE=1 \
      CUMAC_ONLINE_SOCKET="$SOCKET" \
      CUMAC_ONLINE_PERSISTENT=1 \
      CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 \
      CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
      CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
      CUMAC_SERVICE_BALANCE_TRACE_CSV="$ACTOR_DIR/service_balance_trace.csv" \
      CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=1 \
      CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="$ACTOR_DIR/traffic_offered_packets.csv" \
      CUMAC_CELL_ASSOC_MODE=strongest \
      CUMAC_UE_PLACEMENT_MODE=coop_boundary \
      CUMAC_UE_VORONOI_CLIP=1 \
      CUMAC_TOPOLOGY_DUMP=1 \
      CUMAC_BS_TX_PATTERN=omni \
      CUMAC_TOPOLOGY_SEED="$SEED" \
      CUMAC_TRAFFIC_SEED="$SEED" \
      CUMAC_TRAFFIC_ARRIVAL_RATE=1.0 \
      CUMAC_PACKET_TTL_TTI=0 \
      CUMAC_PACKET_TTL_MS=200 \
      CUMAC_COMPACT_TTI_LOG=1 \
      CUMAC_PROGRESS_TTI_INTERVAL=0 \
      CUMAC_COMPARE_TTI_INTERVAL=0 \
      CUMAC_PACKET_DELAY_SAMPLES=1 \
      CUMAC_PACKET_DELAY_SAMPLES_CSV="$ACTOR_DIR/packet_delay_samples.csv" \
      CUMAC_PACKET_OUTCOME_SAMPLES_CSV="$ACTOR_DIR/packet_outcome_samples.csv" \
      CUMAC_EXEC_MODE=both \
      "$SIM_BIN" -d 1 -b 2 -x 0 -f 0 -g 100 -r 3000
  ) > "$ACTOR_DIR/run.log" 2>&1 &
  SIM_PID=$!

  set +e
  python -m training.stageb_hgraph.eval_checkpoint_online \
    --checkpoint "$CHECKPOINT" \
    --socket-path "$SOCKET" \
    --episode-horizon 4000 \
    --max-steps 4000 \
    --stats-warmup-steps 1000 \
    --decode-mode argmax \
    --seed "$SEED" \
    --sample-seed "$SEED" \
    --device auto \
    --print-every-steps 500 \
    --save-steps 1 \
    --sim-bin "$SIM_BIN" \
    --outcome-samples "$ACTOR_DIR/packet_outcome_samples.csv" \
    --out-dir "$ACTOR_DIR" \
    --ue-prg-topk 4 \
    --ue-prg-diversity-extra 0 \
    --independent-policy-diagnostics 1 \
    --independent-runtime-candidate-width 0 \
    --independent-headroom-accounting ctbs-v1 \
    --independent-non-starvation-guard off \
    --slot-repair-max-per-cell 0 \
    --safe-fill-max-per-cell 0 \
    --candidate-blank-budget-max-per-cell -1 \
    --candidate-decode-demand-slack-bytes 3000 \
    --candidate-marginal-headroom-slack-bytes -1 \
    --candidate-headroom-rank-mode stable \
    --candidate-budgeted-select-mode off \
    > "$ACTOR_DIR/eval.log" 2>&1
  EVAL_RC=$?
  wait "$SIM_PID"
  SIM_RC=$?
  set -e
  printf 'eval_rc=%s\nsim_rc=%s\n' "$EVAL_RC" "$SIM_RC" > "$ACTOR_DIR/exit_codes.txt"
  test "$EVAL_RC" -eq 0
  test "$SIM_RC" -eq 0
  ```

  Expected: simulator and evaluator exit zero, exactly 4000 evaluator step rows,
  and service trace coverage from TTI 1 through 4000 for all 36 UEs.

- [ ] **Step 4: Generate full KPI and both service-health windows**

  ```bash
  python cuMAC/scripts/summarize_stageA_kpi.py \
    --output-dir "$ACTOR_DIR" \
    --slot-duration-ms 0.5 \
    --traffic-percent 100 \
    --packet-size-bytes 3000 \
    --traffic-arrival-rate 1.0 \
    --prbs-per-group 16 \
    --prg-count 17

  set +e
  python -m training.stageb_hgraph.seed45_starvation_audit \
    --seed "$SEED" \
    --service-trace "$ACTOR_DIR/service_balance_trace.csv" \
    --measured-first-tti 1 \
    --measured-last-tti 4000 \
    --expected-ue-count 36 \
    --streak-threshold-tti 400 \
    --eval-summary "$ACTOR_DIR/eval_summary.json" \
    --outcome-samples "$ACTOR_DIR/packet_outcome_samples.csv" \
    --out "$ACTOR_DIR/starvation_full_4000.json"
  FULL_STARVATION_RC=$?
  python -m training.stageb_hgraph.seed45_starvation_audit \
    --seed "$SEED" \
    --service-trace "$ACTOR_DIR/service_balance_trace.csv" \
    --measured-first-tti 1001 \
    --measured-last-tti 4000 \
    --expected-ue-count 36 \
    --streak-threshold-tti 400 \
    --eval-summary "$ACTOR_DIR/eval_summary.json" \
    --outcome-samples "$ACTOR_DIR/packet_outcome_samples.csv" \
    --out "$ACTOR_DIR/starvation_steady_3000.json"
  STEADY_STARVATION_RC=$?
  set -e
  test "$FULL_STARVATION_RC" -eq 0 -o "$FULL_STARVATION_RC" -eq 1
  test "$STEADY_STARVATION_RC" -eq 0 -o "$STEADY_STARVATION_RC" -eq 1
  ```

  A return code of one from either starvation audit is retained as a valid
  model result.

- [ ] **Step 5: Extract seed41 metrics and validate the measurement path**

  ```bash
  python -m training.stageb_hgraph.dpa_v1_historical_comparison extract-seed \
    --campaign-manifest "$CAMPAIGN_ROOT/campaign_manifest.json" \
    --seed "$SEED" \
    --actor-dir "$ACTOR_DIR" \
    --out "$CAMPAIGN_ROOT/seed${SEED}/seed_metrics.json"
  python -m json.tool "$CAMPAIGN_ROOT/seed${SEED}/seed_metrics.json" >/dev/null
  test "$(wc -l < "$ACTOR_DIR/eval_steps.jsonl")" -eq 4000
  test "$(($(wc -l < "$ACTOR_DIR/service_balance_trace.csv") - 1))" -eq 144000
  ```

  Inspect the JSON and require structural validity, full count 4000, steady
  count 3000, outcome-delay metrics, expiry-aware per-UE rate metrics, and a
  per-UE service table. Only then proceed to Task 4.

---

### Task 4: Sequential Seed42--50 DPA-v1 Rollouts

**Files:**
- Create outputs under: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/seed42/` through `seed50/`

**Interfaces:**
- Consumes: the unchanged manifest, checkpoint, simulator, and seed41-validated command.
- Produces: nine additional structurally valid `seed_metrics.json` files without adapting the actor between seeds.

- [ ] **Step 1: Define and run the frozen per-seed command sequentially**

  In the same activated container shell, run this complete function and loop:

  ```bash
  run_dpa_seed() {
    local seed="$1"
    local actor_dir="$CAMPAIGN_ROOT/seed${seed}/actor"
    local socket="/tmp/cumac_dpa_v1_seed${seed}_4000.sock"

    python -m training.stageb_hgraph.dpa_v1_historical_comparison preflight \
      --verify-only \
      --worktree-root "$WORKTREE" \
      --legacy-root "$LEGACY_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --sim-bin "$SIM_BIN" \
      --v9c-root "$V9C_ROOT" \
      --pfq-root "$PFQ_ROOT" \
      --out "$CAMPAIGN_ROOT/campaign_manifest.json"

    test ! -e "$actor_dir"
    mkdir -p "$actor_dir"
    (
      cd "$actor_dir"
      env \
        CUMAC_ONLINE_BRIDGE=1 \
        CUMAC_ONLINE_SOCKET="$socket" \
        CUMAC_ONLINE_PERSISTENT=1 \
        CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1 \
        CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1 \
        CUMAC_ONLINE_EXPORT_PHY_TBS_AUDIT=1 \
        CUMAC_SERVICE_BALANCE_TRACE_CSV="$actor_dir/service_balance_trace.csv" \
        CUMAC_SERVICE_BALANCE_TRACE_FIRST_TTI=1 \
        CUMAC_TRAFFIC_OFFERED_SAMPLES_CSV="$actor_dir/traffic_offered_packets.csv" \
        CUMAC_CELL_ASSOC_MODE=strongest \
        CUMAC_UE_PLACEMENT_MODE=coop_boundary \
        CUMAC_UE_VORONOI_CLIP=1 \
        CUMAC_TOPOLOGY_DUMP=1 \
        CUMAC_BS_TX_PATTERN=omni \
        CUMAC_TOPOLOGY_SEED="$seed" \
        CUMAC_TRAFFIC_SEED="$seed" \
        CUMAC_TRAFFIC_ARRIVAL_RATE=1.0 \
        CUMAC_PACKET_TTL_TTI=0 \
        CUMAC_PACKET_TTL_MS=200 \
        CUMAC_COMPACT_TTI_LOG=1 \
        CUMAC_PROGRESS_TTI_INTERVAL=0 \
        CUMAC_COMPARE_TTI_INTERVAL=0 \
        CUMAC_PACKET_DELAY_SAMPLES=1 \
        CUMAC_PACKET_DELAY_SAMPLES_CSV="$actor_dir/packet_delay_samples.csv" \
        CUMAC_PACKET_OUTCOME_SAMPLES_CSV="$actor_dir/packet_outcome_samples.csv" \
        CUMAC_EXEC_MODE=both \
        "$SIM_BIN" -d 1 -b 2 -x 0 -f 0 -g 100 -r 3000
    ) > "$actor_dir/run.log" 2>&1 &
    local sim_pid=$!

    set +e
    python -m training.stageb_hgraph.eval_checkpoint_online \
      --checkpoint "$CHECKPOINT" \
      --socket-path "$socket" \
      --episode-horizon 4000 \
      --max-steps 4000 \
      --stats-warmup-steps 1000 \
      --decode-mode argmax \
      --seed "$seed" \
      --sample-seed "$seed" \
      --device auto \
      --print-every-steps 500 \
      --save-steps 1 \
      --sim-bin "$SIM_BIN" \
      --outcome-samples "$actor_dir/packet_outcome_samples.csv" \
      --out-dir "$actor_dir" \
      --ue-prg-topk 4 \
      --ue-prg-diversity-extra 0 \
      --independent-policy-diagnostics 1 \
      --independent-runtime-candidate-width 0 \
      --independent-headroom-accounting ctbs-v1 \
      --independent-non-starvation-guard off \
      --slot-repair-max-per-cell 0 \
      --safe-fill-max-per-cell 0 \
      --candidate-blank-budget-max-per-cell -1 \
      --candidate-decode-demand-slack-bytes 3000 \
      --candidate-marginal-headroom-slack-bytes -1 \
      --candidate-headroom-rank-mode stable \
      --candidate-budgeted-select-mode off \
      > "$actor_dir/eval.log" 2>&1
    local eval_rc=$?
    wait "$sim_pid"
    local sim_rc=$?
    set -e
    printf 'eval_rc=%s\nsim_rc=%s\n' "$eval_rc" "$sim_rc" > "$actor_dir/exit_codes.txt"
    test "$eval_rc" -eq 0
    test "$sim_rc" -eq 0

    python cuMAC/scripts/summarize_stageA_kpi.py \
      --output-dir "$actor_dir" \
      --slot-duration-ms 0.5 \
      --traffic-percent 100 \
      --packet-size-bytes 3000 \
      --traffic-arrival-rate 1.0 \
      --prbs-per-group 16 \
      --prg-count 17

    set +e
    python -m training.stageb_hgraph.seed45_starvation_audit \
      --seed "$seed" \
      --service-trace "$actor_dir/service_balance_trace.csv" \
      --measured-first-tti 1 \
      --measured-last-tti 4000 \
      --expected-ue-count 36 \
      --streak-threshold-tti 400 \
      --eval-summary "$actor_dir/eval_summary.json" \
      --outcome-samples "$actor_dir/packet_outcome_samples.csv" \
      --out "$actor_dir/starvation_full_4000.json"
    local full_starvation_rc=$?
    python -m training.stageb_hgraph.seed45_starvation_audit \
      --seed "$seed" \
      --service-trace "$actor_dir/service_balance_trace.csv" \
      --measured-first-tti 1001 \
      --measured-last-tti 4000 \
      --expected-ue-count 36 \
      --streak-threshold-tti 400 \
      --eval-summary "$actor_dir/eval_summary.json" \
      --outcome-samples "$actor_dir/packet_outcome_samples.csv" \
      --out "$actor_dir/starvation_steady_3000.json"
    local steady_starvation_rc=$?
    set -e
    test "$full_starvation_rc" -eq 0 -o "$full_starvation_rc" -eq 1
    test "$steady_starvation_rc" -eq 0 -o "$steady_starvation_rc" -eq 1

    python -m training.stageb_hgraph.dpa_v1_historical_comparison extract-seed \
      --campaign-manifest "$CAMPAIGN_ROOT/campaign_manifest.json" \
      --seed "$seed" \
      --actor-dir "$actor_dir" \
      --out "$CAMPAIGN_ROOT/seed${seed}/seed_metrics.json"
    test "$(wc -l < "$actor_dir/eval_steps.jsonl")" -eq 4000
    test "$(($(wc -l < "$actor_dir/service_balance_trace.csv") - 1))" -eq 144000
  }

  for seed in 42 43 44 45 46 47 48 49 50; do
    run_dpa_seed "$seed"
  done
  ```

  Expected: the loop stops only on provenance/execution/trace/structural error.
  Starvation findings remain in `seed_metrics.json` and do not interrupt later
  seeds.

- [ ] **Step 2: Verify that the frozen cohort is complete**

  ```bash
  for seed in 41 42 43 44 45 46 47 48 49 50; do
    test -f "$CAMPAIGN_ROOT/seed${seed}/seed_metrics.json"
    python -m json.tool "$CAMPAIGN_ROOT/seed${seed}/seed_metrics.json" >/dev/null
  done
  test "$(find "$CAMPAIGN_ROOT" -mindepth 2 -maxdepth 2 -name seed_metrics.json | wc -l)" -eq 10
  ```

  Expected: ten structurally valid seed artifacts. Model-level failures are
  present as data rather than missing runs.

---

### Task 5: Historical Comparison and Final Scorecard

**Files:**
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/per_seed_metrics.csv`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/per_seed_metrics.json`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/aggregate_metrics.csv`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/aggregate_metrics.json`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/historical_comparability.json`
- Create: `output/pfq_independent_reuse/formal/dpa_v1_seed41_50_historical_comparison_v1/final_report.json`

**Interfaces:**
- Consumes: ten immutable DPA seed metrics plus the frozen PFQ/v9c artifacts.
- Produces: full-4000 and steady-3000 comparison tables, aggregate distributions, win counts, stress-seed callouts, worst seed, service-health result, and one descriptive classification.

- [ ] **Step 1: Run the frozen aggregate command**

  ```bash
  python -m training.stageb_hgraph.dpa_v1_historical_comparison aggregate \
    --campaign-manifest "$CAMPAIGN_ROOT/campaign_manifest.json" \
    --campaign-root "$CAMPAIGN_ROOT" \
    --legacy-root "$LEGACY_ROOT" \
    --v9c-root "$V9C_ROOT" \
    --pfq-root "$PFQ_ROOT" \
    --out-root "$CAMPAIGN_ROOT"
  ```

  Expected: command exits zero only when all ten seed artifacts and historical
  hashes match.

- [ ] **Step 2: Validate every report artifact and comparison limitation**

  ```bash
  for name in \
    per_seed_metrics.json \
    aggregate_metrics.json \
    historical_comparability.json \
    final_report.json; do
    python -m json.tool "$CAMPAIGN_ROOT/$name" >/dev/null
  done
  test -s "$CAMPAIGN_ROOT/per_seed_metrics.csv"
  test -s "$CAMPAIGN_ROOT/aggregate_metrics.csv"
  rg -n 'strict_paired_superiority_claim_allowed.*false' \
    "$CAMPAIGN_ROOT/final_report.json" \
    "$CAMPAIGN_ROOT/historical_comparability.json"
  ```

  Confirm in `final_report.json`:

  - DPA has ten full and ten steady rows;
  - PFQ has ten full rows and no fabricated steady rows;
  - v9c has ten full rows and six-block steady summaries for each seed;
  - headline tables include goodput, BLER, expiry, PRG utilization, blank/fill,
    outcome or historical packet delay, per-UE tail rate, Jain, and service
    health where available;
  - mean/std/median/min/max and worst-seed identity are present;
  - seeds 43, 45, and 48 are called out explicitly;
  - missing historical outcome metrics are null, not zero;
  - DPA zero-service and TTL-streak findings affect classification before
    aggregate goodput.

- [ ] **Step 3: Run a fresh consistency check against source summaries**

  Select seeds 41, 45, and 48 and manually verify that:

  ```text
  DPA full goodput == eval_summary.mean_goodput_mbps
  DPA steady goodput == eval_summary.stats_mean_goodput_mbps
  v9c full goodput == historical eval_summary.mean_goodput_mbps
  PFQ full goodput == per-seed rr_vs_pfq_compare.json PF value
  v9c steady goodput == arithmetic mean of log blocks ending 1500--4000
  ```

  Record these checks in `final_report.json` under
  `spot_check_provenance` with source paths and exact values.

---

### Task 6: Reporting Brief and Mainline Research Record

**Files:**
- Create: `Doc/dpa_v1_seed41_50_historical_comparison_brief.md`
- Modify: `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`
- Modify: `Doc/README.md`

**Interfaces:**
- Consumes: `final_report.json`, per-seed table, aggregate table, and comparability statement.
- Produces: a concise presentation-ready report plus a durable project-history entry.

- [ ] **Step 1: Write the result brief from frozen artifacts**

  The brief must contain these sections in this order:

  ```text
  1. Executive conclusion and final classification
  2. Frozen DPA-v1 identity and evaluation windows
  3. 10-seed aggregate DPA-v1 / PFQ / v9c table
  4. Seed41--50 headline table
  5. Outcome delay and expiry-aware per-UE rate
  6. Goodput, BLER, expiry, and PRG/blank efficiency
  7. Fairness, worst UE, and starvation evidence
  8. Historical stress seeds 43/45/48
  9. Comparability limits
  10. Evidence-supported next research decision
  ```

  State explicitly that low PRG utilization is beneficial only when business
  service metrics remain acceptable. Keep outcome delay including expired
  packets separate from historical served-packet and backlog-estimated delay.

- [ ] **Step 2: Update the project experiment report and document index**

  Add one dated section to
  `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md` containing
  the final classification, the aggregate headline table, worst seed/UE,
  stress-seed interpretation, and next research question. Add one link and a
  one-sentence description to `Doc/README.md`.

  Do not rewrite earlier v9c, iter21, seed45, seed48/51, or guard conclusions.
  The new section appends evidence; it does not retroactively change historical
  metric definitions.

- [ ] **Step 3: Run final verification**

  ```bash
  python -m pytest \
    training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py \
    training/stageb_hgraph/tests/test_seed45_starvation_audit.py \
    training/stageb_hgraph/tests/test_service_balance_audit.py \
    -q
  python -m json.tool "$CAMPAIGN_ROOT/final_report.json" >/dev/null
  python -m json.tool "$CAMPAIGN_ROOT/aggregate_metrics.json" >/dev/null
  git diff --check
  rg -n "DPA-v1|seed41--50|outcome delay|expiry-aware|historical" \
    Doc/dpa_v1_seed41_50_historical_comparison_brief.md \
    Doc/stageB_boundary_intelligent_algorithm_experiment_report.md \
    Doc/README.md
  ```

  Expected: tests PASS, JSON parses, no whitespace errors, and all three docs
  identify DPA-v1 consistently.

- [ ] **Step 4: Commit only the newly created brief**

  Because both existing project documents already contain unrelated dirty-tree
  changes, leave their new hunks unstaged for later integration. Commit only
  the new brief:

  ```bash
  git add Doc/dpa_v1_seed41_50_historical_comparison_brief.md
  git diff --cached --name-only
  git commit -m "docs: report DPA-v1 10-seed historical comparison"
  ```

  Expected staged set: exactly the new brief. Report the two intentionally
  unstaged mainline-document paths in the handoff.

---

## Final Verification Checklist

- [ ] Checkpoint, actor-state, simulator, historical v9c, and historical PFQ hashes match the frozen protocol.
- [ ] DPA-v1 is named consistently; `iter8` appears only in provenance.
- [ ] Exactly one 4000-TTI DPA rollout exists for each seed 41--50.
- [ ] Every seed has full `1--4000` and steady `1001--4000` evidence from the same uninterrupted trajectory.
- [ ] Every `eval_summary.json` has `completed_steps=4000`, `stats_warmup_steps=1000`, and `stats_count=3000`.
- [ ] Guard is OFF, argmax and CTBS-v1 are active, candidate truncation is zero, and repair/fill/budget interventions remain disabled.
- [ ] Illegal action, replay mismatch, mask reactivation, and CTBS MCS/layer/TBS mismatch counts are zero.
- [ ] Outcome delay includes delivered and expired outcomes and is not conflated with served-packet delay.
- [ ] Expiry-aware per-UE rate, per-UE goodput tail, scheduled-dequeue service, zero-service UEs, and maximum service gaps are reported.
- [ ] PFQ has no fabricated steady window; v9c steady metrics are limited to fields recoverable from preserved equal-length blocks.
- [ ] Full and steady horizons are never mixed in a paired win count.
- [ ] Seeds 43, 45, and 48 plus the worst DPA seed are explicit.
- [ ] The report states `strict_paired_superiority_claim_allowed=false`.
- [ ] Model failures do not invalidate traces; infrastructure or provenance failures do.
- [ ] No training, tuning, new checkpoint selection, guard run, PFQ rerun, or v9c rerun occurred.
- [ ] Execution stops after the trustworthy 10-seed comparison and documentation.

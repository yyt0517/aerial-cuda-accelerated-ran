# Stage-B Baseline Simulation Verification Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a project-acceptance simulation verification report, reproducible figures, and validation tests for the RR, PF, RRQ, and PFQ Stage-B scheduling baselines.

**Architecture:** Treat the existing full-KPI suite CSV files as immutable sources of truth. A small Python plotting module validates the wide-form aggregate schema and emits report-only figures; the Markdown report cites those figures, existing packet-delay plots, source data, and executable reproduction commands without merging independent experiment batches.

**Tech Stack:** Python 3, standard-library `csv`/`argparse`/`unittest`, NumPy, Matplotlib with the `Agg` backend, Markdown, existing Bash experiment wrappers.

## Global Constraints

- Cover RR, PF, RRQ, and PFQ; define Q as this project's queue-aware implementation.
- Use `output/stageB_baseline_eval_suite_fullkpi_seed41_50/` as the sole source for the four-algorithm main comparison.
- Use the full-KPI boundary RRQ/PFQ leg and the active boundary gate as separately labeled supporting batches; never splice their values into one result table.
- Do not claim that these baselines directly optimize transmit power or perform joint cross-cell actions.
- Do not modify existing algorithm implementations or experiment outputs.
- Preserve all pre-existing dirty-worktree changes, especially the current edits in `Doc/README.md`.
- Record missing run-time Git hash, dirty patch, container digest, compiler options, and unified configuration hash as provenance limitations.

---

### Task 1: Add metric validation and report plotting tests

**Files:**
- Create: `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`
- Test: `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`

**Interfaces:**
- Consumes: the future module at `cuMAC/scripts/plot_stageB_baseline_verification_report.py` loaded with `importlib.util.spec_from_file_location`.
- Produces: executable behavioral requirements for `load_metric_table(path, expected_baselines, expected_seed_count)`, `beneficial_relative_change(old, new, direction)`, and `generate_figures(metrics, output_dir, dpi)`.

- [ ] **Step 1: Write tests for valid input, missing algorithms, non-finite values, seed mismatch, improvement direction, and generated artifacts**

```python
from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "cuMAC/scripts/plot_stageB_baseline_verification_report.py"


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Expected plotting module to exist: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("stageb_report_plot", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PlotStageBBaselineVerificationReportTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_fixture(self, *, pfq_mean="4.0", common_seed_count="10"):
        path = self.root / "metrics.csv"
        fields = ["scenario", "metric", "unit", "direction", "common_seed_count", "common_seeds"]
        for baseline in ("rr", "pf", "rrq", "pfq"):
            fields += [f"{baseline}_mean", f"{baseline}_std", f"{baseline}_seed_count", f"{baseline}_seeds"]
        metrics = [
            ("traffic.goodput_mbps", "Mbps", "higher_better", (1, 2, 3, pfq_mean)),
            ("traffic.packet_delay_p95_ms", "ms", "lower_better", (40, 80, 30, 20)),
            ("traffic.expiry_drop_rate", "%", "lower_better", (0.01, 0.30, 0.02, 0.001)),
            ("global_kpi.ue_goodput_jain", "", "higher_better", (0.9, 0.8, 0.95, 0.99)),
            ("global_kpi.global_tb_bler", "%", "lower_better", (0.10, 0.12, 0.09, 0.11)),
            ("global_kpi.prg_utilization_ratio", "%", "higher_better", (1.00, 1.00, 0.90, 0.95)),
            ("global_kpi.ue_goodput_p5_mbps", "Mbps", "higher_better", (20, 5, 25, 30)),
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for metric, unit, direction, values in metrics:
                row = {
                    "scenario": "RAYLEIGH",
                    "metric": metric,
                    "unit": unit,
                    "direction": direction,
                    "common_seed_count": common_seed_count,
                    "common_seeds": "41|42|43|44|45|46|47|48|49|50",
                }
                for baseline, value in zip(("rr", "pf", "rrq", "pfq"), values):
                    row[f"{baseline}_mean"] = value
                    row[f"{baseline}_std"] = "0.1"
                    row[f"{baseline}_seed_count"] = common_seed_count
                    row[f"{baseline}_seeds"] = row["common_seeds"]
                writer.writerow(row)
        return path

    def test_loads_complete_four_baseline_table(self):
        table = self.module.load_metric_table(self.write_fixture(), ("rr", "pf", "rrq", "pfq"), 10)
        self.assertEqual(table["traffic.goodput_mbps"]["pfq"]["mean"], 4.0)
        self.assertEqual(table["traffic.expiry_drop_rate"]["pf"]["mean"], 30.0)

    def test_rejects_missing_algorithm_columns(self):
        path = self.write_fixture()
        text = path.read_text(encoding="utf-8").replace(",pfq_mean", ",missing_pfq_mean", 1)
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pfq_mean"):
            self.module.load_metric_table(path, ("rr", "pf", "rrq", "pfq"), 10)

    def test_rejects_non_finite_metric(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.module.load_metric_table(self.write_fixture(pfq_mean="nan"), ("rr", "pf", "rrq", "pfq"), 10)

    def test_rejects_seed_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "10 seeds"):
            self.module.load_metric_table(self.write_fixture(common_seed_count="9"), ("rr", "pf", "rrq", "pfq"), 10)

    def test_beneficial_change_respects_metric_direction(self):
        self.assertAlmostEqual(self.module.beneficial_relative_change(100, 110, "higher_better"), 10.0)
        self.assertAlmostEqual(self.module.beneficial_relative_change(100, 80, "lower_better"), 20.0)

    def test_generates_all_png_and_svg_artifacts(self):
        table = self.module.load_metric_table(self.write_fixture(), ("rr", "pf", "rrq", "pfq"), 10)
        outputs = self.module.generate_figures(table, self.root / "figures", dpi=72)
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail because the plotting module does not exist**

Run:

```bash
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_plot_stageB_baseline_verification_report.py' -v
```

Expected: FAIL with `Expected plotting module to exist`; this is the deliberate TDD red state before production code exists.

- [ ] **Step 3: Commit the failing behavioral specification**

```bash
git add cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py
git commit -m "test: specify Stage-B verification report plots"
```

### Task 2: Implement the validated report plotting command

**Files:**
- Create: `cuMAC/scripts/plot_stageB_baseline_verification_report.py`
- Modify: `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`
- Test: `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`

**Interfaces:**
- Consumes: wide-form aggregate CSV with `<baseline>_mean`, `<baseline>_std`, `<baseline>_seed_count`, scenario, metric, unit, direction, and common seed metadata.
- Produces: `load_metric_table(Path, tuple[str, ...], int) -> dict`, `beneficial_relative_change(float, float, str) -> float`, `generate_figures(dict, Path, int) -> list[Path]`, and a CLI with `--suite-dir`, `--output-dir`, and `--dpi`.

- [ ] **Step 1: Implement strict CSV loading and improvement-direction helpers**

Create the module with these constants and functions:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINES = ("rr", "pf", "rrq", "pfq")
COLORS = {"rr": "#2563EB", "pf": "#D97706", "rrq": "#059669", "pfq": "#7C3AED"}
REQUIRED_METRICS = (
    "traffic.goodput_mbps",
    "traffic.packet_delay_p95_ms",
    "traffic.expiry_drop_rate",
    "global_kpi.ue_goodput_jain",
    "global_kpi.global_tb_bler",
    "global_kpi.prg_utilization_ratio",
    "global_kpi.ue_goodput_p5_mbps",
)


def load_metric_table(path: Path, expected_baselines=BASELINES, expected_seed_count: int = 10) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate metrics CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"metric", "unit", "direction", "common_seed_count", "common_seeds"}
        for baseline in expected_baselines:
            required_columns.update({f"{baseline}_mean", f"{baseline}_std", f"{baseline}_seed_count"})
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        table = {}
        for row in reader:
            metric = row["metric"]
            if metric not in REQUIRED_METRICS:
                continue
            if int(row["common_seed_count"]) != expected_seed_count:
                raise ValueError(f"Metric {metric} does not contain {expected_seed_count} seeds")
            values = {"unit": row["unit"], "direction": row["direction"]}
            for baseline in expected_baselines:
                if int(row[f"{baseline}_seed_count"]) != expected_seed_count:
                    raise ValueError(f"Metric {metric}/{baseline} does not contain {expected_seed_count} seeds")
                scale = 100.0 if row["unit"] == "%" else 1.0
                mean = float(row[f"{baseline}_mean"]) * scale
                std = float(row[f"{baseline}_std"]) * scale
                if not math.isfinite(mean) or not math.isfinite(std):
                    raise ValueError(f"Metric {metric}/{baseline} must contain finite mean/std values")
                values[baseline] = {"mean": mean, "std": std}
            if metric in table:
                raise ValueError(f"Duplicate metric row: {metric}")
            table[metric] = values
    missing_metrics = sorted(set(REQUIRED_METRICS) - set(table))
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {', '.join(missing_metrics)}")
    return table


def beneficial_relative_change(old: float, new: float, direction: str) -> float:
    if old == 0:
        raise ValueError("Cannot calculate relative change from a zero baseline")
    raw = (new - old) / abs(old) * 100.0
    if direction == "higher_better":
        return raw
    if direction == "lower_better":
        return -raw
    raise ValueError(f"Unsupported direction: {direction}")
```

- [ ] **Step 2: Implement deterministic PNG/SVG figures without dual axes**

Add helpers `_save_both`, `_plot_four_baselines`, and `generate_figures`. `generate_figures` must create exactly these stems:

```python
def generate_figures(metrics: dict, output_dir: Path, dpi: int = 220) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs += _plot_core_kpis(metrics, output_dir / "uniform_core_kpis", dpi)
    outputs += _plot_resource_tradeoffs(metrics, output_dir / "uniform_resource_tradeoffs", dpi)
    outputs += _plot_queue_aware_improvements(metrics, output_dir / "queue_aware_improvements", dpi)
    return outputs
```

`_plot_core_kpis` is a 2x2 bar figure for goodput, p95 delay, expiry, and Jain; `_plot_resource_tradeoffs` is a 1x2 figure for BLER and PRG utilization; `_plot_queue_aware_improvements` compares RR→RRQ and PF→PFQ beneficial relative changes for goodput, p95, expiry, and UE p5. Each chart uses `BASELINES` order, `COLORS`, cross-seed standard-deviation error bars where applicable, English labels that render with the default Matplotlib font, `fig.tight_layout()`, and both PNG and SVG outputs.

Implement the CLI:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-B baseline verification report figures.")
    parser.add_argument("--suite-dir", default="output/stageB_baseline_eval_suite_fullkpi_seed41_50")
    parser.add_argument("--output-dir", default="Doc/figures/stageB_baseline_verification")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.suite_dir) / "suite_aggregate/uniform_four_baseline_metrics.csv"
    table = load_metric_table(csv_path, BASELINES, expected_seed_count=10)
    for output in generate_figures(table, Path(args.output_dir), dpi=args.dpi):
        print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the focused tests and verify all cases pass**

Run:

```bash
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_plot_stageB_baseline_verification_report.py' -v
```

Expected: 6 tests run, `OK`.

- [ ] **Step 4: Run the plotting command against the real full-KPI suite**

Run:

```bash
python3 cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  --suite-dir output/stageB_baseline_eval_suite_fullkpi_seed41_50 \
  --output-dir Doc/figures/stageB_baseline_verification
```

Expected: exit 0 and six non-empty paths: PNG and SVG for `uniform_core_kpis`, `uniform_resource_tradeoffs`, and `queue_aware_improvements`.

- [ ] **Step 5: Commit the plotting implementation and generated report figures**

```bash
git add cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py \
  Doc/figures/stageB_baseline_verification
git commit -m "feat: add Stage-B baseline report figures"
```

### Task 3: Write the formal acceptance report and technical appendix

**Files:**
- Create: `Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md`
- Reference: `Doc/stageB_cochannel_baseline_analysis_reference.md`
- Reference: `Doc/stageB_packet_delay_distribution_analysis.md`
- Reference: `Doc/current_stageB_effective_configuration.md`

**Interfaces:**
- Consumes: full-KPI aggregate CSV/JSON, active boundary gate aggregates, three generated figure stems, and existing packet-delay distribution figures.
- Produces: a self-contained project-acceptance Markdown report with 13 main sections and appendices.

- [ ] **Step 1: Write the executive summary, scope, algorithm principles, and multidimensional mapping**

Use the approved section order. State explicitly that the four algorithms are representative baselines, not AI algorithms, and that they provide the comparison foundation for later AI scheduling. Include the exact mapping table from the approved design and the caveat that power/spatial dimensions are observed through link quality, MIMO/precoding, and interference conditions rather than direct joint power control or cross-cell action optimization.

- [ ] **Step 2: Write the simulation environment, traffic model, controlled variables, and acceptance criteria**

Record: 3 cells, 36 active UEs, 12 UEs/cell, 4T4R, Rayleigh, SVD, Type-0 bitmap, 2.5 GHz, 30 kHz SCS, 0.5 ms TTI, 3000-byte fixed packets, Poisson `1.0 pkt/TTI/UE`, 4000 TTIs, 200 ms TTL, and paired seeds 41--50. State the theoretical offered load calculation:

```text
36 UE × 3000 Byte × 8 bit/Byte × 1 pkt/TTI / 0.0005 s = 1.728 Gbps
```

For PRG granularity, report the value recorded by each batch's actual command/manifest and explicitly separate the full-KPI suite from the active gate if their resource grouping differs.

- [ ] **Step 3: Add the uniform four-baseline main result table and interpretation**

The table must include these full-KPI values as `mean ± std` across 10 seeds:

| Metric | RR | PF | RRQ | PFQ |
|---|---:|---:|---:|---:|
| goodput (Mbps) | 1486.396 ± 31.771 | 925.656 ± 50.922 | 1512.253 ± 35.026 | 1527.240 ± 16.945 |
| p95 delay (ms) | 136.100 ± 66.598 | 199.200 ± 0.245 | 98.550 ± 90.659 | 38.000 ± 49.942 |
| expiry (%) | 1.080 ± 0.961 | 31.023 ± 2.986 | 1.179 ± 1.207 | 0.117 ± 0.215 |
| UE goodput p5 (Mbps) | 34.891 ± 3.594 | 8.542 ± 1.280 | 35.374 ± 4.763 | 39.190 ± 1.570 |
| Jain fairness | 0.993380 ± 0.004774 | 0.805345 ± 0.024030 | 0.993267 ± 0.005662 | 0.997899 ± 0.001119 |
| TB BLER (%) | 11.706 ± 0.665 | 12.411 ± 0.656 | 11.240 ± 0.634 | 11.422 ± 0.703 |
| PRG utilization (%) | 99.999 ± 0.002 | 100.000 ± 0.000 | 89.193 ± 5.149 | 94.509 ± 2.974 |

Interpret PF's poor result as a property of the current high-load queue-unaware implementation, not a general refutation of proportional fairness. Interpret RRQ and PFQ as queue-aware modifications with measurable tradeoffs, and identify PFQ as the strongest comprehensive baseline for this test condition.

- [ ] **Step 4: Add packet-delay and boundary evidence without batch mixing**

Embed the three existing focused distribution images and explain completed-only delay sampling together with expired and pending packet counts. Add the boundary RRQ/PFQ table with:

| Metric | Boundary RRQ | Boundary PFQ | PFQ - RRQ |
|---|---:|---:|---:|
| goodput (Mbps) | 1494.513 ± 33.711 | 1516.222 ± 17.190 | +21.710 |
| p95 delay (ms) | 94.700 ± 73.961 | 41.850 ± 30.602 | -52.850 |
| expiry (%) | 1.282 ± 1.427 | 0.083 ± 0.112 | -1.199 pp |
| UE p5 (Mbps) | 34.829 ± 4.704 | 38.954 ± 1.084 | +4.124 |
| Jain fairness | 0.992767 ± 0.006513 | 0.998177 ± 0.000836 | +0.005409 |
| PRG utilization (%) | 94.350 ± 6.055 | 98.598 ± 2.218 | +4.248 pp |

Then summarize the active gate as a separate cross-check: PFQ vs RRQ goodput `+6.903 Mbps`, expiry `-0.592 pp`, p95 `-30.428 ms`, UE p5 `+1.935 Mbps`, and Jain `+0.002475`. Label the values as a separate batch and do not place them in the full-KPI boundary table.

- [ ] **Step 5: Write acceptance conclusions, limitations, reproduction appendix, and evidence index**

The acceptance matrix must mark functional execution, paired comparison, KPI completeness, and traceability as satisfied, while marking bit-exact reproducibility provenance as partially satisfied. Include the exact commands:

```bash
cuMAC/scripts/run_stageB_baseline_eval_suite.sh \
  --build-method phase4 \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --suite-output-dir output/stageB_baseline_eval_suite_reproduce

python3 cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  --suite-dir output/stageB_baseline_eval_suite_fullkpi_seed41_50 \
  --output-dir Doc/figures/stageB_baseline_verification
```

State that the current repository proves script- and manifest-level traceability but lacks the original run-time commit hash, dirty patch, image digest, compiler-options hash, and unified configuration hash needed for bit-exact reproduction.

- [ ] **Step 6: Check the report for placeholders, unsupported claims, and mixed-batch numbers**

Run:

```bash
rg -n 'T[B]D|T[O]DO|RRF|显著提升系统容量|直接优化发射功率|跨小区联合动作' \
  Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md
```

Expected: no unresolved placeholder or `RRF`; any matches for prohibited claims occur only inside explicit negations or limitations and are manually confirmed.

- [ ] **Step 7: Commit the formal report**

```bash
git add Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md
git commit -m "docs: add Stage-B baseline verification report"
```

### Task 4: Add the report to the existing documentation index

**Files:**
- Modify: `Doc/README.md`
- Test: `Doc/README.md`

**Interfaces:**
- Consumes: the existing dirty-worktree version of `Doc/README.md`.
- Produces: one additional uncommitted bullet linking the formal verification report without reverting, rewriting, staging, or committing the user's current edits.

- [ ] **Step 1: Insert one report link under `## 当前文档`**

Add this bullet before the intelligent-algorithm report entry:

```markdown
- [`stageB_multidimensional_resource_allocation_simulation_verification_report.md`](./stageB_multidimensional_resource_allocation_simulation_verification_report.md)：RR、PF、RRQ、PFQ 四类代表性资源分配技术的项目验收仿真验证报告，包含统一场景对比、同频边缘增强验证和复现附录。
```

- [ ] **Step 2: Verify the existing user changes remain present**

Run:

```bash
git diff -- Doc/README.md
rg -n 'stageB_intelligent_algorithm_evolution_training_report|stageB_intelligent_algorithm_data_manifest|stageB_multidimensional_resource_allocation_simulation_verification_report' Doc/README.md
```

Expected: all three links exist; the diff still contains the user's pre-existing intelligent-algorithm index updates plus the new verification-report bullet.

- [ ] **Step 3: Leave the combined index change unstaged after reviewing it**

Run `git status --short Doc/README.md` and verify it remains ` M Doc/README.md`. Report the exact inserted line in the handoff; do not stage or commit this file because it already contains user-authored changes.

### Task 5: Run final acceptance verification

**Files:**
- Verify: `Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md`
- Verify: `Doc/figures/stageB_baseline_verification/`
- Verify: `cuMAC/scripts/plot_stageB_baseline_verification_report.py`
- Verify: `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`
- Verify: `Doc/README.md`

**Interfaces:**
- Consumes: all report deliverables and immutable source data.
- Produces: fresh test, artifact, link, source-number, and worktree evidence suitable for the final handoff.

- [ ] **Step 1: Run the focused automated tests**

```bash
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_plot_stageB_baseline_verification_report.py' -v
```

Expected: 6 tests, 0 failures, 0 errors.

- [ ] **Step 2: Regenerate all report-specific figures from the real suite**

```bash
python3 cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  --suite-dir output/stageB_baseline_eval_suite_fullkpi_seed41_50 \
  --output-dir Doc/figures/stageB_baseline_verification
```

Expected: exit 0 and six printed artifact paths.

- [ ] **Step 3: Verify all local Markdown image and document links resolve**

```bash
python3 - <<'PY'
import re
from pathlib import Path

report = Path('Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md')
text = report.read_text(encoding='utf-8')
missing = []
for target in re.findall(r'!?(?:\[[^]]*\])\(([^)]+)\)', text):
    clean = target.split('#', 1)[0]
    if not clean or '://' in clean:
        continue
    path = (report.parent / clean).resolve()
    if not path.exists():
        missing.append(clean)
if missing:
    raise SystemExit(f'Missing report links: {missing}')
print('all local report links resolve')
PY
```

Expected: `all local report links resolve`.

- [ ] **Step 4: Verify the report's main values directly against the aggregate CSV**

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path('output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/uniform_four_baseline_metrics.csv')
rows = {row['metric']: row for row in csv.DictReader(path.open(encoding='utf-8'))}
expected = {
    ('traffic.goodput_mbps', 'rr_mean'): 1486.3956544,
    ('traffic.goodput_mbps', 'pf_mean'): 925.6556064,
    ('traffic.goodput_mbps', 'rrq_mean'): 1512.253024,
    ('traffic.goodput_mbps', 'pfq_mean'): 1527.2395264,
    ('traffic.packet_delay_p95_ms', 'pfq_mean'): 38.0,
    ('traffic.expiry_drop_rate', 'pf_mean'): 0.3102319,
    ('global_kpi.ue_goodput_jain', 'pfq_mean'): 0.997899,
}
for (metric, field), value in expected.items():
    actual = float(rows[metric][field])
    if abs(actual - value) > 5e-6:
        raise SystemExit(f'{metric}/{field}: expected {value}, got {actual}')
print('core source values verified')
PY
```

Expected: `core source values verified`.

- [ ] **Step 5: Inspect scope and worktree state before the final completion claim**

```bash
git status --short
git log --oneline -5
git diff --check
```

Expected: report-related changes are committed, pre-existing unrelated user changes remain untouched, and `git diff --check` reports no whitespace errors.

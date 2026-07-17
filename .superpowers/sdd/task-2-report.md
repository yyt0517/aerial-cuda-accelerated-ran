# Task 2 Report: Stage-B Baseline Verification Figures

## Status

Implemented the validated Stage-B plotting command and generated the six requested
PNG/SVG report artifacts from the real full-KPI suite. The plotting command only
reads `output/stageB_baseline_eval_suite_fullkpi_seed41_50/` for the four-algorithm
source data.

## Focused GREEN verification

Command:

```bash
python3 -m unittest discover -s cuMAC/scripts/tests -p 'test_plot_stageB_baseline_verification_report.py' -v
```

Output:

```text
test_beneficial_change_respects_metric_direction ... ok
test_generates_all_png_and_svg_artifacts ... ok
test_loads_complete_four_baseline_table ... ok
test_rejects_missing_algorithm_columns ... ok
test_rejects_non_finite_metric ... ok
test_rejects_seed_count_mismatch ... ok

----------------------------------------------------------------------
Ran 6 tests in 1.459s

OK
```

The initial RED run had 6 failures, all caused by the expected missing plotting
module.

## Real-suite CLI verification

Command:

```bash
python3 cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  --suite-dir output/stageB_baseline_eval_suite_fullkpi_seed41_50 \
  --output-dir Doc/figures/stageB_baseline_verification
```

The command exited 0 and printed:

```text
Doc/figures/stageB_baseline_verification/uniform_core_kpis.png
Doc/figures/stageB_baseline_verification/uniform_core_kpis.svg
Doc/figures/stageB_baseline_verification/uniform_resource_tradeoffs.png
Doc/figures/stageB_baseline_verification/uniform_resource_tradeoffs.svg
Doc/figures/stageB_baseline_verification/queue_aware_improvements.png
Doc/figures/stageB_baseline_verification/queue_aware_improvements.svg
```

Artifact size check:

```text
queue_aware_improvements.png 76017 bytes
queue_aware_improvements.svg 39911 bytes
uniform_core_kpis.png 125599 bytes
uniform_core_kpis.svg 73318 bytes
uniform_resource_tradeoffs.png 55281 bytes
uniform_resource_tradeoffs.svg 41348 bytes
```

## Task file list

- `cuMAC/scripts/plot_stageB_baseline_verification_report.py`
- `cuMAC/scripts/tests/test_plot_stageB_baseline_verification_report.py`
- `Doc/figures/stageB_baseline_verification/uniform_core_kpis.{png,svg}`
- `Doc/figures/stageB_baseline_verification/uniform_resource_tradeoffs.{png,svg}`
- `Doc/figures/stageB_baseline_verification/queue_aware_improvements.{png,svg}`
- `.superpowers/sdd/task-2-report.md`

## Self-check

- Strict loading rejects a missing CSV, missing required columns/metrics,
  duplicate required rows, seed-count mismatches, and non-finite mean/std values.
- Source values with unit `%` are multiplied by 100 during loading; the real-suite
  BLER, PRG utilization, and expiry plots therefore display percent values rather
  than 0-1 ratios.
- Every four-baseline chart uses the required RR, PF, RRQ, PFQ order and fixed
  color map, with cross-seed standard-deviation error bars.
- All panels use a single y-axis; there are no dual axes and no radar plots.
- Queue-aware bars apply metric direction so positive values always mean a
  beneficial relative change.
- Plot text is English/default-font compatible, layout is tightened, and all
  figures are closed after deterministic PNG/SVG saving.
- Visual inspection confirmed readable labels, expected panel layouts, correct
  baseline order, and percent-scale axes.
- `git diff --check` completed with no whitespace errors.
- Existing unrelated workspace modifications were neither edited nor staged.

## Concerns

None. The test expectation for the resource figure was corrected from
`resource_tradeoffs` to the brief-mandated `uniform_resource_tradeoffs` stem.

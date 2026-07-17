from __future__ import annotations

import csv
import importlib.util
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
        common_seeds = "|".join(str(seed) for seed in range(41, 41 + int(common_seed_count)))
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
                    "common_seeds": common_seeds,
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
        self.assertAlmostEqual(self.module.beneficial_relative_change(100, 80, "higher_better"), -20.0)
        self.assertAlmostEqual(self.module.beneficial_relative_change(100, 80, "lower_better"), 20.0)
        self.assertAlmostEqual(self.module.beneficial_relative_change(100, 110, "lower_better"), -10.0)

    def test_generates_all_png_and_svg_artifacts(self):
        table = self.module.load_metric_table(self.write_fixture(), ("rr", "pf", "rrq", "pfq"), 10)
        outputs = self.module.generate_figures(table, self.root / "figures", dpi=72)
        self.assertEqual(len(outputs), 6)
        self.assertEqual(len(set(outputs)), 6)
        self.assertEqual(
            {path.name for path in outputs},
            {
                "uniform_core_kpis.png",
                "uniform_core_kpis.svg",
                "resource_tradeoffs.png",
                "resource_tradeoffs.svg",
                "queue_aware_improvements.png",
                "queue_aware_improvements.svg",
            },
        )
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))


if __name__ == "__main__":
    unittest.main()

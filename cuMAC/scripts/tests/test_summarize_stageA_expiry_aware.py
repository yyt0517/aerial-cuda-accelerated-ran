from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cuMAC.scripts.summarize_stageA_kpi import (
    compute_packet_outcome_sample_window_stats,
    parse_traffic_packet_expiry_aware_rate_kpi,
    parse_traffic_packet_outcome_delay_kpi,
    parse_traffic_packet_outcome_samples_kpi,
    parse_ue_packet_expiry_aware_rate_kpi,
    parse_ue_packet_outcome_delay_kpi,
    summarize_global_kpi,
)


class ExpiryAwareKpiParserTest(unittest.TestCase):
    def test_parses_cluster_and_per_ue_records(self) -> None:
        log = """
TRAFFIC_PKT_OUTCOME_DELAY delivered_pkt_count=10 expired_pkt_count=2 outcome_pkt_count=12 pending_pkt_count=3 mean_ms=34.25 p50_ms=1.0 p90_ms=180.0 p95_ms=200.0 max_ms=200.0
TRAFFIC_PKT_RATE_EA delivered_pkt_count=10 expired_pkt_count=2 pending_pkt_count=3 total_delivered_bits=240000 delivered_packet_system_time_ms=5.0 expired_packet_system_time_ms=400.0 outcome_packet_system_time_ms=405.0 r_pkt_ea_mbps=0.592593 r_pkt_ea_per_packet_mean_mbps=40.0
UE_PKT_OUTCOME_DELAY ue_id=4 delivered_pkt_count=5 expired_pkt_count=1 outcome_pkt_count=6 pending_pkt_count=2 outcome_delay_mean_ms=34.0 outcome_delay_p50_ms=1.0 outcome_delay_p90_ms=180.0 outcome_delay_p95_ms=200.0 outcome_delay_max_ms=200.0
UE_PKT_RATE_EA ue_id=4 delivered_pkt_count=5 expired_pkt_count=1 pending_pkt_count=2 total_delivered_bits=120000 delivered_packet_system_time_ms=2.5 expired_packet_system_time_ms=200.0 outcome_packet_system_time_ms=202.5 packet_expiry_aware_rate_mbps=0.592593 packet_expiry_aware_rate_per_packet_mean_mbps=40.0
"""

        delay = parse_traffic_packet_outcome_delay_kpi(log)
        rate = parse_traffic_packet_expiry_aware_rate_kpi(log)
        ue_delay = parse_ue_packet_outcome_delay_kpi(log)[4]
        ue_rate = parse_ue_packet_expiry_aware_rate_kpi(log)[4]

        self.assertEqual(delay["packet_outcome_count"], 12)
        self.assertEqual(delay["packet_outcome_expired_pkt_count"], 2)
        self.assertEqual(delay["packet_outcome_delay_p95_ms"], 200.0)
        self.assertEqual(rate["packet_expiry_aware_total_delivered_bits"], 240000)
        self.assertEqual(rate["packet_expiry_aware_outcome_system_time_ms"], 405.0)
        self.assertAlmostEqual(rate["packet_expiry_aware_rate_mbps"], 0.592593)
        self.assertEqual(ue_delay["packet_outcome_count"], 6)
        self.assertEqual(ue_delay["packet_outcome_delay_mean_ms"], 34.0)
        self.assertEqual(ue_rate["packet_expiry_aware_expired_pkt_count"], 1)
        self.assertEqual(ue_rate["packet_expiry_aware_rate_per_packet_mean_mbps"], 40.0)

    def test_filters_exact_terminal_tti_window_and_computes_p95(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            csv_path = out_dir / "packet_outcome_samples.csv"
            csv_path.write_text(
                "ue_id,packet_id,arrival_tti,packet_size_bytes,remaining_bytes_at_terminal,"
                "terminal_tti,outcome,system_time_ms\n"
                "0,1,601,3000,0,1000,delivered,1.0\n"
                "0,2,602,3000,0,1001,delivered,20.0\n"
                "1,3,2101,3000,1200,2500,expired,200.0\n"
                "2,4,4601,3000,0,5000,delivered,100.0\n"
                "2,5,4602,3000,3000,5001,expired,200.0\n",
                encoding="utf-8",
            )
            log = f"TRAFFIC_PKT_OUTCOME_SAMPLES_CSV path={csv_path.name} rows=5\n"

            parsed = parse_traffic_packet_outcome_samples_kpi(log, out_dir)
            stats = compute_packet_outcome_sample_window_stats(
                Path(parsed["packet_outcome_samples_csv"]),
                terminal_tti_exclusive=1000,
                terminal_tti_inclusive=5000,
            )

            self.assertEqual(parsed["packet_outcome_samples_rows"], 5)
            self.assertEqual(stats["packet_outcome_measured_count"], 3)
            self.assertEqual(stats["packet_outcome_measured_expired_count"], 1)
            self.assertAlmostEqual(stats["packet_outcome_measured_p95_ms"], 190.0)

    def test_global_summary_exposes_ue_expiry_aware_tail_metrics(self) -> None:
        ue_rows = [
            {
                "avg_thr_mbps": 1.0,
                "packet_expiry_aware_rate_mbps": 1.0,
                "packet_outcome_delay_mean_ms": 10.0,
                "packet_outcome_delay_p95_ms": 100.0,
            },
            {
                "avg_thr_mbps": 2.0,
                "packet_expiry_aware_rate_mbps": 3.0,
                "packet_outcome_delay_mean_ms": 30.0,
                "packet_outcome_delay_p95_ms": 200.0,
            },
        ]

        summary = summarize_global_kpi(ue_rows, [], traffic_kpi={})

        self.assertEqual(summary["ue_macro_packet_expiry_aware_rate_mbps"], 2.0)
        self.assertAlmostEqual(summary["ue_p10_packet_expiry_aware_rate_mbps"], 1.2)
        self.assertEqual(summary["ue_min_packet_expiry_aware_rate_mbps"], 1.0)
        self.assertEqual(summary["ue_macro_packet_outcome_delay_mean_ms"], 20.0)
        self.assertAlmostEqual(summary["ue_p95_packet_outcome_delay_p95_ms"], 195.0)


if __name__ == "__main__":
    unittest.main()

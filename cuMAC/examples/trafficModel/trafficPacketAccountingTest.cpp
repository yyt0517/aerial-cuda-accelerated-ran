/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trafficGenerator.hpp"
#include "serviceBalanceTrace.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 1.0e-9;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, const std::string& message)
{
    if (std::abs(actual - expected) > kTolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

struct Fixture {
    explicit Fixture(size_t n_flow)
        : buffer_size(n_flow, 0U)
        , config(n_flow)
        , radio(&ue_status)
    {
        ue_status.bufferSize = buffer_size.data();
        for (size_t i = 0; i < n_flow; ++i) {
            config[i].flow_id = static_cast<int>(i);
        }
        radio.Configure(config);
        radio.SetSlotDurationMs(0.5);
    }

    void enqueue(size_t flow_id, int packet_bytes, int arrival_tti)
    {
        std::vector<FlowData> arrivals(config.size());
        for (size_t i = 0; i < arrivals.size(); ++i) {
            arrivals[i].flow_id = static_cast<int>(i);
            arrivals[i].arrival_tti = arrival_tti;
        }
        arrivals[flow_id].packet_bytes = {packet_bytes};
        radio.Enqueue(arrivals);
    }

    cumac::cumacCellGrpUeStatus ue_status {};
    std::vector<uint32_t> buffer_size;
    std::vector<FlowData> config;
    RadioResource radio;
};

void testDeliveredPacketMatchesDeliveredOnlyMetrics()
{
    Fixture fixture(1);
    fixture.enqueue(0, 3000, 1);
    fixture.radio.RecordServedBytes({3000ULL}, 0);

    PacketServiceRateSummary delivered_only;
    std::vector<PacketServiceRateSummary> delivered_only_per_flow;
    fixture.radio.GetPacketServiceRateStats(delivered_only, delivered_only_per_flow);

    PacketOutcomeDelaySummary outcome_delay;
    std::vector<PacketOutcomeDelaySummary> outcome_delay_per_flow;
    fixture.radio.GetPacketOutcomeDelayStats(outcome_delay, outcome_delay_per_flow);

    PacketExpiryAwareRateSummary expiry_aware;
    std::vector<PacketExpiryAwareRateSummary> expiry_aware_per_flow;
    fixture.radio.GetPacketExpiryAwareRateStats(expiry_aware, expiry_aware_per_flow);

    require(expiry_aware.delivered_packets == 1ULL, "delivered packet must be an EA outcome");
    require(expiry_aware.expired_packets == 0ULL, "delivered packet must not expire");
    require(expiry_aware.total_delivered_bits == 24000ULL, "3000 bytes must produce 24000 delivered bits");
    requireNear(expiry_aware.delivered_packet_system_time_ms, 0.5, "delivery time");
    requireNear(expiry_aware.outcome_packet_system_time_ms, 0.5, "delivery outcome time");
    requireNear(expiry_aware.packet_expiry_aware_rate_mbps, 48.0, "delivery EA aggregate rate");
    requireNear(expiry_aware.packet_expiry_aware_rate_per_packet_mean_mbps, 48.0,
                "delivery EA per-packet rate");
    requireNear(expiry_aware.packet_expiry_aware_rate_mbps,
                delivered_only.packet_effective_service_rate_mbps,
                "delivered-only and EA aggregate rate must match without expiry");
    require(outcome_delay.outcome_packets == 1ULL, "delivery must terminate one outcome");
    requireNear(outcome_delay.mean_delay_ms, 0.5, "delivery outcome delay");

    std::vector<PacketOutcomeSample> samples;
    fixture.radio.GetPacketOutcomeSamples(samples);
    require(samples.size() == 1U, "delivery must append one outcome sample");
    require(samples[0].arrival_tti == 1, "delivery sample must retain arrival TTI");
    require(samples[0].packet_size_bytes == 3000U, "delivery sample must retain packet size");
    require(samples[0].remaining_bytes_at_terminal == 0U,
            "delivery sample must terminate with no remaining bytes");
    require(samples[0].terminal_tti == 1, "delivery sample TTI must be one-based");
    require(!samples[0].expired, "delivery sample must not be marked expired");
    requireNear(samples[0].system_time_ms, 0.5, "delivery sample system time");
}

void testExpiryUsesExactTtlBoundary()
{
    Fixture fixture(1);
    fixture.radio.SetPacketTtlTti(400);
    fixture.enqueue(0, 3000, 1);
    fixture.radio.AdvanceToTti(401);

    PacketOutcomeDelaySummary outcome_delay;
    std::vector<PacketOutcomeDelaySummary> outcome_delay_per_flow;
    fixture.radio.GetPacketOutcomeDelayStats(outcome_delay, outcome_delay_per_flow);
    PacketExpiryAwareRateSummary expiry_aware;
    std::vector<PacketExpiryAwareRateSummary> expiry_aware_per_flow;
    fixture.radio.GetPacketExpiryAwareRateStats(expiry_aware, expiry_aware_per_flow);

    require(expiry_aware.delivered_packets == 0ULL, "expired packet must deliver zero packets");
    require(expiry_aware.expired_packets == 1ULL, "expired packet must terminate one outcome");
    require(expiry_aware.total_delivered_bits == 0ULL, "expired packet must deliver zero bits");
    requireNear(expiry_aware.expired_packet_system_time_ms, 200.0, "expiry time must equal TTL");
    requireNear(expiry_aware.outcome_packet_system_time_ms, 200.0, "expiry outcome time");
    requireNear(expiry_aware.packet_expiry_aware_rate_mbps, 0.0, "expired aggregate rate");
    requireNear(expiry_aware.packet_expiry_aware_rate_per_packet_mean_mbps, 0.0,
                "expired per-packet rate");
    requireNear(outcome_delay.mean_delay_ms, 200.0, "expiry outcome delay");
    requireNear(outcome_delay.p95_delay_ms, 200.0, "expiry p95 outcome delay");

    std::vector<PacketOutcomeSample> samples;
    fixture.radio.GetPacketOutcomeSamples(samples);
    require(samples.size() == 1U, "expiry must append one outcome sample");
    require(samples[0].arrival_tti == 1, "expiry sample must retain arrival TTI");
    require(samples[0].packet_size_bytes == 3000U, "expiry sample must retain packet size");
    require(samples[0].remaining_bytes_at_terminal == 3000U,
            "expiry sample must retain remaining bytes at expiry");
    require(samples[0].terminal_tti == 400, "expiry sample must use exact last eligible TTI");
    require(samples[0].expired, "expiry sample must be marked expired");
    requireNear(samples[0].system_time_ms, 200.0, "expiry sample system time");
}

void testMixedOutcomesAndPerFlowIdentities()
{
    Fixture fixture(2);
    fixture.radio.SetPacketTtlTti(400);
    fixture.enqueue(0, 3000, 1);
    fixture.enqueue(1, 3000, 1);
    fixture.radio.RecordServedBytes({3000ULL, 0ULL}, 0);
    fixture.radio.AdvanceToTti(401);

    PacketOutcomeDelaySummary outcome_delay;
    std::vector<PacketOutcomeDelaySummary> outcome_delay_per_flow;
    fixture.radio.GetPacketOutcomeDelayStats(outcome_delay, outcome_delay_per_flow);
    PacketExpiryAwareRateSummary expiry_aware;
    std::vector<PacketExpiryAwareRateSummary> per_flow;
    fixture.radio.GetPacketExpiryAwareRateStats(expiry_aware, per_flow);

    require(expiry_aware.delivered_packets == 1ULL, "mixed outcomes delivered count");
    require(expiry_aware.expired_packets == 1ULL, "mixed outcomes expired count");
    require(outcome_delay.outcome_packets == 2ULL, "mixed outcomes total count");
    requireNear(expiry_aware.outcome_packet_system_time_ms, 200.5, "mixed total outcome time");
    requireNear(expiry_aware.packet_expiry_aware_rate_mbps, 24000.0 / (200.5 * 1000.0),
                "mixed aggregate EA rate");
    requireNear(expiry_aware.packet_expiry_aware_rate_per_packet_mean_mbps, 24.0,
                "expired packet contributes zero to mixed per-packet mean");
    requireNear(outcome_delay.mean_delay_ms, 100.25, "mixed mean outcome delay");

    require(per_flow.size() == 2U, "two per-flow EA summaries");
    require(per_flow[0].delivered_packets + per_flow[1].delivered_packets == expiry_aware.delivered_packets,
            "per-flow delivered counts sum to cluster");
    require(per_flow[0].expired_packets + per_flow[1].expired_packets == expiry_aware.expired_packets,
            "per-flow expired counts sum to cluster");
    require(per_flow[0].total_delivered_bits + per_flow[1].total_delivered_bits ==
                expiry_aware.total_delivered_bits,
            "per-flow delivered bits sum to cluster");
    requireNear(per_flow[0].outcome_packet_system_time_ms + per_flow[1].outcome_packet_system_time_ms,
                expiry_aware.outcome_packet_system_time_ms,
                "per-flow outcome times sum to cluster");
}

void testPartialPacketExpiresAsOneZeroCompletedBitOutcome()
{
    Fixture fixture(1);
    fixture.radio.SetPacketTtlTti(400);
    fixture.enqueue(0, 3000, 1);
    fixture.radio.RecordServedBytes({1000ULL}, 0);
    fixture.radio.AdvanceToTti(401);

    PacketExpiryAwareRateSummary expiry_aware;
    std::vector<PacketExpiryAwareRateSummary> per_flow;
    fixture.radio.GetPacketExpiryAwareRateStats(expiry_aware, per_flow);

    require(fixture.radio.GetTotalExpiredBytes() == 2000ULL,
            "only the partially consumed packet remainder expires");
    require(expiry_aware.expired_packets == 1ULL, "partial packet terminates as one expiry");
    require(expiry_aware.delivered_packets == 0ULL, "partial packet is not delivered");
    require(expiry_aware.total_delivered_bits == 0ULL, "partial packet contributes zero completed bits");
    requireNear(expiry_aware.outcome_packet_system_time_ms, 200.0, "partial packet expiry time");
}

void testPerTtiSummaryKeepsExpiryAndLaterDelivery()
{
    Fixture fixture(2);
    fixture.radio.SetPacketTtlTti(400);
    fixture.enqueue(0, 3000, 1);

    fixture.radio.AdvanceToTti(401);
    fixture.enqueue(1, 3000, 401);
    fixture.radio.RecordServedBytes({0ULL, 3000ULL}, 400);

    const PacketExpiryAwareRateSummary last = fixture.radio.GetLastPacketExpiryAwareRateStats();
    require(last.delivered_packets == 1ULL, "per-TTI summary keeps later delivery");
    require(last.expired_packets == 1ULL, "per-TTI summary keeps earlier expiry");
    require(last.pending_packets == 0ULL, "per-TTI summary reports no pending packets");
    require(last.total_delivered_bits == 24000ULL, "per-TTI delivered bits");
    requireNear(last.delivered_packet_system_time_ms, 0.5, "per-TTI delivery time");
    requireNear(last.expired_packet_system_time_ms, 200.0, "per-TTI expiry time");
    requireNear(last.outcome_packet_system_time_ms, 200.5, "per-TTI outcome time");
    requireNear(last.packet_expiry_aware_rate_mbps, 24000.0 / (200.5 * 1000.0),
                "per-TTI aggregate EA rate");
    requireNear(last.packet_expiry_aware_rate_per_packet_mean_mbps, 24.0,
                "per-TTI mean includes zero-rate expiry");
}

void testOfferedPacketAuditHasStableIdentityWithoutChangingTrafficTotals()
{
    TrafficType traffic(Arrival_t::uniform, 3000.0F, 0.0F, 1.0F);
    TrafficConfig config(traffic, 1);
    TrafficGenerator generator(config);
    generator.Seed(42);
    generator.Generate();
    std::vector<PacketOfferSample> disabledSamples;
    generator.GetPacketOfferSamples(disabledSamples);
    require(disabledSamples.empty(), "offered packet audit must be disabled by default");

    generator.SetPacketOfferAuditEnabled(true);
    generator.Generate();
    generator.Generate();

    std::vector<PacketOfferSample> samples;
    generator.GetPacketOfferSamples(samples);
    require(samples.size() == 2U, "two TTIs must produce two offered packet samples");
    require(samples[0].packet_id == 1ULL && samples[1].packet_id == 2ULL,
            "offered packet IDs must be stable and contiguous");
    require(samples[0].flow_id == 0U && samples[1].flow_id == 0U,
            "offered packet samples must retain flow identity");
    require(samples[0].arrival_tti == 2 && samples[1].arrival_tti == 3,
            "offered packet samples must retain one-based arrival TTI");
    require(samples[0].packet_size_bytes == 3000U && samples[1].packet_size_bytes == 3000U,
            "offered packet samples must retain generated packet size");
    require(generator.GetTotalGeneratedPkts() == 3ULL,
            "offered packet instrumentation must not change generated packet count");
    require(generator.GetTotalGeneratedBytes() == 9000ULL,
            "offered packet instrumentation must not change generated bytes");
}

void testServiceBalanceCsvRejectsBrokenConservationAndFreezesSchema()
{
    ServiceBalanceTraceRow row;
    row.tti = 602;
    row.ueId = 2;
    row.servingCell = 2;
    row.offeredBytes = 3000;
    row.acceptedBytes = 3000;
    row.rejectedBytes = 0;
    row.expiredBytes = 0;
    row.previousPostQueueBytes = 1000;
    row.preQueueBytes = 4000;
    row.scheduledDequeueBytes = 1500;
    row.postQueueBytes = 2500;
    row.successfulBytes = 1200;
    row.failedBytes = 300;
    row.demandValidPrgCount = 17;
    row.actualPrgCount = 2;
    row.actualPrgMask = 3;
    row.tbsCapacityBytes = 1500;
    row.tbsCapacityBits = 12000;
    row.mcs = 4;
    row.layer = 2;
    row.tbTxCount = 1;
    row.tbErrCount = 0;
    row.holDelayMs = 1.5F;
    row.ttlSlackMs = 198.5F;

    require(row.conservationHolds(), "valid service-balance row must satisfy every identity");
    ServiceBalanceTraceRow broken = row;
    broken.failedBytes = 299;
    require(!broken.conservationHolds(), "broken successful/failed identity must fail closed");

    std::ostringstream stream;
    writeServiceBalanceTraceHeader(stream);
    require(writeServiceBalanceTraceRow(stream, row), "valid row must serialize");
    require(
        stream.str().find(
            "tti,ue_id,serving_cell,offered_bytes,accepted_bytes,rejected_bytes,expired_bytes,") == 0U,
        "service-balance CSV header must freeze traffic and queue fields first");
    require(stream.str().find("602,2,2,3000,3000,0,0,1000,4000,1500,2500,1200,300,") !=
                std::string::npos,
            "service-balance CSV row must preserve exact integer byte fields");
}

} // namespace

int main()
{
    try {
        testDeliveredPacketMatchesDeliveredOnlyMetrics();
        testExpiryUsesExactTtlBoundary();
        testMixedOutcomesAndPerFlowIdentities();
        testPartialPacketExpiresAsOneZeroCompletedBitOutcome();
        testPerTtiSummaryKeepsExpiryAndLaterDelivery();
        testOfferedPacketAuditHasStableIdentityWithoutChangingTrafficTotals();
        testServiceBalanceCsvRejectsBrokenConservationAndFreezesSchema();
    } catch (const std::exception& error) {
        std::cerr << "trafficPacketAccountingTest failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "trafficPacketAccountingTest passed\n";
    return 0;
}

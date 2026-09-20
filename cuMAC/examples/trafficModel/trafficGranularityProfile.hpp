/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

struct TrafficGranularityFlowSpec
{
    float packetSizeBytes;
    float arrivalRatePacketsPerTti;
    std::string classLabel;
};

inline std::vector<TrafficGranularityFlowSpec> buildTrafficGranularityProfile(
    const std::string& profile,
    int flowCount,
    float defaultPacketSizeBytes,
    float defaultArrivalRatePacketsPerTti)
{
    std::vector<TrafficGranularityFlowSpec> flows;
    flows.reserve(static_cast<size_t>(flowCount));

    if (profile == "homogeneous") {
        for (int ue = 0; ue < flowCount; ++ue) {
            flows.push_back(
                {defaultPacketSizeBytes, defaultArrivalRatePacketsPerTti, "homogeneous"});
        }
        return flows;
    }

    if (profile == "mixed_granularity_v1") {
        if (flowCount % 2 != 0) {
            throw std::invalid_argument("mixed_granularity_v1 requires an even flow count");
        }
        for (int ue = 0; ue < flowCount; ++ue) {
            if (ue % 2 == 0) {
                flows.push_back({1500.0F, 2.0F, "high_frequency_short"});
            } else {
                flows.push_back({12000.0F, 0.25F, "low_frequency_large"});
            }
        }
        return flows;
    }

    throw std::invalid_argument("unknown traffic granularity profile: " + profile);
}

inline double expectedOfferedBytesPerTti(
    const std::vector<TrafficGranularityFlowSpec>& flows)
{
    double total = 0.0;
    for (const auto& flow : flows) {
        total += static_cast<double>(flow.packetSizeBytes) *
                 static_cast<double>(flow.arrivalRatePacketsPerTti);
    }
    return total;
}

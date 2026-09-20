/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trafficGranularityProfile.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool nearlyEqual(double lhs, double rhs)
{
    return std::fabs(lhs - rhs) < 1.0e-9;
}

}  // namespace

int main()
{
    const auto homogeneous = buildTrafficGranularityProfile("homogeneous", 36, 3000.0F, 1.0F);
    assert(homogeneous.size() == 36U);
    assert(nearlyEqual(expectedOfferedBytesPerTti(homogeneous), 108000.0));
    for (const auto& flow : homogeneous) {
        assert(flow.classLabel == "homogeneous");
        assert(nearlyEqual(flow.packetSizeBytes, 3000.0));
        assert(nearlyEqual(flow.arrivalRatePacketsPerTti, 1.0));
    }

    const auto mixed = buildTrafficGranularityProfile("mixed_granularity_v1", 36, 3000.0F, 1.0F);
    assert(mixed.size() == 36U);
    assert(nearlyEqual(expectedOfferedBytesPerTti(mixed), 108000.0));
    for (size_t ue = 0; ue < mixed.size(); ++ue) {
        if (ue % 2U == 0U) {
            assert(mixed[ue].classLabel == "high_frequency_short");
            assert(nearlyEqual(mixed[ue].packetSizeBytes, 1500.0));
            assert(nearlyEqual(mixed[ue].arrivalRatePacketsPerTti, 2.0));
        } else {
            assert(mixed[ue].classLabel == "low_frequency_large");
            assert(nearlyEqual(mixed[ue].packetSizeBytes, 12000.0));
            assert(nearlyEqual(mixed[ue].arrivalRatePacketsPerTti, 0.25));
        }
    }

    bool rejectedOddFlowCount = false;
    try {
        (void)buildTrafficGranularityProfile("mixed_granularity_v1", 35, 3000.0F, 1.0F);
    } catch (const std::invalid_argument&) {
        rejectedOddFlowCount = true;
    }
    assert(rejectedOddFlowCount);

    bool rejectedUnknownProfile = false;
    try {
        (void)buildTrafficGranularityProfile("unknown", 36, 3000.0F, 1.0F);
    } catch (const std::invalid_argument&) {
        rejectedUnknownProfile = true;
    }
    assert(rejectedUnknownProfile);

    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace cumac::online {

// Historical slots 12--15, used only by the explicit reproduction stage mode.
inline std::array<float, 4> buildLegacyDeadlineFeatures(
    float pendingBytes, float ttlSlackMs, float holDelayMs, float ttlMs, int successfulServiceAgeTti)
{
    const float ageWeight = ttlMs > 0.0F
        ? std::min(2.0F, std::max(0.0F, holDelayMs / ttlMs)) : 0.0F;
    return {{(ttlSlackMs >= 0.0F && ttlSlackMs <= 20.0F) ? pendingBytes : 0.0F,
             (ttlSlackMs >= 0.0F && ttlSlackMs <= 60.0F) ? pendingBytes : 0.0F,
             pendingBytes * ageWeight,
             static_cast<float>(successfulServiceAgeTti)}};
}

struct DeadlineServiceFeatures {
    float log1pDrainPressure = 0.0F;
    float scheduledDequeueAdequacy = 0.0F;
    float ttlNormalizedServiceAge = 0.0F;
};

inline float updateScheduledDequeueEwma(const float previousBytesPerTti,
                                        const float scheduledDequeueBytes,
                                        const float alpha = 0.05F)
{
    const float safeAlpha = std::max(0.0F, std::min(1.0F, alpha));
    const float previous = std::max(0.0F, previousBytesPerTti);
    const float dequeue = std::max(0.0F, scheduledDequeueBytes);
    return (1.0F - safeAlpha) * previous + safeAlpha * dequeue;
}

inline DeadlineServiceFeatures buildDeadlineServiceFeatures(const float queueBytes,
                                                             const float scheduledBytesPerTti,
                                                             const int ttlSlackTti,
                                                             const int serviceAgeTti,
                                                             const int packetTtlTti)
{
    DeadlineServiceFeatures result;
    const float queue = std::max(0.0F, queueBytes);
    if (queue <= 0.0F) {
        return result;
    }

    const float service = std::max(0.0F, scheduledBytesPerTti);
    const float slack = static_cast<float>(std::max(1, ttlSlackTti));
    const float drainableBytes = service * slack;
    const float drainPressure = queue / (drainableBytes + 1.0F);
    result.log1pDrainPressure = std::log1p(std::max(0.0F, drainPressure));
    result.scheduledDequeueAdequacy =
        std::max(0.0F, std::min(1.0F, drainableBytes / std::max(queue, 1.0F)));
    if (packetTtlTti > 0) {
        result.ttlNormalizedServiceAge = std::max(
            0.0F,
            std::min(
                1.0F,
                static_cast<float>(std::max(0, serviceAgeTti)) /
                    static_cast<float>(packetTtlTti)));
    }
    return result;
}

} // namespace cumac::online

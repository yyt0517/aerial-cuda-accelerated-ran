/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeadlineServiceFeatures.h"

#include <cassert>
#include <cmath>

namespace {

bool closeEnough(float lhs, float rhs, float tolerance = 1.0e-6F)
{
    return std::fabs(lhs - rhs) <= tolerance;
}

} // namespace

int main()
{
    using cumac::online::buildDeadlineServiceFeatures;
    using cumac::online::updateScheduledDequeueEwma;

    const auto empty = buildDeadlineServiceFeatures(0.0F, 3000.0F, 4, 100, 400);
    assert(empty.log1pDrainPressure == 0.0F);
    assert(empty.scheduledDequeueAdequacy == 0.0F);
    assert(empty.ttlNormalizedServiceAge == 0.0F);

    const auto adequate = buildDeadlineServiceFeatures(12000.0F, 3000.0F, 4, 100, 400);
    assert(closeEnough(adequate.log1pDrainPressure, 0.6931055F));
    assert(closeEnough(adequate.scheduledDequeueAdequacy, 1.0F));
    assert(closeEnough(adequate.ttlNormalizedServiceAge, 0.25F));

    const auto starved = buildDeadlineServiceFeatures(1200000.0F, 0.0F, 0, 800, 400);
    assert(std::isfinite(starved.log1pDrainPressure));
    assert(starved.log1pDrainPressure > 13.0F);
    assert(starved.scheduledDequeueAdequacy == 0.0F);
    assert(starved.ttlNormalizedServiceAge == 1.0F);

    assert(closeEnough(updateScheduledDequeueEwma(1000.0F, 3000.0F), 1100.0F));
    const auto legacy = cumac::online::buildLegacyDeadlineFeatures(3000.0F, 15.0F, 100.0F, 200.0F, 9);
    assert(legacy[0] == 3000.0F && legacy[1] == 3000.0F);
    assert(legacy[2] == 1500.0F && legacy[3] == 9.0F);
    const auto noDeadline = cumac::online::buildLegacyDeadlineFeatures(1000.0F, -1.0F, 800.0F, 200.0F, 17);
    assert(noDeadline[0] == 0.0F && noDeadline[1] == 0.0F);
    assert(noDeadline[2] == 2000.0F && noDeadline[3] == 17.0F);
    return 0;
}

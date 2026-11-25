/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <iomanip>
#include <ostream>

struct ServiceBalanceTraceRow
{
    int tti = 0;
    int ueId = -1;
    int servingCell = -1;
    uint64_t offeredBytes = 0;
    uint64_t acceptedBytes = 0;
    uint64_t rejectedBytes = 0;
    uint64_t expiredBytes = 0;
    uint64_t previousPostQueueBytes = 0;
    uint64_t preQueueBytes = 0;
    uint64_t scheduledDequeueBytes = 0;
    uint64_t postQueueBytes = 0;
    uint64_t successfulBytes = 0;
    uint64_t failedBytes = 0;
    uint32_t demandValidPrgCount = 0;
    uint32_t actualPrgCount = 0;
    uint32_t actualPrgMask = 0;
    uint64_t tbsCapacityBytes = 0;
    uint64_t tbsCapacityBits = 0;
    int mcs = -1;
    int layer = -1;
    uint32_t tbTxCount = 0;
    uint32_t tbErrCount = 0;
    float holDelayMs = -1.0F;
    float ttlSlackMs = -1.0F;
    int schedSlot = -1;
    float phySinrDbCpu = -999.0F;
    float predictedBlerCpu = -1.0F;
    float phySinrDbGpu = -999.0F;
    float predictedBlerGpu = -1.0F;

    bool conservationHolds() const
    {
        // Equivalent to 8 * dequeue_bytes <= raw TBS bits, without overflow.
        if (scheduledDequeueBytes > tbsCapacityBits / 8U ||
            tbsCapacityBytes != tbsCapacityBits / 8U) {
            return false;
        }
        if (offeredBytes != acceptedBytes + rejectedBytes) {
            return false;
        }
        if (previousPostQueueBytes + acceptedBytes < expiredBytes) {
            return false;
        }
        if (preQueueBytes != previousPostQueueBytes + acceptedBytes - expiredBytes) {
            return false;
        }
        if (preQueueBytes < scheduledDequeueBytes ||
            postQueueBytes != preQueueBytes - scheduledDequeueBytes) {
            return false;
        }
        return successfulBytes + failedBytes == scheduledDequeueBytes;
    }
};

inline void writeServiceBalanceTraceHeader(std::ostream& out)
{
    out << "tti,ue_id,serving_cell,offered_bytes,accepted_bytes,rejected_bytes,expired_bytes,"
           "previous_post_queue_bytes,pre_queue_bytes,scheduled_dequeue_bytes,post_queue_bytes,"
           "successful_bytes,failed_bytes,demand_valid_prg_count,actual_prg_count,actual_prg_mask,"
           "tbs_capacity_bytes,mcs,layer,tb_tx_count,tb_err_count,hol_delay_ms,ttl_slack_ms,tbs_capacity_bits,"
           "sched_slot,phy_sinr_db_cpu,predicted_bler_cpu,phy_sinr_db_gpu,predicted_bler_gpu\n";
}

inline bool writeServiceBalanceTraceRow(std::ostream& out, const ServiceBalanceTraceRow& row)
{
    if (!row.conservationHolds()) {
        return false;
    }
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrecision = out.precision();
    out << row.tti << ',' << row.ueId << ',' << row.servingCell << ','
        << row.offeredBytes << ',' << row.acceptedBytes << ',' << row.rejectedBytes << ','
        << row.expiredBytes << ',' << row.previousPostQueueBytes << ',' << row.preQueueBytes << ','
        << row.scheduledDequeueBytes << ',' << row.postQueueBytes << ',' << row.successfulBytes << ','
        << row.failedBytes << ',' << row.demandValidPrgCount << ',' << row.actualPrgCount << ",0x"
        << std::hex << std::setw(5) << std::setfill('0') << row.actualPrgMask << std::dec << ','
        << row.tbsCapacityBytes << ',' << row.mcs << ',' << row.layer << ',' << row.tbTxCount << ','
        << row.tbErrCount << ',' << std::fixed << std::setprecision(6) << row.holDelayMs << ','
        << row.ttlSlackMs << ',' << row.tbsCapacityBits << ',' << row.schedSlot << ','
        << row.phySinrDbCpu << ',' << row.predictedBlerCpu << ','
        << row.phySinrDbGpu << ',' << row.predictedBlerGpu << '\n';
    out.flags(oldFlags);
    out.precision(oldPrecision);
    return static_cast<bool>(out);
}

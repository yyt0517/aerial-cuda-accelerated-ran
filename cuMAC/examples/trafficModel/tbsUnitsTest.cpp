#include "serviceBalanceTrace.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <sstream>

int main()
{
    // Independent literal: 16 PRBs, 11 symbols, MCS0, one layer => 504 bits.
    ServiceBalanceTraceRow row;
    row.offeredBytes = row.acceptedBytes = row.preQueueBytes = 1000;
    row.tbsCapacityBits = 504;
    row.tbsCapacityBytes = 63;
    row.scheduledDequeueBytes = row.successfulBytes = 63;
    row.postQueueBytes = 937;
    assert(row.conservationHolds());
    std::ostringstream valid;
    assert(writeServiceBalanceTraceRow(valid, row));
    // Queue conservation alone accepts the original bits-as-bytes bug.
    row.scheduledDequeueBytes = row.successfulBytes = 504;
    row.postQueueBytes = 496;
    assert(!row.conservationHolds());
    std::ostringstream invalid;
    assert(!writeServiceBalanceTraceRow(invalid, row));
    // Even if the byte export shares the bug, the independent raw bits catch it.
    row.tbsCapacityBytes = 504;
    assert(!row.conservationHolds());
    row.tbsCapacityBytes = 63;
    row.scheduledDequeueBytes = row.successfulBytes = 40;
    row.postQueueBytes = 960;
    assert(row.conservationHolds());
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "OnlineProtocol.h"

namespace cumac::online {

struct StepAction {
    int32_t stepIdx = 0;
    int64_t snapshotId = 0;
    std::vector<int32_t> ueAction;
    std::vector<int16_t> prgAction;
};

struct PreActionRequest {
    int32_t tti = 0;
    std::array<uint8_t, 32> configHash {};
};

struct PreActionSnapshot {
    PreActionRspHeader header;
    std::vector<int32_t> ueIds;
    std::vector<int32_t> servingCell;
    std::vector<float> wbSinrFirstLayerDb;
    std::vector<float> ollaDelta;
    std::vector<float> ollaDeltaUp;
    std::vector<float> ollaDeltaDown;
    std::vector<int32_t> priorTbErr;
    std::vector<uint8_t> prgLayerCap;
};

struct StepState {
    StepStateHeader header;
    std::vector<float> obsCellFeatures;
    std::vector<float> obsUeFeatures;
    std::vector<float> obsPrgFeatures;
    std::vector<int16_t> obsEdgeIndex;
    std::vector<float> obsEdgeAttr;
    std::vector<uint8_t> actionMaskUe;
    std::vector<uint8_t> actionMaskCellUe;
    std::vector<uint8_t> actionMaskPrgCell;
    std::vector<float> postEqSinr;
    std::vector<int32_t> teacherActionUe;
    std::vector<int16_t> teacherActionPrg;
    std::vector<int32_t> acceptedPrePdschUePrgCount;
    std::vector<int32_t> nativeRejectWrongCellSlotCount;
    std::vector<int32_t> nativeRejectEmptySlotCount;
    std::vector<int32_t> nativeRejectPrgMaskCount;
    std::vector<int32_t> nativeRejectOobSlotCount;
    std::vector<int32_t> nativeRejectDuplicateUeCount;
    std::vector<int32_t> nativeSlotCountPerCell;
    std::vector<int32_t> nativeCellSlotStart;
    std::vector<int32_t> nativeSlotToCell;
    std::vector<int32_t> nativeSlotValid;
    std::vector<int32_t> actualUePrgCount;
    std::vector<float> actualUeServedBytes;
    std::vector<float> actualUeGoodputBytes;
    std::vector<int32_t> actualUeTbTxCount;
    std::vector<int32_t> actualUeTbErrCount;
    std::vector<int32_t> actualUePacketDeliveredPackets;
    std::vector<int32_t> actualUePacketPendingPackets;
    std::vector<float> actualUePacketDeliveredBits;
    std::vector<float> actualUePacketSystemTimeMs;
    std::vector<float> actualUePacketServiceRateMbps;
    std::vector<float> actualUePacketServiceRatePerPacketMeanMbps;
    std::vector<int32_t> actualUePacketExpiredPackets;
    std::vector<float> actualUePacketExpiredSystemTimeMs;
    std::vector<float> actualUePacketOutcomeSystemTimeMs;
    std::vector<float> actualUePacketExpiryAwareRateMbps;
    std::vector<float> actualUePacketExpiryAwareRatePerPacketMeanMbps;
    std::vector<uint8_t> acceptedPrePdschUePrgMask;
    std::vector<uint8_t> actualUePrgMask;
    std::vector<float> decisionPreWbSinrFirstLayerDb;
    std::vector<float> decisionPreOllaDelta;
    std::vector<float> decisionPreOllaDeltaUp;
    std::vector<float> decisionPreOllaDeltaDown;
    std::vector<int32_t> decisionPreTbErr;
    std::vector<uint8_t> decisionPrePrgLayerCap;
    std::vector<float> phyTbsConfig;
    std::vector<int32_t> phyTbsAuditProvenance;
    std::vector<int32_t> actualUeMcs;
    std::vector<int32_t> actualUeLayer;
    std::vector<int32_t> actualUeTbsCapacityBytes;
    std::vector<float> ctbsConfig;
};

class OnlineBridgeServer {
public:
    explicit OnlineBridgeServer(std::string socketPath);
    ~OnlineBridgeServer();

    bool initialize();

    bool recvResetReq(ResetReqPayload* req);
    bool sendResetRsp(const StepState& state);

    bool recvStepReq(StepAction* req, bool* closeReq = nullptr);
    bool sendStepRsp(const StepState& state);

    bool recvPreActionReq(PreActionRequest* req, bool* closeReq = nullptr);
    bool sendPreActionRsp(const PreActionSnapshot& snapshot);

    bool recvCloseReq();
    bool sendCloseRsp();

    bool sendError(int32_t code, const std::string& message);

private:
    std::string m_socketPath;
    int m_listenFd = -1;
    int m_clientFd = -1;

    bool readExact(int fd, void* buf, size_t len) const;
    bool writeExact(int fd, const void* buf, size_t len) const;

    bool recvMessage(MsgType expectedType, std::vector<uint8_t>* payload);
    bool sendMessage(MsgType type, const std::vector<uint8_t>& payload);

    std::vector<uint8_t> buildStatePayload(const StepState& state) const;

    void closeAll();
};

} // namespace cumac::online

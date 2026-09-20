# MCS and capacity-units audit

> GPU MCS follow-up: [fixed-input synchronization closure](../gpu_mcs_sync_v1/report.md) confirms a separate locally introduced initialization race (official Release 25-3 retained the barrier). Restoring only synchronization removes observed raw count/MCS errors; paired CPU service traces and KPIs remain identical. Training remains paused.

> 2026-09-20 follow-up: [PF/RR PHY slot-invariance and index-fix experiment](../phy_slot_invariance_v1/report.md) verifies the official example's unit-mismatch provenance separately from the locally introduced slot/UE indexing bug. Index correction substantially changes MCS and throughput; do not infer physical capacity from the earlier faulty-index results below.

Read-only investigation of the existing granularity campaign. No simulator or training changes.

## MCS distribution

All actual TB transmissions across seeds 101–110. Steady window: TTI 1001–4000. Shares pool transmissions (not equal-weight cell averages).

| Profile/policy | Full-run maximum | Steady maximum | Steady MCS 0 % | Steady TB count |
|---|---:|---:|---:|---:|
| homogeneous/dpa_v1 | 27 | 16 | 73.695 | 744173 |
| homogeneous/pfq | 27 | 20 | 76.974 | 815187 |
| homogeneous/rrq | 27 | 18 | 76.865 | 843332 |
| mixed_granularity_v1/dpa_v1 | 27 | 19 | 73.493 | 661539 |
| mixed_granularity_v1/pfq | 27 | 22 | 77.121 | 756099 |
| mixed_granularity_v1/rrq | 27 | 21 | 78.207 | 788294 |

## Why low MCS persists

CPU MCS selection caps first-layer wideband SINR at 25.99 dB, subtracts OLLA delta, and selects an index from the descending LUT. A prior TB error increases delta by 1 dB; success reduces it by 0.1 dB. There is no delta clamp. The implied zero-drift BLER is 0.1/1.1 = 9.09%, rather than exactly 10%. MCS cannot go below zero.

Seed101 DPA UE26 starts MCS12 at TTI1, experiences repeated errors, and first reaches MCS0 at TTI37. Its steady MCS0 BLER is 11.48%; UE34 steady MCS0 BLER is 16.96%. These exceed the OLLA zero-drift rate and support persistent downward selection pressure. This is not proof that all UEs have the same physical cause.

MCS selection uses first-layer wideband SINR, while the realized CPU BLER path averages the allocated PRGs and selected layers and recomputes interference. These are different inputs. A controlled same-state PHY audit is still needed to isolate the source of sustained low-SINR/error conditions; do not label this solely as poor radio coverage or solely as an OLLA bug.

## Confirmed capacity-unit mismatch

All 324,334 actual transmissions in seed101 homogeneous DPA/PFQ/RRQ have tbs_capacity_bytes equal to the independently calculated TBS bit count, not TBS/8. In 321,133 transmissions, recorded queue dequeue exceeds the correctly converted physical capacity.

Example: seed101 homogeneous DPA, TTI1001, UE0, one PRG (16 PRBs), 11 data symbols, MCS0 (Qm=2, R=120/1024), three layers. N_info=1485 bits; quantized TBS=1480 bits=185 bytes. Archived capacity, dequeue and successful payload are each 1480 in fields measured in bytes. This proves the issue affects historical execution, not just a current field name.

The C++ determineTbsPdsch function returns bits; the CPU dequeue path consumes it as bytes. scheduledTbsCapacityByActiveUe exposes the same number as byte capacity. The Python determine_tbs_bytes mirror also preserves this mismatch. Existing simulator/reconstruction parity checks therefore do not validate dimensional correctness. network::getAllocBytes separately divides by 8, providing an internal correct conversion example.

This inflates nominal byte capacity by 8x, but does not imply each observed goodput is precisely 8x or that relative policy rankings survive correction. Queue evolution, offered-load severity, actions, expiry, and learned policies are affected. Existing absolute spectral-efficiency results should not be presented as calibrated physical performance.

## Relevant source locations

- cuMAC/src/4T4R/mcsSelectionLUTCpu.cpp: constructor and type0 selection (OLLA and LUT).
- cuMAC/examples/network.cu: determineTbsPdsch, updateDataRatePdschCpu, scheduledTbsCapacityByActiveUe, getAllocBytes.
- training/stageb_hgraph/phy_tbs_alignment_audit.py: determine_tbs_bytes.

Full counts, histograms, examples and scope limits: audit.json.

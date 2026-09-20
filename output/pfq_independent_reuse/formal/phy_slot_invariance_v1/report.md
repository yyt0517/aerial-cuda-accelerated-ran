# RR/PF：PHY slot 置换不变量与索引修正对照

> 后续闭环：[GPU MCS 同步竞争验证](../gpu_mcs_sync_v1/report.md) 已用原始计数/MCS 固定输入重放确认独立竞争，恢复初始化同步后消除观测错误，CPU 业务指标不变；下文保留本轮当时的证据边界。

2026-09-20。范围：一个 DL Type-0 场景的可证伪诊断，不训练、不降低负载、不改变 RR/PF 分配规则。

## 1. 两个问题的来源不同

- **容量单位问题确实存在于官方公开的 Release 25-3 示例代码。** 已核对 NVIDIA 官方仓库 commit `654c2491461e8073321ef472b68deb116241e8a6`（提交标题 `Release 25-3`）的 [`cuMAC/examples/network.cu`](https://raw.githubusercontent.com/NVIDIA/aerial-cuda-accelerated-ran/654c2491461e8073321ef472b68deb116241e8a6/cuMAC/examples/network.cu)。官方远端内容与本地该提交的 Git blob 都为 `81da7bef68d594c06f65ec0a60d30f93e6b3764c`。`determineTbsPdsch` 返回 bit，CPU 路径却直接与 byte 队列取 min 并扣减；原文约 1860 行可见。结论限定于这一公开示例的流量/PHY 抽象接缝，不等于官方所有 PHY、所有部署均有此问题。
- **本次 slot/active-UE 索引混用不是上述官方版本原有的写法。** 本地 commit `4b05d6b30f024f936976286c6a82d7fee4faa8a9`（`goodput breakthrouh`）把 Type-0 PHY 的信道/预编码访问从 slot `ueIdx` 改成 `activeUeId`，但 `ueDownSel4TrKernel` 仍按 slot 打包这些数组。因此官方来源的 RR/PF 调度规则也会被共同的本地 PHY 执行路径污染。

本实验前后**都已经保留 TBS bits→bytes 修正**；前后比较不混入容量单位回退。

## 2. 最小变更与实验控制

修正 `network.cu` 的 CPU/GPU Type-0 PHY：信道与预编码数组使用 scheduled slot；UL 干扰信道使用 allocation 中的 interferer slot。关联、队列和随机误码样本继续使用 active UE ID。Type-1 不改。

PF 打分、RR 的平均 PRG 分配、余数分配及 shuffle 规则均未改变。额外发现原生 CPU RR 每次调用 `srand(time(0))`，仅固定业务/信道 seed 不能固定 RR 动作。因此增加默认关闭的 `CUMAC_RR_AUDIT_SEED`；本次统一为 `61 + 调用序号`，仅控制随机序列。第一轮未控制 RR 时间种子的失败尝试保留在 `before_uncontrolled_rr_attempt1/`，不纳入结论。

### A. 固定物理输入的 slot 置换

同一 seed61 的首个 TTI，先由原 PF/RR 决定动作及 CPU link adaptation，再在同 serving cell 内反转有效 slot，同步搬运 UE 映射、PRG allocation、MCS、layer，并重建 slot-packed channel/precoder。没有重新运行 scheduler/MCS。

检查 36 个 UE 的 canonical UE/PRG、队列、offered、MCS、layer、raw TBS 和 scheduled dequeue 相等；offered 文件、拓扑及 raw channel 指纹也相等。实际有 20 个 slot 改变。

为隔离 PHY，正式 `controlled_before/`、`controlled_after/` 使用 `CUMAC_PHY_AUDIT_COMMON_MCS=1`，将同一 CPU-selected MCS/layers 提供给两条 PHY 路径；这**不是**对原生 GPU link adaptation 正确性的验收。长场景不开此开关。

| policy | 修正前最大 SINR 差（CPU/GPU） | 修正后最大 SINR 差 | 修正后最大 BLER 差（CPU/GPU） | 修正后 CPU TB error/goodput 变化 UE 数 |
|---|---:|---:|---:|---:|
| PF | 4.617641 dB | 0 | 0 | 0 / 0 |
| RR | 15.621666 dB | 0 | 0 | 0 / 0 |

修正前 RR predicted BLER 最大差为 0.019088。首 TTI 的 CPU TB outcome/goodput 恰好未变化，不能把这一点误写为“物理结果无问题”：SINR 已违反不变量，而概率变化未必改变某一次随机采样结果。对应实测断言：修正前 **2 failed**，修正后 **2 passed**。

### B. 原生 PF/RR 同场景长对照

3 cell / 36 UE；seed61；strongest association、coop_boundary、omni；17 PRG/cell × 16 PRB；TTI 0.5 ms；3000-byte packet、Poisson 1 packet/UE/TTI；TTL 200 ms。`1000 warmup + 4000 measured`，统计 TTI 1001–5000。offered = **1723.44 Mbps**，没有缩小负载。RR/PF 均在 online bridge 的 native reference action 路径执行，实际业务统计来自 CPU PHY/队列路径。

两个 policy 的 before/after offered identity、topology hash、逐 TTI raw channel 指纹分别完全一致；全部 4 条轨迹共 **720,000** 个 UE-TTI 行通过独立 TBS 数值、`8 × dequeue_bytes ≤ TBS_bits` 与流量守恒检查。

| 指标 | PF 修正前 | PF 修正后 | RR 修正前 | RR 修正后 |
|---|---:|---:|---:|---:|
| Goodput Mbps | 251.56 | **903.67** | 372.55 | **1022.74** |
| Scheduled dequeue Mbps | 288.62 | 1000.19 | 419.21 | 1131.40 |
| TB BLER | 14.164% | **9.094%** | 12.234% | **9.100%** |
| Expired bytes / offered bytes | 83.257% | **41.574%** | 75.654% | **34.283%** |
| Completion delay mean ms | 198.66 | 83.23 | 50.87 | 47.72 |
| Completion delay P95 ms | 199.5 | 199.5 | 199.5 | 199.5 |
| Outcome delay mean（包含过期）ms | 199.93 | 143.09 | 183.50 | 122.21 |
| 总 backlog mean MB（十进制） | 43.049 | 30.777 | 39.499 | 26.288 |
| 总 backlog P95 MB | 43.448 | 31.235 | 39.875 | 26.714 |
| 每 TTI 正出队 UE 数 | 22.898 | 17.077 | 35.519 | 31.312 |
| PRG 利用率 | 100% | 100% | 100% | 100% |
| 实际 TB 的 MCS0 占比 | 73.326% | **0%** | 66.405% | **0.0184%** |
| 平均 MCS（按 TB 加权） | 1.132 | 9.852 | 1.615 | 6.532 |

平均/P95 completion delay 仅统计 delivered packet，不能脱离 expiry 和 backlog 解读。服务 UE 数下降也不直接等于退化：更多 UE 队列被排空时，正出队次数会下降。

## 3. 能得出的结论和剩余边界

1. 假设“仅改变内部存储排列不应改变 PHY”在坏索引版本被否证，修正后在这个受控 DL 场景成立。问题并非 RR/PF 调度规则自身必须重写。
2. 错误索引显著污染了先前的低 MCS、吞吐与承载判断。不能用修正前 252/373 Mbps 宣称当前物理系统只能承载这么多，也不能为了恢复好数字撤销 bits→bytes 修正。
3. 仍有 34–42% expiry，未验证当前负载可稳定承载。PF 修正后 raw scheduled TBS 为 2898.24 Mbps，但 demand-clipped dequeue 仅 1000.19 Mbps；RR 分别为 1309.36 和 1131.40 Mbps。raw TBS 汇总不是可持续系统容量，不能忽略用户/小区的异质性、需求匹配和 BLER，直接与总 offered 相除得出可承载结论。
4. 原始非受控 GPU 检查中，RR 曾出现同 SINR、不同 predicted BLER。审查发现 GPU `mcsSelectionLUT.cu` Type-0 `numAllocPrg` 的 shared-memory 初始化与跨 warp atomicAdd 之间缺少 barrier，是独立的竞争嫌疑；本次没有修正它，也没有声称原生 GPU MCS 已验证正确。`before/after/invariance.json` 的 input equality 只覆盖 CPU canonical 参数；完整的受控 CPU+GPU 结论以 `controlled_*` 为准。
5. UL 的对称索引修正尚未运行 UL 场景；单个 DL seed 的通过不代表所有拓扑都已验证。没有扩展训练、没有重采 BC，旧训练仍保持暂停。以前五策略容量诊断（包含旧 DPA）是在坏索引下运行，不能用作修正后系统性能。

## 4. 归档与复现

- 汇总、MCS histogram、输入 identity、四份原始 report：[`final_report.json`](final_report.json)。
- 长场景原始报告：[`before/report.json`](before/report.json)、[`after/report.json`](after/report.json)。
- 受控不变量：[`controlled_before/invariance.json`](controlled_before/invariance.json)、[`controlled_after/invariance.json`](controlled_after/invariance.json)。
- 命令、环境、窗口与 trace：各 arm 的 `command.json`、`reference_manifest.json` 和 CSV。
- 执行脚本：[`run_probe.py`](run_probe.py)；只读聚合：[`summarize_probe.py`](summarize_probe.py)。

二进制 SHA-256：

```text
long before       107b9b29ce6b722370b7f2fdb210e39f2363330ae86412a782577069a776cc2f
long after        670159b0e380e33fe76be5dc4c23d1c7892878be24262f8f66c646e2a5a38716
controlled before b692de2be360fd7ae9df8a4cdcac7ad470d4cf32e81d865d558c1944bffd0697
controlled after  2f21175cb2bd0b883c08fa8662f72be19d7bd44ec88f8e41be41f99646cc2ccf
common libcumac   67b4219dd2e623de74a5a55cceeaa5e992b667bfe7593dd8216e0e4925176ec8
```

受控检查在 `build.phy_slot_v3`，没有覆盖 `build.capacity_units_v2`。新增审计开关默认关闭；当前主二进制包含 common-MCS 控制块，原长对照的 after hash 不应误当当前主二进制 hash。

验证：受控修正后不变量 + capacity decision probe + CTBS headroom 共 **7 passed**；C++ `tbsUnitsTest`、`trafficPacketAccountingTest` **2/2 passed**。独立代码审查复核了输入控制与索引修正范围；没有进行全库回归或生产级验收。

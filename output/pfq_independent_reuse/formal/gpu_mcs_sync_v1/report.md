# GPU MCS 同步竞争：固定输入复现、最小修正与 CPU 对照

2026-09-20，研究闭环完成。未恢复训练，未重采 BC，未改变 PF/RR 调度规则、负载、TTL 或 PHY 模型。

## 结论

GPU Type-0 WB-SINR 的共享 PRG 计数确实存在初始化/累加竞争。实际仿真输入重复执行时，出现“有 PRG，却计数为 0、MCS=-1”；只恢复初始化后的同步后，PF/RR 各 128 次固定输入重放均与独立计数/MCS 参考一致。CPU 业务轨迹和所有汇总指标保持完全不变。

这里证明的是 **GPU 计数/MCS 正确性及重放稳定性改善**，不是 GPU-only 吞吐增益，也不是 DPA 性能恢复或组件增益证明。

## 来源与修正范围

官方 `Release 25-3`（commit `654c2491461e8073321ef472b68deb116241e8a6`）的 [mcsSelectionLUT.cu](https://raw.githubusercontent.com/NVIDIA/aerial-cuda-accelerated-ran/654c2491461e8073321ef472b68deb116241e8a6/cuMAC/src/4T4R/mcsSelectionLUT.cu) 在清零后、atomicAdd 前有同步。该同步被本地 commit `6c31b20aa1ce277a7906b788b5c9b15bae4485f6`（`7 site senario completed`）移除共享 allocation 缓存时一并删除。它不是官方容量单位问题的另一种表现，也不是前一轮 PHY slot 索引错误。

`before/mcsSelectionLUT.cu` 与 `after/mcsSelectionLUT.cu` 的差异**仅有两行**：WB-SINR 与 CQI Type-0 kernel 中，`numAllocPrg[threadIdx.x] = 0` 后各恢复一个无条件 `__syncthreads()`。CQI 修改对称但本轮没有执行 CQI 场景。

审计读出、固定输入重放与独立参考在两个版本中完全相同。默认不启用审计；没有插入 sleep、warp 延迟、合成竞态或调整调度算法来放大故障。两版均已包含正确的 bits→bytes 和 PHY slot 索引修正。

## 固定输入检查

采用 seed61、3 cell / 36 UE、PF 与 RR 的第一帧真实 WB-SINR 输入。第一帧有 21 个有效候选 UE，包含有分配及无分配的候选；每个 policy 重放 128 次，共 2688 条有效 UE 记录。

固定 slot→UE、cell IDs、allocations、WB SINR、prior TB error、OLLA 参数、可选波束增益及 kernel 配置。每次执行前恢复 kernel 可能修改的 OLLA、MCS 输出缓冲与 `beamformGainLastTx`；完成审计后再次恢复，然后才进行正常业务调用。不是在持续变化的 OLLA 上重复调用。

原始 PRG 计数在 kernel 现有的“累加完成”屏障之后读出；独立 host 参考直接数 allocation，再按 WB SINR、OLLA 与 LUT0 阈值计算 MCS，不使用有问题的 GPU 计数器。

| 检查 | PF 修正前 | PF 修正后 | RR 修正前 | RR 修正后 |
|---|---:|---:|---:|---:|
| 重放次数 | 128 | 128 | 128 | 128 |
| 有效 UE 记录 | 2688 | 2688 | 2688 | 2688 |
| 原始 PRG 计数错误 | 1 | **0** | 6 | **0** |
| 原始 MCS 错误 | 1 | **0** | 6 | **0** |
| 有分配但 MCS 为负的记录 | 1 | **0** | 6 | **0** |
| 同输入输出发生变化的 slot 数 | 1 | **0** | 6 | **0** |

例：PF、repeat0、slot5、UE22，分配 **12 PRG**、WB SINR **25.40284 dB**，GPU 原始计数却为 **0**、MCS 为 **−1**；独立参考为计数 **12**、MCS **26**。RR 也出现了 UE9 应为 2 PRG/MCS15，却得到 0/−1 的记录。

这些错误都出现在首次重放，其余重放可恢复正确。这正说明不能用“重复多次后正常”代替初始化时序保证；但审计会影响编译与线程调度，**不能把这里的次数除以总次数当作未插桩生产路径的故障概率**。

输入快照 SHA-256 在修正前后逐 policy 完全相等：

```text
PF 6f91c3acce68b5ecd26e7a655f4802bbddbe9e906fb0d850c9a0263756fe7254
RR 4447f2fbc19f8a98783e9c16f1eeee6ac2a8215f660087a04144c5403a02dd27
```

修复前同一个行为测试为 **2 failed**，修复后为 **2 passed**。初始 prior TB error 均为 −1、OLLA delta 为 0；本轮记录/恢复了 OLLA，但没有覆盖非零历史下的 OLLA 更新分支。

## CPU 业务指标隔离检查

每个版本分别跑 PF/RR：`1000 warmup + 4000 measured`，seed61，TTI 0.5 ms、17 PRG/cell、3000-byte Poisson 1 packet/UE/TTI、TTL200ms。measured offered 为 **1723.44 Mbps**。RR 使用已建立的固定审计随机种子61；slot 置换与 common-MCS 强制覆盖都关闭。

四条运行均使用 `CUMAC_EXEC_MODE=both`，CPU 独立选择 MCS、执行 PHY 出队并产生业务指标，不从 GPU 复制 MCS。每个 policy 前后 **180,000 行 UE-TTI 的全部非 GPU 诊断字段逐字段一致**，包括动作、MCS、层数、TBS、误码、出队、goodput、队列和 CPU SINR/BLER。所有聚合指标字典也精确相等；offered、topology、raw-channel identity 均一致。两 policy 合计验证 360,000 对行；四条轨迹合计 720,000 行通过容量单位与流量守恒检查。

| CPU measured 指标 | PF：前后相同 | RR：前后相同 |
|---|---:|---:|
| Goodput Mbps | 903.668040 | 1022.735504 |
| BLER | 9.093970% | 9.100345% |
| Expired bytes / offered bytes | 41.573997% | 34.283132% |
| Completion delay mean ms | 83.225186 | 47.719932 |
| Completion delay P95 ms | 199.5 | 199.5 |
| 总 backlog mean MB（十进制） | 30.776991 | 26.288239 |

长轨迹记录的 GPU predicted BLER 也没有发生变化。这些长运行在首帧先做了重放预检，存在温启动效应；因此不能把它解读为“未插桩 GPU 路径没有风险”或“GPU 性能提升为零”。本次可确认的是首帧固定输入下错误被消除，以及 CPU 路径没有受到性能干预。没有测量 GPU-only goodput 增益。

## 验证与归档

- 原始重放：`before/{pf,rr}/gpu_mcs_replay.csv`、`after/{pf,rr}/gpu_mcs_replay.csv`；对应 `.inputs` 为输入快照。
- 汇总及逐项配对检查：[`final_report.json`](final_report.json)，状态：[`status.json`](status.json)。
- 运行及环境：各 policy 的 `command.json`、`reference_manifest.json`、service CSV。
- 可复核源码、实际 simulator 与 `libcumac.so` 均归档于 `before/` 和 `after/`；共享库哈希必须随二进制记录，不能只看 executable hash。
- 执行：[`run_probe.py`](run_probe.py)；断言式聚合：[`summarize_probe.py`](summarize_probe.py)。

```text
executable（两版相同） b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644
before libcumac       bff73e78746333e3cc338e3fa8fedcafdc90bd76fb45a834828ad4b98cf6c81d
after libcumac        079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e
```

新构建目录 `build.gpu_mcs_sync_v1`，没有替换旧实验构建和归档数据。若重放旧版本，必须同时使用对应的归档 `libcumac.so`；原命令中的可变构建目录不能独自标识历史版本。

验收：修复后 GPU 重放测试、capacity decision probe、CTBS headroom 共 **7 passed**；C++ `tbsUnitsTest`、`trafficPacketAccountingTest` **2/2 passed**；逐行配对聚合断言通过，独立代码审查未发现阻塞问题。按研究任务范围未进行全库/生产级回归。

停止条件已满足：本轮不继续调负载、不扩展多 seed、不恢复 DPA 训练。剩余高 expiry 不能归因于这个 GPU-only 同步缺陷已经得到本场景的直接支持；如何改善承载与调度需要另一个实验问题。

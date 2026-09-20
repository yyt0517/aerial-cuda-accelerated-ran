# Stage-B 修正环境：验证与实验报告

> 更新于 2026-09-20。旧训练和性能结论撤出当前报告；[完整旧报告原文](archive/pre_correction_2026_09_20/originals/Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)保留历史观察。
> 版本与来源以[重大变更记录](stageB_corrected_environment_v1.md)为准，不把修正前后差异归因为调度算法增益。

## 1. 三项修正的验证结论

1. TBS bits→bytes：修正容量、出队、统计及 Python CTBS，增加独立数值和流量守恒检查。[容量审计](../output/pfq_independent_reuse/formal/mcs_capacity_units_audit_v1/report.md)。
2. PHY slot 索引：固定 canonical UE 动作，仅置换内部 slot，修正后受控 DL CPU/GPU SINR、BLER 不变。seed61 同输入 PF goodput 251.560→903.668 Mbps，RR 372.552→1022.736 Mbps；这是执行错误修正的效果，不是 PF/RR 调度规则变化。[完整对照](../output/pfq_independent_reuse/formal/phy_slot_invariance_v1/report.md)。
3. GPU MCS 初始化同步：每 policy 128 次首帧固定输入重放，PF/RR 原始计数和 MCS 错误分别从 1/6 条降至 0/0。CPU 对照所有业务指标及逐行轨迹不变；未测 GPU-only goodput 增益。[同步闭环](../output/pfq_independent_reuse/formal/gpu_mcs_sync_v1/report.md)。

这些验证限定于所测模式；CQI、UL 及更大拓扑不能据此声称完成性能验证。

## 2. 三项修正后的 seed42 PFQ/RRQ

相同 topology、offered stream 和 raw channel；1000 warmup + 4000 measured，TTI 0.5 ms，3 cell / 36 UE，TTL 200 ms。实际 offered 为 **1729.548 Mbps**。正常运行路径，没有 slot 置换、common-MCS 覆盖或 GPU 固定输入重放。业务指标来自 CPU 路径，goodput 为三个小区总计。

| measured 指标 | PFQ | RRQ |
|---|---:|---:|
| Goodput Mbps | 1388.112544 | 1197.038240 |
| Goodput / offered | 80.2587% | 69.2110% |
| Scheduled dequeue Mbps | 1528.722224 | 1323.872308 |
| TB BLER | 8.9579% | 8.9768% |
| Expired bytes / offered bytes | 10.5367% | 22.7203% |
| Completion delay mean ms | 49.3977 | 55.8018 |
| Completion delay P95 ms | 199.0 | 199.5 |
| 总 backlog mean MB | 16.0776 | 22.6772 |
| 总 backlog P95 MB | 17.2311 | 23.3470 |
| 每 TTI 正出队 UE 数 | 32.5635 | 32.35575 |
| PRG 利用率 | 100% | 100% |

每组 180,000 行、共 360,000 个 UE-TTI 通过独立 TBS 数值、bits→bytes 上限与流量守恒检查；配对输入及运行前后构建哈希一致。完整结果、命令和身份见 [seed42 报告](../output/pfq_independent_reuse/formal/seed42_pfq_rrq_corrected_v1/report.md)及 [JSON](../output/pfq_independent_reuse/formal/seed42_pfq_rrq_corrected_v1/final_report.json)。

本 seed 下 PFQ goodput 比 RRQ 高 191.074304 Mbps，但两者仍有过期和 backlog，不能宣称该 offered load 已稳定承载。表中 completion delay 的历史字段名保留：它实际来自 MAC 出队完成的 packet tracker `delivered` 记录，没有按 TB 成功筛选，不是端到端成功交付时延，也不包含未完成包。单 seed 不能证明普遍排序或最大可达容量。此处没有修正环境的新 DPA 对照。

## 3. 旧结果的地位

seeds41–50 的 DPA/PFQ、seeds101–110 packet granularity、随机初始化等预算消融、checkpoint 适应、冻结 zero-shot 删除、Full 复现/BC 及仅修单位的中间实验均已归档。旧模型优势及 attention/deadline 的正负贡献均不能作为当前结论。

历史结果用于理解旧环境与排查，不抹除原始数字，也不把所有失败都归咎于三个 bug。旧 checkpoint 只能作为明确标注的历史或迁移诊断对象。[文档/数据归档入口](archive/pre_correction_2026_09_20/README.md)。

## 4. 当前下一步

重新校准场景并冻结新协议 → 新采集 BC → 修正环境 Full 训练与 argmax 验收 → 同流程成对消融。禁止缩放旧轨迹后复用，禁止自动恢复旧队列。验收须覆盖 goodput、expiry、尾时延、弱 UE 和 backlog，不继承错误容量下的旧绝对门槛。

本轮是归档和文档更新，未启动这条新训练流程。

# 当前 Stage-B 修正环境配置

> 更新于 2026-09-20。详见[三项修正与有效性分界](stageB_corrected_environment_v1.md)。
> “当前”指下列已测试研究环境，不表示有一个已验收的当前 DPA 模型。

## 代码与模型身份

环境版本：`stageb_corrected_2026_09_20_v1`，构建：`build.gpu_mcs_sync_v1`。

| 对象 | SHA-256 / 标识 |
|---|---|
| simulator | `b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644` |
| libcumac.so | `079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e` |
| Git 基底 | `f567a839e07ba5f622b54fbacd388ecf9427411e`，dirty，必须叠加[版本快照](../output/pfq_independent_reuse/versions/stageb_corrected_2026_09_20_v1/README.md) |
| DPA | 尚无修正环境中新训练并验收的 checkpoint；旧模型仅作历史参考/迁移诊断 |

旧 `build.capacity_units_v2` 仅修单位，不能作为完整修正版；旧训练脚本默认路径也不能代替运行时 hash 验证。本轮不修改启动默认值，不恢复旧队列。

## 已完成 seed42 校准点

- 3 cell / 36 UE；coop_boundary、strongest association、omni、Rayleigh、DL Type-0、SVD。
- 每 cell 17 PRG × 16 PRB；TTI 0.5 ms。
- topology/traffic seed42；3000-byte packet，Poisson 1 packet/UE/TTI，homogeneous，TTL 200 ms。
- 1000 warmup + 4000 measured，总计 5000 TTI；统计 TTI 1001–5000，共 2 秒。
- PFQ `-b 2`、RRQ `-b 3`；RR audit seed42 固定随机序列，不改变调度规则。
- `CUMAC_EXEC_MODE=both`，业务指标来自 CPU PHY/队列；slot permutation、common-MCS、GPU fixed-input replay 均关闭。
- offered/topology/raw-channel identity 两组配对一致；实际 offered 为三个小区合计 1729.548 Mbps。
- 这是一个单 seed 基线点，不是以后所有负载/期限压力场景的默认验收标准。

[执行命令、配置及完整结果](../output/pfq_independent_reuse/formal/seed42_pfq_rrq_corrected_v1/report.md)。

## 单位及指标

- PHY TBS 原始量是 bits，在消费边界转换为 bytes；队列、出队、统计导出和 Python CTBS 使用一致的 byte 语义。
- 不变量：`8 * scheduled_dequeue_bytes <= raw_TBS_bits`。独立数值重建和流量守恒同时检查，不能只比较两个共享错误公式的实现。
- 内部 average-rate 保留 bit/s 语义；不能无差别对所有字段再除以 8。
- goodput 为成功业务字节 × 8 / measured 时间，三个小区合计，单位 Mbps。
- expiry 为 expired bytes / offered bytes；不是包数比。
- 现有 completion delay 来自 packet tracker 的 `delivered` 记录，但该标记表示 MAC 出队完成，没有按 TB 成功筛选；准确名称为 MAC dequeue-completion delay，不是端到端成功交付时延。必须同时报告成功 goodput、失败出队、expiry 和 backlog。
- backlog 为总队列量；报告 MB 采用十进制，P95 与均值分别统计。
- 源码现有 BC schema `independent-bc-ctbs-physical-bytes-v2` 只约束数据契约，不能证明采集已使用三项修正；还须校验采集构建身份。

## 数据使用约束

旧数据、旧 checkpoint、旧归一化状态与旧门槛均不默认复用。新训练必须重新采集 BC、重新校准基线并冻结协议；不得将旧轨迹字段除以 8 充当新数据。原始历史配置保存在[归档原文](archive/pre_correction_2026_09_20/originals/Doc/current_stageB_effective_configuration.md)。

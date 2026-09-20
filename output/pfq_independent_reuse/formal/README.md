# Formal 输出索引 — 2026-09-20 有效性分界

三项关键错误修正后，本目录中的旧 campaign 已迁往 `../archive/pre_correction_2026_09_20/formal/`，原名现在是兼容符号链接。**保留旧路径不代表数据有效或允许续跑。**

当前保留的四个实体目录：

- [mcs_capacity_units_audit_v1](mcs_capacity_units_audit_v1/report.md)：容量单位历史审计；含修正前证据。
- [phy_slot_invariance_v1](phy_slot_invariance_v1/report.md)：slot 不变量与索引前后对照；含 before/after。
- [gpu_mcs_sync_v1](gpu_mcs_sync_v1/report.md)：GPU 原始计数/MCS 同步闭环；含 before/after。
- [seed42_pfq_rrq_corrected_v1](seed42_pfq_rrq_corrected_v1/report.md)：三项修正后正常路径 PFQ/RRQ 基线，1000 warmup + 4000 measured。

前三者是故障证据目录，不可将其所有子运行合并为修正后基线。当前尚无修正环境中新训练并验收的 DPA 模型。

[重大变更记录](../../../Doc/stageB_corrected_environment_v1.md)；[归档数据清单](../archive/pre_correction_2026_09_20/manifest.json)。新实验必须新建 campaign 并记录完整修正构建身份。本次未启动训练或仿真。

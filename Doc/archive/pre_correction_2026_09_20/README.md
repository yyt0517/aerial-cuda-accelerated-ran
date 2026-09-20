# 2026-09-20 三项修正前历史归档

本归档用于追溯，不用于当前性能证明。见[有效性分界与修正来源](../../stageB_corrected_environment_v1.md)。旧文件中的“当前”“有效”“通过”只表示当时记录。

## 保存内容

修改前的 50 份文档原文位于 `originals/`，包括：

- [旧综合实验报告](originals/Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)
- [旧 DPA 主设计](originals/docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)
- [旧有效配置](originals/Doc/current_stageB_effective_configuration.md)
- [等预算消融](originals/Doc/dpa_equal_budget_ablation_experiment.md)
- [checkpoint 适应消融](originals/Doc/dpa_checkpoint_adaptation_ablation.md)
- [Full 复现 preflight](originals/Doc/dpa_training_reproduction_ablation_v1_preflight.md)、[协议 JSON](originals/Doc/dpa_training_reproduction_ablation_v1_protocol.json)、[训练历史 JSON](originals/Doc/dpa_training_reproduction_ablation_v1_history.json)
- [仅容量单位迁移的中间记录](originals/Doc/capacity_units_bytes_v2_migration.md)
- 其他早期规格、执行计划、README 和平台/时延报告。

[文档清单](manifest.json)记录每份原文的 SHA-256。原文不修改；其中的相对引用按历史上下文理解，当前导航以本索引为准。

[归档核验记录](verification.json)：迁移后逐文件重算全部数据与原文 SHA-256，验证兼容链接、旧 checkpoint、96 份源码快照及构建身份；当前文档和新增历史告示的链接通过存在性检查。未重新运行仿真或训练。

旧 formal 中 70 个顶层条目已迁移到[实验数据归档](../../../output/pfq_independent_reuse/archive/pre_correction_2026_09_20/README.md)。7,780 个普通文件、12,374,575,291 bytes，全部保留，包括日志、原始轨迹、BC、checkpoint、成功和失败尝试。原 formal 路径保留兼容符号链接；没有删除历史数据。归档仍在同一工作区，不等于异地备份。

特别需要保留的研究链：

- 冻结 DPA 的 BC/多阶段 PPO、dimensionless seed45 iteration8 与 checkpoint 身份；
- 标准 seeds41–50 paired、packet-granularity seeds101–110；
- 随机初始化等预算、旧 Full 最小诊断、适应删除与冻结 zero-shot 删除；
- 生命周期复现、教师/CTBS 对齐及容量单位单独修复的中间尝试。

## 使用限制

可以只读回溯训练流程、比较旧实现、复现故障，或把旧 checkpoint 作为明确标注的迁移诊断对象。不能将旧数据当作修正环境的数据，不能缩放后复用轨迹，不能自动恢复暂停队列。

`archive_history.py` 是本次执行的一次性归档脚本，不是训练入口；它拒绝覆盖现有归档。数据和版本完整清单在各自输出目录；Git 跟踪策略可能忽略 output，大数据需另行备份。

当前资料：[修正版本配置](../../current_stageB_effective_configuration.md)、[当前结果](../../stageB_boundary_intelligent_algorithm_experiment_report.md)。

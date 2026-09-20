# Stage-B 文档入口

> 2026-09-20：三项关键错误已修正。此前的训练、性能、消融和泛化结果已归档，不再作为修正环境的有效性能证据。当前尚无重新训练并验收的 DPA。

## 当前入口

- [重大变更与实验有效性分界](stageB_corrected_environment_v1.md)：错误来源、修正证据、归档政策与新实验要求。
- [修正环境配置](current_stageB_effective_configuration.md)：构建哈希、单位和 seed42 协议。
- [当前综合实验报告](stageB_boundary_intelligent_algorithm_experiment_report.md)：修正验证与新的 PFQ/RRQ 基线。
- [DPA 算法结构说明](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)：保留算法契约，不代表历史 checkpoint 已在新环境验收。
- [训练代码说明](../training/stageb_hgraph/README.md)。

## 历史入口

- [修正前文档与数据归档](archive/pre_correction_2026_09_20/README.md)
- [更早研究归档](archive/README.md)
- [旧等预算消融](dpa_equal_budget_ablation_experiment.md)
- [旧 checkpoint 适应消融](dpa_checkpoint_adaptation_ablation.md)
- [旧训练复现计划](dpa_training_reproduction_ablation_v1_preflight.md)

旧文件中的“当前”“通过”“冻结”等均属于当时记录，不能覆盖本次有效性分界。新实验必须使用修正版代码和新采集的数据；本次未启动实验。

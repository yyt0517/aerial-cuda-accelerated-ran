# Stage-B 项目文档

更新至 **2026-10-21**。当前研究主线是 `3cell / 36UE / boundary / rate0.8` 下，修正物理模型与输入时序后，对冻结的 PFQ、DPA 和 current 策略进行同条件比较。

`current` 指 **PFQ + CTBS 裁剪 + 冻结定向协调器 + 服务缺口加权局部搜索**。它不是旧 v9c/v10 checkpoint，也不是已经完成通用部署的 PPO 模型。最新实现位于独立实验 worktree；本次整理同步文档和小型证据，不代表实验代码已合入 `main`。

## 阅读顺序

1. [当前进展与结论](stageB_current_progress.md)：版本演进、十 seed 结果、证据边界与剩余问题；项目进展的唯一维护入口。
2. [当前配置与指标口径](current_stageB_effective_configuration.md)：冻结策略、场景、窗口、版本身份和本地复现入口；配置的唯一维护入口。
3. [2026-10-21 证据快照](evidence/2026-10-21/README.md)：逐 seed/UE 数据、协议、验证记录、构建/源码哈希与原始路径。
4. [历史归档](archive/README.md)：旧 HGraph、基线、设计和机制实验；只用于追溯。

## 维护约定

- 当前结果只在进展文档维护，配置只在配置文档维护；训练与脚本 README 链接到这两个入口。
- 历史报告中的“当前”“最好”“已通过”按其原始版本理解，不继承为新环境结论。
- [历史计划索引](../docs/superpowers/README.md)保留设计判据和实施记录；计划复选框不等于实际完成状态。
- `output/`、训练 checkpoint、`deliverables/` 的压缩包和导出文件留在本地；长期引用的小型证据进入 `Doc/evidence/`，记录来源和 SHA-256。
- 合并去向见[归档索引](archive/README.md)。独有负结果、逐 seed 数据和审计证据保留，重复摘要、写作底稿与重复导航删除。

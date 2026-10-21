# 历史设计与实施计划

这里保留2026年7—8月的Stage-B设计判据、实施步骤和命令。**当前项目路线与有效实验结论统一见 [Doc/stageB_current_progress.md](../../Doc/stageB_current_progress.md)，当前配置见[配置说明](../../Doc/current_stageB_effective_configuration.md)。**

本目录是过程归档。`specs/`说明假设、边界和验收条件，`plans/`记录实施接口与步骤，二者不是仅按同名即可删除的重复文件。已有实现的计划也可能仍为未勾选状态，不用复选框推断研究进度；完成状态须由代码与产物核验。

| 日期/主题 | 保留理由 | 当前定位 |
|---|---|---|
| 07-17 baseline verification / CUDA rerun、07-21 dual-package | 报告结构、交付方法、归档哈希 | 历史报告和交付记录 |
| 07-30 HGraph architecture / delivery / causal evidence | 图结构、因果动作/证据、实施边界 | 旧HGraph探索路线 |
| 07-30 PFQ residual / canonical parity / adaptive bucket / PRG ablation | 分支实验判据、接口与否证过程 | 独立历史实验，不自动续跑 |
| 08-12 PFQ-v2 marginal delivery | 数学假设、固定实验与实现步骤 | 独立研究分支 |
| 08-13 independent reuse actor / blank diagnostics | 独立策略与blank诊断协议 | 8月代码与诊断过程 |
| 08-14 hierarchical decode / rescue cohort | 条件解码与cohort对账判据 | 8月后续机制实验 |

旧命令可能引用生成产物、旧checkpoint或修正前环境，不能作为10月修正版的直接复现命令。旧报告路径已统一指向[合并后的历史资料](../../Doc/archive/README.md)。

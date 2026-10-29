# Stage-B 项目文档

更新至 **2026-10-29**。当前有效算法是 **PFQ＋CTBS＋因果服务缺口加权物理局部搜索，出队下界 ρ=0.90**；boundary/rate0.8 与 uniform/rate1.0 的五 seed 长窗口对照均已完成。神经路径已完成若干成本与质量试验，尚未替换有效算法。

## 主文档

1. [当前进展与结论](stageB_current_progress.md)：维护已完成工作、有效结果、未通过门槛与下一步。
2. [当前配置与指标口径](current_stageB_effective_configuration.md)：维护版本身份、两场景协议、运行依赖及统计定义。
3. [算法注册表](../training/stageb_hgraph/current_algorithm.json)与[启动入口](../cuMAC/scripts/run_stageB_current_algorithm.sh)：当前 boundary 研究入口；完整运行仍依赖本地研究 worktree 与 CUDA 容器。

## 按日期查证据

| 日期 | 内容 |
|---|---|
| [2026-10-29](evidence/2026-10-29/README.md) | 过程价值蒸馏、近似物理残差、增量缓存、搜索收益诊断及联合修复 PHY 补表 |
| [2026-10-28](evidence/2026-10-28/README.md) | uniform 五 seed、风险直接修复、解码缓存等价降本、配对机理与前缀验证 |
| [2026-10-27](evidence/2026-10-27/README.md) | ρ0.90 选型、boundary 五 seed、外部下界负结果及联合块修复 |
| [2026-10-21](evidence/2026-10-21/README.md) | 历史含协调器版本三臂十 seed、物理修正、跨版本比较及身份快照 |

[历史归档](archive/README.md)保存独有设计、负结果、旧实验和本次合并去向；[历史计划](../docs/superpowers/README.md)中的复选框不代表当前实现或验收状态。

## 维护约定

- 进展只在进展文档持续维护，配置只在配置文档持续维护；其它 README 提供入口，冻结报告不追写为“最新结论”。
- 历史报告中的 `current`、“最好”和 `passed` 按原版本及检查范围解释，不能继承为新环境、质量晋级或生产就绪结论。
- `output/`、`deliverables/`、checkpoint、大轨迹与演示导出留在本地；长期引用的小型证据进入 `Doc/evidence/`，保留来源及 SHA-256。来源清单可能同时列出未收录的大工件，具体见日期索引。
- 删除已被最终报告覆盖的临时摘要，保留独有协议、逐 seed/UE 数据、负结果及审计记录。实际合并去向见[归档索引](archive/README.md)。

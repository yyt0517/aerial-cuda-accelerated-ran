# Stage-B 项目文档

更新至 **2026-11-25**。最新二阶段研究交付为 **DA-PLS（M-LWDF-PF 权重＋精确剪枝局部搜索）**，已完成十 seed 主场景、G1–G4 泛化、七小区扩展及 RSRP/协方差输入简化研究。注册默认入口仍为 10 月 27 日的 PFQ＋CTBS＋缺口权重、ρ=0.90 版本；报告实验不做 CTBS 裁剪，两者分别解释。

十 seed、同 10 秒到达窗口的 Boundary/rate0.8 对比：DA-PLS 完整包吞吐 **749.497→1257.499 Mbps（+67.78%）**，成功包平均时延 **121.577→2.624 ms（−97.84%）**，P95 **200→5.15 ms**，十条轨迹均零过期。七小区和邻区输入简化未保持该结论；负结果与成本边界见[当前进展](stageB_current_progress.md)。本次仅整理既有实验，没有重跑完整 GPU 实验或更换默认算法。

推荐从[最新证据索引](evidence/2026-11-25/README.md)和[二阶段最小交付包](evidence/2026-11-25/minimal_delivery/README.md)开始；后者支持标准库数值复核及 CPU 离线状态重放。旧短窗结果和已过期计划移入[截至 11 月 4 日的进展快照](archive/progress/stageB_progress_20261104.md)。

## 主文档

1. [当前进展与结论](stageB_current_progress.md)：维护已完成工作、有效结果、未通过门槛与下一步。
2. [当前配置与指标口径](current_stageB_effective_configuration.md)：维护版本身份、两场景协议、运行依赖及统计定义。
3. [算法注册表](../training/stageb_hgraph/current_algorithm.json)与[启动入口](../cuMAC/scripts/run_stageB_current_algorithm.sh)：当前 boundary 研究入口；完整运行仍依赖本地研究 worktree 与 CUDA 容器。

## 按日期查证据

| 日期 | 内容 |
|---|---|
| [2026-11-25](evidence/2026-11-25/README.md) | 归档 11-04 至 11-23 的十 seed 收尾、指标修订、泛化、七小区与 RSRP 研究；包括完整最小交付包、报告辅助数据和来源哈希 |
| [2026-11-04](evidence/2026-11-04/README.md) | [M-LWDF-PF 同窗口 PFQ 对比](evidence/2026-11-04/mlwdf_pfq_boundary_comparison/report.txt)：完整包吞吐+68.96%、平均成功时延−97.76%、三seed零过期；推理加速仅有相对完整DPLS的旧权重回放证据。[跨负载网格、文献规则与条件浅树](evidence/2026-11-04/crossload_weights_v2/report.txt)保留全部结果、负结果与计时边界 |
| [2026-11-03](evidence/2026-11-03/README.md) | 最新[零PHY容量上界与规范惰性精评诊断](evidence/2026-11-03/certified_bounds/report.txt)：上界排序条件评分次数界23738，未实跑惰性PHY/墙钟；此前精确缓存等价通过，守卫cap4闭环失败保留 |
| [2026-11-02](evidence/2026-11-02/README.md) | 排序、轨迹/预算、资源条件、连续弱服务及定向重排；定向路径开发三门槛通过，验证服务/成本仍失败，下一步候选级服务风险选择与同根候选/代理缓存 |
| [2026-10-30](evidence/2026-10-30/README.md) | 跨时隙服务选择 PPO、包龄奖励续训及物理教师少候选实验 |
| [2026-10-29](evidence/2026-10-29/README.md) | 过程价值蒸馏、近似物理残差、增量缓存、搜索收益诊断及联合修复 PHY 补表 |
| [2026-10-28](evidence/2026-10-28/README.md) | uniform 五 seed、风险直接修复、解码缓存等价降本、配对机理与前缀验证 |
| [2026-10-27](evidence/2026-10-27/README.md) | ρ0.90 选型、boundary 五 seed、外部下界负结果及联合块修复 |
| [2026-10-21](evidence/2026-10-21/README.md) | 历史含协调器版本三臂十 seed、物理修正、跨版本比较及身份快照 |

[历史归档](archive/README.md)保存独有设计、负结果、旧实验和本次合并去向；[历史计划](../docs/superpowers/README.md)中的复选框不代表当前实现或验收状态。

## 维护约定

- 进展只在进展文档持续维护，配置只在配置文档持续维护；其它 README 提供入口，冻结报告不追写为“最新结论”。
- 历史报告中的 `current`、“最好”和 `passed` 按原版本及检查范围解释，不能继承为新环境、质量晋级或生产就绪结论。
- `output/`、`outputs/`、`deliverables/`、checkpoint、大轨迹与演示导出留在本地；长期引用的小型证据进入 `Doc/evidence/`，保留来源及 SHA-256。来源清单可能同时列出未收录的大工件，具体见日期索引。
- 删除已被最终报告覆盖的临时摘要，保留独有协议、逐 seed/UE 数据、负结果及审计记录。实际合并去向见[归档索引](archive/README.md)。

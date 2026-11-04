# Stage-B 项目文档

更新至 **2026-11-04**。当前有效算法是 **PFQ＋CTBS＋因果服务缺口加权物理局部搜索，出队下界 ρ=0.90**；boundary/rate0.8 与 uniform/rate1.0 的五 seed 长窗口对照均已完成。神经路径已完成若干成本与质量试验，尚未替换有效算法。

最新 Boundary/rate0.8 研究候选为 **M-LWDF-PF 权重＋上界剪枝的精确 DPLS 内层**。与 PFQ 的同三 seed、同 2 秒到达窗口比较，完整包吞吐 **742.516→1254.580 Mbps（+68.96%）**，成功包平均时延 **122.502→2.743 ms（−97.76%）**，P95 **200→5.5 ms**，过期率 **31.754%→0%**。三个 seed 均零过期；108/108 个匹配 UE 的吞吐和平均成功时延改善。它是经典时延权重与精确启发式搜索的组合，没有模型训练；高负载不保持最优，未替换注册表中的长窗口默认版本。[完整同窗口对比及计时边界](evidence/2026-11-04/mlwdf_pfq_boundary_comparison/report.txt)。

上界剪枝相对完整 DPLS 的公平串行加速为 **2.78/2.30/2.06 倍**（×1/×1.4/×1.6 的旧 κ0 保存输入）；缺少 M-LWDF-PF 与 PFQ 的同口径串行计时，不能将此写成相对 PFQ 的推理加速。

## 主文档

1. [当前进展与结论](stageB_current_progress.md)：维护已完成工作、有效结果、未通过门槛与下一步。
2. [当前配置与指标口径](current_stageB_effective_configuration.md)：维护版本身份、两场景协议、运行依赖及统计定义。
3. [算法注册表](../training/stageb_hgraph/current_algorithm.json)与[启动入口](../cuMAC/scripts/run_stageB_current_algorithm.sh)：当前 boundary 研究入口；完整运行仍依赖本地研究 worktree 与 CUDA 容器。

## 按日期查证据

| 日期 | 内容 |
|---|---|
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
- `output/`、`deliverables/`、checkpoint、大轨迹与演示导出留在本地；长期引用的小型证据进入 `Doc/evidence/`，保留来源及 SHA-256。来源清单可能同时列出未收录的大工件，具体见日期索引。
- 删除已被最终报告覆盖的临时摘要，保留独有协议、逐 seed/UE 数据、负结果及审计记录。实际合并去向见[归档索引](archive/README.md)。

# Stage-B 当前项目进展

截至 **2026-10-21**，项目已从旧 HGraphPPO checkpoint 优化，推进到**修正仿真环境下的多小区联合资源搜索评估**。当前结果支持冻结 current 策略在指定 boundary 场景和切换窗口内优于同环境 PFQ 与 DPA；稳态容量、尾部可靠性、学习组件的独立贡献与通用部署仍未完成验证。

## 1. 当前版本

`current` 的执行链为 **PFQ → CTBS 裁剪 → 冻结定向协调器 → 服务缺口加权局部搜索**。DPA 对照为冻结 BC actor 加同一协调器，current 不执行该 BC actor。整体改进不能称为纯 PPO 或纯神经网络收益。

最新有效实验 `boundary08_corrected_physics_20261020_v1` 完成 PFQ、DPA、current 三臂 × seeds 41–50，共30条轨迹。三臂均使用实际服务链路 SVD、显式 3 dB 阴影和同一步动作前输入；没有重训或调参。

最新实现仍位于 `.worktrees/pfq-independent-reuse-actor`，包含未提交修正及 `output/` 内专用研究代码，尚未合入主仓库。本次提交收口现状、数据和版本身份，不代表代码集成完成。详见[配置与版本边界](current_stageB_effective_configuration.md)。

## 2. 研究进度与有效性分界

| 阶段 | 已获得的资产或证据 | 当前定位 |
|---|---|---|
| 6—7 月 HGraphPPO | v9c–v19 训练、候选动作、ONNX 与多 seed 评测链路 | 历史实现与机制参考；旧窗口/基线不完全配对，不继承旧绝对收益 |
| 7—8 月 PFQ residual / independent actor | 因果动作对齐、blank、hierarchical decode、rescue cohort 诊断 | 保留代码与否证过程；计划状态不等于验收状态 |
| 9 月环境修正 | 容量 bit/byte、PHY slot 索引、GPU MCS 同步；实验分支提交 `7f3b700` | 修正前性能退出当前证据，不能按常数缩放复用旧轨迹 |
| 10 月策略演进 | 有向协调器、联合候选搜索、PFQ+CTBS 初选、缺口加权与搜索加速 | 保留冻结身份；75% boundary 等旧场景不与本轮混合 |
| 10 月20—21日复核 | 实际关联 SVD、阴影 dB→ln 单位、fresh preaction；30条新轨迹及统计/数值核验 | 当前主结果，仅支持已测场景和窗口 |
| 10 月21日历史对照 | 旧 PFQ 与修正 current 共20条轨迹重算，另复用已核验修正 PFQ | 描述整个版本变化，与同环境算法比较分开 |

旧阴影实现将 `3` 当作 lognormal 自然对数标准差，约等于 13.03 dB；修正版明确为 3 dB，会改变传播模型和服务小区，不保证 PFQ 分数上升。SVD 与时序问题属于本地扩展仿真/接口路径，不能概括为 PFQ 公式的三个错误。

## 3. 当前同环境比较

3小区、36 UE、全 boundary、每小区17 PRG、arrival rate 0.8；每臂4000 TTI共同PFQ预热、2000 TTI测量、400 TTI随访。下表为 **10 seed 等权均值 ± 样本标准差**；完整设置见[当前配置](current_stageB_effective_configuration.md)。

| 指标 | 修正 PFQ | 修正 DPA（full） | 修正 current |
|---|---:|---:|---:|
| 分段 goodput / Mbps | 994.97 ± 50.59 | 1046.82 ± 51.49 | **1398.20 ± 23.20** |
| 完整包 payload / Mbps | 752.41 ± 68.27 | 808.68 ± 63.60 | **1212.88 ± 19.00** |
| 整包成功率 / % | 54.34 ± 4.95 | 58.40 ± 4.59 | **87.59 ± 1.38** |
| 成功包平均时延 / ms | 118.22 ± 19.49 | 103.09 ± 12.05 | **50.68 ± 12.79** |
| 包过期率 / % | 31.17 ± 6.12 | 27.30 ± 5.78 | **2.60 ± 1.41** |
| 小区–PRG占用 / % | 99.88 ± 0.34 | 97.91 ± 0.40 | **78.79 ± 2.41** |

来源：[逐 seed 数据](evidence/2026-10-21/corrected/per_seed_metrics.csv)、[完整均值与标准差](evidence/2026-10-21/corrected/mean_std.csv)。

| 配对比较 | 均值之比增益 | current − baseline / Mbps | 名义95% t区间 / Mbps | 正向seed |
|---|---:|---:|---:|---:|
| 对PFQ：分段goodput | +40.53% | +403.23 | [366.08, 440.37] | 10/10 |
| 对PFQ：完整包payload | +61.20% | +460.47 | [414.16, 506.78] | 10/10 |
| 对DPA：分段goodput | +33.57% | +351.38 | [312.93, 389.83] | 10/10 |
| 对DPA：完整包payload | +49.98% | +404.21 | [359.10, 449.32] | 10/10 |

来源：[配对区间](evidence/2026-10-21/corrected/paired_mean_ci.csv)。它们支持完整冻结策略在本场景的收益，没有隔离协调器、CTBS、搜索或缺口权重各自的因果贡献。

## 4. 与历史原始PFQ的区别

“历史原始PFQ”指本仓库扩展后保存的旧基线，**不是未经修改的 NVIDIA 原版**。修正 current 相对它的分段goodput为 +17.41%，完整包吞吐为 +15.13%；成功包平均时延却从31.71 ms增至50.68 ms，配对增加18.96 ms，95% CI [6.09, 31.84]，8/10 seed变差。

跨版本流量与拓扑CSV相同，但阴影变化使59/360个UE-seed的服务小区改变，H/W、预热队列及OLLA也不同。因此只能描述整体版本变化，不能称为算法收益或单独SVD收益。全包平均终态时延与Jain公平性差值区间均跨零；360个匹配UE中仍有39个分段goodput下降。

来源：[三组逐seed对照](evidence/2026-10-21/historical_comparison/per_seed_comparison.csv)、[历史比较报告](evidence/2026-10-21/historical_comparison/report.txt)、[核验记录](evidence/2026-10-21/historical_comparison/validation.json)。

## 5. 已回答的问题与剩余不确定性

**已回答：**三项修正后，在共同预热和同一步输入下，冻结 current 在这个1秒窗口的两类吞吐、整包成功率、平均成功时延和资源占用优于同环境PFQ/DPA。既有验证覆盖动作合法性、FIFO/字节守恒、配对关系和成熟cohort无截尾；物理独立数值核验覆盖90个保存状态。

**尚不能推出：**

- **稳态容量。** current 队列由24.44 MB降至5.66 MB，goodput 1398.20 Mbps略高于实际到达1384.74 Mbps。四个250 ms窗口依次为1536.92、1424.13、1337.89、1293.86 Mbps，随后200 ms为1279.67 Mbps，均含短期动态。
- **尾部可靠性已解决。** 所有seed全包终态p99仍为200 ms；current成功包p95的seed均值193.2 ms。最差UE整包成功约50.96%、过期44.09%。
- **所有UE、所有指标都改善。** 相对修正PFQ/DPA，仍分别有1/6个匹配UE的分段goodput下降。包失败12.41%与TB BLER 8.80%分母不同。
- **泛化、学习贡献或生产就绪。** 未证明7cell和其它负载的泛化或学习组件独立收益；通用训练bridge尚未推广专用runner的时序修正，本轮也未确证实时部署预算。

若继续研究，应先选择一个未回答问题（如稳态是否仍有收益），冻结最小实验与停止条件，再扩展。本次整理不追加seed、仿真或训练。

## 6. 证据与文档收口

[证据快照](evidence/2026-10-21/README.md)保存协议、逐seed/UE表、验证JSON、构建与策略身份、来源SHA-256。快照中的 `passed` 是既有验证记录；本次另做统计算术和文件完整性复核，不宣称重新执行全部轨迹或CUDA数值检查。

旧智能算法报告、阶段实验报告、独立数据清单合并为[HGraph历史回顾](archive/hgraph_experiments/stageB_hgraph_v9c_v19_retrospective.md)；旧验收报告、时延分布分析、同频资料底稿合并为[历史基线报告](archive/baselines/stageB_baseline_verification_legacy.md)。长篇7cell/HGraph设计保留为[历史设计与实验日志](archive/hgraph_design/stageB_7cell_tar_gnn_pfq_design.md)，独有负结果未删除。详见[归档索引](archive/README.md)。

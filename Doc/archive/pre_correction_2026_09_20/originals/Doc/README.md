# Stage-B / DPA-v1 文档入口

> 更新于 2026-09-07。当前文档只保留一个算法定义、一个综合实验结论和一个生效配置入口；阶段性 brief 已合并，避免同一指标在多个文件中漂移。

## 当前结论

当前冻结算法是 **DPA-v1（Deadline-Pressure Actor v1）**，代码兼容名为 `PFQIndependentReuseActor` / `pfq_independent_reuse_v1`。在线决策不读取 PFQ，使用无量纲 deadline/service features、CTBS-v1 累计 TBS headroom、deterministic argmax，正式配置为 guard off。

- standard seeds 41--50：DPA-v1 相对 PFQ mean goodput `+15.433 Mbps`（`+1.01%`，10/10 seed 胜），BLER 相对下降 `6.30%`，outcome p95 delay `29.10 → 1.90 ms`，UE p10 expiry-aware rate `6.888 → 19.487 Mbps`。
- unseen fixed-load mixed-granularity seeds 101--110：相对 PFQ goodput `+14.195 Mbps`，95% CI `[5.396, 22.994]`；相对 RRQ `+17.101 Mbps`，CI `[8.309, 25.893]`。时延、backlog 和 active-PRG 效率同时改善。
- 不能宣称 zero expiry：mixed DPA 有 4/10 seed 非零 expiry，mean `1.278%`，seed104 为 `9.080%`。
- CTBS-legal non-starvation guard 已实现并在 seed51 评估，但合法 Blank 机会不足，`override_count=0`；它没有进入当前 DPA-v1。

## 三个当前入口

1. [DPA-v1 主设计](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)：当前最详细、唯一生效的算法说明；包含图结构、候选/动作、无量纲特征、CTBS-v1、PFQ 独立性、reward、guard 和正确性不变量。
2. [DPA-v1 / PFQ / RRQ 综合实验报告](./stageB_boundary_intelligent_algorithm_experiment_report.md)：合并 standard 41--50、fixed-load packet-granularity 101--110，以及 iter21 → dimensionless → guard 的关键实验沿革。
3. [当前生效配置与指标口径](./current_stageB_effective_configuration.md)：冻结 checkpoint、场景参数、DPA/PFQ/RRQ 定义、业务 profile、统计窗口和 KPI 解释。

新结论应更新以上对应入口，不再创建单 seed 或中间阶段 brief。原始结果继续保留在 `output/`，设计/执行协议继续保留在 `docs/superpowers/`。

## 实验协议与原始证据

### DPA 消融研究（2026-09-11 更新）

- [原随机初始化等预算实验](dpa_equal_budget_ablation_experiment.md)：9 次训练完成，剩余评估暂停；两项 Full 诊断已完成。
- [共同冻结 checkpoint 的短程适应消融与直接消融](dpa_checkpoint_adaptation_ablation.md)：三组 8 轮适应及评估已完成；当前仅补两项冻结权重、零新增训练的直接消融评估，区分移除模块与后续适应的影响。
- 此研究尚未形成正式结论，不替代下列冻结 checkpoint 结果。

### Standard DPA/PFQ，seeds 41--50

- [严格 aligned comparison 设计](../docs/superpowers/specs/2026-09-01-dpa-v1-pfq-aligned-expiry-aware-comparison-design.md)
- [历史比较/窗口设计](../docs/superpowers/specs/2026-08-31-dpa-v1-seed41-50-historical-comparison-design.md)
- [最终 JSON](../output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/final_report.json)
- [聚合 CSV](../output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/aggregate_comparison.csv)
- [逐 seed CSV](../output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/per_seed_comparison.csv)

### Fixed-load packet granularity，DPA/PFQ/RRQ，seeds 101--110

- [10-seed 最终 JSON](../output/pfq_independent_reuse/formal/granularity_newseed_stage1_v1/campaign_report_10seed.json)
- 每 seed 的 `granularity_report.json`、`headline_steady.csv` 和 `per_ue_steady.csv` 位于 `output/pfq_independent_reuse/formal/granularity_newseed_stage1_v1/seed101..seed110/`。
- `campaign_report.json` 和 `campaign_report_3seed.json` 是中间停止点快照，不用于最终结论。

### 表示与 guard 沿革

- [无量纲 service feature 设计](../docs/superpowers/specs/2026-08-27-seed45-dimensionless-service-features-design.md)
- [seed48/51 frozen generalization 协议](../docs/superpowers/specs/2026-08-27-iter8-seed48-seed51-frozen-generalization-protocol.md)
- [CTBS-legal guard 设计](../docs/superpowers/specs/2026-08-28-iter8-ctbs-legal-non-starvation-guard-design.md)
- [seed48/51 frozen generalization JSON](../output/pfq_independent_reuse/formal/iter8_dimensionless_frozen_generalization_seed48_seed51_v1/final_report.json)
- [seed51 guard JSON](../output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/final_report.json)

## 代码入口

- `training/stageb_hgraph/independent_actor.py`：DPA-v1 actor 与模型配置。
- `training/stageb_hgraph/independent_policy.py`：在线打分、固定 PRG 解码与动作。
- `training/stageb_hgraph/feature_snapshot.py`：Cell/UE/PRG graph snapshot。
- `training/stageb_hgraph/ctbs_headroom.py`：CTBS-v1 causal cumulative-TBS reconstruction。
- `training/stageb_hgraph/independent_reward.py`：treatments A--E；DPA checkpoint 使用 E。
- `training/stageb_hgraph/pfq_dpa_aligned_comparison.py`：standard DPA/PFQ paired 汇总。
- `training/stageb_hgraph/granularity_experiment.py`：DPA/PFQ/RRQ packet-granularity 汇总。
- `cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h`：online deadline/service state。

## 仍保留的专题与历史文档

- [基线 packet-delay 分布与 RR/RRQ/PF/PFQ 机制](./stageB_packet_delay_distribution_analysis.md)
- [7cell/TAR-GNN/PFQ 历史设计](./stageB_7cell_tar_gnn_pfq_design.md)：不是当前 DPA-v1 已验证配置。
- [多维资源分配仿真验证报告](./stageB_multidimensional_resource_allocation_simulation_verification_report.md)
- [archive 索引](./archive/README.md)：旧 HGraph、legacy GNNRL、平台审计、会议记录和早期计划。

## 维护规则

- 算法语义只写入主设计；实验数字只写入综合报告；复现参数只写入生效配置。
- 中间结果不另建长期 brief；确需保留时放入 raw JSON/CSV 或 `archive/`，并标明不代表当前结论。
- 任何 paired 优势必须同时核验 checkpoint、simulator binary、offered stream、topology、horizon 和统计窗口。
- stress seed 不从总体统计中剔除；CI 跨 0 的差异不描述为稳定提升。
- `output/` 是生成目录。长期证据优先保留 manifest、final/campaign report、aggregate/per-seed 表和对应 checkpoint。

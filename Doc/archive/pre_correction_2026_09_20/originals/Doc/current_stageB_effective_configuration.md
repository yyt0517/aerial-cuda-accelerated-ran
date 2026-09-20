# Stage-B 当前生效配置与 DPA/PFQ/RRQ 口径

> 2026-09-20 容量审计更新：以下历史配置/结果使用了 TBS bits 被当作 bytes 的旧环境，不能视为校准物理性能。旧训练队列已停用；修正实现及新实验边界见[容量单位迁移说明](capacity_units_bytes_v2_migration.md)。

> 更新于 2026-09-07。本文只维护当前 DPA-v1 比较所需的生效配置和指标口径；算法语义见 [DPA-v1 主设计](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)，结果见 [综合实验报告](./stageB_boundary_intelligent_algorithm_experiment_report.md)。

## 1. 当前算法与部署配置

| 项目 | 当前值 |
|---|---|
| 报告名称 | DPA-v1（Deadline-Pressure Actor v1） |
| 代码 actor/family | `PFQIndependentReuseActor` / `pfq_independent_reuse_v1` |
| checkpoint | `deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt` |
| checkpoint file SHA-256 | `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a` |
| checkpoint tensor SHA-256 | `6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be` |
| action selection | deterministic `argmax` |
| headroom | CTBS-v1 cumulative TBS |
| non-starvation guard | off |
| online PFQ input | none |
| action | per `(cell, local PRG)` choose associated UE or Blank |
| candidate width | runtime topology；truncation forbidden |

相同 checkpoint 但改变 feature slot、CTBS、decode 顺序、guard 或 candidate policy，都不是同一个 DPA-v1 实验臂。

## 2. 共同仿真配置

当前两组正式 campaign 的共同核心配置：

- `3 cell / 36 UE`
- `cell_radius=500 m`，`site_spacing=1000 m`
- `ue-placement=coop_boundary`
- `cell-assoc-mode=strongest`
- legacy shared shadow RNG
- Rayleigh + SVD + Type-0 bitmap
- `17` local PRG/cell
- packet TTL `200 ms`
- continuous horizon `4000 TTI`
- headline steady window：TTI `1001..4000`，共 `3000 TTI`
- supplementary full window：TTI `1..4000`

paired 声明还要求同 seed 的 scheduler arms 使用同一 simulator binary 和同一 offered stream。只对配置相同不够。

## 3. 业务 profile

### 3.1 Standard homogeneous（seeds 41--50）

```text
每个 UE：Poisson(1 packet/TTI)
packet size：3000 bytes
期望 offered load：3000 bytes/TTI/UE
网络期望 offered load：108000 bytes/TTI
```

本 campaign 只比较 DPA-v1 与 PFQ。

### 3.2 Fixed-load homogeneous（seeds 101--110）

与 standard 的期望负载定义相同：全部 36 UE 使用 `Poisson(1) × 3000B`。

### 3.3 Fixed-load mixed granularity（seeds 101--110）

```text
18 个偶数 UE：Poisson(2) × 1500B
18 个奇数 UE：Poisson(0.25) × 12000B
```

两组 UE 的期望 offered load 都是 `3000 bytes/TTI/UE`，因此网络期望负载仍是 `108000 bytes/TTI`。该对照改变 packet 到达频率/尺寸，不改变每 UE 期望字节负载。

此 campaign 比较 DPA-v1、PFQ 和 RRQ，共 `10 seeds × 2 profiles × 3 schedulers = 60 arms`。

## 4. Scheduler 口径

### 4.1 DPA-v1

使用冻结 checkpoint，从 causal graph state 直接预测 UE/Blank。PFQ 仅是训练祖先和实验基线，不进入在线 observation 或动作。

### 4.2 PFQ

PFQ 是 queue-aware proportional-fair native reference。它使用 native 调度路径，在 Type-0 下产生 UE selection/bitmap；它不是 DPA-v1 的 online fallback。

### 4.3 RRQ

RRQ 是 queue-aware round-robin native reference，用于区分“比例公平排序的收益”和“一般 queue-aware 可行调度”的收益。

旧 RR/PF（非 queue-aware）在 boundary 场景中可能把资源分给无数据 UE，仍可用于历史机制分析，但不再是 DPA-v1 的主要 baseline。

## 5. 指标定义

| 指标 | 口径 | 方向 |
|---|---|---|
| goodput | 成功交付 bytes 按窗口时长换算 Mbps | 高 |
| offered-normalized goodput | goodput bytes / offered bytes | 高 |
| BLER | failed TB/bytes 对应的实验汇总口径 | 低 |
| expiry drop rate | expired bytes / offered bytes | 低 |
| outcome delay | served + expired packet outcome 的 system time | 低 |
| end backlog | 窗口/运行终点未清队列 bytes | 低 |
| UE expiry-aware rate | goodput 与 expired outcome 联合计入的逐 UE 服务结果 | 高 |
| mean-cell spectral efficiency | 三 cell 平均成功 bytes / active PRG 的归一化量 | 高 |
| PRG utilization | non-Blank/总合法 PRG 机会 | 语境指标 |
| successful bytes/active PRG | 每个使用 PRG 的交付效率 | 高 |
| cell Jain | cell-level 资源/结果公平性 | 高，但需结合吞吐与 weak-cell 指标 |

`goodput_mbps`、scheduled dequeue、TBS capacity 和 packet outcome 是不同量：

- scheduled dequeue 表示调度器本 TTI 尝试服务的 bytes；
- cumulative TBS 是按 MCS/layer/累计 PRB 重建的容量；
- goodput 只计成功交付；
- outcome delay 必须包含 expired outcomes，否则会把过期包从时延统计中删除并产生乐观偏差。

## 6. 汇总与统计

- 表格中的 `mean ± std` 是跨 10 个 seed 的样本均值和标准差。
- paired delta 定义为 `DPA - baseline` 或 `mixed - homogeneous`。
- 95% CI 基于逐 seed paired delta；CI 跨 0 时不写成稳定优势/劣势。
- win count 按每个 seed 的指标方向计数；语境指标不强制判胜。
- stress seed 不能从总体均值中删除，必须单独报告。

standard campaign 的 headline 为 steady 3000；full 4000 只作一致性补充。packet-granularity campaign 也以 steady 3000 为 headline，以减少冷启动对因果解释的污染。

## 7. 完整性检查

正式比较至少满足：

- 每个 arm `4000` decision records；
- 每个 arm `36 × 4000 = 144000` UE-service records；
- 同 seed/profile 的 binary hash、offered stream hash 和 topology 相同；
- DPA illegal action、mask violation 和 replay mismatch 为 0；
- CTBS pre-action snapshot/config hash 完整；
- 统计窗口边界一致；
- expiry-aware outcome 包含 served 与 expired packet。

packet-granularity campaign 共核验 `240000` decision records 和 `8640000` UE-service records。

## 8. 数据与代码入口

当前正式数据：

- standard DPA/PFQ：`output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/`
- fixed-load DPA/PFQ/RRQ：`output/pfq_independent_reuse/formal/granularity_newseed_stage1_v1/`
- frozen checkpoint：`output/pfq_independent_reuse/formal/deadline_dimensionless_features_seed45_v1/train_attempt2/`
- guard audit：`output/pfq_independent_reuse/formal/iter8_ctbs_legal_non_starvation_guard_seed51_v1/final_report.json`

当前比较/汇总代码：

- `training/stageb_hgraph/pfq_dpa_aligned_comparison.py`
- `training/stageb_hgraph/granularity_experiment.py`
- `training/stageb_hgraph/dpa_v1_historical_comparison.py`

旧 baseline 机制和 packet-delay 分布仍见 [`stageB_packet_delay_distribution_analysis.md`](./stageB_packet_delay_distribution_analysis.md)，但其中历史场景结果不覆盖本文的当前 campaign 配置。

## 9. 已知边界

- 只完成 3cell/36UE；不能外推为 7cell/84UE 结论。
- standard 和 packet-granularity campaign 的 simulator binary hash 不同，不能跨 campaign 做严格绝对值 paired 比较。
- mixed profile 保持期望字节负载相同，但到达方差、packet count 和单包尺寸不同；这是 packet granularity + arrival process 的联合变化。
- DPA-v1 在 mixed seeds104/105/107/110 出现非零 expiry；当前配置没有 worst-seed 零过期保证。
- guard off 是当前正式配置；已有 guard 实验不构成有效安全层证据。

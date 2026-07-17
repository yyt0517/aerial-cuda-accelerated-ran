# Stage-B 多维协同资源分配基线仿真验证报告设计

## 1. 目标与交付定位

本设计用于生成项目验收材料《Stage-B 多维协同资源分配代表性技术仿真验证报告》。报告覆盖 RR、PF、RRQ、PFQ 四类调度算法，以项目已有代码、10-seed 仿真结果、聚合数据和图表为证据，说明代表性时频资源调度技术的实现方法、验证过程、量化表现、适用边界与复现方式。

报告面向项目验收和技术评审，不采用论文式长篇理论推导。正文优先回答“验证了什么、如何验证、结果如何、是否达到验证目的”，技术细节集中放入附录，以兼顾可读性和审计追溯。

四类算法的角色定义如下：

- RR：原始轮询式 PRG 分配，作为基础时频资源均分基线。
- PF：根据瞬时可达速率和历史平均速率进行比例公平分配，作为信道与公平性联合权衡基线。
- RRQ：在 RR 基础上引入队列需求约束和 demand cap，作为 queue-aware RR 增强算法。
- PFQ：在 PF 基础上引入 buffer weight、demand cap 和 TTI 内衰减，作为 queue-aware PF 增强算法。

Q 明确表示本项目的 queue-aware 实现，不将 RRQ/PFQ表述为标准化算法名称，也不把四类传统或增强基线宣称为 AI 算法。它们用于验证代表性资源分配机制，并为后续 AI 调度方案提供可量化参考。

## 2. 交付物

实施阶段生成以下文件：

- 正式报告：`Doc/stageB_multidimensional_resource_allocation_simulation_verification_report.md`
- 图表生成脚本：`cuMAC/scripts/plot_stageB_baseline_verification_report.py`
- 报告图表目录：`Doc/figures/stageB_baseline_verification/`
- 实施计划：`docs/superpowers/plans/2026-07-17-stageb-baseline-verification-report.md`

正式报告采用“主报告 + 技术附录”结构。图表生成脚本只读取现有聚合数据，不修改原始 run、CSV、JSON 或 packet sample。

## 3. 报告结构

### 3.1 主报告

正文包含以下章节：

1. 项目背景与验证目标
2. 验证对象及多维资源分配映射
3. RR、PF、RRQ、PFQ 算法原理
4. 仿真平台与软硬件环境
5. 同频三小区场景及业务模型
6. 验证指标与验收判据
7. 试验设计、变量控制及随机种子
8. 四算法统一对比结果
9. 时延分布、吞吐、公平性和资源利用分析
10. Boundary 边缘用户增强验证
11. 多维协同能力分析
12. 验证结论与验收符合性
13. 局限性及后续 AI 技术演进建议

正文开头设置执行摘要，给出验证范围、核心发现和验收判断。正文避免堆叠全部逐 seed 数值，只保留能支持验收结论的核心表格和图形。

### 3.2 技术附录

附录包含：

- 仿真参数配置总表
- KPI 定义、公式、统计层级和解释限制
- seed 41--50 配对设计与统计方法
- 代码模块、算法入口和调用链
- 一键运行、聚合和绘图命令
- CSV、JSON、日志及图表证据索引
- 可复现性声明和 provenance 缺口

附录中的路径统一使用仓库相对路径，确保在其他 checkout 中仍可理解和执行。绝对路径只在说明当前工作区时使用，不作为复现接口。

## 4. 证据架构

### 4.1 四算法统一主证据

四算法横向比较以同一 full-KPI suite 为数据源：

`output/stageB_baseline_eval_suite_fullkpi_seed41_50/`

主要聚合文件为：

- `suite_aggregate/uniform_four_baseline_metrics.csv`
- `suite_aggregate/uniform_four_baseline_metrics.json`
- `suite_aggregate/figures/packet_delay_distribution_bins.csv`
- `suite_aggregate/figures/packet_delay_distribution_by_seed.csv`

该批次用于 RR、PF、RRQ、PFQ 的统一横向比较。报告不得从其他批次抽取单个 KPI 拼入四算法主表。

### 4.2 同频边缘增强证据

Boundary RRQ/PFQ 结果来自同一 full-KPI suite 的：

- `suite_aggregate/boundary_rrq_pfq_metrics.csv`
- `suite_aggregate/figures/packet_delay_distribution_boundary_rrq_vs_pfq.png`

该结果用于验证边缘 UE 和簇内同频干扰条件下，queue-aware PF 相对 queue-aware RR 的吞吐、尾时延、过期率、弱 UE 吞吐和公平性表现。

### 4.3 交叉验证证据

active boundary gate 数据来自：

`output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/`

该独立批次仅用于检查结论方向和跨 seed 稳健性，不与 full-KPI suite 的绝对数值拼接为同一次实验结果。报告必须明确两批次均有效但运行独立。

### 4.4 资料底稿

以下现有文档用于核对场景、指标和结论边界，不替代原始数据：

- `Doc/stageB_cochannel_baseline_analysis_reference.md`
- `Doc/stageB_packet_delay_distribution_analysis.md`
- `Doc/current_stageB_effective_configuration.md`
- `Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`

## 5. 场景与多维资源映射

主场景按现有证据写为 3 个相邻同频小区、36 个 active UE、每小区 12 UE、4T4R、Rayleigh、SVD、Type-0 bitmap、固定 3000 B 下行业务包、Poisson 到达、4000 TTI、TTL 200 ms、paired seed 41--50。资源粒度必须以对应结果批次的实际 manifest 和运行参数为准；若 full-KPI suite 与 active gate 使用不同 PRG 粒度，应在各自表格中分列，不得写成同一配置。

“时、频、功、空”等多维协同按可验证证据映射，而不扩大算法能力：

| 维度 | 仿真中的可验证对象 | 主要指标 |
|---|---|---|
| 时间 | TTI 级调度、队列等待、包 TTL | mean/p90/p95 delay、expiry、pending packet |
| 频率 | PRG bitmap、PRG 占用与同频复用 | PRG utilization、goodput、served throughput |
| 功率/链路 | 信道质量、SVD 后可达率、传输成败 | TB BLER、goodput、可达速率相关 PF 权重 |
| 空间 | 3-cell 同频、4T4R、边缘 UE、strongest association | UE p5/p10、Jain、公平性、boundary 对比 |
| 业务/队列 | buffer、demand cap、TTL 压力 | completed/expired/pending packet、tail delay |

报告不得宣称 RR、PF、RRQ、PFQ 已直接联合优化发射功率或执行跨小区联合动作。这些算法的“功、空”体现为调度决策对链路质量、MIMO/预编码结果和空间干扰场景的适应与响应。

## 6. 验收判据

技术验证从四个方面验收：

1. 功能性：四类算法均可通过现有 GPU Type-0 调度链路执行并产出完整 KPI。
2. 公平性：比较使用一致的拓扑、信道、业务、时长和随机种子，并采用 paired seed 统计。
3. 完整性：结果覆盖 goodput、served throughput、包时延、TTL expiry、UE p5/p10、公平性、BLER、PRG utilization 和 packet completion 状态。
4. 可追溯性：核心数值可追溯到聚合 CSV/JSON、运行目录、执行脚本和参数说明。

验收不要求某一算法在所有指标上获胜。每类算法按“收益—代价”联合解释，特别区分系统吞吐、完成包时延、过期业务、末端残留和资源占用。若 PFQ 在吞吐、尾时延、过期率和弱 UE 指标的综合表现优于其他基线，可将其认定为后续智能算法的强传统参考；不能仅凭 PRG utilization 或单一 packet-rate 指标下结论。

## 7. 图表设计

正文使用以下图表：

1. Uniform 四算法 goodput 对比图，显示 mean 和标准差。
2. Uniform 四算法 p95 delay 对比图，显示 mean 和标准差。
3. Uniform 四算法 expiry 对比图，显示 mean 和标准差；跨度较大时使用对数坐标并说明零值处理。
4. Uniform 四算法 Jain fairness 与 PRG utilization 对比，使用两个独立子图避免双轴误导。
5. RR 相对 RRQ、PF 相对 PFQ 的关键指标改进量图，明确每项指标的优劣方向。
6. 现有 Uniform RR/RRQ、PF/PFQ 和 Boundary RRQ/PFQ packet-delay 分布图。
7. 时间、频率、功率/链路、空间、队列维度与指标的映射表。

不使用雷达图。不同单位、不同优劣方向和数量级的指标不做无说明归一化。所有误差线必须说明是跨 seed 标准差，不能暗示为置信区间。

## 8. 数据流与实现边界

数据流为：

```text
原始 run KPI / packet samples
        -> 现有 suite 聚合 CSV/JSON
        -> 报告专用绘图脚本
        -> 验收图表与数值一致性检查
        -> 正式 Markdown 报告及技术附录
```

绘图脚本仅负责读取、验证和绘图，不重新定义 KPI，不回写聚合结果。脚本应提供清晰的命令行参数，默认读取本仓库既有 full-KPI suite，并允许显式指定 suite 根目录和输出目录。

脚本遇到以下情况时以非零状态退出并给出具体错误：

- 输入文件不存在；
- 缺少 RR、PF、RRQ、PFQ 中任一算法；
- 核心指标列缺失或无法转为有限数值；
- 同一算法重复或聚合结构不符合预期；
- 输出目录不可创建或图表无法写入。

脚本不删除既有图表，不修改源数据，不自动搜索并混用其他实验批次。

## 9. 质量控制与验证

实施完成前执行以下验证：

1. 对源 CSV 进行 schema 和四算法完整性检查。
2. 用脚本重新生成全部报告专用图表，并确认命令退出码为 0。
3. 对报告核心表格中的数值逐项与源 CSV/JSON 比较。
4. 检查正文图表与本地 Markdown 链接全部存在。
5. 搜索报告中的未完成占位符、未解释缩写和无法追溯的绝对结论。
6. 检查 full-KPI suite 与 active gate 是否被明确标为独立批次。
7. 检查“同频”“功率维”“空间维”“AI 基线”等术语没有超出代码和数据证据。
8. 运行绘图脚本的测试或最小验证命令，覆盖正常输入和至少一个缺列/缺算法错误输入。

现有结果未固化运行时精确 Git commit、dirty patch、容器 digest、完整编译选项和统一配置 hash。正式报告应将这些列为 provenance 缺口，并表述为“基于当前可追溯脚本和运行 manifest 复核”，不作 bit-exact reproduction 声明。

## 10. 变更范围

实施仅新增报告、绘图脚本、测试或验证辅助文件以及报告图表；如需更新 `Doc/README.md`，只增加新报告索引并保留现有未提交修改。不得修改现有算法实现、仿真结果或用户工作区中的其他未提交文件。

报告使用已有本地证据完成，不需要互联网资料。若正文需要外部标准或文献引用，应另行确认引用范围后再补充，不在本设计的当前验收交付范围内。

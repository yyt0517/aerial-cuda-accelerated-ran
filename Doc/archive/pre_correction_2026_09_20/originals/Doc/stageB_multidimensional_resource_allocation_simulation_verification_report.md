# Stage-B 多维协同资源分配代表性技术仿真验证报告

**文档性质：** 项目验收材料

**验证对象：** RR、PF、RRQ、PFQ 四类 Type-0 时频资源调度基线

**主证据批次：** `output/stageB_baseline_eval_suite_fullkpi_seed41_50/`

**统计范围：** paired seeds 41--50；表中 `mean ± std` 为先按 seed 统计、再跨 10 个 seed 求算术均值与总体标准差；误差不是置信区间

**结论适用范围：** 本报告所列三小区、Rayleigh、SVD、4T4R、Type-0、固定高负载业务与 4000 TTI 仿真条件

## 执行摘要

本次验证在可追溯的 Stage-B GPU Type-0 调度链路上，对 RR、PF、RRQ、PFQ 四类代表性基线进行了同拓扑、同信道、同业务、同仿真时长及同随机种子的配对比较。四类算法均能完成运行并产出吞吐、包时延、TTL 过期、弱 UE 吞吐、公平性、TB BLER、PRG 利用率及包完成状态等 KPI，满足功能执行、配对比较、KPI 完整性和结果追溯要求。

统一四算法主线采用 full-KPI suite 的 **uniform** 场景。PFQ 在该条件下取得 1527.240 ± 16.945 Mbps goodput、38.000 ± 49.942 ms p95 时延、0.117 ± 0.215% 过期率、39.190 ± 1.570 Mbps UE goodput p5 和 0.997899 ± 0.001119 Jain 公平性，是本次工况下综合表现最强的基线；其 PRG 利用率为 94.509 ± 2.974%，TB BLER 为 11.422 ± 0.703%，说明结论应按收益和资源/可靠性代价联合理解。RRQ 相对 RR 改善 goodput 和已完成包尾时延，但过期率均值未同步改善。当前 PF 路径在该高负载、queue-unaware 实现下发生明显退化；该现象只描述当前实现与工况，不能外推为比例公平理论在一般条件下失效。

full-KPI **boundary** 结果作为同一 suite 内的独立边缘场景证据：PFQ 相对 RRQ 的 goodput 为 +21.710 Mbps、p95 为 -52.850 ms、过期率为 -1.199 个百分点、UE p5 为 +4.124 Mbps。另一个独立的 active boundary gate 仅用于方向交叉核验，得到 goodput +6.903 Mbps、p95 -30.428 ms、过期率 -0.592 个百分点、UE p5 +1.935 Mbps、Jain +0.002475。两批 boundary 数值未拼接为同一次实验结果。

综合判断：本次代表性基线仿真验证达到既定技术验证目的，可作为后续 AI 调度方案的量化比较基础；但现有结果缺少原始运行时 Git commit hash、dirty patch、容器镜像 digest、编译选项 hash 和统一配置 hash，因此 bit-exact 可复现性仅部分满足。

## 1. 项目背景与验证目标

项目需要在后续智能资源调度研发之前建立可执行、可量化、可审计的传统与增强基线。本报告的目标是：

1. 验证四类基线在当前 CUDA 加速 Type-0 调度链路上均可运行；
2. 在统一条件下比较系统吞吐、时延尾部、包过期、弱 UE、公平性、可靠性和资源占用；
3. 说明 queue-aware 改造在高负载业务中的收益及代价；
4. 建立后续 AI 调度必须在同一场景、同一 KPI 口径下超过的参考基准；
5. 明确证据边界和可复现性缺口，避免从有限仿真结果作普遍化外推。

RR、PF、RRQ、PFQ 均为本项目选取的代表性基线，**不是 AI 算法**。其中 Q 仅表示本项目的 queue-aware 实现，不是标准化算法名称。

## 2. 验证对象及多维资源分配映射

当前 Type-0 路径先将所有 active UE 放入候选集合，四类基线的核心差异主要落在逐 PRG bitmap 分配，而非旧 Type-1 语义下完整的“UE selection + 连续资源分配”两阶段流程。

| 维度 | 仿真中的可验证对象 | 主要指标 |
|---|---|---|
| 时间 | TTI 级调度、队列等待、包 TTL | mean/p90/p95 delay、expiry、pending packet |
| 频率 | PRG bitmap、PRG 占用与同频复用 | PRG utilization、goodput、served throughput |
| 功率/链路 | 信道质量、SVD 后可达率、传输成败 | TB BLER、goodput、可达速率相关 PF 权重 |
| 空间 | 3-cell 同频、4T4R、边缘 UE、strongest association | UE p5/p10、Jain、公平性、boundary 对比 |
| 业务/队列 | buffer、demand cap、TTL 压力 | completed/expired/pending packet、tail delay |

这里的“功率/链路”和“空间”维度体现为调度器对链路质量、MIMO/预编码结果及空间干扰场景的观测和响应。四类基线**不直接优化发射功率，也不执行跨小区联合动作**；各小区内的调度动作在共同载频上形成簇内同频干扰。

## 3. RR、PF、RRQ、PFQ 算法原理

| 算法 | 项目实现语义 | 主要机制 | 预期收益与风险 |
|---|---|---|---|
| RR | 原始轮询式 PRG 分配 | 按小区关联 UE 尽量均分 PRG，余数顺序补齐 | 机制简单、占用充分；不显式利用队列需求与信道机会 |
| PF | 当前原始比例公平路径 | 以瞬时可达率与历史平均速率构造 PF 权重 | 权衡信道机会与长期服务历史；当前路径缺少高负载业务所需的充分队列约束 |
| RRQ | 本项目 queue-aware RR | 在 RR 基础上引入队列需求约束和 demand cap | 可减少超需求分配并改善部分时延；不保证过期、公平性等所有指标同步改善 |
| PFQ | 本项目 queue-aware PF | PF 基础分数叠加 buffer weight、demand cap 和 TTI 内衰减 | 抑制超需求集中分配和危险长尾；可能增加占用或牺牲部分极短时延包/单包速率指标 |

PFQ 的基础权重保留 `rate^beta / avgRate` 形式，并叠加 buffer 权重；demand cap 依据 buffer、单 PRG 瞬时可服务字节和 slack 限制超需求分配，intra-TTI decay 则随同一 UE 在本 TTI 已获 PRG 数增加而衰减。当前定稿不包含历史试验中的二阶段 relaxed idle-PRG fill。

## 4. 仿真平台与软硬件环境

验证使用仓库现有 CUDA 加速 cuMAC Stage-B 仿真链路，执行 GPU Type-0 bitmap 调度。PHY/场景的已确认配置如下。

| 类别 | 配置 |
|---|---|
| 小区与 UE | 3 个相邻全向小区，36 个 active UE，12 UE/cell |
| 天线 | BS/UE 4T4R；不等同于每次传输固定 4 层 |
| 信道 | Rayleigh |
| 预编码 | SVD |
| 资源分配 | Type-0 bitmap |
| 载频 / SCS | 2.5 GHz / 30 kHz |
| TTI | 0.5 ms |
| 总资源 | 272 PRB/cell |
| 执行链路 | native GPU scheduler；聚合数据按 run log 与 manifest 追溯 |

现有运行产物没有完整固化 GPU 型号、驱动/工具链组合、容器镜像 digest 及编译选项 hash。本报告不据此声称已恢复原始运行时的精确软硬件镜像。

### 4.1 实际 PRG 粒度核验

批准设计要求按各批实际命令/manifest 记录资源分组。full-KPI 批次保留 60 份、active boundary gate 批次保留 20 份，共 80 份 seed wrapper log，均记录 `nPrbGrpsConst 17`；结合 272 PRB，可核验为 17 PRG/cell、16 PRB/PRG。

| 数据批次 | 实际记录 | 报告使用方式 |
|---|---|---|
| full-KPI uniform 四算法 | 17 PRG × 16 PRB = 272 PRB/cell | 四算法主线 |
| full-KPI boundary RRQ/PFQ | 17 PRG × 16 PRB = 272 PRB/cell | boundary 完整 KPI 与逐包分布 |
| active boundary gate | 17 PRG × 16 PRB = 272 PRB/cell | 独立方向交叉核验 |

三批实际粒度相同，但仍属于独立运行批次；相同资源粒度不构成混合其绝对数值的理由。

## 5. 同频三小区场景及业务模型

三个小区共享同一载频和同一 PRG 编号空间，频率复用因子可按 1 理解。不同小区在同一 TTI、同一 PRG index 上的并发传输形成协调簇内部的同频干扰。场景没有簇外干扰小区，也没有 handover 或动态负载均衡证据。

uniform 主场景用于四算法横向验收；boundary 场景将 UE 布置在更具边缘性的区域并采用 strongest-cell association，用于核验弱 UE 与不均衡链路条件。placement 同时改变关联、信道和队列演化，因此不把跨 placement 的单个 KPI 变化全部归因于干扰强弱。

业务为每 UE 一条固定 3000 Byte 下行 flow，逐 UE 按 Poisson 过程到达，均值 `1.0 pkt/TTI/UE`。单 TTI 可到达 0、1 或多个包。理论 offered load 为：

```text
36 UE × 3000 Byte × 8 bit/Byte × 1 pkt/TTI / 0.0005 s = 1.728 Gbps
```

每个 seed 仿真 4000 TTI，即 2.0 s；TTL 为 200 ms，即 400 TTI；三批运行的 warm-up 均为 0 TTI。有限时窗内实际生成量随 Poisson realization 波动。

## 6. 验证指标与验收判据

### 6.1 KPI 口径

| KPI | 统计含义 | 解读限制 |
|---|---|---|
| goodput | PHY 侧 `tbErr == 0` 的有效字节除以仿真时间 | 系统级主吞吐指标；不同于 MAC served estimate |
| p95 delay | 对 MAC tracker 中完全出队的 completed packet 统计 enqueue-to-completion 时延的 seed 内 p95 | expired/pending 不进入样本，必须联读状态计数 |
| expiry | TTL 到期移除字节占生成字节比例 | 源 CSV 为 0--1 ratio，报告乘 100 显示为 `%` |
| UE goodput p5 | per-UE goodput 的 5% 分位 | 弱 UE 体验代理，不等同于几何位置标签 |
| Jain fairness | per-UE goodput 的 Jain 指数 | 接近 1 表示分布更均衡 |
| TB BLER | 调度 TB 的错误比例 | 源 ratio 乘 100；低 BLER 不必然对应最高 goodput |
| PRG utilization | 已分配 PRG opportunity / 总 PRG capacity | 源 ratio 乘 100；高占用不等同于高频谱效率 |
| completed/expired/pending | 完全离开 MAC tracker / TTL 移除 / 仿真结束仍残留的包或残片 | 用于识别 completed-only 时延的选择性 |

所有百分比均按源 ratio × 100 转换；百分点差值记为 pp。跨 seed 标准差反映随机场景离散性，不作统计显著性声明。

### 6.2 验收判据

1. **功能性：** 四算法均可通过现有 GPU Type-0 链路执行并产出 KPI；
2. **公平比较：** 同批比较使用一致拓扑、信道、业务、时长、资源粒度及 paired seeds；
3. **完整性：** 覆盖 goodput、served、时延、expiry、UE p5/p10、公平性、BLER、PRG utilization 和 packet completion 状态；
4. **可追溯性：** 核心数值可追溯到聚合 CSV/JSON、seed manifest、运行日志和脚本；
5. **可复现性：** 区分脚本/manifest 级复现与 bit-exact 复现。

验收不要求单一算法在所有 KPI 上获胜，而是要求机制可运行、比较受控、结果完整且结论边界清楚。

## 7. 试验设计、变量控制及随机种子

full-KPI suite 分为三个配对子任务：uniform RR/PF、uniform RRQ/PFQ、boundary RRQ/PFQ，再由 suite aggregate 形成统一表。每个子任务都包含 seeds 41--50，聚合文件的 `common_seed_count=10`，各算法 seed 列与共同 seed 列一致。

控制变量包括：3 cells、36 active UEs、12 UE/cell、4T4R、Rayleigh、SVD、Type-0、2.5 GHz、30 kHz SCS、0.5 ms TTI、272 PRB、17 PRG、3000 Byte 固定包、Poisson 1.0 pkt/TTI/UE、4000 TTI、200 ms TTL 与相同 seed。算法是同批配对中的主要变化量。

统计流程为：每个 seed 独立计算 KPI，再对 10 个 seed 的 KPI 求均值和总体标准差。差值使用 paired seed 聚合字段；逐包直方图则先在每个 seed 内计算桶占比，再对 seed 等权平均，避免包数较多的 seed 获得更大权重。

证据层次严格如下：

- **full-KPI uniform：** RR/PF/RRQ/PFQ 四算法验收主线；
- **full-KPI boundary：** 同一 suite 内 RRQ/PFQ 的完整边缘 KPI 和 completed-packet 分布；
- **active boundary gate：** 独立运行批次，仅检查 PFQ 相对 RRQ 的结论方向。

## 8. 四算法统一对比结果

下表只取 full-KPI suite 的 uniform 聚合 CSV，未混入 boundary 或 active gate 数据。

| 指标 | RR | PF | RRQ | PFQ |
|---|---:|---:|---:|---:|
| goodput (Mbps) | 1486.396 ± 31.771 | 925.656 ± 50.922 | 1512.253 ± 35.026 | 1527.240 ± 16.945 |
| p95 delay (ms) | 136.100 ± 66.598 | 199.200 ± 0.245 | 98.550 ± 90.659 | 38.000 ± 49.942 |
| expiry (%) | 1.080 ± 0.961 | 31.023 ± 2.986 | 1.179 ± 1.207 | 0.117 ± 0.215 |
| UE goodput p5 (Mbps) | 34.891 ± 3.594 | 8.542 ± 1.280 | 35.374 ± 4.763 | 39.190 ± 1.570 |
| Jain fairness | 0.993380 ± 0.004774 | 0.805345 ± 0.024030 | 0.993267 ± 0.005662 | 0.997899 ± 0.001119 |
| TB BLER (%) | 11.706 ± 0.665 | 12.411 ± 0.656 | 11.240 ± 0.634 | 11.422 ± 0.703 |
| PRG utilization (%) | 99.999 ± 0.002 | 100.000 ± 0.000 | 89.193 ± 5.149 | 94.509 ± 2.974 |

![Uniform 四算法核心 KPI](figures/stageB_baseline_verification/uniform_core_kpis.png)

![Uniform 四算法资源与可靠性权衡](figures/stageB_baseline_verification/uniform_resource_tradeoffs.png)

![Queue-aware 改造的方向一致化改进量](figures/stageB_baseline_verification/queue_aware_improvements.png)

### 8.1 结果解释

- RR 保持近满 PRG 占用和较高 goodput，但 completed-only p95 与 expiry 表明高负载下仍存在尾部压力。
- RRQ 相对 RR 的 goodput 增加 25.857 Mbps，p95 降低 37.550 ms，PRG 利用率下降 10.806 pp；但 expiry 由 1.080% 增至 1.179%，说明 queue-aware demand cap 的收益不能概括为所有包可靠性指标全面改善。
- PF 当前路径 goodput 为 925.656 Mbps、expiry 为 31.023%、p95 接近 TTL。该结果说明当前高负载 queue-unaware 实现发生队列失配；它不是对 PF 理论的一般性否定。
- PFQ 相对当前 PF 同时改善 goodput、尾时延、过期率、弱 UE 和 Jain，并降低 5.491 pp PRG 占用；这体现当前 queue-aware 机制对该实现的修正效果。
- PFQ 在四算法中具有最均衡的综合 KPI 和较小的 goodput、UE p5、Jain 离散性，因而被认定为本工况下后续 AI 调度应优先对标的强基线，而非宣称其在所有场景普遍最优。

## 9. 时延分布、吞吐、公平性和资源利用分析

### 9.1 Uniform RR 与 RRQ

![Uniform RR 与 RRQ completed-packet 时延分布](figures/stageB_baseline_verification/packet_delay_uniform_rr_vs_rrq.png)

RRQ 将 `0--1 ms` completed-packet 占比从 65.552% 提高到 77.063%，将 `50--200 ms` 尾部占比从 11.224% 降到 5.485%。与此同时，每 seed completed packet 从 139,148.7 增至 139,813.3，pending 从 2,282.3 降至 1,353.7，但 expired packet 从 2,641.2 增至 2,905.2。由此只能确认 RRQ 改善已完成包分布和末端残留，不能据 completed-only p95 推断所有到达包都获得改善。

### 9.2 Uniform PF 与 PFQ

![Uniform PF 与 PFQ completed-packet 时延分布](figures/stageB_baseline_verification/packet_delay_uniform_pf_vs_pfq.png)

当前 PF 的 `150--200 ms` completed-packet 占比为 37.944%，PFQ 为 1.269%；completed packet 从每 seed 83,639.1 增至 143,178.7，expired 从 49,763.0 降至 283.5，pending 从 10,670.1 降至 610.0。三类状态与分布共同说明 PFQ 避免了当前 PF 实现中的 TTL 临界堆积；仅看 PF 的 100% PRG 占用会掩盖业务层退化。

### 9.3 completed-only 统计限制

普通 delay 样本只来自完全离开 MAC packet tracker 的包。达到 TTL 而被移除的包以及仿真结束仍 pending 的包不进入分布，因此图中 `≥200 ms` 为 0 并不代表没有 TTL 风险。尤其当算法间 completed、expired、pending 数差异很大时，p95 必须与三类状态计数联读；本报告所有时延结论均遵循该限制。

goodput 是 PHY 成功解码的系统级有效吞吐，而 completed packet 是 MAC tracker 的完全出队事件。两者不能按固定包长简单当作同一完成语义。PRG utilization 只表示资源机会被分配的比例，高利用率也不自动表示单位资源效率更高。

## 10. Boundary 边缘用户增强验证

### 10.1 full-KPI boundary：正式边缘场景表

下表仅使用 full-KPI suite 中 `boundary_rrq_pfq_metrics.csv`。差值为 paired `PFQ - RRQ`；百分比差值使用百分点。

| 指标 | Boundary RRQ | Boundary PFQ | PFQ - RRQ |
|---|---:|---:|---:|
| goodput (Mbps) | 1494.513 ± 33.711 | 1516.222 ± 17.190 | +21.710 |
| p95 delay (ms) | 94.700 ± 73.961 | 41.850 ± 30.602 | -52.850 |
| expiry (%) | 1.282 ± 1.427 | 0.083 ± 0.112 | -1.199 pp |
| UE p5 (Mbps) | 34.829 ± 4.704 | 38.954 ± 1.084 | +4.124 |
| Jain fairness | 0.992767 ± 0.006513 | 0.998177 ± 0.000836 | +0.005409 |
| PRG utilization (%) | 94.350 ± 6.055 | 98.598 ± 2.218 | +4.248 pp |

![Boundary RRQ 与 PFQ completed-packet 时延分布](figures/stageB_baseline_verification/packet_delay_boundary_rrq_vs_pfq.png)

PFQ 的 `0--1 ms` completed 占比低于 RRQ（55.729% 对 63.839%），但 `150--200 ms` 从 3.770% 降至 0.760%。这说明 PFQ 并非让每个包都更快，而是减少两极化并把部分危险长尾移入 1--100 ms 区间。每 seed completed 从 139,301.3 增至 143,025.5，expired 从 3,103.7 降至 186.4，pending 从 1,667.2 降至 860.3，与 tail 治理方向一致。代价是 PRG utilization 增加 4.248 pp，TB BLER 亦由 11.635% 增至 11.823%（+0.188 pp）。

### 10.2 active boundary gate：独立交叉核验

active gate 位于：

`output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/`

该批次与 full-KPI boundary 独立运行，绝对均值不同，故只列方向核验，不将其 KPI 填入上一表：

| PFQ 相对 RRQ | active gate 差值 |
|---|---:|
| goodput | +6.903 Mbps |
| expiry | -0.592 pp |
| p95 delay | -30.428 ms |
| UE goodput p5 | +1.935 Mbps |
| Jain fairness | +0.002475 |

五项方向均与 full-KPI boundary 一致，支持 PFQ 在 paired seeds 41--50 下的 tail、expiry 和弱 UE 改善具有跨独立运行批次的方向稳定性；但本报告不把这种一致性表述为统计显著性或所有 seed 逐一获胜。

## 11. 多维协同能力分析

时间维上，RRQ/PFQ 的队列约束以及 PFQ 的单 TTI 衰减改变服务次序和等待尾部，证据为 p95、expiry 及 pending 状态。频率维上，Type-0 bitmap 和 demand cap 改变 PRG 分配密度，证据为 PRG utilization 与 goodput 的联合变化。功率/链路维上，PF/PFQ 使用 SVD 后可达率相关信息，TB BLER 与 goodput 反映调度选择在给定链路条件下的结果。空间维上，三小区同频和 boundary strongest association 提供链路不均衡与簇内复用环境，UE p5/p10 与 Jain 反映弱 UE 和分配均衡性。业务维上，buffer、TTL 和 Poisson 到达通过 completed/expired/pending 及 tail delay 得到验证。

上述能力属于“给定链路和空间场景下的时频与队列感知调度”。本次基线动作空间不包含发射功率变量，也没有跨小区联合动作优化；因此“多维协同”不能扩展解释为功控、波束与跨小区动作的统一联合最优化。

## 12. 验证结论与验收符合性

| 验收项 | 状态 | 证据与判断 |
|---|---|---|
| 功能执行 | 满足 | RR、PF、RRQ、PFQ 均完成 10-seed 运行并生成聚合 KPI |
| 配对比较 | 满足 | common seeds 与算法 seed 均为 41--50，同批变量受控 |
| KPI 完整性 | 满足 | 覆盖吞吐、时延、expiry、弱 UE、公平性、BLER、资源占用和包状态 |
| 可追溯性 | 满足 | 可追溯至 CSV/JSON、seed manifest、wrapper log、脚本与图表 |
| bit-exact reproduction provenance | 部分满足 | 具备脚本/manifest 级路径，但缺少五项原始运行时 provenance |

验收结论如下：

1. 四类代表性资源分配基线均已在当前 Type-0 GPU 链路完成多 seed 功能及 KPI 验证；
2. full-KPI uniform 数据足以建立四算法统一比较基准；
3. PFQ 是当前工况下综合最强基线，主要价值是高负载 queue-aware tail/expiry 治理及弱 UE 稳定性，而不是数量级容量增益；
4. full-KPI boundary 与独立 active gate 给出一致方向，支持将 PFQ 作为后续 AI 调度的强比较基准；
5. 本次验收结论为“验证目的达到、运行时 bit-exact provenance 部分满足”。

## 13. 局限性及后续 AI 技术演进建议

### 13.1 局限性

- 结论仅覆盖三小区协调簇内部同频干扰，不含簇外干扰、handover 或动态负载均衡；
- 信道为 Rayleigh、预编码为 SVD；不能外推到 CDL/TDL、不同天线或所有传播条件；
- 单一高负载、固定包长、固定 TTL 和 2 s horizon 不能代表所有业务负载；
- 4T4R 不表示固定四层，实际层选择与传输结果仍受链路条件影响；
- PF 的退化限定于当前 queue-unaware 路径和高负载工况，不能外推为 PF 理论普遍失效；
- completed-only delay 存在选择性，必须与 expired/pending 联读；
- 没有统计假设检验或置信区间，`mean ± std` 只表示 10 个 seed 的描述统计；
- 结果产物未固化部分 PHY、队列容量及运行环境细节，禁止据此作未记录机制的因果解释。

### 13.2 后续 AI 演进建议

后续智能算法首先应复用同一 full-KPI uniform/boundary 配置、paired seeds 和 KPI 口径，与 PFQ 而非当前 raw PF 作主比较。评价函数至少联合 goodput、p95、expiry、UE p5/Jain、BLER 和 PRG utilization，避免通过只优化单一 completed-packet 指标获得表面改善。若未来扩展功率、波束或跨小区动作，应显式扩大动作空间并新增相应约束、合法性检查和独立验收项，不能把本次基线观察维度当作已实现的联合控制能力。

# 技术附录

## 附录 A：仿真参数配置总表

| 参数 | full-KPI uniform | full-KPI boundary | active boundary gate |
|---|---:|---:|---:|
| cells / active UE | 3 / 36 | 3 / 36 | 3 / 36 |
| UE per cell | 12 | 12 | 12 |
| antenna | 4T4R | 4T4R | 4T4R |
| fading / precoding | Rayleigh / SVD | Rayleigh / SVD | Rayleigh / SVD |
| allocation | Type-0 bitmap | Type-0 bitmap | Type-0 bitmap |
| carrier / SCS | 2.5 GHz / 30 kHz | 2.5 GHz / 30 kHz | 2.5 GHz / 30 kHz |
| TTI / horizon | 0.5 ms / 4000 TTI | 0.5 ms / 4000 TTI | 0.5 ms / 4000 TTI |
| PRB / PRG | 272 / 17 | 272 / 17 | 272 / 17 |
| PRB per PRG | 16 | 16 | 16 |
| packet / arrival | 3000 B / Poisson 1.0 pkt/TTI/UE | 同左 | 同左 |
| TTL / warm-up | 200 ms / 0 TTI | 200 ms / 0 TTI | 200 ms / 0 TTI |
| seeds | 41--50 paired | 41--50 paired | 41--50 paired |
| scheduler | RR, PF, RRQ, PFQ | RRQ, PFQ | RRQ, PFQ |

## 附录 B：统计层级与公式

对算法 `a`、seed `s` 的 KPI 记为 `x(a,s)`，报告均值与总体标准差为：

```text
mean_a = (1/10) × Σ_s x(a,s)
std_a  = sqrt[(1/10) × Σ_s (x(a,s) - mean_a)^2]
```

paired 差值先在共同 seed 上作算法差，再聚合。百分比源字段如 `expiry_drop_rate=0.0108002`，报告显示为 `1.08002%`；ratio 差 `-0.0119901` 显示为 `-1.19901 pp`。

Jain 公平性为：

```text
J = (Σ_i x_i)^2 / (N × Σ_i x_i^2)
```

理论 offered load 为 1.728 Gbps；期望生成包数为 `36 × 4000 × 1 = 144,000 packet/seed`，但实际值受有限 Poisson realization 影响。

## 附录 C：代码模块与调用链

```text
run_stageB_baseline_eval_suite.sh
  -> run_stageB_rr_pf_multi_seed_compare.sh
    -> run_stageB_main_experiment.sh
      -> main.cpp: Type-0 active UE population / 参数注入 / KPI 日志
      -> multiCellScheduler.cu / multiCellSchedulerCpu.cpp: PF/PFQ bitmap 分配
      -> roundRobinScheduler.cu / roundRobinSchedulerCpu.cpp: RR/RRQ bitmap 分配
  -> seed compare CSV/JSON
  -> suite aggregate CSV/JSON + packet-delay samples/bins
  -> plot_stageB_baseline_verification_report.py
  -> 本报告图表与表格
```

主要代码入口：

- `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- `cuMAC/src/4T4R/multiCellScheduler.cu`
- `cuMAC/src/4T4R/multiCellSchedulerCpu.cpp`
- `cuMAC/src/4T4R/roundRobinScheduler.cu`
- `cuMAC/src/4T4R/roundRobinSchedulerCpu.cpp`

## 附录 D：一键运行与绘图命令

按批准设计保留的 suite 复现命令：

```bash
cuMAC/scripts/run_stageB_baseline_eval_suite.sh \
  --build-method phase4 \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --suite-output-dir output/stageB_baseline_eval_suite_reproduce
```

从现有 full-KPI 聚合数据生成报告专用图：

```bash
python3 cuMAC/scripts/plot_stageB_baseline_verification_report.py \
  --suite-dir output/stageB_baseline_eval_suite_fullkpi_seed41_50 \
  --output-dir Doc/figures/stageB_baseline_verification
```

复现时必须检查新 run 的实际 `nPrbGrpsConst`、PRB/PRG 配置及 manifest，不因命令入口相同而假设与本报告保留批次 bit-exact 一致。

## 附录 E：证据索引

| 证据 | 仓库相对路径 | 用途 |
|---|---|---|
| uniform 四算法 CSV | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/uniform_four_baseline_metrics.csv` | 第 8 节主表 |
| uniform 四算法 JSON | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/uniform_four_baseline_metrics.json` | 机器可读交叉核对 |
| boundary CSV | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/boundary_rrq_pfq_metrics.csv` | 第 10.1 节 |
| packet bins | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/figures/packet_delay_distribution_bins.csv` | 三张时延分布图 |
| packet by seed | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/figures/packet_delay_distribution_by_seed.csv` | seed 等权分桶审计 |
| packet samples manifest | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/suite_aggregate/packet_delay_samples_manifest_all.csv` | 样本行与来源追踪 |
| full-KPI seed manifests | `output/stageB_baseline_eval_suite_fullkpi_seed41_50/*/scenario_seed_manifest.csv` | seeds/run 路径追踪 |
| active gate CSV | `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/RAYLEIGH/rrq_vs_pfq_compare_mean.csv` | 第 10.2 节独立交叉核验 |
| active gate seed manifest | `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/scenario_seed_manifest.csv` | paired seeds 追踪 |
| 报告绘图脚本 | `cuMAC/scripts/plot_stageB_baseline_verification_report.py` | 专用图生成 |

## 附录 F：可复现性声明与 provenance 缺口

当前仓库可证明：运行入口脚本存在，聚合 manifest 可列出 seed 41--50 和 run/compare 路径，CSV/JSON 可追溯核心数值，wrapper log 可核验实际 PRG 常量，绘图可由保留 CSV 重新生成。因此本报告具备**脚本级和 manifest 级可追溯性**。

现有原始运行产物未固化以下五项信息：

1. 原始实验运行时 Git commit hash；
2. 原始实验运行时 dirty patch；
3. 原始实验容器镜像 digest；
4. 原始实验 compiler options hash；
5. 原始实验 unified configuration hash。

上述缺口意味着无法对上游原始测量作 bit-exact reproduction 声明。当前可执行的是按现有脚本、manifest 和保留配置进行同口径重跑，并比较 KPI 方向与统计范围；未来验收批次应在 run 启动时自动固化这五项 provenance。

## 附录 G：报告内图表索引

- `Doc/figures/stageB_baseline_verification/uniform_core_kpis.png`
- `Doc/figures/stageB_baseline_verification/uniform_resource_tradeoffs.png`
- `Doc/figures/stageB_baseline_verification/queue_aware_improvements.png`
- `Doc/figures/stageB_baseline_verification/packet_delay_uniform_rr_vs_rrq.png`
- `Doc/figures/stageB_baseline_verification/packet_delay_uniform_pf_vs_pfq.png`
- `Doc/figures/stageB_baseline_verification/packet_delay_boundary_rrq_vs_pfq.png`

---

**最终声明：** 本报告验收的是四类代表性基线在所列 Stage-B 工况下的可执行性、受控比较、KPI 完整性与可追溯性。量化结果是后续 AI 调度的比较基准，不构成对未验证拓扑、信道、负载、功控或跨小区动作空间的性能承诺。

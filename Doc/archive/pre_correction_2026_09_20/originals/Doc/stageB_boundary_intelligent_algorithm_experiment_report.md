# Stage-B DPA-v1、PFQ 与 RRQ 综合实验报告

> 状态：当前主实验报告，更新于 2026-09-07。
> 算法定义以 [DPA-v1 主设计](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md) 为准，实验参数与 KPI 口径以 [当前生效配置](./current_stageB_effective_configuration.md) 为准。本文合并并替代原有 iter21、feature-only、seed48/51、standard DPA/PFQ 和 packet-granularity 阶段 brief。

## 1. 执行摘要

消融研究更新（2026-09-11）：[随机初始化等预算实验](dpa_equal_budget_ablation_experiment.md) 剩余评估保持暂停；[共同冻结 checkpoint 的短程适应消融](dpa_checkpoint_adaptation_ablation.md) 与冻结删除属于诊断，不能替代全生命周期训练消融。当前启动[成功训练路径复现与四版本成对消融](dpa_training_reproduction_ablation_v1_preflight.md)：先完成 P0 并验收 Full801，通过后再展开 Full／−Attention／−Features／−Reward × 3 个训练 seed。当前尚无该新协议的性能结论，不覆盖本报告已有系统基线证据。

当前冻结算法 DPA-v1 在两组严格对齐实验中得到以下结论：

1. **标准业务 seeds 41--50，DPA-v1 相对 PFQ 全面占优。** Steady 3000 TTI 中，mean goodput `1545.803` vs `1530.370 Mbps`（`+1.01%`，10/10 seed 胜），BLER 相对下降 `6.30%`，outcome p95 delay 从 `29.10` 降到 `1.90 ms`，UE p10 expiry-aware rate 从 `6.888` 提高到 `19.487 Mbps`。
2. **未见 seeds 101--110 的 mixed packet-granularity 中，DPA-v1 相对 PFQ/RRQ 的收益仍成立。** 相对 PFQ，mean goodput `+14.195 Mbps`，95% CI `[5.396, 22.994]`；相对 RRQ 为 `+17.101 Mbps`，CI `[8.309, 25.893]`。mean/p95 outcome delay 和 end backlog 同时下降。
3. **收益不是靠更多 PRG 换来的。** mixed 场景相对 PFQ，DPA-v1 PRG utilization 少 `6.635 pp`，但 successful bytes/active PRG 多 `167.6 B`，mean-cell spectral efficiency 多 `0.0483`。
4. **packet granularity 本身没有显示稳定的总体性能损失。** DPA mixed 相对 homogeneous 的 goodput delta 为 `+1.057 Mbps`，CI `[-6.457, 8.571]`；offered-normalized goodput、expiry 和 mean-cell spectral efficiency 的 CI 也跨 0。可确认的变化是 utilization 下降而每 active PRG 效率上升。
5. **不能宣称 zero expiry 或 worst-seed 保证。** mixed DPA 在 seeds104/105/107/110 有非零 expiry，10-seed mean `1.278%`，seed104 达 `9.080%`；seed107 goodput 也低于 PFQ/RRQ。
6. **实验 guard 不构成保护证据。** seed51 的 CTBS-legal non-starvation guard 已实现并运行，但 `override_count=0`，原因是合法 Blank 机会不足。当前 DPA-v1 仍明确使用 guard off。

因此，核心假设“PFQ-independent actor 能在相同负载与合法动作约束下优于 queue-aware baselines”得到支持；更强的“对所有 seed 零过期”假设被否定。

## 2. 算法与比较对象

### 2.1 DPA-v1

DPA-v1（Deadline-Pressure Actor v1）对应代码类 `PFQIndependentReuseActor` / family `pfq_independent_reuse_v1`。它使用 Cell--UE--Cell-PRG 图、同 local PRG 跨小区 reuse attention、无量纲 deadline/service features，以及 CTBS-v1 累计 TBS headroom，直接选择 UE 或 Blank。

当前冻结 checkpoint：

```text
output/pfq_independent_reuse/formal/
  deadline_dimensionless_features_seed45_v1/
  train_attempt2/ppo_actor_last.pt
```

- file SHA-256：`94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`
- tensor SHA-256：`6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`
- deployment：iter8、argmax、CTBS-v1、guard off

PFQ teacher 信息只参与训练沿革，不进入 DPA-v1 的在线 observation/action。

### 2.2 PFQ 与 RRQ

- **PFQ**：queue-aware proportional-fair native reference。
- **RRQ**：queue-aware round-robin native reference。

二者都走 native Type-0 调度执行路径。旧 RR/PF 非 queue-aware baseline 只保留为历史机制分析，不是本报告的主要结论对象。

## 3. 实验矩阵与有效性

### 3.1 Campaign A：standard DPA/PFQ

- seeds：41--50
- schedulers：DPA-v1、PFQ
- traffic：36 UE 全部 `Poisson(1 packet/TTI) × 3000B`
- horizon：连续 `4000 TTI`
- headline：TTI `1001..4000`
- supplementary：TTI `1..4000`
- simulator binary SHA-256：`4bdfee54c3203a808f42fd85367f6c46457bbf20dd1314e0c5855d2f07400a10`

每个 seed 的两臂满足 binary、offered stream、topology 和 horizon 对齐。全部 20 arms 均有 4000 decision records 和 `36 × 4000` service records；DPA illegal action、candidate truncation 和 deterministic replay error 为 0。

### 3.2 Campaign B：fixed-load packet granularity

- seeds：101--110，训练和 earlier selection 均未使用
- profiles：homogeneous、mixed granularity
- schedulers：DPA-v1、PFQ、RRQ
- horizon/headline：连续 4000 / TTI `1001..4000`
- simulator binary SHA-256：`28f971218b50c6e5315310a8543c13894e07adfa12ca0303ff438decd56593d2`

业务定义：

| Profile | UE 组 | 到达与包长 | 每 UE 期望负载 |
|---|---|---|---:|
| homogeneous | 全 36 UE | `Poisson(1) × 3000B` | 3000 B/TTI |
| mixed | 18 个偶数 UE | `Poisson(2) × 1500B` | 3000 B/TTI |
| mixed | 18 个奇数 UE | `Poisson(0.25) × 12000B` | 3000 B/TTI |

共 60 arms，核验 `240000` decision records 和 `8640000` UE-service records。同 seed/profile 的三 scheduler 共享 binary 和 offered stream；全部 arm 完成 4000 TTI。

### 3.3 可比性边界

Campaign A 与 B 的 simulator binary hash 不同，因此：

- campaign 内的 paired 比较是正式因果证据；
- campaign 间只能比较方向和稳健性，不能把绝对值相减后当成 scheduler 效果。

## 4. Campaign A：DPA-v1 对 PFQ

### 4.1 Headline：steady 3000 TTI

以下是 seeds 41--50 的跨 seed 均值：

| 指标 | DPA-v1 | PFQ | DPA - PFQ / 相对变化 | DPA 胜 seed |
|---|---:|---:|---:|---:|
| goodput (Mbps) | 1545.803 | 1530.370 | `+15.433` / `+1.01%` | 10/10 |
| BLER | 10.712% | 11.433% | 相对 `-6.30%` | 10/10 |
| outcome delay mean (ms) | 0.934 | 4.359 | `-78.57%` | 10/10 |
| outcome delay p95 (ms) | 1.90 | 29.10 | `-93.47%` | 10/10 |
| UE macro expiry-aware rate (Mbps) | 29.238 | 27.424 | `+6.61%` | 9/10 |
| UE p10 expiry-aware rate (Mbps) | 19.487 | 6.888 | `+182.93%` | 10/10 |
| UE min expiry-aware rate (Mbps) | 14.126 | 2.562 | `+451.45%` | 10/10 |
| PRG utilization | 84.868% | 95.300% | `-10.432 pp` | context |

DPA-v1 在 10 个 seed 的 steady window 都是零 expiry。PFQ 仅 seed43 出现 expiry，seed43 为 `0.7353%`，因此 PFQ 跨 seed mean 为 `0.0735%`。

结果的关键不是单一 goodput：DPA 同时降低 BLER、长尾时延和弱 UE 退化，并使用更少的 active PRG。这与“通过无条件占满资源提高吞吐”的解释不相容。

### 4.2 Full 4000 TTI 一致性检查

| 指标 | DPA-v1 | PFQ | DPA - PFQ / 相对变化 |
|---|---:|---:|---:|
| goodput (Mbps) | 1535.879 | 1518.640 | `+17.240 Mbps`，10/10 seed 胜 |
| BLER | 11.131% | 11.822% | `-0.691 pp` |
| expiry drop rate | 0 | 0.0552% | `-0.0552 pp` |
| outcome delay mean (ms) | 0.922 | 3.838 | `-2.916 ms` |
| outcome delay p95 (ms) | 1.950 | 25.968 | `-24.018 ms` |
| UE macro expiry-aware rate (Mbps) | 29.378 | 27.459 | `+1.919 Mbps` |
| UE p10 expiry-aware rate (Mbps) | 19.742 | 7.138 | `+12.604 Mbps` |
| UE min expiry-aware rate (Mbps) | 14.472 | 2.689 | `+11.783 Mbps` |
| PRG utilization | 84.353% | 95.078% | `-10.725 pp` |

full-window 与 headline 的方向完全一致，说明结论不是只靠去除前 1000 TTI 后才成立。

## 5. Campaign B：DPA-v1 对 PFQ/RRQ

### 5.1 Steady 3000 跨 seed 汇总

数值为 `mean ± sample std`；expiry/BLER/goodput-offered 使用百分比。

| Profile | Scheduler | Goodput (Mbps) | Goodput/offered | BLER | Expiry | Delay mean (ms) | Delay p95 (ms) | End backlog (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | DPA-v1 | 1513.109 ± 38.953 | 87.525 ± 2.319% | 11.369 ± 0.601% | 1.271 ± 2.580% | 5.812 ± 6.240 | 24.70 ± 61.94 | 1.307 ± 1.340 |
| homogeneous | PFQ | 1498.374 ± 42.315 | 86.672% | 11.804% | 1.278% | 17.081 | 108.95 | 4.272 |
| homogeneous | RRQ | 1492.844 | 86.352% | 11.374% | 1.935% | 24.101 | 155.95 | 6.157 |
| mixed | DPA-v1 | 1514.166 ± 44.914 | 87.672% | 11.311% | 1.278 ± 2.813% | 6.733 | 41.35 | 1.082 |
| mixed | PFQ | 1499.971 | 86.849% | 11.716% | 1.236% | 17.387 | 117.90 | 4.040 |
| mixed | RRQ | 1497.065 | 86.681% | 11.426% | 1.536% | 15.400 | 107.10 | 5.490 |

为保持表格可读，仅 DPA 行展开了主要 std；PFQ/RRQ 的完整 std、逐 seed 值和所有资源指标在 `campaign_report_10seed.json` 与各 seed CSV 中。

### 5.2 Mixed：DPA-v1 相对 PFQ

| 指标 | Paired mean delta | 95% CI | DPA 胜 seed | 判断 |
|---|---:|---:|---:|---|
| goodput | +14.195 Mbps | [5.396, 22.994] | 8/10 | 稳定提高 |
| goodput/offered | +0.823 pp | [0.312, 1.334] | 8/10 | 稳定提高 |
| expiry | +0.042 pp | [-0.491, 0.575] | — | 无稳定差异 |
| outcome delay mean | -10.653 ms | [-17.153, -4.153] | 10/10 | 稳定降低 |
| outcome delay p95 | -76.550 ms | [-132.828, -20.272] | — | 稳定降低 |
| end backlog | -2.958 MB | [-4.788, -1.128] | — | 稳定降低 |
| mean-cell spectral efficiency | +0.0483 | [0.0184, 0.0783] | — | 稳定提高 |
| PRG utilization | -6.635 pp | [-8.985, -4.285] | — | 更少资源占用 |
| successful bytes/active PRG | +167.6 B | [104.9, 230.3] | — | 稳定提高 |
| cell BLER | -0.387 pp | [-0.716, -0.057] | — | 稳定降低 |

min-cell spectral efficiency 没有稳定差异；cell Jain 略低但 CI 跨 0，不能写成确定的公平性退化。

### 5.3 Mixed：DPA-v1 相对 RRQ

| 指标 | Paired mean delta | 95% CI | DPA 胜 seed | 判断 |
|---|---:|---:|---:|---|
| goodput | +17.101 Mbps | [8.309, 25.893] | 9/10 | 稳定提高 |
| goodput/offered | +0.991 pp | [0.481, 1.501] | 9/10 | 稳定提高 |
| expiry | -0.258 pp | [-0.814, 0.297] | — | 无稳定差异 |
| outcome delay mean | -8.667 ms | [-13.413, -3.920] | 10/10 | 稳定降低 |
| outcome delay p95 | -65.750 ms | [-110.124, -21.376] | 8/10，2 tie | 稳定降低 |
| end backlog | -4.408 MB | [-6.488, -2.328] | 10/10 | 稳定降低 |
| mean-cell spectral efficiency | +0.0582 | [0.0283, 0.0881] | 9/10 | 稳定提高 |
| PRG utilization | -2.757 pp | [-4.433, -1.080] | — | 更少资源占用 |
| successful bytes/active PRG | +87.7 B | [45.3, 130.2] | 10/10 | 稳定提高 |
| cell Jain | -0.00321 | [-0.00619, -0.00022] | — | 小幅但稳定降低 |

与 RRQ 比较显示 DPA 的吞吐/效率优势仍在，但 cell Jain 有小幅代价。该结果应与 mean/min-cell SE、weak-UE rate 和 expiry 联合解读，不能用单一 Jain 指标替代服务质量结论。

## 6. Packet granularity 影响

同一个 DPA-v1 checkpoint 下，mixed - homogeneous 的 paired 对照为：

| 指标 | Mean delta | 95% CI | mixed 胜 seed | 结论 |
|---|---:|---:|---:|---|
| goodput | +1.057 Mbps | [-6.457, 8.571] | 7/10 | 无稳定差异 |
| goodput/offered | +0.147 pp | [-0.263, 0.558] | 8/10 | 无稳定差异 |
| expiry | +0.006 pp | [-0.337, 0.349] | — | 无稳定差异 |
| mean-cell spectral efficiency | +0.0036 | [-0.0220, 0.0292] | 7/10 | 无稳定差异 |
| PRG utilization | -1.249 pp | [-1.862, -0.636] | — | mixed 稳定更低 |
| successful bytes/active PRG | +31.5 B | [17.5, 45.5] | 10/10 | mixed 稳定更高 |

这组实验支持“在保持每 UE 期望字节负载不变时，DPA 的总体性能没有因 packet granularity 改变而系统性下降”。但 mixed 同时改变包长与 Poisson 到达率，不能把差异严格归因于包长单一因素。

## 7. Stress seeds 与被否定的强结论

mixed DPA 的非零 expiry seeds：

| Seed | Expiry | 观察 |
|---:|---:|---|
| 104 | 9.080% | 主要 stress case；三 scheduler 都受场景压力影响，DPA 仍高于 PFQ goodput，但离 zero-expiry 很远 |
| 105 | 0.741% | DPA goodput 略低于 PFQ |
| 107 | 1.755% | DPA goodput低于 PFQ/RRQ，需优先做因果诊断 |
| 110 | 1.201% | 轻度非零 expiry |

因此：

- “标准 seeds 41--50 上 DPA zero expiry”是准确的 campaign 内事实；
- “DPA-v1 普遍 zero expiry”不准确；
- 总体 goodput/delay 优势不能掩盖 seed104/107 的 worst-seed 风险。

## 8. 从 actor reuse 到 DPA-v1 的实验沿革

本节保留被合并阶段 brief 中仍有解释价值的最小证据。

### 8.1 Deadline-tail-rate iter21

seed42 formal evaluation 中，actor 相对 PFQ：

- goodput `1541.572` vs `1521.627 Mbps`
- BLER `11.009%` vs `12.128%`
- expiry `0`
- outcome p95 delay `2.5` vs `30 ms`
- UE p10 expiry-aware rate `13.663` vs `3.601 Mbps`
- PRG utilization `84.134%` vs `98.513%`

但随后 seed43/45 出现 starvation。诊断发现 deadline pressure、service gap 和 drain deficit 已能标出失败 UE，actor logits 却仍把它们排在可服务动作之后。因此问题不是“再增加一个告警量”，而是绝对量纲和排序泛化不足。

### 8.2 Dimensionless feature-only

在 seed45、固定 iter8、1000 TTI evaluation 中：

- goodput `1543.612 Mbps`
- BLER `10.427%`
- expiry `0`
- outcome mean/p95 delay `0.921/2 ms`
- UE macro/p10/min expiry-aware rate `27.115/14.075/10.995 Mbps`
- PRG utilization `70.600%`
- max backlogged zero-dequeue streak `9 TTI`

旧失败 UE4/UE11 恢复服务，支持将 `log1p pressure`、scheduled-dequeue adequacy 和 normalized service age 固定为 DPA-v1 表示。

### 8.3 Frozen seed48/51 generalization

- seed48：DPA 相对 PFQ goodput `1550.059` vs `1545.480 Mbps`，BLER `10.615%` vs `10.623%`，expiry 均为 0，mean delay `1.188` vs `5.616 ms`，UE p10 rate `13.484` vs `3.897 Mbps`；Gate 通过。
- seed51：DPA goodput `1481.583` vs `1469.662 Mbps`，BLER `10.198%` vs `10.866%`，mean delay `12.203` vs `43.721 ms`，UE p10 rate `9.040` vs `0.101 Mbps`，但 expiry `4.8509%` 高于 PFQ `4.0893%`；Gate 失败。

seed51 将剩余问题定位到 UE4/UE6 的 PRG 与 scheduled dequeue 塌陷，说明 feature-only 可以显著改善平均和尾部服务，却不能提供 worst-UE 保证。

### 8.4 CTBS-legal non-starvation guard

针对 seed51 实现了只能替换 Blank、不能抢占 actor 非 Blank 分配的合法 guard。steady measured window 的实际审计：

```text
actor_nonblank_choices = 159087
no_pressure_candidate  = 44913
override_count         = 0
classification         = insufficient_blank_opportunities
```

guarded 与 unguarded 动作/结果相同，所以该实验没有支持 guard 有效性。它只证明当前“仅替换 Blank”的干预空间不足；正式 DPA-v1 保持 guard off。

## 9. 综合判断

### 9.1 得到支持

- DPA-v1 的 PFQ-independent online 路径可以在标准 10-seed 中稳定超过 PFQ。
- 优势同时覆盖 goodput、BLER、outcome delay、weak-UE rate 与资源效率，不是单指标取舍。
- 在未见 seed、相同期望负载但不同 packet granularity/arrival process 下，DPA 对 PFQ/RRQ 的总体优势仍可复现。
- 无量纲 deadline/service 表示是从旧 actor reuse 到 DPA-v1 的关键有效改动。

### 9.2 未得到支持或已被否定

- 所有 seed zero expiry：已被 seeds104/105/107/110 否定。
- 仅替换 Blank 的 guard 能修复 starvation：seed51 中没有触发机会，尚未得到支持。
- 已证明 7cell/84UE 泛化：未做。
- mixed 与 homogeneous 性能完全相同：只能说大多数 delta 的 CI 跨 0，不能证明等价。
- DPA 已达到生产级安全保证：当前仍是研究原型。

### 9.3 下一项最小实验

先对 seed104 与 seed107 做只读逐 UE/逐 PRG 因果归因，检查：

1. deadline pressure 是否及时进入高位；
2. 高 pressure UE 是否有合法候选与可用 layer cap；
3. actor 是选择 Blank、选择其他 UE，还是被 CTBS headroom 提前封顶；
4. expiry 前的 service-age、logit margin 和 reuse conflict 如何演化。

停止条件是能把失败归入“表示不足、actor 排序、CTBS/顺序状态或必须抢占的安全约束”之一。得到这一证据前，不扩建新的通用 guard 或 7cell 架构。

## 10. 原始证据与复现入口

### 10.1 Standard DPA/PFQ

```text
output/pfq_independent_reuse/formal/
  dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/
    campaign_manifest.json
    final_report.json
    aggregate_comparison.{json,csv}
    per_seed_comparison.{json,csv}
    seed41..seed50/
```

### 10.2 Packet granularity DPA/PFQ/RRQ

```text
output/pfq_independent_reuse/formal/
  granularity_newseed_stage1_v1/
    campaign_report_10seed.json
    seed101..seed110/
      granularity_report.json
      headline_steady.csv
      per_ue_steady.csv
      homogeneous/{dpa_v1,pfq,rrq}/
      mixed_granularity_v1/{dpa_v1,pfq,rrq}/
```

`campaign_report.json` / `campaign_report_3seed.json` 是 stage-1 停止点快照；最终结论只使用 `campaign_report_10seed.json`。

### 10.3 Stage evidence

```text
output/pfq_independent_reuse/formal/
  deadline_dimensionless_features_seed45_v1/
  iter8_dimensionless_frozen_generalization_seed48_seed51_v1/final_report.json
  iter8_ctbs_legal_non_starvation_guard_seed51_v1/final_report.json
```

### 10.4 代码与验证

- standard comparator：`training/stageb_hgraph/pfq_dpa_aligned_comparison.py`
- packet experiment：`training/stageb_hgraph/granularity_experiment.py`
- historical/window metrics：`training/stageb_hgraph/dpa_v1_historical_comparison.py`
- focused tests：
  - `training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py`
  - `training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py`
  - `training/stageb_hgraph/tests/test_granularity_experiment.py`

2026-09-07 文档复核时在实验容器内运行上述测试：`19 passed in 1.49s`。

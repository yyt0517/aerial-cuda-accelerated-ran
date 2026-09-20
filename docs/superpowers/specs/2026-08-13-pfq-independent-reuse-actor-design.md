# DPA-v1（PFQ-independent reuse actor）主设计

> 状态：保留算法结构与接口契约，更新于 2026-09-20；历史性能结论撤回，尚无修正环境中新训练并验收的 DPA。
> 容量单位、PHY slot 索引、GPU MCS 同步已修正，见[重大变更与有效性分界](../../../Doc/stageB_corrected_environment_v1.md)。[修改前原文](../../../Doc/archive/pre_correction_2026_09_20/originals/docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)保留历史记录；当前结果统一见[修正环境实验报告](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)。

## 1. 定位与结论

DPA-v1（Deadline-Pressure Actor v1）是 Stage-B 历史冻结的、在线不依赖 PFQ 的图策略。代码中的兼容名称仍为：

- actor class：`PFQIndependentReuseActor`
- actor family：`pfq_independent_reuse_v1`
- 报告名称：`DPA-v1`

三者指向同一个算法，不代表三个版本。保留旧代码名是为了 checkpoint 和已有实验目录兼容。

DPA-v1 回答的核心问题是：在不读取 PFQ score、PFQ ranking 或 PFQ action 的前提下，能否从因果可用的队列、截止期、信道和历史服务状态直接生成 Type-0 PRG 分配，并在吞吐、可靠性、时延和弱 UE 服务上超过 PFQ/RRQ。

当前证据边界：

- 旧 seeds41–50 paired 结果、seeds101–110 packet-granularity 结果，以及训练/消融结果来自存在关键错误的环境，已归档，不能证明修正环境中 DPA 超过 PFQ/RRQ。
- 修正环境当前完成 seed42 PFQ/RRQ 短基线与三项错误的限定验证，尚未证明 DPA 或 reuse attention/deadline 组件的性能贡献。
- 旧 guard 的零 override 是历史观察，不是修正环境的 non-starvation 保证；重新实验时须明确 guard 配置。
- 下文保留结构、特征和解码语义，但不继承旧容量、旧训练门槛及旧 checkpoint 的部署有效性。

## 2. 冻结身份与复现边界

历史 DPA-v1 checkpoint（原路径现为归档兼容链接，仅限历史研究/明确标注的迁移诊断）：

```text
output/pfq_independent_reuse/formal/
  deadline_dimensionless_features_seed45_v1/
  train_attempt2/ppo_actor_last.pt
```

- 文件 SHA-256：`94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`
- tensor-state SHA-256：`6039c59e92ecda5ed75731fef86b71f567fe0c227142d9ed267434934b1e56be`
- 冻结 iteration：`8`
- 训练初始化：`deadline_tail_rate_alignment_v1/.../ppo_actor_absolute_best.pt`
- 训练 reward：treatment `E`，deadline-tail potential + tail-rate utility
- 历史评估配置：deterministic `argmax`，CTBS-v1，guard off；不代表当前部署验收

所有性能声明必须同时给出 checkpoint、二进制 hash、seed、业务 profile、窗口和解码配置。不同二进制或不同 offered stream 的结果只能作为跨 campaign 参考，不能做严格 paired 因果比较。

新实验还必须记录共享库 hash 和三项修正的代码身份。旧 trajectory/BC/归一化状态不能通过字段除以 8 复用；修正环境需重新采集并训练。

## 3. 在线算法契约

每个 TTI 的决策流程如下：

1. 从 online bridge 读取一次 pre-action snapshot。
2. 构造 Cell、UE 和 Cell-PRG 三类实体及稀疏关系。
3. 对整张图编码，并在相同 local PRG 的不同 cell token 之间做 reuse attention。
4. 按固定 local PRG 顺序 `0..16` 解码。
5. 每个 local PRG 同时为三个 cell 从同一份 pre-update 状态计算 logits。
6. 以 `argmax` 选择本 cell 的某个 UE 或 Blank。
7. 三个 cell 的选择确定后，按固定 cell 顺序更新累计 CTBS headroom 与 sequential features。
8. materializer 将选择写成合法 Type-0 bitmap，送入现有仿真执行路径。

同一 reuse group 必须“先统一打分、后顺序更新”。如果 cell 0 的选择提前污染 cell 1/2 的打分输入，便破坏了同 PRG 跨小区决策的可比性。

### 3.1 候选集合

每个 cell 的候选集合是：

```text
所有与该 cell 关联、可调度且 pre-action demand > 0 的 UE + Blank
```

候选宽度由运行时拓扑决定，禁止截断。冻结 checkpoint/eval metadata 记录的 `training_observed_max_candidates=13` 是 UE 候选观察值，不含额外 Blank，也不是部署上限。例如 seed45 正式评估的 association candidate counts 为 `[7, 20, 9]`，运行时 `K=20`，加入 Blank 后 logit 宽度为 21。源码 `IndependentActorConfig` 的默认 `max_candidates=12` 同样不能解释为截断上限；正式复现应以 checkpoint metadata 和 runtime full-width validation 为准。

候选合法性来自 mask；无 demand、未关联、不可调度 UE 不得通过低 logit“软排除”，而必须被硬 mask。

### 3.2 动作

动作是每个 `(cell, local_prg)` 的一个离散 token：

```text
UE id | Blank
```

一个 UE 可在同 TTI 获得多个 local PRG。Blank 表示该 cell 在该 PRG 上不发射，属于资源效率/干扰控制动作，不等价于“没有候选”。

## 4. 状态表示

### 4.1 图结构

- Cell node：小区负载、活跃 UE 数、平均宽带 SINR、平均速率和 bridge 当前 snapshot 的 TB-error rate。
- UE node：队列、长期速率、宽带信道、HARQ/新数据、时延与服务历史，共 16 维。
- Cell-PRG node：局部/邻区信道、历史占用、reuse conflict/ICI proxy、post-EQ 和 expected-goodput 类特征，共 12 维。
- Cell--Cell edge：拓扑邻接及两维 edge attribute。
- UE--Cell-PRG edge：候选 UE 与对应 PRG token 的关系及五维 edge attribute。

模型先分别编码三类 node，经过两层 `HeteroMessageLayer`；随后对同一 local PRG 的跨 cell embedding 使用 4-head self-attention，得到 reuse context。

当前模型参数：

| 参数 | 值 |
|---|---:|
| cell / UE / PRG feature dim | 5 / 16 / 12 |
| hidden dim | 128 |
| message layers | 2 |
| UE--PRG edge attr dim | 5 |
| reuse attention heads | 4 |
| dropout | 0 |

UE scorer 输入 reuse、cell、UE、candidate-edge context 和 3 维 sequential features；Blank 使用独立 scorer。

实现说明：`FeatureSnapshotBuilder` 当前会维护私有 `_cell_tb_err_ema`，但 `cell_values` 实际直接传递 observation 的前 5 维，并未用该 EMA 覆盖 slot 4。因此算法文档必须按“当前 TB-error rate”描述；若未来真的切换为 EMA，需要作为输入分布变更重新版本化和评估。

### 4.2 无量纲 deadline/service 特征

旧版直接使用绝对 backlog、service gap 和手工阈值，能识别 starvation，却不能稳定改变策略排序。DPA-v1 将 UE slot 12--15 固定为：

```text
S_hat(t+1) = 0.95 * S_hat(t) + 0.05 * scheduled_dequeue_bytes(t)
slack_slots  = max(TTL slack in TTI, 1)
pressure     = Q / (S_hat * slack_slots + 1)

f12 = log1p(pressure)
f13 = clip(S_hat * slack_slots / max(Q, 1), 0, 1)
f14 = clip(service_age_tti / packet_ttl_tti, 0, 1)
f15 = 0
```

当前 `TTL=200 ms`、TTI `0.5 ms`，所以 `packet_ttl_tti=400`。当 `Q=0` 时，四个 slot 全部为 0。这里使用 scheduled dequeue，而不是 successful goodput，目的是让状态反映调度器可控的服务行为，同时避免由 PHY 成败引入错误的“未调度”解释。

三者分别表达：按剩余期限归一化的排空压力、期限内预计可服务比例、归一化连续未服务年龄。`f15` 保留为零是版本契约，不应悄悄复用。

### 4.3 Sequential features

同一 TTI 内，一个 UE 获得 PRG 后，其后续候选打分会读取 3 维顺序状态，用于表达已分配资源和剩余需求。它们只由当前 TTI 已经确定的动作及 causal CTBS state 更新，不得读取 post-action label。

## 5. CTBS-v1：累计 TBS headroom

早期设计使用“单 PRG expected service”累加，会把 TBS 当线性函数，和真实 PHY 的累计 PRB/TBS 过程不一致。该描述已经废止。DPA-v1 使用 CTBS-v1：

1. bridge 在动作前导出 CPU OLLA 状态、第一层 WB SINR、prior TB error、每 UE/PRG layer cap、PHY TBS 配置与 snapshot/config hash。
2. Python 侧按与 PHY 对齐的规则重建 UE MCS。
3. 对同一 UE 已选 local PRG 集合，取有效 layer cap，并按累计 PRB 数重新计算 TBS。
4. headroom 为：

```text
remaining_demand = max(0, initial_queue_bytes - cumulative_tbs_bytes)
```

5. 每增加一个 PRG 都重算 cumulative TBS，而不是将单 PRG 容量简单相加。

2026-09-20 单位契约：PHY TBS 原始值为 bits，进入 byte headroom 前转换为 bytes；不得将 bit 数直接扣减 byte 队列。需要独立数值检查及 `8 * scheduled_dequeue_bytes <= raw_TBS_bits`，不仅是 Python 与 C++ 相互一致。旧 CTBS 对齐测试不能单独证明物理单位正确。

CTBS 输入必须带正的 snapshot id、64 位 config hash、canonical UE id 和合法 layer cap。post-action actual MCS/layer/TBS 只用于审计重建一致性，不可回流到当次动作。

## 6. PFQ 独立性

PFQ 只允许出现在两处：

- 早期 behavior cloning 的 teacher label；
- 实验中的 baseline arm。

DPA-v1 在线前向不得读取：

- PFQ score、ranking 或 selected UE；
- teacher action；
- PFQ 内部 moving average 或队列权重；
- 从 post-action 结果反推的当前动作标签。

`actor_family` 声明 `requires_pfq_features=False`。删除 PFQ/teacher 字段前后，actor-visible tensors 和动作必须相等；相关 fail-closed 测试由 `test_independent_pfq_deletion.py` 和 online-path tests 覆盖。

## 7. 训练目标与版本沿革

冻结 checkpoint 使用 treatment E。基础 reward 为：

```text
R_base = goodput_bytes / offered_bytes
       - failed_bytes / offered_bytes
       - 2 * expired_bytes / offered_bytes
```

再加入：

- deadline-tail potential，`eta=0.25`，history window `100`，tail fraction `0.10`；
- expiry-aware tail-rate utility，weight `0.25`，tail fraction `0.10`。

代码仍保留 treatments A--D 供消融，但它们不是 DPA-v1 冻结身份。

2026-09-08 的[等预算重训消融](../../../Doc/dpa_equal_budget_ablation_experiment.md)使用 `full` / `no_reuse_attention` / `no_dimensionless`，三组从随机初始化使用 reward E，不改变历史冻结 checkpoint 的训练沿革。`IndependentActorConfig.ablation` 默认 `full`，旧 checkpoint 缺省时仍加载完整 DPA。该 campaign 已因环境错误归档，不作为修正环境中 attention 或 deadline 特征的贡献证据，也不自动续跑。

关键研究沿革（均为错误环境中的历史观察，不是修正后的有效性能证据）：

| 阶段 | 当时观察 | 历史解释（须重验） |
|---|---|---|
| deadline-tail-rate iter21 | seed42 明显改善，但 seed43/45 出现 actor 排序导致的 starvation | 被 DPA-v1 表示修正替代 |
| dimensionless feature-only seed45 | 旧失败 UE 恢复，最长 backlog 零服务 streak 降至 9 TTI | 支持采用当前 UE slots 12--15 |
| frozen seed48/51 | seed48 全门通过；seed51 大部分 KPI 改善但 expiry 失败 | 暴露 worst-seed 风险 |
| CTBS-legal guard seed51 | guard 已实现，但合法 Blank 机会不足，0 次 override | 不进入当前部署 |
| DPA-v1 seeds 41--50 | 旧环境 paired 优于 PFQ | 已归档，须重新验证 |
| packet granularity seeds 101--110 | 旧环境相对 PFQ/RRQ 优势总体保持，但非零 expiry 暴露 | 已归档，须重新验证 |

历史完整数字保留在[原综合报告](../../../Doc/archive/pre_correction_2026_09_20/originals/Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)。后续不得自动续跑旧随机初始化、适应或生命周期复现队列。

## 8. Experimental guard

guard 的研究目标是在不制造非法 Type-0 动作的前提下，为高 pressure/长 service-age UE 替换一个 Blank。它不能抢占已由 actor 分配给其他 UE 的 PRG。

旧错误环境 seed51 steady measured window 的历史记录：

- actor non-Blank choices：`159087`
- `no_pressure_candidate`：`44913`
- guard override：`0`
- 分类：`insufficient_blank_opportunities`

因此该历史实验中 guard 没有改变动作和 KPI，也没有验证“能降低 expiry”的假设。旧 DPA-v1 配置为 `guard=off`；不得把“guard 代码存在”写成“guard 已生效”。

## 9. 正确性不变量

实现或评估 DPA-v1 时必须检查：

1. snapshot 是 pre-action 且 TTI/snapshot id 单调。
2. 图 shape、UE id、serving cell、PRG index 与运行时拓扑一致。
3. 候选全集完整，禁止静默 truncation。
4. illegal、masked 或跨 serving-cell UE 永不落入 materialized action。
5. 每个 TTI 恰有 `n_cell * n_prg` 个 UE/Blank 决策。
6. 三个 cell 在同 local PRG 使用同一 pre-update 状态打分。
7. CTBS cumulative TBS 与审计 reference 对齐，且 reference-label deletion equality 成立。
8. PFQ/teacher 删除前后 actor 输入和动作相等。
9. deterministic replay 在同 snapshot/checkpoint 上动作一致。
10. baseline/treatment 的二进制、offered stream、拓扑和统计窗口在 paired 声明中完全对齐。
11. bits/bytes 单位和流量守恒通过独立数值检查；slot 重排不改变固定物理输入结果；GPU 路径需独立检查原始 PRG 计数/MCS，不用 CPU goodput 替代验证。

聚合代码的最小验证集：

```text
training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py
training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py
training/stageb_hgraph/tests/test_granularity_experiment.py
```

2026-09-07 历史复核结果：`19 passed`，仅说明当时聚合测试通过，不代表三项环境错误不存在。修正验证以新报告为准。

## 10. 当前证据与不主张事项

主结果见：

- [`Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)
- [`Doc/current_stageB_effective_configuration.md`](../../../Doc/current_stageB_effective_configuration.md)
- [修正前报告与数据归档](../../../Doc/archive/pre_correction_2026_09_20/README.md)
- [修正后 seed42 PFQ/RRQ 基线](../../../output/pfq_independent_reuse/formal/seed42_pfq_rrq_corrected_v1/report.md)

当前不主张：

- 修正环境中 DPA 已超过 PFQ/RRQ，或 attention/deadline 组件已获正/负贡献证明；
- 对所有 seed 或所有负载实现 zero expiry；
- guard 已提供 non-starvation 保证；
- 已验证 7cell/84UE 的性能或 checkpoint 泛化；
- mixed packet granularity 与 homogeneous 之间存在显著 goodput 差异；
- 不同 campaign 的绝对数值可直接做 paired 比较；
- DPA-v1 已达到生产级鲁棒性或安全保证。

当前优先顺序改为：固定修正环境 → 重新校准负载和基线 → 新采集 BC → Full 重训与独立 argmax 验收 → 同流程组件消融。旧 seed104/107 机制诊断不再优先于环境重建；不能预设修复后组件一定有增益。本次文档更新不启动新训练。

## 11. 代码入口

- actor/model：`training/stageb_hgraph/independent_actor.py`
- online policy/decode：`training/stageb_hgraph/independent_policy.py`
- graph snapshot：`training/stageb_hgraph/feature_snapshot.py`
- CTBS-v1：`training/stageb_hgraph/ctbs_headroom.py`
- reward：`training/stageb_hgraph/independent_reward.py`
- bridge deadline/service features：`cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h`
- standard paired aggregation：`training/stageb_hgraph/pfq_dpa_aligned_comparison.py`
- packet-granularity aggregation：`training/stageb_hgraph/granularity_experiment.py`

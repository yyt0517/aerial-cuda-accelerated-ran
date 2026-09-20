# DPA-v1（PFQ-independent reuse actor）主设计

> 状态：当前生效设计，更新于 2026-09-07。
> 本文是 DPA-v1 算法语义的唯一主设计稿；训练过程、消融和最终基线结果统一见 [`Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)。早期同目录 spec/plan 只记录当时的研究问题与执行协议，不覆盖本文。

## 1. 定位与结论

DPA-v1（Deadline-Pressure Actor v1）是 Stage-B 当前冻结的、在线不依赖 PFQ 的图策略。代码中的兼容名称仍为：

- actor class：`PFQIndependentReuseActor`
- actor family：`pfq_independent_reuse_v1`
- 报告名称：`DPA-v1`

三者指向同一个算法，不代表三个版本。保留旧代码名是为了 checkpoint 和已有实验目录兼容。

DPA-v1 回答的核心问题是：在不读取 PFQ score、PFQ ranking 或 PFQ action 的前提下，能否从因果可用的队列、截止期、信道和历史服务状态直接生成 Type-0 PRG 分配，并在吞吐、可靠性、时延和弱 UE 服务上超过 PFQ/RRQ。

当前证据支持：

- 在严格同二进制、同 offered stream 的标准业务 seeds 41--50 中，DPA-v1 相对 PFQ 同时提高 goodput、降低 BLER，并显著改善 outcome delay 和弱 UE expiry-aware rate。
- 在未见 seeds 101--110、固定期望负载的 homogeneous/mixed packet-granularity 场景中，DPA-v1 相对 PFQ/RRQ 的 goodput、效率、时延和 backlog 优势仍然成立。
- DPA-v1 不是“零过期保证”：packet-granularity 10-seed 中 mixed 场景有 4 个 seed 出现非零 expiry，最差 seed104 为 `9.080%`。
- 实验性 non-starvation guard 已实现并在 seed51 运行，但没有合法触发机会，`override_count=0`；当前正式评估保持 guard 关闭。

## 2. 冻结身份与复现边界

当前 DPA-v1 checkpoint：

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
- 正式部署：deterministic `argmax`，CTBS-v1，guard off

所有性能声明必须同时给出 checkpoint、二进制 hash、seed、业务 profile、窗口和解码配置。不同二进制或不同 offered stream 的结果只能作为跨 campaign 参考，不能做严格 paired 因果比较。

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

2026-09-08 新增 [等预算重训消融协议与进度](../../../Doc/dpa_equal_budget_ablation_experiment.md)：`full` / `no_reuse_attention` / `no_dimensionless`。这是独立研究 campaign，三组均从随机初始化使用 reward E；不改变上述冻结 checkpoint 的训练沿革。`IndependentActorConfig.ablation` 默认 `full`，旧 checkpoint 缺省时仍加载完整 DPA。实验尚未完成前，不将其作为 attention 或 deadline 特征的性能贡献证据。

关键研究沿革：

| 阶段 | 结论 | 当前地位 |
|---|---|---|
| deadline-tail-rate iter21 | seed42 明显改善，但 seed43/45 出现 actor 排序导致的 starvation | 被 DPA-v1 表示修正替代 |
| dimensionless feature-only seed45 | 旧失败 UE 恢复，最长 backlog 零服务 streak 降至 9 TTI | 支持采用当前 UE slots 12--15 |
| frozen seed48/51 | seed48 全门通过；seed51 大部分 KPI 改善但 expiry 失败 | 暴露 worst-seed 风险 |
| CTBS-legal guard seed51 | guard 已实现，但合法 Blank 机会不足，0 次 override | 不进入当前部署 |
| DPA-v1 seeds 41--50 | 严格 paired 优于 PFQ | 当前标准业务证据 |
| packet granularity seeds 101--110 | 相对 PFQ/RRQ 优势总体保持，但非零 expiry 暴露 | 当前外推证据与限制 |

完整数字只维护在综合实验报告中，避免阶段 brief 与主设计重复漂移。

## 8. Experimental guard

guard 的研究目标是在不制造非法 Type-0 动作的前提下，为高 pressure/长 service-age UE 替换一个 Blank。它不能抢占已由 actor 分配给其他 UE 的 PRG。

seed51 steady measured window 的实验记录：

- actor non-Blank choices：`159087`
- `no_pressure_candidate`：`44913`
- guard override：`0`
- 分类：`insufficient_blank_opportunities`

因此 guard 没有改变动作和 KPI，也没有验证“能降低 expiry”的假设。正式 DPA-v1 明确为 `guard=off`；不得把“guard 代码存在”写成“guard 已生效”。

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

聚合代码的最小验证集：

```text
training/stageb_hgraph/tests/test_dpa_v1_historical_comparison.py
training/stageb_hgraph/tests/test_pfq_dpa_aligned_comparison.py
training/stageb_hgraph/tests/test_granularity_experiment.py
```

2026-09-07 复核结果：`19 passed`。

## 10. 当前证据与不主张事项

主结果见：

- [`Doc/stageB_boundary_intelligent_algorithm_experiment_report.md`](../../../Doc/stageB_boundary_intelligent_algorithm_experiment_report.md)
- [`Doc/current_stageB_effective_configuration.md`](../../../Doc/current_stageB_effective_configuration.md)
- standard paired raw report：`output/pfq_independent_reuse/formal/dpa_v1_pfq_seed41_50_aligned_expiry_aware_v1/final_report.json`
- packet-granularity raw report：`output/pfq_independent_reuse/formal/granularity_newseed_stage1_v1/campaign_report_10seed.json`

当前不主张：

- 对所有 seed 或所有负载实现 zero expiry；
- guard 已提供 non-starvation 保证；
- 已验证 7cell/84UE 的性能或 checkpoint 泛化；
- mixed packet granularity 与 homogeneous 之间存在显著 goodput 差异；
- 不同 campaign 的绝对数值可直接做 paired 比较；
- DPA-v1 已达到生产级鲁棒性或安全保证。

下一项最小研究问题应针对 seed104/107 的 expiry 机制做因果诊断，优先判断是压力表示、Blank/干扰权衡还是 CTBS 分配次序导致，再决定是否需要重新训练或可抢占型安全层；不应先扩展完整架构。

## 11. 代码入口

- actor/model：`training/stageb_hgraph/independent_actor.py`
- online policy/decode：`training/stageb_hgraph/independent_policy.py`
- graph snapshot：`training/stageb_hgraph/feature_snapshot.py`
- CTBS-v1：`training/stageb_hgraph/ctbs_headroom.py`
- reward：`training/stageb_hgraph/independent_reward.py`
- bridge deadline/service features：`cuMAC/examples/onlineTrainBridge/DeadlineServiceFeatures.h`
- standard paired aggregation：`training/stageb_hgraph/pfq_dpa_aligned_comparison.py`
- packet-granularity aggregation：`training/stageb_hgraph/granularity_experiment.py`

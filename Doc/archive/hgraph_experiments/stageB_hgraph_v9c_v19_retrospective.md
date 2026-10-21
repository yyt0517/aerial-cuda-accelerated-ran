# Stage-B HGraph v9c–v19 历史实验复盘

> **历史归档，不是当前算法或当前性能证据。** 本文合并旧《智能算法演进与训练完整报告》《Boundary 智能算法实验报告》和对应数据清单。2026-09-20 后续路线及 2026-10-20 物理/时序修正之后，这些旧结果已退出当前性能证据链；保留它们是为了追溯架构、训练收敛、负结果和原始产物。当前状态见 `Doc/stageB_current_progress.md`。
>
> 旧结果使用当时的 SVD、阴影参数解释、动作时序和 packet accounting，不能与修正版指标直接拼接。特别是 `v9c post-1000 1530.558 Mbps` 对 `PFQ full-run 1520.536 Mbps` 的 `+10.022 Mbps` 是**不同窗口的参考差值**；`v9c seed41 +10.662 Mbps` 又使用了 PFQ 十 seed 均值作为参考。两者都不证明同 seed、同窗口下胜出，更不能据此证明稳态容量。旧十 seed full-run 数字为 `1518.248 vs 1520.536 Mbps`。

原报告节标题尽量保留，编号按来源保留；下文的“最好”“推荐”和实现描述均限定于历史探索阶段，不代表当前选型。重复摘要、基线总述和失效的后续训练建议已移除；原始聚合与 checkpoint 路径集中在末尾。


## 2. 实验口径与数据来源

### 2.1 标准场景

| 项目 | 取值 |
|---|---|
| 拓扑 | `3cell / 36 UE` |
| 用户场景 | `coop_boundary` |
| 接入规则 | `strongest association` |
| 几何参数 | `cell_radius=500m`, `site_spacing=1000m` |
| 信道与 PHY | Rayleigh fading, SVD, Type-0 bitmap |
| 业务模型 | `packet_size_bytes=3000`, `arrival_rate=1.0 pkt/TTI/UE`, `TTL=200ms` |
| 评测时长 | `4000 TTI` |
| 正式多 seed | `41-50` |
| 聚焦评测 seed | `41-43` |
| 主参考基线 | `PFQ` |
| 辅助基线 | `RRQ`, raw `RR`, raw `PF` |

### 2.3 指标定义

| 指标 | 含义 | 主要用途 |
|---|---|---|
| `goodput_mbps` | 成功交付吞吐 | 主性能指标 |
| `served_mbps` | 调度服务吞吐 | 资源服务能力 |
| `global_tb_bler` | TB 级错误率 | PHY 成功率与链路风险 |
| `expiry_drop_rate` | TTL 过期丢包率 | 包级可靠性 |
| `traffic.packet_effective_service_rate_mbps` | 包级有效服务率 | 队列服务质量 |
| `packet_delay_mean_ms` / `p95_ms` | 包时延均值与尾部 | 时延质量 |
| `ue_goodput_p5/p10` | UE 低分位吞吐 | 边缘 UE 保障 |
| `ue_goodput_jain` | UE 吞吐公平性 | 全局公平性 |
| `prg_utilization_ratio` | PRG 使用率 | 资源效率 |


## 4. 智能算法架构

### 4.1 总体数据流

```text
cuMAC / Stage-B simulator
  -> OnlineTrainBridge feature extraction
  -> HGraph snapshot: cell / UE / PRG features + masks + edge graph
  -> HGraphPPO policy inference
  -> UE selection logits + PRG allocation logits/actions
  -> C++ bridge fallback and scheduler buffer finalization
  -> Type-0 PRG bitmap scheduling
  -> KPI / reward / rollout logs
```

智能策略不直接替代全部 MAC 流程，而是在 Stage-B 的可调度窗口内输出 UE 选择和 PRG 分配建议。原生 C++ 侧仍负责合法性约束、fallback、资源 bitmap 写入与 KPI 采集。

### 4.2 输入特征

特征由 `training/stageb_hgraph/feature_snapshot.py` 组织，当时核心维度为：

| 类型 | 张量形状 | 维度 | 特征 |
|---|---|---:|---|
| Cell | `[3, 5]` | 5 | `cell_load_bytes`, `active_ue_count`, `mean_wb_sinr_lin`, `mean_avg_rate_mbps`, `tb_err_rate` |
| UE | `[36, 12]` | 12 | `buffer_bytes`, `avg_rate_mbps`, `wb_sinr_lin`, `cqi`, `ri`, `tb_err_last`, `new_data_flag`, `stale_slots`, `hol_delay_ms`, `ttl_slack_ms`, `recent_scheduled_ratio`, `recent_goodput_deficit_norm` |
| PRG | `[3, 17, 12]` | 12 | `top1_sinr_db`, `top2_gap_db`, `prev_prg_assigned`, `prev_prg_reuse_ratio`, `neighbor_max_top1_sinr_db`, `neighbor_mean_top1_sinr_db`, `same_prg_conflict_ratio`, `ici_proxy`, `post_eq_p10_sinr_db`, `post_eq_mean_sinr_db`, `backlog_weighted_expected_goodput_mbps`, `low_sinr_hol_ttl_weight` |

额外输入包括：

- `action_mask_cell_ue`：每个 cell 可服务 UE 的合法性。
- `action_mask_prg_cell`：PRG 与 cell 的可用性。
- `action_mask_ue`：UE 级合法性。
- `post_eq_sinr`：可选的后均衡 SINR，用于候选集生成和 PRG 质量判断。
- teacher actions：可选 imitation/对齐信号。

### 4.3 图结构

`training/stageb_hgraph/edge_generator.py` 构建异构图关系：

| 边类型 | 含义 | 作用 |
|---|---|---|
| `E_CC` | cell-cell 邻接/干扰关系 | 建模跨 cell 协同与干扰 |
| `E_CU` | cell-UE 服务关系 | 连接 UE 队列状态与 serving cell |
| `E_CP` | cell-PRG 资源关系 | 表示 cell 对 PRG 的可用资源 |
| `E_UP` | UE-PRG 质量关系 | 表示 UE 在 PRG 上的后均衡质量与候选优先级 |
| `E_PP` | PRG-PRG 邻接/同频关系 | 建模频域资源连续性与冲突 |

候选生成有两条路径：

- 存在 `post_eq_sinr` 时，使用 `post_eq_topk` 候选。
- 不存在后均衡矩阵时，退化到 `shared_cell_pool`。

### 4.4 模型结构

核心模型在 `training/stageb_hgraph/model.py`：

- Cell encoder：输入 5 维 cell 特征。
- UE encoder：输入 12 维 UE 特征。
- PRG encoder：输入 12 维 PRG 特征。
- Heterogeneous message passing：默认 hidden dim `128`，默认 message layers `2`。
- Action heads：
  - UE selection head：决定每个调度位置/UE 是否被选中。
  - PRG allocation head：决定每个 cell/PRG 分配给哪个 UE 或空置。

当时主要训练使用 `candidate` action head mode，并配合 `ue_prg_topk`、`max_prg_candidates` 控制动作空间。这样可以降低 `[cell, PRG, UE]` 全连接动作空间的搜索难度，同时保留基于后均衡 SINR 的局部精细选择。

### 4.5 ONNX 部署接口

`training/stageb_hgraph/export_onnx.py` 提供两类导出模式。

Logits 模式输入：

| 名称 | 形状 |
|---|---|
| `obs_cell_features` | `[1, 3, 5]` |
| `obs_ue_features` | `[1, 36, 12]` |
| `obs_prg_features` | `[1, 3, 17, 12]` |
| `obs_edge_index` | `[1, 6, 2]` |
| `obs_edge_attr` | `[1, 6, 2]` |
| `action_mask_ue` | `[1, 36]` |
| `action_mask_cell_ue` | `[1, 3, 36]` |

Logits 模式输出：

| 名称 | 形状 | 含义 |
|---|---|---|
| `ue_logits` | `[1, 36, 37]` | UE 选择 logits，含空动作 |
| `prg_logits` | `[1, 3, 17, 37]` | PRG 分配 logits，含空动作 |

Action 模式额外输入：

| 名称 | 形状 |
|---|---|
| `obs_post_eq_sinr` | `[1, 36, 17]` |

Action 模式输出：

| 名称 | 形状 | 含义 |
|---|---|---|
| `action_ue_select` | `[1, 36]` | UE 选择动作 |
| `action_prg_alloc` | `[1, 51]` | 3 cell x 17 PRG 的分配动作 |

C++ 侧会继续做 demand cap、合法性 fallback 和 Type-0 scheduler buffer finalization，因此 ONNX 策略只需要保证偏好排序和候选质量，而不承担全部调度合法性。

历史接口说明：上面的 12 维 UE 输入及 ONNX wrapper 是该阶段接口。wrapper 数值一致不等于 Python evaluator 与 C++ 最终动作一致；coverage repair、second-forward 与 native finalization 的差异见历史基线归档的“21.1 原生 17-PRG ONNX 同轨迹复跑”。


## 4. 智能算法迭代

| 阶段 | 核心变化 | 代表结果 | 结论 |
|---|---|---:|---|
| seed41 i70 / v8 早期 | HGraphPPO 初始长训，偏高 fill | seed41 goodput 1512.368，BLER 12.460% | 训练能动作闭环，但 high-risk fill 导致 BLER 偏高，低于 PFQ |
| v9 reliability gate | 加 reliability-gated goodput，限制 TB err | seed41 best4 deployment 1514.988，BLER 12.166% | 训练态有提升，但 deployment 仍输 PFQ |
| v9b stricter gate | 更硬 TB gate，greedy/budgeted decode | seed41 last 1514.967；slack0/nosafe override 1526.139 | 证明 decoder 的 demand slack/safe fill 是关键变量 |
| v9c PRG efficiency | 对齐少发高效策略，强化 harmful packing 惩罚 | seed41 best 1531.197，BLER 11.387%，p95 2.5 ms；10-seed post-1000 1530.558 | 单 seed 正式部署最好；10-seed full-run 未通过；post-1000 缺少同窗 PFQ |
| v10 rate2 | goodput + packet-rate 第二主线，保存 iter20/24 | iter24 w1000 seed41-43 eval goodput 1528.135 | 当时 balanced 智能 baseline |
| v11 rate-tail gate | checkpoint gate 加 R_pkt / per-packet rate / delay 硬门 | iter20 w1000 seed41-43 eval goodput 1525.170 | seed41 gate 有效，但不能解决 seed42/43 service-time collapse |
| v12 multi-seed rate | 训练 seed 41/42/43 list-cycle | last w1000 eval goodput 1526.673 | 多 seed 方向正确，但训练窗口受冷启动误导 |
| v12b steady warmup | rollout warmup 1000，训练看稳态 | seed42 R_pkt 从 6.140 到 6.801，seed43 未改善 | warmup 必要，但 reward/selector 压力仍不足 |
| v12c tail rescue | tail/rescue reward + selector bias | seed42 R_pkt 到 7.387，seed43 约 2.520 | 对 seed42 有效，seed43 仍是 blocker |
| v13 UE-rate proxy | 引入 UE goodput-rate proxy | seed43 goodput 可到 1540.356，但 R_pkt 降到 1.235 | goodput proxy 不能替代 packet residence time |
| v14 true packet-rate | bridge 导出 true UE packet diagnostics | smoke 通过；seed43 v14/v14b p10 packet-rate 仍塌陷 | 诊断口径正确，但 selector 仍要改 |
| v15-v18 tail feature/quota | tail 特征、strict target、thin/thick quota | v17 seed43 goodput 1542.374，UE p10 0.592 | quota 有效但覆盖/厚度不稳 |
| v19 coverage quota | top8 * 2 PRG coverage quota | seed43 pilot UE p10 packet-rate >1 Mbps；补充 eval goodput 1518.480，PRG util 88.041% | 后续资源效率/coverage 探索；不替代 v9c/v10 正式部署主线 |


## 6. 训练收敛表现

训练指标来自 online PPO rollout 日志，适合作为优化过程诊断，不等同于最终部署评测。

| 版本 | 迭代 | 初始化 | rollout goodput Mbps | TB err | PRG util | R_pkt Mbps | weighted delay ms | best score |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v9c | 12 | v9b last | 1449.4 -> 1538.6, max 1547.4 | 15.36% -> 11.32%, min 10.71% | 84.36% -> 88.69% | 24.31 -> 22.14 | 1.032 -> 1.094 | max 1545.97 |
| v10 | 24 | v9c best | 1449.1 -> 1544.6, max 1567.9 | 15.33% -> 10.11%, min 9.83% | 84.57% -> 86.62% | 24.21 -> 23.33 | 1.036 -> 1.038 | max 1576.29 |
| v11 | 24 | v10 iter24 | 1444.9 -> 1540.9, max 1564.1 | 15.67% -> 10.69% | 85.05% -> 87.31% | 23.65 -> 22.85 | 1.064 -> 1.060 | max 1575.23 |
| v12 | 24 | v11 iter20 | 1443.5 -> 1417.2, max 1454.9 | 15.72% -> 14.94% | 84.97% -> 88.89% | 24.09 -> 14.11 | 1.040 -> 2.508 | 1277.03 -> 1214.67 |
| v19 | 3 | v19 i2 last | 1515.6 -> 1545.9 | 10.58% -> 9.83% | about 70.1% | 13.70 -> 3.92 | 1.792 -> 6.866 | 1392.34 -> 1111.39 |

训练层面的判断：

- v9c、v10、v11 都出现了明确的 goodput 收敛与 TB error 下降，是有效训练链路。
- v10 的 rollout 峰值最高，只说明该训练轨迹的 rollout 峰值更高，不能作为部署容量证据。
- v11 没有明显突破 v10，但也没有在训练中失稳。
- v12 虽然 PRG utilization 上升，但 packet service 和 delay 明显变差，说明多 seed 训练需要重新平衡 reward。
- v19 的 TB error 与 resource efficiency 方向有价值，但 packet service 代价过高，需要继续约束队列目标。


## 7. 部署评测结果

### 7.1 智能策略与 PFQ 对比

以下表保留历史测量值；`vs PFQ` 使用 active gate 十 seed full-run mean `1520.536 Mbps`，并非所有行的同 seed、同窗口配对差。v9/v9b/v9c best/last 的部署窗口为该行全程；带 warmup 的多 seed 行才使用 evaluator 的 post-1000 窗口。`-` 表示该原表没有报告，不以零补齐。

| 策略 | seed | full-run goodput | full vs PFQ | 部署窗口 goodput | 部署窗口 vs PFQ | TB BLER | expiry | R_pkt Mbps | delay mean / p95 ms | UE p5 / p10 Mbps | Jain | PRG util |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9 best4 | 41 | 1514.988 | -5.547 | 1514.988 | - | 12.166% | 0.000% | 17.339 | 1.384 / 5.0 | - | - | - |
| v9b best | 41 | 1511.768 | -8.768 | 1511.768 | - | 12.245% | 0.000% | 15.892 | 1.510 / 6.0 | - | - | - |
| v9b last | 41 | 1514.967 | -5.569 | 1514.967 | - | 12.116% | 0.000% | 15.139 | 1.585 / 6.5 | - | - | - |
| v9b last slack0/nosafe | 41 | 1526.139 | +5.604 | 1526.139 | - | 11.733% | 0.000% | 22.533 | 1.065 / 2.5 | - | - | - |
| v9c last | 41 | 1530.625 | +10.089 | 1530.624 | - | 11.602% | 0.000% | 22.478 | 1.068 / 2.5 | - | - | - |
| v9c best | 41 | 1531.197 | +10.662 | 1531.197 | +10.662 | 11.387% | 0.000% | 22.847 | 1.050 / 2.50 | 40.022 / 40.961 | 0.999094 | 86.730% |
| v9c threshold | 41-50 | 1518.248 | -2.287 | 1530.558 | +10.022 | 11.377% | 0.285% | 8.503 | 5.853 / 37.65 | 38.472 / 40.220 | 0.997839 | 87.973% |
| v10 iter20 | 41-43 | 1514.965 | -5.570 | 1526.088 | +5.552 | 11.666% | 0.191% | 10.175 | 5.792 / 38.83 | 38.808 / 40.102 | 0.998317 | - |
| v10 iter24 | 41-43 | 1516.414 | -4.122 | 1528.135 | +7.599 | 11.519% | 0.226% | 10.559 | 5.757 / 37.50 | 38.792 / 40.139 | 0.998237 | 88.944% |
| v11 iter20 | 41-43 | 1515.117 | -5.419 | 1525.170 | +4.635 | 11.558% | 0.236% | 10.394 | 5.866 / 42.33 | 38.442 / 40.098 | 0.998167 | - |
| v12 last | 41-43 | 1514.837 | -5.699 | 1526.673 | +6.138 | 11.640% | 0.170% | 10.183 | 5.916 / 44.33 | 38.738 / 39.987 | 0.998312 | - |
| v19 i3 | 41-50 | 1518.480 | -2.056 | 1531.037 | +10.501 | 11.407% | 0.201% | 8.170 | 6.693 / 51.10 | 38.815 / 40.066 | 0.998091 | 88.041% |

TB BLER、expiry、R_pkt、delay、UE fairness 和 PRG utilization 保留各 compare/KPI 文件中的 full-run 指标。不同窗口的 goodput 不能与这些指标拼成一份统一窗口测量。


### 5.2 单 seed 正式部署最好：v9c best / iter4

对比口径：

- model：`ppo_actor_best.pt`，seed41，t4000，argmax + greedy。
- 输出目录：`training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0`
- PFQ reference：同一 active boundary PFQ compare 口径。

| 指标 | HGraph v9c | vs PFQ reference |
|---|---:|---:|
| goodput | `1531.197 Mbps` | `+10.662 Mbps` |
| served | `1730.643 Mbps` | `+8.583 Mbps` |
| global TB BLER | `11.387%` | `-0.391 pp` |
| PRG util / fill | `86.730%` | `-11.793 pp` |
| expiry | `0` | 优于 PFQ `0.0173%` |
| average UE goodput | `42.533 Mbps` | `+0.296 Mbps` |
| UE goodput p5 | `40.022 Mbps` | `+0.497 Mbps` |
| UE goodput p10 | `40.961 Mbps` | `+0.829 Mbps` |
| UE goodput Jain | `0.999094` | `+0.000502` |
| UE throughput p5 | `46.995 Mbps` | `+0.820 Mbps` |
| packet delay mean | `1.050 ms` | `-3.640 ms` |
| packet p90 delay | `2.0 ms` | `-9.0 ms` |
| packet p95 delay | `2.5 ms` | `-23.15 ms` |
| packet effective rate R_pkt | `22.847 Mbps` | `+13.875 Mbps` |
| per-packet mean rate | `31.906 Mbps` | `-2.472 Mbps` |
| UE-macro packet delay mean | `1.049 ms` | `-3.864 ms` |

口径修正：表中的 PFQ reference 为十 seed 聚合参考，故这些差值不是 seed41 配对处理效应。“最好”仅指当时保留的智能 checkpoint，不代表已证明优于同轨迹 PFQ。


### 5.3 10-seed 部署证据：v9c full-run 与 post-1000

full-run 10 seed 其实还没有过 PFQ：

| 指标 | HGraph v9c full | PFQ full | 差值 |
|---|---:|---:|---:|
| mean goodput | `1518.248 Mbps` | `1520.536 Mbps` | `-2.287 Mbps` |
| served | `1715.513 Mbps` | `1722.060 Mbps` | `-6.547 Mbps` |
| global TB BLER | `11.377%` | `11.778%` | `-0.401 pp` |
| expiry | `0.285%` | `0.017%` | `+0.268 pp` |
| PRG utilization | `87.973%` | `98.523%` | `-10.550 pp` |
| UE goodput p5 | `38.472 Mbps` | `39.524 Mbps` | `-1.052 Mbps` |
| UE goodput p10 | `40.220 Mbps` | `40.132 Mbps` | `+0.088 Mbps` |
| UE goodput Jain | `0.997839` | `0.998592` | `-0.000754` |
| packet p95 delay | `37.650 ms` | `25.650 ms` | `+12.000 ms` |
| R_pkt | `8.503 Mbps` | `8.972 Mbps` | `-0.469 Mbps` |

去掉前 `1000 TTI` 冷启动后，按 HGraph `checkpoint_eval.log` 的等长 `500 TTI` 窗口重聚合：

| 指标 | HGraph full 4000 | HGraph post-1000 | PFQ full 4000 | post-1000 HGraph vs PFQ |
|---|---:|---:|---:|---:|
| mean goodput | `1518.248 Mbps` | `1530.558 Mbps` | `1520.536 Mbps` | `+10.022 Mbps` |
| mean TB err | `11.553%` | `10.996%` | n/a | n/a |
| mean PRG util | `87.973%` | `88.226%` | `98.523%` | `-10.298 pp` |
| mean blank PRG / TTI | `6.134` | `6.005` | n/a | n/a |

口径修正：PFQ 只有 full-run KPI，没有同粒度 post-1000 数据。因此 `+10.022 Mbps` 只能记录为不同窗口参考差；不能用它宣称稳态胜出，也不能把 full-run 失利归因于冷启动。旧日志记录 post-1000 对 PFQ full-run 为 7/10 正差，落后 seed 为 43、45、48；这仍不是同窗口 gate。


### 5.4 Rate-aware 三 seed：v10 iter24 warm1000

v10 iter24 是当时 rate-aware 正式部署较好的三 seed 结果：

| 指标 | HGraph v10 iter24, seed41-43 |
|---|---:|
| full-run goodput | `1516.414 Mbps` |
| full-run TB BLER | `11.519%` |
| expiry | `0.226%` |
| R_pkt | `10.559 Mbps` |
| per-packet mean rate | `28.753 Mbps` |
| UE-macro R_pkt | `25.834 Mbps` |
| packet delay mean | `5.757 ms` |
| packet p95 delay | `37.500 ms` |
| UE goodput p5 | `38.792 Mbps` |
| UE goodput p10 | `40.139 Mbps` |
| UE goodput Jain | `0.998237` |
| PRG utilization | `88.944%` |
| warmup-excluded evaluator goodput | `1528.135 Mbps` |
| warmup-excluded TB err | `11.094%` |
| warmup-excluded PRG util | `89.203%` |

相对 PFQ seed41-43，v10 iter24 的 goodput `+1.091 Mbps`、TB BLER `-0.250 pp`、packet delay mean `-1.313 ms`、p95 delay `-8.0 ms`、UE goodput p10 `+0.483 Mbps`、Jain `+0.000160`；但 UE goodput p5 `-0.062 Mbps`，UE-macro R_pkt `-4.742 Mbps`。


### 5.5 v19 的当时定位

v19 是 packet-tail coverage quota 的后续方向。历史实验记录中它主要是 online rollout/pilot：seed43 上 UE p10 true packet-rate 已经能稳定到 `>1 Mbps`，但那不是正式 t4000 deployment gate。

本轮补充跑出的 v19 seed41-50 checkpoint eval 可以作为资源效率观察：

| 指标 | PFQ active | HGraph v19 i3 | v19 - PFQ |
|---|---:|---:|---:|
| full-run goodput | `1520.536 Mbps` | `1518.480 Mbps` | `-2.056 Mbps` |
| TB BLER | `11.778%` | `11.407%` | `-0.371 pp` |
| PRG utilization | `98.523%` | `88.041%` | `-10.482 pp` |
| UE goodput p5 | `39.524 Mbps` | `38.815 Mbps` | `-0.709 Mbps` |
| UE goodput p10 | `40.132 Mbps` | `40.065 Mbps` | `-0.067 Mbps` |
| UE throughput p5 | `46.175 Mbps` | `45.000 Mbps` | `-1.175 Mbps` |
| R_pkt | `8.972 Mbps` | `8.170 Mbps` | `-0.802 Mbps` |
| UE-macro R_pkt | `30.425 Mbps` | `23.927 Mbps` | `-6.498 Mbps` |
| packet p95 delay | `25.650 ms` | `51.100 ms` | `+25.450 ms` |

结论：v19 的资源效率很有价值，但它没有在 UE tail rate 和 packet service tail 上超过 PFQ，因此不能替代 v9c/v10-v11 作为“当时正式部署最好结果”。


## 7. 机制分析

### 7.1 v9c 单 seed 与 full-run 的历史差异

v9c 的关键成功在于少发高效：

- `safeFill=0`、低 demand slack 让模型避免把 marginal PRG 塞给高风险 UE。
- seed41 中 PRG utilization 降到约 `86.73%`，但 goodput 反而高于 PFQ。
- BLER 从 PFQ 参考约 `11.78%` 降到 `11.39%`，说明 harmful packing 被抑制。

但 10-seed threshold 结果显示：

- full-run goodput 均值低于 PFQ，`1518.248 Mbps` vs `1520.536 Mbps`。
- 去掉前 `1000 TTI` 后，HGraph mean goodput 升到 `1530.558 Mbps`，对 PFQ full-run 为 `+10.022 Mbps`。
- R_pkt 从 seed41 的 `22.847 Mbps` 降到 10-seed 的 `8.503 Mbps`。
- p95 delay 从 `2.5 ms` 扩大到 `37.65 ms`。

这些历史数据说明资源占用与 packet service 存在权衡，并暴露 seed43/45/48 的 tail/expiry 风险。由于 PFQ 缺少同窗口统计，它们不能证明稳态容量超过 PFQ，也不足以确认冷启动是 full-run 差距的主因。

### 7.2 v10 与 v19 的定位

v10 把 packet effective rate 作为第二主线：

- seed41 训练 absolute best iter14 goodput 达 `1567.854 Mbps`，但 R_pkt 只有 `21.262 Mbps`。
- iter24 goodput 不如 iter14，但 R_pkt 和 per-packet rate 更好，因此更适合作为 balanced/rate candidate。
- 跨 seed w1000 评估中，iter24 比 iter20 goodput 更高、TB err 更低、R_pkt 更高。

v10 的价值在于 balanced：

- seed41-43 matched full-run 中，它对 PFQ 有 `+1.091 Mbps` goodput、`-0.250 pp` TB BLER、`-8.0 ms` p95 delay、UE goodput p10 `+0.483 Mbps`。
- 但它只有 seed41-43 证据，且 UE-macro packet-rate、per-packet mean rate、backlog-free UE ratio 低于 PFQ/RRQ。

v19 的价值在于探索资源效率和 coverage quota：

- seed41-50 full-run 中，它对 PFQ goodput 仅 `-2.056 Mbps`，TB BLER `-0.371 pp`，PRG utilization `-10.482 pp`。
- `goodput / PRG utilization` 比 PFQ 高约 `11.75%`，说明它更接近“少用频谱资源拿到接近同等吞吐”的目标。
- 但 UE goodput p5/p10 没有明显优于 PFQ，UE-macro R_pkt 和 packet delay p95 明显落后。

因此，当时正式部署最好主线仍是 v9c/v10-v11：v9c 负责 goodput/时延/UE tail 的最好证据，v10 负责 rate-aware 三 seed 证据；v19 只作为后续资源效率和 packet-tail quota 的补充方向。

### 7.3 v11-v19 的主线

v11-v12 证明单 seed gate 不足：

- v11 在 seed41 内可以筛出 R_pkt `>23 Mbps`、weighted delay 约 `1 ms` 的候选。
- 但 seed42/43 的 R_pkt 仍塌陷，goodput/TB err 看起来可用，packet service-time 已经失败。

v12b/v12c 证明 warmup 对齐是必要条件：

- 加 `rollout_warmup_steps=1000` 后训练看到稳态窗口。
- tail/rescue 对 seed42 有改善，但 seed43 仍是 blocker。

v13-v14 证明必须用 true packet residence-time：

- UE goodput-rate proxy 在 seed43 看起来很好，但真实 R_pkt 会降到 `1.235 Mbps`。
- v14 协议导出 true UE packet diagnostics 后，诊断路径正确，但 reward/selector 仍需要更直接的 tail service 约束。

v15-v19 的方向是显式 packet-tail quota：

- v17 的 thin quota 证明“被服务次数”不是问题，问题是服务太薄。
- v18 thick quota 改善第一轮，但覆盖范围不稳。
- v19 coverage quota 用 top8 * 2 PRG 改善覆盖，seed43 pilot 能保持 UE p10 packet-rate 大于 `1 Mbps` 多轮。
- 本轮补充 seed41-50 checkpoint eval 后，v19 有资源效率证据：它用 `88.041%` PRG utilization 达到 `1518.480 Mbps` full-run goodput，BLER 也低于 PFQ。
- 但 v19 的 seed43/45/48 仍暴露 packet-rate collapse 和 delay tail；它还不能替代 v9c/v10-v11 的正式部署最好结论。


### 3.3 Boundary 原始 RR/PF 补跑

按容器口径补跑：

```text
output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50
```

10-seed mean：

| 指标 | RR | PF | PF - RR |
|---|---:|---:|---:|
| served Mbps | 1666.277 | 1086.953 | -579.324 |
| goodput Mbps | 1467.894 | 940.790 | -527.104 |
| TB BLER | 11.889% | 12.872% | +0.982 pp |
| expiry drop | 1.287% | 29.484% | +28.197 pp |
| R_pkt | 1.112 Mbps | 0.256 Mbps | -0.857 |
| per-packet rate mean | 28.189 Mbps | 14.544 Mbps | -13.644 |
| UE-macro R_pkt | 22.224 Mbps | 6.180 Mbps | -16.045 |
| UE-macro per-packet rate | 26.969 Mbps | 8.734 Mbps | -18.235 |
| packet delay mean | 24.510 ms | 95.127 ms | +70.617 |
| packet delay p95 | 151.750 ms | 199.450 ms | +47.700 |
| average UE goodput | 40.775 Mbps | 26.133 Mbps | -14.642 |
| UE goodput p5 | 34.234 Mbps | 11.009 Mbps | -23.226 |
| UE goodput p10 | 35.607 Mbps | 12.399 Mbps | -23.208 |
| UE goodput Jain | 0.992659 | 0.842600 | -0.150059 |
| backlog-free UE ratio | 43.611% | 15.278% | -28.333 pp |
| PRG utilization | 99.998% | 100.000% | +0.002 pp |

结论：在 boundary 场景下，原始 PF 几乎满 PRG 使用，但 UE tail 和 packet tail 严重恶化，p95 packet delay 接近 TTL 上限，expiry 接近 `29.5%`。原始 RR 明显好于原始 PF，但仍显著弱于 RRQ/PFQ，尤其 R_pkt、packet delay、UE p5/p10 都落后。这也说明 RRQ/PFQ 的 queue-aware 改良不是小修，而是 boundary 稳态可用性的关键。


### 5.6 原始 RR/PF 补跑状态

本轮已按用户给定环境完成 boundary 原始 RR/PF 10-seed 补跑。

运行环境：

- 先进入容器：`./cuPHY-CP/container/attach_aerial.sh`
- 容器内仓库路径：`/opt/nvidia/cuBB`
- 激活虚拟环境：`source .venv/bin/activate`
- 输出目录：`output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50`

实际命令：

```bash
./cuPHY-CP/container/attach_aerial.sh
cd /opt/nvidia/cuBB
source .venv/bin/activate

./cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh \
  --seed-list 41,42,43,44,45,46,47,48,49,50 \
  --topology-scenario 3cell \
  --total-ue-count 36 \
  --tti 4000 \
  --cell-radius 500 \
  --cell-assoc-mode strongest \
  --ue-placement coop_boundary \
  --packet-size-bytes 3000 \
  --traffic-arrival-rate 1.0 \
  --packet-ttl-ms 200 \
  --prbs-per-group 16 \
  --precoding svd \
  --packet-delay-samples 1 \
  --reference-baseline rr \
  --pf-baseline pf
```

宿主机 shell 环境缺少旧二进制依赖的 `libhdf5_serial_cpp.so.103`、`libhdf5_serial.so.103`、CUDA/TensorRT 等动态库，不能直接补跑。本报告纳入的 RR/PF 数字均以容器内 `/opt/nvidia/cuBB` 路径下的输出为准。

补跑过程中部分原始 PF seed 出现 CPU/GPU per-UE throughput CDF perf-check WARN/FAIL，但每个 seed 都完成全部 4000 TTI，并写出 `kpi_summary.json`、`ue_kpi.csv` 和 per-seed compare 文件；聚合表使用这些已完成 KPI。

## 历史产物与代码证据索引

B/T/E/A/D 编号沿用原 data manifest。目录记录是历史定位信息；未在本次整理中声称所有本地产物仍存在。

## 2. Baseline Data Sources

| ID | Role | Seeds | Source directory | Core raw files | Report use |
|---|---|---:|---|---|---|
| B1 | Active boundary `RRQ` vs `PFQ` gate | 41-50 | `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50` | `RAYLEIGH/rrq_vs_pfq_compare_mean.csv`; `RAYLEIGH/rrq_vs_pfq_compare_mean.json`; `aggregate_manifest.csv`; `scenario_seed_manifest.csv`; `seed_runs.csv` | Primary native reference for HGraph deployment comparisons |
| B2 | Boundary raw `RR` vs raw `PF` rerun | 41-50 | `output/stageB_rr_pf_multiseed_compare_boundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50` | `RAYLEIGH/rr_vs_pf_compare_mean.csv`; `RAYLEIGH/rr_vs_pf_compare_mean.json`; `RAYLEIGH/packet_delay_samples_manifest.csv`; `aggregate_manifest.csv`; `scenario_seed_manifest.csv`; `seed_runs.csv` | Demonstrates why raw `RR/PF` are supporting context rather than the main gate |
| B3 | Full KPI baseline suite | 41-50 | `output/stageB_baseline_eval_suite_fullkpi_seed41_50` | `suite_aggregate/boundary_rrq_pfq_metrics.csv`; `suite_aggregate/boundary_rrq_pfq_metrics.json`; `suite_aggregate/uniform_four_baseline_metrics.csv`; `suite_aggregate/packet_delay_samples_manifest_all.csv`; `suite_summary.txt` | Packet-delay, UE KPI, and broader baseline evidence |
| B4 | Per-seed active boundary native outputs | 41-50 | `output/stageB_main_experiment_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_s*_rrq_*`; `output/stageB_main_experiment_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_s*_pfq_*` | Per-run `kpi_summary.json`; KPI CSV/TXT files emitted by native runner | Drill-down source for seed-level native behavior |
| B5 | Per-seed raw boundary native outputs | 41-50 | `output/stageB_main_experiment_boundary_assocstrongest_legacyshadow_r500_t4000_w0_rr_pf_s41_s50_s*_rr_*`; `output/stageB_main_experiment_boundary_assocstrongest_legacyshadow_r500_t4000_w0_rr_pf_s41_s50_s*_pf_*` | Per-run `kpi_summary.json`; packet delay sample files when enabled | Raw `RR/PF` reproducibility and failure-mode context |

## 3. Training Run Data Sources

| ID | Version / purpose | Seeds | Training directory | Checkpoints to reference | Core raw files | Report use |
|---|---|---:|---|---|---|---|
| T1 | Early long conservative HGraphPPO training | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_i70_from_i30_long_conservative_thread1` | `ppo_actor_best.pt`; `ppo_actor_last.pt`; `ppo_actor_threshold_best.pt`; `candidate_checkpoints/ppo_actor_iter_*.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log`; `sim_runtime/ue_topology.csv`; `sim_runtime/snr.txt` | Early closed-loop training baseline and pre-v9 comparison |
| T2 | v9 reliability gate | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9_relgate_i40_from_v8best_thread1` | `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log` | Reliability-gated reward stage |
| T3 | v9b reliability + greedy decoder | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9b_relgate_greedy_i20_r512_from_v9best_thread1` | `ppo_actor_best.pt`; `ppo_actor_last.pt`; `ppo_actor_threshold_best.pt`; `candidate_checkpoints/ppo_actor_iter_*.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log`; `sim_runtime/ue_topology.csv` | Bridge from reliability-gated training to PRG-efficiency training |
| T4 | v9c PRG efficiency / harmful packing penalty | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_prgeff_greedy_i12_r512_from_v9blast_thread1` | `ppo_actor_best.pt`; `ppo_actor_last.pt`; `ppo_actor_threshold_best.pt`; `candidate_checkpoints/ppo_actor_iter_0004.pt`; `candidate_checkpoints/ppo_actor_iter_0012.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log`; `sim_runtime/ue_topology.csv`; `sim_runtime/snr.txt` | Main v9c training-convergence evidence and deployment checkpoint source |
| T5 | v10 rate-aware objective | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v10_rate2_greedy_i24_r512_from_v9cbest_thread1` | `candidate_checkpoints/ppo_actor_iter_0020.pt`; `candidate_checkpoints/ppo_actor_iter_0024.pt`; `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log`; per-iteration diagnostics if present | Rate-aware evolution stage and balanced deployment candidate source |
| T6 | v11 rate-tail checkpoint gate | 41 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v11_rate_tail_greedy_i24_r512_from_v10i24_thread1` | `candidate_checkpoints/ppo_actor_iter_0020.pt`; `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log` | Checkpoint-selection and rate-tail gate evidence |
| T7 | v12 multi-seed rate training | 41-43 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s43_v12_multiseed_rate_greedy_i24_r512_from_v11i20_thread1` | `ppo_actor_last.pt`; candidate checkpoints if retained | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `sim_runtime/online_bridge_imitation.log`; seed-cycle diagnostics if present | First formal multi-seed training stage |
| T8 | v12b steady-rate pilot | 41-43 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v12b_steady_rate_greedy_i6_r512_from_v12last_pilot` | `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; per-UE diagnostics if present; `sim_runtime/online_bridge_imitation.log` | Pilot evidence for steady-rate reward/selector changes |
| T9 | v12c steady-tail-rescue pilot | 41-43 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v12c_steady_tail_rescue_greedy_i6_r512_from_v12last_pilot` | `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; per-UE diagnostics if present; native reject diagnostics if present | Tail-rescue reward/selector pilot evidence |
| T10 | v13-v18 packet-tail mechanism pilots | 43 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s43_v13_ue_rate_proxy_greedy_i3_r512_from_v12last_pilot`; `...v14_true_packet_rate...`; `...v14b_true_packet_tail_guard...`; `...v15_packet_tail_feature...`; `...v16_packet_tail_strict...`; `...v17_packet_tail_quota...`; `...v18_packet_tail_quota_thick...` | `ppo_actor_last.pt` in each retained pilot directory | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `per_ue_diagnostics/*.csv`; `native_reject_diagnostics/*.csv`; bridge logs | Mechanism-evolution evidence for true packet-rate and packet-tail features |
| T11 | v19 packet-tail quota coverage | 43 | `training/stageb_hgraph/runs/online_ppo_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s43_v19_packet_tail_quota_coverage_i3_r512_from_v19last_pilot` | `ppo_actor_last.pt` | `online_ppo_metrics.csv`; `online_ppo_summary.json`; `per_ue_diagnostics/iter_*.csv`; `per_ue_diagnostics/per_ue_latest.csv`; `native_reject_diagnostics/iter_*.csv`; `sim_runtime/online_bridge_imitation.log` | Resource-efficiency and coverage-quota mechanism evidence |

## 4. Deployment / Checkpoint Evaluation Data Sources

| ID | Model / checkpoint | Seeds | Evaluation directory | Core aggregate files | Per-seed files | Report use |
|---|---|---:|---|---|---|---|
| E1 | early HGraph `i70 best14` | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_hgraph_s41_i70_best14_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/*_kpi_mean.csv`; `RAYLEIGH/*_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41/eval_summary.json`; `per_seed_eval/s41/main_experiment.log`; `per_seed_eval/s41/manifest_rows.csv` | Early deployment reference |
| E2 | v9 best4 | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9_i15r128_best4_argmax_greedy_rerun_coop_boundary_r500_t4000_w0` | `RAYLEIGH/*_kpi_mean.csv`; `RAYLEIGH/*_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41/*` | Reliability-gated deployment comparison |
| E3 | v9b best | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9b_i20r512_best_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/*_kpi_mean.csv`; `RAYLEIGH/*_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41/*` | v9b best deployment source |
| E4 | v9b last | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9b_i20r512_last_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/*_kpi_mean.csv`; `RAYLEIGH/*_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41/*` | v9b last deployment source |
| E5 | v9c best | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_best_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/hgraph_s41_v9c_i12r512_best_coop_boundary_r500_t4000_w0_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_s41_v9c_i12r512_best_coop_boundary_r500_t4000_w0_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv`; `scenario_seed_manifest.csv` | `per_seed_eval/s41/eval_summary.json`; `per_seed_eval/s41/checkpoint_eval.log`; `per_seed_eval/s41/main_experiment.log`; `per_seed_eval/s41/manifest_rows.csv` | Main single-seed deployment evidence |
| E6 | v9c last | 41 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_v9c_i12r512_last_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/*_kpi_mean.csv`; `RAYLEIGH/*_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41/*` | v9c last checkpoint comparison |
| E7 | v9c threshold best | 41-50 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50_v9c_i12r512_best_threshold_argmax_greedy_coop_boundary_r500_t4000_w0` | `RAYLEIGH/hgraph_s41_v9c_i12r512_best_threshold_same_coop_boundary_r500_t4000_w0_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_s41_v9c_i12r512_best_threshold_same_coop_boundary_r500_t4000_w0_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv`; `scenario_seed_manifest.csv` | `per_seed_eval/s41..s50/eval_summary.json`; `checkpoint_eval.log`; `main_experiment.log`; `manifest_rows.csv` | Main multi-seed v9c deployment data |
| E8 | v10 rate2 iter20 | 41-43 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v10_rate2_iter20_argmax_greedy_coop_boundary_r500_t4000_w1000` | `RAYLEIGH/hgraph_v10_rate2_iter20_balanced_w1000_coop_boundary_r500_t4000_w1000_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_v10_rate2_iter20_balanced_w1000_coop_boundary_r500_t4000_w1000_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41..s43/*` | v10 checkpoint comparison |
| E9 | v10 rate2 iter24 | 41-43 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v10_rate2_iter24_argmax_greedy_coop_boundary_r500_t4000_w1000` | `RAYLEIGH/hgraph_v10_rate2_iter24_rate_w1000_coop_boundary_r500_t4000_w1000_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_v10_rate2_iter24_rate_w1000_coop_boundary_r500_t4000_w1000_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41..s43/*` | Main rate-aware deployment data |
| E10 | v11 rate-tail iter20 | 41-43 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v11_rate_tail_iter20_argmax_greedy_coop_boundary_r500_t4000_w1000` | `RAYLEIGH/hgraph_v11_rate_tail_iter20_w1000_coop_boundary_r500_t4000_w1000_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_v11_rate_tail_iter20_w1000_coop_boundary_r500_t4000_w1000_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41..s43/*` | v11 rate-tail deployment source |
| E11 | v12 multi-seed rate last | 41-43 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s43_v12_multiseed_rate_last_argmax_greedy_coop_boundary_r500_t4000_w1000` | `RAYLEIGH/hgraph_v12_multiseed_rate_last_w1000_coop_boundary_r500_t4000_w1000_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_v12_multiseed_rate_last_w1000_coop_boundary_r500_t4000_w1000_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv` | `per_seed_eval/s41..s43/*` | v12 multi-seed deployment source |
| E12 | v19 packet-tail quota coverage | 41-50 | `training/stageb_hgraph/runs/eval_3cell_coopboundary_assocstrongest_legacyshadow_r500_t4000_w1000_s41_s50_v19_packet_tail_i3_argmax_greedy_rerun_skip_coop_boundary_r500_t4000_w1000` | `RAYLEIGH/hgraph_v19_packet_tail_i3_coop_boundary_r500_t4000_w1000_kpi_mean.csv`; `RAYLEIGH/rr_vs_pfq_vs_hgraph_v19_packet_tail_i3_coop_boundary_r500_t4000_w1000_compare_mean.csv`; `aggregate_summary.txt`; `aggregate_manifest.csv`; `scenario_seed_manifest.csv` | `per_seed_eval/s41..s50/eval_summary.json`; `checkpoint_eval.log`; `main_experiment.log`; `manifest_rows.csv` | v19 resource-efficiency / coverage-quota supplemental deployment data |

## 5. Architecture / Code Evidence Sources

| ID | Area | Source files | What to extract for the report |
|---|---|---|---|
| A1 | Scenario runner and native KPI output | `cuMAC/scripts/run_stageB_main_experiment.sh`; `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`; `cuMAC/examples/network.cu` | Scenario knobs, topology/traffic overrides, exported KPI families, Type-0 execution path |
| A2 | Native baseline aggregation | `cuMAC/scripts/run_stageB_rr_pf_compare.sh`; `cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh`; `cuMAC/scripts/rr_vs_pf_compare.py`; `cuMAC/scripts/aggregate_rr_pf_compare.py`; `cuMAC/scripts/aggregate_stageB_baseline_eval_suite.py` | How native runs are paired, aggregated, and written to `*_compare_mean.*` |
| A3 | Online bridge protocol | `cuMAC/examples/onlineTrainBridge/OnlineProtocol.h`; `OnlineObservationTypes.h`; `OnlineObservationExtrasBuilder.h`; `OnlineBridgeServer.cpp`; `OnlineBridgeServer.h` | Observation payload, action payload, packet diagnostics, post-eq SINR export |
| A4 | HGraph features and edges | `training/stageb_hgraph/feature_snapshot.py`; `training/stageb_hgraph/edge_generator.py`; `training/stageb_hgraph/template_encoders.py` | `cell_features`, `ue_features`, `prg_features`, `Cell-Cell`, `Cell-UE`, `Cell-PRG`, `UE-PRG`, `PRG-PRG` construction |
| A5 | Policy model and decoder | `training/stageb_hgraph/model.py`; `training/stageb_hgraph/graph_ops.py`; `training/stageb_hgraph/export_onnx.py` | `StageBHGraphPolicy`, Type-0 action head, `slot -> UE`, `PRG -> slot`, demand cap, deployment export ABI |
| A6 | Training loop and rewards | `training/stageb_hgraph/online_ppo.py`; `training/stageb_hgraph/reward_utils.py`; `training/stageb_hgraph/bridge_env.py`; `training/stageb_hgraph/sanity_logger.py` | PPO loop, reward components, environment bridge, sanity logging, training metric columns |
| A7 | Deployment evaluator | `training/stageb_hgraph/eval_checkpoint_online.py`; `cuMAC/scripts/run_stageB_hgraph_checkpoint_multi_seed_compare.sh`; `cuMAC/scripts/aggregate_model_kpi_compare.py` | Checkpoint evaluation command flow, per-seed manifest, HGraph-vs-PFQ compare aggregation |

## 6. Existing Report / Design Notes To Cite

| ID | Document | Use |
|---|---|---|
| D1 | `Doc/archive/hgraph_experiments/stageB_hgraph_v9c_v19_retrospective.md` | Historical experiment conclusions, baseline references, and report-ready metric definitions |
| D2 | `Doc/archive/hgraph_design/stageB_7cell_tar_gnn_pfq_design.md` | Algorithm evolution log from v9c through v19 and 7-cell design background |
| D3 | `Doc/current_stageB_effective_configuration.md` | Configuration reference; historical scope must be distinguished |
| D4 | `Doc/archive/baselines/stageB_baseline_verification_legacy.md` | Packet-delay distribution methodology and references |
| D5 | `training/stageb_hgraph/README.md` | Module boundary, feature shapes, graph stack, and run hygiene policy |
| D6 | `training/stageb_hgraph/runs/README.md` | Run directory keep/delete policy and naming convention |

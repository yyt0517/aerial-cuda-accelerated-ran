# 修正环境：随机 UE 基线与 DPA 重训实验计划

> 状态：2026-09-20 方案草案，等待用户确认；本轮不启动采集、仿真或训练。
> 执行方式：确认后按 `superpowers:executing-plans` 在当前工作树逐项执行，不引入新的通用实验框架。
>
> 本计划遵循研究优先：先回答一个固定场景下的基线与学习问题，保留完整证据；不扩展 7cell、多负载扫描或消融。

**Goal:** 在三项错误修正后的环境中，完成随机分布 UE 的 seed42/43 × PF/RR/PFQ/RRQ 比较，并用全新数据训练、评估一版 DPA Full。

**Architecture:** 复用现有 native-reference、service-balance trace、packet tracker、指标聚合、BC 和 PPO。仅补充随机放置参数的贯通与缺失的原始指标导出；冻结后四基线和 DPA 使用同一环境与统计协议。

**Tech Stack:** cuMAC C++/CUDA、Python/PyTorch、现有 Aerial 容器、CSV/JSON；原始大型 CSV 可无损 gzip。

**Spec:** 用户本轮要求及[修正环境有效性分界](../../../Doc/stageB_corrected_environment_v1.md)。这是新的修正环境训练协议，不声称逐阶段复刻历史错误环境。

**代码基底：** `7f3b700c6147c1ff738188d837c6c8b99623110b`；后续导出/配置修改另行提交并记录新 hash。现有修正构建身份见 Spec，不把文档提交误当作重建二进制。

## 1. 核心问题与路径选择

- 基线问题：保持资源、业务和 PHY 语义，仅改为随机 UE 放置后，两 seed 上 PF/RR 与 queue-aware PFQ/RRQ 的吞吐、可靠性、时延、服务公平性分别如何？
- 学习问题：全新 BC 初始化 + PPO 的 42/43 交替采样，是否能产生有效的 deterministic argmax DPA？正确执行与性能优越是两回事，不能预设 DPA 必胜。
- 推荐顺序：指标/输入核验 → 8 个正式基线运行 → 固定训练及选模协议 → BC → 8 轮 PPO 检查点 → 最多 100 轮 Full。
- 不采用直接恢复旧队列/旧 checkpoint 的路径；也不先并行铺开多 seed 消融。前者轨迹无效，后者目前不能回答 Full 是否可训练这个首要问题。

## 2. 场景与配对约束

| 项目 | 本轮固定值 |
|---|---|
| 拓扑/用户 | 3 cell / 36 UE，基站布局不变，半径 500 m、站间距 1000 m |
| 用户放置 | `CUMAC_UE_PLACEMENT_MODE=uniform`；现有实现为各生成小区内按面积随机采样，保留最小站距和 Voronoi clipping |
| 关联/天线/信道 | strongest / omni / Rayleigh / SVD / DL Type-0，其他 PHY 参数保持当前配置 |
| 资源/时隙 | 每 cell 17 PRG × 16 PRB，SCS 30 kHz，TTI 0.5 ms |
| 业务 | homogeneous，3000 bytes/packet，Poisson 1 packet/UE/TTI，TTL 200 ms |
| offered | 期望 108,000 bytes/TTI，即三小区合计 1728 Mbps；报告实测到达量，不强制等于期望 |
| 场景 seed | 42、43；topology/traffic/channel 随该 seed 生成，同 seed 四策略严格复用外生输入 |
| 策略 | PF `-b 0`、RR `-b 1`、PFQ `-b 2`、RRQ `-b 3`；启动日志及实际 reference action 路径双重核验 |
| 窗口 | 每次独立重置，1000 warmup + 4000 measured，共 5000 TTI；主统计 TTI1001–5000 |
| 执行口径 | `CUMAC_EXEC_MODE=both`；CPU PHY/队列业务结果为主，GPU 诊断单列，不混成一组 goodput |
| 控制开关 | guard off；关闭 slot permutation、common-MCS override、GPU 固定输入重放；HARQ 保持当前禁用模式 |
| RR 随机性 | RR/RRQ 均显式设置 `CUMAC_RR_AUDIT_SEED=场景seed`，保持原调度规则 |
| 模型 seed | 第一版建议 801，与场景 seed42/43 分开记录；只训练一个 Full |

“场景不变”在此解释为除用户放置从 coop_boundary 改为 uniform 外不改业务、资源或 PHY。随机放置并不保证 strongest 关联后每 cell 恰好 12 UE；保留实际 association counts，不截断候选。

运行前比较同 seed 的 offered packet 内容、UE 坐标/关联、逐 TTI raw-channel fingerprint；策略改变干扰及队列属于内生差异，不要求 PHY SINR 或队列轨迹相等。5001 帧若只是 bridge 预备输入，不计入 5000 帧统计。

本计划不降低到达量、不延长 TTL。若基线仍过载，如实报告，不通过改业务隐藏问题。所选动作 TBS 总和是 scheduled PHY capacity，不是最优/可持续容量上界。

## 3. 必须先处理的两个执行问题

### 3.1 uniform 参数必须贯穿执行链

现有 `pfq_reference_calibration.py` 会直接设置 coop_boundary，manifest 也写死该名字；不能仅在外层设置环境变量。最小改动是增加明确的 placement 参数并用于子进程环境及 manifest，保持旧默认兼容。

训练的现有 `base_env`/runner 也需要显式使用 uniform。启用 `CUMAC_TOPOLOGY_DUMP=1`，因为 uniform 默认不一定导出拓扑。以 sim.log、实际 ue_topology.csv 和 manifest 三者一致为验收条件。

旧 reproduction runner 含有 seed41、41/42、45 的阶段配置以及 4000 总 TTI 的评估身份；不直接续跑或套用。新计划明确使用用户指定的 42/43 和 5000 TTI 评估。

### 3.2 区分 MAC 出队完成与真正成功交付

现有 PHY 会扣除本次调度字节，goodput 只在 TB 成功时累计；packet tracker 的 `delivered` 却由 MAC 出队完成驱动。

- 保持本轮调度和队列模型不变，不另引入 HARQ/重传机制。
- 原 `completion_delay` 保留作历史兼容字段，主名称改为 `mac_dequeue_completion_delay`；不是成功端到端 packet delay。
- 明确记录成功字节、失败出队字节、MAC-completed、expired、pending/censored。
- 没有逐包成功交付证据的指标标为 unavailable/NA，不用出队时延或 0 冒充。
- expiry-aware packet rate 继承 packet tracker 的完成定义；真正按成功字节计算的 UE goodput 单列，不混用 delivered bits 的两种含义。

## 4. 原始数据与汇总交付

不是只保留一张均值表。每个 arm 建独立目录，所有行带 seed、policy、TTI 或窗口、canonical UE ID；所有原始数据覆盖 warmup + measured，可重算完整/稳态两个窗口。

| 层级/文件 | 保留内容 |
|---|---|
| `command.json`、`manifest.json`、`sim.log` | 命令、实际环境、Git commit/dirty diff、sim/lib hash、配置/数据 schema、时间、seed、退出状态和统计窗口 |
| `ue_topology.csv`、`traffic_offered_packets.csv`、channel fingerprint CSV | 完整坐标/关联、逐包 offered stream、逐 TTI 外生信道 identity |
| `service_balance_trace.csv` | 每 UE×TTI 的 offered/accepted/rejected/expired、前后 backlog、scheduled/successful/failed bytes、raw TBS bits/bytes、PRG count/mask、MCS/layer、TB count/error、HOL/slack、CPU/GPU PHY SINR/BLER |
| `ue_tti_metrics.csv` | 对齐同一 UE×TTI 的 WB/CQI/RI、历史速率、service age、deadline 特征及原有统计依赖的可观测量；原始值和派生速率明确区分 |
| `cell_prg_actions.csv` | 每 TTI×cell×PRG 的 UE/Blank、利用率/reuse 依据；可据此重算动作一致率和首次分歧 |
| `packet_delay_samples.csv`、`packet_outcome_samples.csv`、`packet_lifecycle.csv` | 包 ID、UE、大小、arrival、完成/过期、累计服务/剩余 bytes；结束时未完成包的 age/remaining，保留右删失状态 |
| `tti_metrics.csv` | 每 TTI 的 cell/cluster offered、goodput、served、failed、expiry、backlog、BLER、服务 UE 数、PRG 利用/reuse、谱效 |
| `per_ue_metrics.csv`、`per_cell_metrics.csv`、`global_metrics.csv` | 按完整/稳态窗口分别输出 UE、cell、cluster 汇总；36 个 UE 全保留，包括零服务 UE |
| `distributions.csv`、`mcs_histograms.csv` | 原有 delay/rate 分位数、MCS 分布（0..27；无发送状态分列）、layers、SINR 与服务分布；原始样本不丢弃 |
| `comparison.csv`、`metric_catalog.csv`、`report.md` | seed×policy×window 汇总比较；每个指标定义、单位、分母、原始来源、聚合方式及有效性 |

复用当前文件名已有的导出；新增文件只补齐缺口，不重复创建一套协议框架。CSV 可无损压缩，但不能只留 JSON 或图。NPY 可辅助保存动作，不替代可读 CSV。无须保存全部复数信道矩阵；原有 SINR/MCS/谱效指标必须可由保留的数值和定义重算。

**指标覆盖清单：**

- 负载/吞吐：offered、accepted/rejected、scheduled dequeue、successful goodput、failed bytes、expiry bytes/packets/rate、goodput/offered、scheduled TBS capacity 及负载比。
- 时延：MAC 出队完成时延、完成+过期 outcome delay、HOL、TTL slack、队列估计时延；各自 mean、P50/P90/P95/P99/max 以及原有 P5/P10 等字段；未完成包数、年龄和剩余 backlog。
- 速率/公平性：每 UE goodput、served rate、expiry-aware/service-rate 原有定义、长时平均速率；UE mean/P5/P10/P50/min/max、Jain、零服务 UE、最长有 backlog 零出队 streak、每 TTI 服务 UE 数。
- PHY/资源：TB BLER、predicted BLER、WB/实际 PHY/post-EQ 已有 SINR 统计、MCS（tx-only 与 all-TTI 各自分母）、Qm/code-rate、layer/RI、PRG utilization、reuse、Blank、每 active PRG 成功字节。
- 谱效分别导出 served 与 successful-goodput 版本；同载波 cluster `ΣG_cell/B` 与每 cell `G_cell/B` 分开，另给 `ΣG_cell/(3B)`。本资源占用带宽 `B=17*16*12*30000=97,920,000 Hz`；如另用标称 100 MHz，单独命名，禁止混分母。

“之前统计过的指标全部放进来”用清单闭环：取 `summarize_stageA_kpi.py` 的指标字典、aligned/historical comparison、granularity、service audit 的输出键并集，逐一映射。旧名含歧义则保留 alias 并标明正确语义；不能计算的非适用项列出原因，不能静默漏列。

聚合规则：global goodput 从成功字节求和；BLER 为总错误 TB/总发送 TB，不直接平均 UE BLER；global P95 从对应原始包/TTI 样本计算，不平均 UE P95。零样本用 NA 和 count=0。outcome 窗口按 terminal TTI，额外标识 warmup 前到达但 measured 内结束的包，不能把末尾 pending 当作完成或过期。

## 5. 最小执行顺序及停止条件

- [ ] **P0：参数与导出自检。** 显式贯通 uniform，补齐指标缺口/目录与字典。先 seed42 PFQ、RRQ 各 100 warmup + 200 measured；此 smoke 不进正式性能表。检查行数/唯一性、实际 placement、MCS/TBS 独立数值、流量守恒、packet 口径和成功字节 goodput 重算。使用同一策略/seed 对比导出开关开/关，确认记录不改变 CPU 动作和业务指标。
- [ ] **P1：8 个基线运行。** 每 seed 依次 PF、RR、PFQ、RRQ，5000 TTI，全部开启相同导出；逐 arm 成功后封存哈希。每 arm 必须有 180,000 个唯一 UE-TTI，合计 1,440,000；51 个 cell-PRG 决策/TTI。按 seed 配对，报告各自结果和两 seed 描述性均值，不给两个 seed 的“显著性”结论。
- [ ] **P2：检查基线并冻结训练协议。** 确认非单位/索引/统计故障导致的性能表现；如果只是过载，保留该负载，不自动降载。记录 PFQ/RRQ 基线、reward scales 和新选模阈值的实际数值及 hash 后才能启动学习，不复用旧低 expiry 门槛。
- [ ] **P3：新 BC + Full PPO。** 按下一节的建议预算；首 8 轮检查后决定是否在同一状态上继续到 100 轮上限，不额外增加消融组。
- [ ] **P4：冻结一份 DPA 并重跑 seed42/43。** 同一 checkpoint、同一 argmax/CTBS/guard 配置，生成与四基线完全相同的原始/聚合 CSV，发布包含所有保留 checkpoint 学习曲线的报告，不按 seed 各挑一个最优模型。

停止条件：输入不配对、uniform 未生效、schema/行数不完整、任一单位/守恒不变量失败、核心指标缺失、模型/梯度非有限，立即停止该阶段；不覆盖失败目录。未超过 PFQ 是研究结果，不是修改统计口径的理由。

## 6. DPA Full 建议训练协议（批准后执行）

### 新初始化与训练设置

- 只训练 Full：reuse attention、当前无量纲 deadline 特征、reward E、CTBS bytes 语义全部保留；PFQ/teacher 不进入在线 actor-visible input，guard off。
- 重采 PFQ 教师数据：seed42、43 各 5000 TTI，前 1000 仅 warmup、保留后 4000 的有效 BC 样本；合计 8000 TTI 样本，使用当前 teacher–CTBS 对齐路径，同时保留原生/对齐后 teacher 动作及改变计数，明确 BC 行为与原生 PFQ 基线可能不同。
- 先各采 64 个有效样本，确认 teacher 全候选合法、CTBS reference/deletion 检查通过，再完整采集；不使用旧 BC、checkpoint、critic、optimizer、normalizer。
- BC 建议一遍有效样本的顺序监督训练，42/43 各 4000；lr=3e-4、weight_decay=1e-5、grad_clip=5，模型 seed801，结束保存完整训练身份和独立 argmax 评估。BC loss/action accuracy 不是调度性能证明。
- 从 BC actor 初始化 PPO，critic/optimizer/normalizer 新建并记录；进入 PPO 后不得每轮偷偷重置。当前特征从 BC 起固定，不回放历史 legacy→dimensionless 的语义切换。
- hidden=128、message_layers=2、reuse_heads=4、dropout=0；全候选，不截断。
- PPO 场景模式 `list_cycle`，`seed-list=42,43`；每轮显式重置场景，奇数轮42、偶数轮43；各轮 warmup1000 + rollout512，warmup 不计入 PPO transitions。
- 建议总预算100轮，即51,200个训练 transitions，另有100,000 warmup transitions；先执行8轮（每 seed4轮），不是之前失败的随机初始化48轮方案。
- actor_lr=1e-4、critic_lr=3e-4、weight_decay=1e-5；gamma=.99、GAE=.95、clip=.2、entropy=.01、value_coef=.5、Huber beta=10、max_grad_norm=1；normalize_state/reward/adv=1。
- update_epochs=4、minibatch64；target_KL=.05、min_updates=4，保留 KL 提前停止并记录实际更新数；不为凑满更新次数关闭控制。
- 保留历史平均子决策 log-prob 目标；标明是该 PPO 变体，不在本轮暗改成联合动作 sum。
- reward E：goodput/offered − failed/offered − 2*expired/offered，加 deadline-tail eta=.25/window100/tail.1 和 expiry-aware tail-rate weight=.25/tail.1；offered归一化108000 bytes/TTI。所需参考尺度从新基线得出，绝不加载旧 reward_scales。

以上保留既有算法/交替采样逻辑，但使用用户指定的42/43与全新数据，属于新训练方案；归档记录主阶段的实际 seed-list 是41/42，不应把新计划写成逐参数复刻旧训练。

### 独立部署验证与选模

- BC结束（iter0）及 PPO每4轮保存并独立 argmax 评估，最终 iter100 也评估；先保存，再绑定 checkpoint SHA 和验证指标。每次均同时评估42、43，各1000+4000 TTI。
- 记录训练采样与部署 argmax 两条曲线：reward 各分项、entropy、KL、clip fraction、actor/critic loss、实际 updates、Blank、goodput、失败/过期、backlog、弱 UE 与服务 streak。
- 建议候选资格（P2 冻结前可由用户调整；训练后不放宽）：对每 seed，相对新 PFQ，goodput ≥0.99×；expiry ≤PFQ+0.005（0.5个百分点）；MAC completion mean/P95 各 ≤max(1.10×PFQ, PFQ+0.5ms)；UE goodput P10 ≥0.90×PFQ；backlog mean/P95 各 ≤1.10×PFQ。分母/基线为0时不做比值，以明确的绝对基线条件报告，不以NA通过。
- 合格候选统一按“两 seed 最差 goodput/PFQ 比值最大 → 最差 expiry 增量最小 → 最差 outcome P95 最小 → iteration最早”选择；同时保留 last 和全部 Pareto trade-off。不用训练 reward 或更新前采样 goodput 给更新后模型冠以 best。
- 没有合格模型就报告未达标，保留 last/曲线诊断，不自动加长或重新开三组训练。
- seed42/43 既参与学习又用于本阶段诊断/选模，所以最终两 seed 的比较是训练场景复现结果，**不是未见场景泛化验证**。后续泛化另留未用于决策的 seed，本轮不扩大运行矩阵。

## 7. 代码触点与交付边界

最小变更范围（确认后才实现）：

- `training/stageb_hgraph/pfq_reference_calibration.py`：placement/manifest、逐 TTI/UE 原始记录、完整窗口身份。
- `cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`、`trafficModel/serviceBalanceTrace.hpp`、必要的 packet tracker 导出：补充已有统计需要的原始量，保持调度/PHY不变；只有确有缺口才新增字段。
- `cuMAC/scripts/summarize_stageA_kpi.py`、`training/stageb_hgraph/pfq_dpa_aligned_comparison.py`、`capacity_units_decision_probe.py`：复用公式、补齐 raw→perUE→perCell→global CSV 和 metric catalog。
- 新建一个范围固定的 `training/stageb_hgraph/corrected_uniform_baselines.py`：8 个 arm 的顺序运行、输入核验和汇总；不改旧 campaign，不建设通用队列框架。
- `online_imitation.py`、`online_ppo.py`、`eval_checkpoint_online.py`：以新配置调用；仅在现有入口无法满足 warmup排除、42/43循环或导出时作最小修改，记录与本修正版基底的差异。

目录建议：`output/pfq_independent_reuse/formal/corrected_uniform_seed42_43_baselines_v1/` 与 `.../corrected_uniform_dpa_full_v1/`；运行完成后不得就地覆盖。记录 schema/源码变化后的新 campaign 身份，不冒用旧 build hash。若需要重新编译，所有8个基线和DPA应基于同一已核验新构建。

Git 提交代码、协议、metric catalog、清单和小型报告；完整 CSV/BC/checkpoint 以工作区数据归档保留并列 SHA-256，另行备份。不默认把 GB 级数据推到 GitHub。未完成正式数据备份前不得删除原始文件。

## Review Focus

1. uniform 环境变量是否被 calibration 覆盖，拓扑是否实际导出。
2. 失败出队被称为成功完成、末尾 pending 被遗漏、不同窗口/分母混用。
3. 聚合丢失零服务 UE，或用均值的均值代替 TB/包/字节加权统计。
4. 新代码只修 C++ 或只修 Python，教师/CTBS、sim/lib 版本发生漂移。
5. 使用训练场景选模却声称未见泛化，或把低时延建立在丢弃/失败样本上。

**待用户确认：** uniform 的场景解释、完整指标与数据保留范围、BC+100轮上限（8轮先检查）的训练预算，以及上述候选资格建议。确认后先完成 P0/P1 并汇报8组基线；P2 冻结实际基线门槛后才进入训练。

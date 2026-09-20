# DPA 训练路径复现与组件消融：P0 核查

更新：2026-09-20。状态：4000 条公共 BC 与 16 条诊断归档检查已通过；源码归档的容器 Git 路径问题已修复，正在复用已完成数据续跑。具体训练阶段以实时状态为准，尚不代表 Full801 性能验收通过。

- [用户计划 v1.0 原文](../docs/superpowers/plans/2026-09-11-dpa-training-reproduction-ablation.md)
- [机器可读 P0 配置](dpa_training_reproduction_ablation_v1_protocol.json)
- [从历史 checkpoint 提取的完整参数与哈希](dpa_training_reproduction_ablation_v1_history.json)
- [实时状态](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/status.json)
- [P0 对照日志](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/driver_p0.log)
- [自动续跑启动记录](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/continuation_launch.json)
- [自动续跑日志](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/driver_full.log)（等待 P0 时为空；接管后开始写入）
- [2026-09-14 对齐小样本验证](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/preflight/alignment_smoke_20260914_attempt2/validation_report.json)
- [修复后续跑日志](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/driver_resume_20260914.log)
- [9 月 20 日归档路径修复后续跑日志](../output/pfq_independent_reuse/formal/dpa_training_reproduction_ablation_v1/driver_resume_20260920.log)

## 2026-09-20：归档路径修复与数据复用

9 月 14 日 09:52（北京时间），完整数据检查完成后，队列在训练前源码归档处退出：共享 worktree 的 `.git` 文件指向宿主机绝对路径，容器内无法访问。未进入 Full801 正式训练。这不是教师动作再次失败或 Full 性能失败。

本次仅修复运行环境，不修改共享 `.git`、算法源码、仿真二进制或验收门槛。容器内续跑进程显式使用：

```text
GIT_DIR=/opt/nvidia/cuBB/.git/worktrees/pfq-independent-reuse-actor
GIT_WORK_TREE=/opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
CUDA_VISIBLE_DEVICES=1
```

后续从容器启动该队列时须保留上述 Git 环境变量。已使用现有 `archive_current_source` 生成真实源码归档：171 个文件的当前源码/归档内容/manifest 哈希全部一致；HEAD 为 `f567a839e07ba5f622b54fbacd388ecf9427411e`，diff SHA-256 为 `8d183b709e3bf5f2da9eb1ef52ff18cfc57399f53ddb912c3bd332145a21b6d2`，与宿主机读取结果相同。

残留空 `source_archive` 及旧失败状态保留在 `preflight/failure_20260914_git_archive/`，没有删除数据。4000 条公共 BC 和 16 条无量纲数据通过原有完整性检查后直接复用，不重新采集；公共 BC dataset SHA-256 仍为 `c185ab97ec533bd685555ed30d6a3cccee081cebf2dbe8a3331d1bd535843cb8`。仍须完成 Full801 训练及 62/63 验收后，才允许扩展其他训练链。

## 2026-09-14：BC 教师动作与 CTBS 对齐修订

9 月 11 日 P0 对照、PFQ 期限校准和原生 PFQ 快照采集已完成，但公共 BC 首条记录因 teacher target 非法停止：UE17 初始队列 12000 bytes，前一 PRG 累计 TBS 26632 bytes，PFQ 仍在 local PRG2 再分配。原生 PFQ 的 Type-0 动作不因此非法，但不满足 DPA 的累计 CTBS 需求 mask。小样本原始动作严格重放复现了完全相同的错误。

按用户授权采用 `ctbs_prefix_prune_v1`：按固定 local PRG 顺序保留 PFQ 动作，仅把已分配前缀的累计 CTBS 覆盖队列后的重复分配改为 Blank，不补分配、不重排其他 UE、不放宽 mask。先检查原始动作的 serving-cell/PRG 可用性，再删减；其他非法目标仍由严格 canonical trace 拒绝。归档同时保存原始动作、删减位置、实际执行的修正动作，以及 pre-action CTBS tuple；逐 TTI 验证 post-action MCS/layer/TBS exact parity，失败不写记录。BC 标签和环境执行必须使用同一份修正动作。

这批 BC 数据应称为 **PFQ 派生的 CTBS 合法教师轨迹**，不能再称“未修改的原生 PFQ 轨迹”，也不能声称与历史 BC 逐动作一致。所有版本共用同一份哈希绑定数据。PFQ/RRQ 原生基线、旧冻结权重、仿真二进制、PPO 平均 log-prob、训练预算和验收阈值均未更改。原协议 JSON 保留为启动前档案；此修订由本文、数据 manifest 的 `teacher_alignment`、运行时 executable manifest 及代码归档绑定。

GPU1 小规模实证：

| 检查 | 结果 |
|---|---|
| legacy 64 TTI | 63 TTI 发生删减，共 523 个 PRG 决策；64 条 CTBS/PHY exact |
| dimensionless 16 TTI | 15 TTI 发生删减，共 185 个 PRG 决策；16 条 CTBS/PHY exact |
| 采集是否训练 actor | 两组初始/最终 tensor hash 相等，零优化器更新 |
| Full / −Attention 离线重放 | 各完成同一份 legacy 数据的 64 次更新；loss 全部有限、权重确实更新 |
| 两阶段特征、奖励差分、同状态同前缀 | 均通过；不据此声称组件性能增益 |
| 定向测试 | 30 passed |

真实采集还暴露并修复了两个执行/审计问题：CPU 归档 trace 与 GPU greedy trace 比较时未对齐 device；服务年龄验算混用了请求 TTI 与响应/traffic TTI。后者按现有 bridge 时序验算：RESET 的 TTI0 已含 traffic-clock1 特征，后续 observation 为响应时刻的 post-action snapshot，dequeue 事件采用 `next_info.tti`。只修正检查器，不改变输入语义或二进制。只读审查提出的“删减不能掩盖不可用 PRG”问题已有回归验证。

失败目录和日志保留；本次小样本不混入正式公共 BC 数据，也不替代 Full801 性能验收。恢复顺序仍为完整 P0 → Full801 → 通过 62/63 验收后扩展其余训练链，任何合法性/完整性错误停止。

## 9 月 11 日启动记录（历史快照）

2026-09-11 19:11（北京时间）核实：容器 `c_aerial_oai2` 的 P0 进程 PID 1077233 正在运行；自动续跑 PID 1085763 已启动，等待 P0 完成。后者的 `CUDA_VISIBLE_DEVICES=1` 已核对。seed61 的 PFQ/旧 DPA 对照、seed62 的 PFQ 已完成，seed62 旧 DPA 正在运行。此处只是启动时快照，后续以实时状态与日志为准。

执行顺序为：P0 对照与 PFQ 期限校准 → PFQ 快照采集不干扰原动作的完整轨迹检查 → 公共 BC 4000 条与无量纲 16 条诊断归档 → 真实特征/奖励/同前缀验证 → Full801 完整训练与 62/63 验收 → 通过后其余 11 条逻辑训练链 → 冻结 150 次测试 → 机制分析与报告。失败停止，不换 seed、模型或阈值重试。最终定向验证为 **56 passed**；这不是对尚未运行的真实归档或 Full 性能作通过声明。

## 已核实的阶段预算

| 阶段 | 场景 | 训练预算 | warmup | PPO 有效 transitions |
|---|---|---|---|---:|
| S0 BC | 41 | 1 episode，4000 个 teacher TTI；末轮权重 | 不套用 PPO warmup | 不计入 PPO |
| S1 pilot-B 前缀 | 41 | 4 × 512；历史完整 campaign 是 20 轮 | 仅第 1 轮 1000 TTI | 2048 |
| S2 主 PPO | 41/42，从 41 起交替 | 100 × 512 | 每轮 1000 TTI | 51200 |
| S3 无量纲特征适应 | 45 | 8 × 512 | 每轮 1000 TTI | 4096 |

每条完整链 PPO 有效样本 57344，warmup 109000 TTI，PPO 环境合计 166344 TTI，另计 BC 4000 TTI。选中 S2 早期权重不减少已经消耗的完整 campaign 预算。S2 有 26 个候选（每 4 轮及 iter21）；加 S0/S1 各一次、S3 四次，共 32 次验证、128000 TTI/链。正式测试另为 150 次、600000 TTI，尚未启动。

历史 BC：lr=3e-4、weight_decay=1e-5、grad_clip=5。三个 PPO 阶段 optimizer 保存值均为 actor lr=1e-4、critic lr=3e-4、AdamW、weight_decay=1e-5；不能把 args 中 actor_lr/critic_lr=0 误读为实际零学习率。其他公共值：gamma=.99、GAE=.95、clip=.2、entropy=.01、target_KL=.05、min_updates=4、max_epochs=4、minibatch=64、grad_norm=1、Huber value loss。阶段之间的来源是 actor-only checkpoint，不能将 initialization 字段的兼容标签 standalone_bc 理解为每次都重新执行 BC。

## 启动前必须解决的冲突

用户计划要求保留历史阶段结构，同时固定 CTBS。早期 BC/pilot-B manifest 指向 62a17a3，该版本 independent_policy.py 按单 PRG expected service 累加 headroom；S2/S3 checkpoint args 则明确为 ctbs-v1。这是源码/元数据证据，并非对所有历史 dirty runtime 的完整证明。

已获用户批准：新协议从 BC 开始统一 CTBS；所有版本一致。这是相对历史早期 headroom 的协议变更，Full801 用来验证该复现方案，不能称逐项精确恢复旧运行。当前 `launch_allowed=false` 表示 P0 实证检查尚未通过，并非仍等待用户选择。

## 特征语义与消融时点

| 阶段 | slot12 | slot13 | slot14 | slot15 |
|---|---|---|---|---|
| 历史 BC/pilot-B；S2 probe 亦确认旧名称 | TTL slack≤20ms 时的 backlog | TTL slack≤60ms 时的 backlog | backlog × clip(HOL/TTL,0,2) | 距上次正 successful goodput 的 TTI |
| S3 | log1p drain pressure | scheduled-dequeue adequacy | scheduled-dequeue service age / TTL，clip[0,1] | 0 |

前两列旧阈值还要求 slack≥0。旧源码与 S2 probe_report.json 提供语义线索；bridge 新增显式 `CUMAC_DPA_FEATURE_MODE=legacy|dimensionless`，默认仍是无量纲输入。旧公式手算检查与编译通过，真实归档状态检查须在 Full801 前完成。−Features 只在 S3 分叉；−Reward 可共享相同 BC、从 S1 删除过期惩罚，S2/S3 再同时删除两个指定塑形项；−Attention 从 BC 开始独立学习。S2/S3 的 reward U 含公平性和过期混合效应，按原计划保留解释边界。

## 阈值与待办边界

已原样固定附件阈值：goodput≥PFQ及旧 DPA 各自99%；expiry≤0.05%且相对PFQ增加≤0.05个百分点；mean/P95≤max(旧值×1.25，旧值+0.5/1ms)；UE P10≥旧值90%；backlog≤旧值+窗口offered的1%；持续可调度且backlog的无dequeue streak严格短于TTL。0.05%=0.0005，不能写成0.05。61选模按附件字典序；62、63逐场景验收，失败不得更换权重重试。阈值不是显著性标准，未观测新 Full 后不得放宽。

统计口径沿用 `pfq_dpa_aligned_comparison.summarize_aligned_window`：expiry 为主窗口实际过期剩余字节 / 同窗口 offered 字节，并非过期包数 / 完成包数；goodput 为成功 payload 字节按 3000×0.5ms 换算 Mbps。时延按 terminal TTI 在 1001–4000 的 delivered+expired 包统计，跨 warmup 的包保留完整 system time，不把到达窗口之外视为零时延。未完成包不进入 terminal delay，而单独报告 backlog、最终 TTI 的 pre-action HOL 和可调度持续零 dequeue 时长。真实期末 post-action 最老待处理包年龄尚无直接字段，不能将 pre-action HOL 冒充为该值。比例分母为零或没有 terminal outcome 时，不以 0 填充正常性能结果。

seed 固定为 801–803 / 61 / 62–63 / 211–215。可用工作区 486 份元数据 JSON 及相关文本扫描未发现先前使用；这不是对不可访问的外部归档作保证。

公共 BC 将以 PFQ 派生的 CTBS 合法教师动作单独采集 4000 TTI、零优化器更新，再由各版本离线重放同一份哈希绑定数据。S2 历史 safety_prefix_iters=8 的采样性能停止不再继承：固定预算训练后，用独立 argmax 验证和 Full801 验收决定是否扩展。阶段之间仅继承 actor；critic、优化器及归一化状态重建。保留历史平均 log-prob 目标口径及 KL 控制，不改成联合动作概率比。

仍需通过的 P0 项包括：公共 BC 完整性/语义/来源绑定、独立验证隔离、真实 transition 奖励差分与同状态同前缀干预、跨 warmup 饥饿统计、配对输入和 PFQ 压力校准。压力目标区间明确按包含端点的 [0.1%, 5%] 处理。历史 dirty runtime 无法完整恢复之处保留为限制，当前代码/二进制另行锁定。

旧 checkpoint 与已暂停的旧队列保持不变。新实验单独使用 GPU1 和上述输出目录。P0 启动不等于 Full 复现成功；新模型与模块贡献结论均须等待后续实测。

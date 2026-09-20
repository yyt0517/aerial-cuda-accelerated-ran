# DPA 等预算消融实验执行计划

> **For agentic workers:** 使用 superpowers:executing-plans 在当前 worktree 内执行；研究优先，仅增加回答本问题所需的开关、校验和串行任务脚本。用户于 2026-09-08 确认执行。

**Goal:** 区分 reuse attention 与无量纲 deadline/service 输入对重新训练后性能的贡献，不预设 Full 必须获胜。

**Architecture:** 复用 independent actor、online PPO、CTBS、在线评估与现有统一指标聚合。checkpoint 保存处理组，训练/评估共用同一模型入口。

**Tech Stack:** Python/PyTorch、现有 CUDA cuMAC 仿真与 bridge。

**Spec:** 本文记录用户已确认的研究协议；算法背景见 [DPA 主设计](../specs/2026-08-13-pfq-independent-reuse-actor-design.md)。

## 固定协议

| 处理组 | 唯一改动 |
|---|---|
| full | 当前 DPA 网络、特征与解码 |
| no_reuse_attention | `reuse_norm(H)` 替代 `reuse_norm(H + Attention(H))`；保留 GNN 跨 cell 消息 |
| no_dimensionless | actor 输入 UE slots 12–14 置零；slot 15 仍保留；不改原始 snapshot、reward 或 CTBS |

同训练 seed 的三组使用相同随机初始化；不继承任何 BC/PPO checkpoint。attention 模块在去除组不执行且不优化，保留其 dormant state 仅为共享初始化的 RNG 顺序和 checkpoint 结构一致。此实验不匹配有效参数量，不能完全区分 attention 结构与容量收益。

- training RNG seeds：701、702、703；训练拓扑/业务 seeds：41、42、45，按 iteration 轮转。
- 每组每 seed：48 iterations × 512 rollout TTI；每 iteration 重新开始对应拓扑，1000 warmup TTI 不进入 PPO。因此每次训练 24576 transitions、72576 simulator interactions。
- PPO：4 epochs、minibatch 64；actor LR 1e-4、critic LR 3e-4、weight decay 1e-5、gamma .99、GAE .95、clip .2、entropy .01、grad clip 1；使用现有 state/reward/adv normalization。
- 关闭 target-KL 与 performance early stop；每 iteration 32 次 actor/critic minibatch 更新，合计各 1536 次。数值失效则中止并报告，不静默跳过。
- reward：deadline_tail / treatment E（eta=.25、window=100、tail fraction=.1、tail-rate weight=.25）；显式允许本 campaign 的 E 从随机初始化，历史 E 默认 BC 契约不变。
- 评估固定最后一个 checkpoint，不用 best、不用测试集选择 iteration。
- 独立测试 topology/traffic seeds：201–205；9 checkpoint × 5 seeds = 45 actor runs；PFQ/RRQ 各 5 runs。三组训练完成后再汇总，不依测试结果调参。
- 场景固定 3cell/36UE/17PRG、coop_boundary/strongest/legacy_shared_shadow、R500/ISD1000、Poisson λ=1/UE/TTI、3000B、TTL200ms、Rayleigh/SVD/Type-0；同二进制与 offered stream。
- deterministic argmax、CTBS-v1、guard off；4000 连续 TTI，headline 窗口 1001–4000；额外给出 1–4000。
- 指标：goodput、BLER、expiry、outcome mean/p95、UE expiry-aware p10/min、PRG utilization、successful bytes/active PRG、backlog。
- 按 `(training_seed, test_seed)` 配对给原始差值；先对 5 个测试 seed 取均值，再展示 3 个训练 seed 的差值及 t 区间（df=2）。另外展示按测试 seed 的结果；不得把 15 个单元当独立重复。仅 3 个训练重复、5 个测试拓扑，区间是条件于测试集合的描述，不主张普适显著性。
- 若 Full 在该预算下未学出有效策略，报告 budget-limited/inconclusive；不能由整体训练失败推断模块无用。
- 不检验去除 CTBS/sequential decode，不宣称已经单独验证“解码贡献”；不扩展 7cell。

## 最小执行任务

- [x] `independent_actor.py` / `online_ppo.py`：checkpoint 内置处理组，先 RED 再 GREEN；验证去除模块与共享初始化。
- [x] `dpa_ablation_experiment.py`：固定参数串行运行器、预算及 paired artifact 校验、自动 JSON/Markdown 报告；保留命令、hash、耗时与日志。
- [x] 真实三组短训练 + checkpoint 评估 + PFQ/RRQ 烟测；只验证链路，不作性能结论。v3 含基线 trace 起点修正记录，74 项相关测试通过。
- [x] 在空闲 GPU 1 启动包含 9 次正式训练和 55 次评估的串行 campaign；容器后台 PID `1071869`，初始任务 `train/701/full`。其余任务自动排队，不占用/终止既有 GPU 0 作业。
- [ ] 完成后检查 actual budget、数值有限、动作合法、binary/traffic/topology hashes、统计窗口；汇总证据，不提前给胜负结论。

## 运行与停止条件

单次运行脚本使用专属输出根目录与 socket。任何子进程非零退出、预算不符、输入版本改变或配对不一致均停止，不输出完成结论。输出目录存在时拒绝覆盖；失败后由人工检查具体原因，不自动将部分训练当最终 checkpoint。正式结果不覆盖冻结 DPA-v1 的身份或历史报告。

基于历史每 iteration 耗时，完整 campaign 可能需要数天；烟测后按实际耗时更新估计。训练尚在运行时只报告进度，不将脚本实现或 smoke 通过写成消融实验完成。

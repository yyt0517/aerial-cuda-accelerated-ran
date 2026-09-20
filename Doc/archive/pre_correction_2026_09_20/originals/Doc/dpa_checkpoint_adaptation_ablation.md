# DPA 冻结 checkpoint 条件下的短程适应消融

> 2026-09-11：三组短程适应训练与评估已全部完成，动作/CTBS/配对审计通过，但未支持 Full 全面占优。用户随后确认仅补做两项“冻结权重直接消融”评估（见末节），不继续训练。两类结果均是单 seed、继承 Full 预训练的探索证据，不是从头训练的架构消融。

## 决策依据

[两项诊断已完成](../output/pfq_independent_reuse/formal/dpa_full_minimal_diagnostics_v1/report.md)。同二进制、拓扑和 offered stream，主窗口 TTI 1001–4000：

| 场景 / 策略 | Goodput Mbps | Expiry % | 平均 outcome delay ms |
|---|---:|---:|---:|
| seed201 / 旧冻结 Full | 1514.524 | 0 | 1.178 |
| seed201 / 随机重训 Full701 | 1285.622 | 16.281 | 47.941 |
| seed201 / PFQ | 1501.294 | 0 | 2.844 |
| seed42 / 随机重训 Full701 | 1364.505 | 11.414 | 39.550 |
| seed42 / PFQ | 1521.884 | 0 | 4.101 |

新 Full 在训练场景也退化，不能仅归因于未见场景泛化。旧 Full 在 seed201 仍有效。诊断不能分别识别初始化、预算、KL 或选模的因果贡献。

核对真实 checkpoint 和历史记录后，旧 Full 的来源链为：

1. `bc_seed41/hgraph_independent_bc_last.pt`：4000 teacher TTIs，BC seed41。
2. `pilot-B/candidate_checkpoints/ppo_actor_iter_0004.pt`：来自 20-iteration campaign 的 iter4。
3. `deadline_tail_rate_alignment_v1/train_seeds41_42/ppo_actor_absolute_best.pt`：41/42 交替，100 iterations 中选择 iter21。
4. `deadline_dimensionless_features_seed45_v1/train_attempt2/ppo_actor_last.pt`：切换无量纲 deadline 特征后，在 seed45 固定适应 8 iterations，选择 last。

历史 BC manifest 的 binary SHA 为 `3ca2aeb195430e9851a3f7be483b28832ad666c12981354e8e7e4280c00bf15c`；当前为 `28f971218b50c6e5315310a8543c13894e07adfa12ca0303ff438decd56593d2`。记录能追溯权重和参数，但只照搬最后 8 轮参数，不能重现完整历史训练/选模经历。当前未恢复、验证各阶段历史执行环境，因此不把“当前环境训练消融 vs 历史冻结 Full”声称为严格等条件的从头训练比较。

采用用户已同意的备选：共同冻结 checkpoint 出发的短程适应对照。避免重跑整个历史训练链，也不启动随机初始化 Full。

## 假设与最小协议

**问题：**在继承同一有效 Full 权重、获得相同短程采样预算后，移除 reuse attention 或无量纲 deadline 特征是否仍导致性能损失？

三组：`full`、`no_reuse_attention`、`no_dimensionless`。原冻结 Full 作为零新增训练的性能参照；短程适应后的 Full 才是两组消融的直接训练预算对照。

- 三组初始 actor tensor 完全相同；只在派生 checkpoint 的 config 中指定既有 ablation 开关，不修改原 checkpoint。
- `no_reuse_attention` 沿用现有旁路，attention 参数保留但不执行、不优化；GNN 消息传递仍存在。
- `no_dimensionless` 沿用现有 UE slots 12–14 置零路径；slot15 按原契约已为零。基础队列、时延和 CTBS/顺序解码仍存在。
- 三组都继承了包含完整模块的预训练知识，不能称为“训练全程从未使用被移除模块”。
- 单训练 RNG seed701；训练场景 41/42 交替，共 8 iterations。
- 每 iteration 1000 warmup + 512 PPO transitions；每组 4096 有效 transitions、12096 总环境步。
- 延续历史主要 PPO 设置：actor LR `1e-4`、critic LR `3e-4`、target KL `0.05`、最多 4 epochs、minibatch64、reward E、entropy `0.01`；其余复用当前既有入口。
- 使用 actor-only checkpoint；critic、optimizer 和归一化统计均重新初始化。三组构造次序和 RNG 相同，启动前核对 actor hash 与重建的 critic hash 相等；这不是恢复历史 optimizer/critic 的训练续跑。
- 等预算指**相同环境样本数与最大更新轮数**。KL 可能提前结束更新，实际 optimizer updates 单独记录，不能写成“完全相同梯度更新次数”。
- 固定最后第 8 轮 checkpoint，不选择 best、不根据 seed201 调参。
- 评估 seed201、4000 连续 TTI，主窗口 1001–4000；argmax、CTBS-v1、guard off。借用已完成的同条件 PFQ/RRQ，重新检查二进制、offered stream、拓扑与窗口对齐。
- 指标：goodput、expiry、outcome mean/P95、UE P10/min expiry-aware rate、BLER、PRG 效率与 backlog。报告 Full 减去消融的原始差值，不计算单 seed 显著性。

## 顺序与停止条件

1. 关键有效性测试和短烟测：三组各 1 iteration（32 warmup + 64 transitions）、96 TTI 评估；只验证加载、训练/评估链路和动作/CTBS 合法性，不提供性能证据。
2. Full 控制组先完成 8 轮训练及 seed201 正式窗口评估。
3. 在新结果出现前固定保护门：goodput 不低于冻结 Full 的 95%，expiry 不高于冻结 Full 加 1 个百分点。失败则写 `stopped_control_degraded`，停止后续消融，不另选 checkpoint 或自动加预算。该门仅防严重控制组崩塌，不表示全部 KPI 保持。
4. 控制组通过后，依次运行两项消融的 8 轮训练及相同评估，生成完整原始报告后停止。

本轮不自动扩展 3 个训练 seed 或 seeds201–205，不恢复原随机初始化队列。seed201 已被用于诊断，本轮仅为探索性机制证据；若需要确认性结论，后续另行预设未查看场景和重复训练。

## 产物

- [独立脚本](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/run_experiment.py)：复用原训练/评估函数；不修改训练源码。
- [有效性测试](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/test_protocol.py)：与既有 `test_dpa_ablation.py` 合计 17 项通过。
- [新实验 status](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/status.json)：正式运行实时状态。
- [烟测 status](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/smoke/status.json)、[烟测最终报告](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/smoke/final_report.json)：三组各完成一次训练和 96 TTI 评估，初始化一致、offered/topology hash 一致，动作/候选覆盖/CTBS 审计通过；不作为性能证据。
- `manifest.json`：锁定初始权重、代码/二进制来源、采样预算和保护门。
- `init/*.pt`：标明继承 Full 预训练的派生初始 checkpoint。
- `train/<variant>/budget_audit.json`：实际采样与梯度更新、CTBS/replay 审计。
- `control_gate.json`：Full 控制组保护门。
- `partial_report.json` / `final_report.json` / `report.md`：逐组结果、配对审计与差值。

既有 `online_ppo` 会把所有 independent checkpoint warm-start 的 `initialization.mode` 兼容标签记为 `standalone_bc`。本实验不修改共享源码；真实来源以派生 checkpoint 的 `derivation` 及本实验 manifest 为准，不能把本次适应训练误写为 BC。

相关入口：[原随机初始化消融与诊断记录](dpa_equal_budget_ablation_experiment.md)、[DPA 主设计](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)。

## 适应结果与后续：冻结权重直接消融（2026-09-11）

[适应实验最终报告](../output/pfq_independent_reuse/formal/dpa_checkpoint_adaptation_v1/final_report.json)已完成。Full / 去 attention / 去 deadline 特征的主窗口平均时延分别为 1.920 / 1.281 / 1.353 ms，三组 expiry 均为 0；Full 吞吐最高，但时延和 UE P10 expiry-aware rate 不占优。旧冻结 Full 平均时延为 1.178 ms，说明 Full 的适应训练本身引入了退化。

保护门通过只代表没有严重吞吐/expiry 崩塌，不表示时延或弱 UE 性能保持。现有结果不能用“尚未完全适应”直接解释，也不支持为使 Full 获胜而继续调参。

用户确认的下一项最小实验仅为**零新增训练的直接消融评估**：

- 使用上轮 `dpa_checkpoint_adaptation_v1/init/no_reuse_attention.pt` 和 `init/no_dimensionless.pt`，不是 `train/` 下的适应后 checkpoint。
- 启动前验证派生文件 hash、所有 actor tensor 与旧冻结 Full bitwise equality、graph config 相等，actor config 唯一差异为 ablation；再经过真实 evaluator loader 确认开关生效。
- 仅新增两次 seed201 评估，各 4000 TTI，主窗口 1001–4000，argmax、CTBS-v1、guard off；复用旧冻结 Full、PFQ/RRQ 和已完成的三组适应结果，不重跑训练。
- 逐组检查动作、候选覆盖、CTBS 和 PFQ/RRQ 的二进制/offered stream/拓扑/窗口对齐。
- 报告三类原始差值：`zero-shot ablation - zero-shot Full`、`adapted ablation - adapted Full`、各组 `adapted - zero-shot`。符号明确；不混合直接移除和适应训练的影响。
- 两次评估完成即停止，不自动微调、不扩展 seed、不恢复旧队列。直接消融同时改变输入/内部表示分布，不能当作无该模块模型的能力上限。

产物根目录为 `output/pfq_independent_reuse/formal/dpa_frozen_zero_shot_ablation_v1/`：

- [实时 status](../output/pfq_independent_reuse/formal/dpa_frozen_zero_shot_ablation_v1/status.json)
- [独立评估脚本](../output/pfq_independent_reuse/formal/dpa_frozen_zero_shot_ablation_v1/run_evaluation.py)，无训练入口调用。
- `manifest.json`：记录零训练 checkpoint 的逐张量验证与借用结果来源。
- `partial_report.json`：每完成一组更新；全部完成后生成 `final_report.json` 和 `report.md`。

差值计算测试经历预期失败后通过，与已有 checkpoint/消融有效性测试合计 18 项通过。正式评估于 16:02 左右（UTC+8）在独立后台启动，进度以 status 为准。

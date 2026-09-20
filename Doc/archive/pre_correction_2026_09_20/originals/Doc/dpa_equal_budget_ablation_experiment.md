# DPA 等训练预算消融实验

> 2026-09-11：9 次随机初始化训练已完成，剩余评估保持暂停。两项 Full 诊断已完成；当前转向[共同冻结 checkpoint 的短程适应消融](dpa_checkpoint_adaptation_ablation.md)。本随机初始化 campaign 尚未形成正式消融结论。

本实验检验 reuse attention 和无量纲 deadline/service 特征的增量价值。详细预设协议见 [执行计划](../docs/superpowers/plans/2026-09-08-dpa-equal-budget-ablation.md)，算法定义见 [DPA 主设计](../docs/superpowers/specs/2026-08-13-pfq-independent-reuse-actor-design.md)。

三组为 `full`、`no_reuse_attention`、`no_dimensionless`。每组从随机初始化训练，3 个训练随机 seed，每次 48 × 512 PPO transitions、4 epochs；相同训练 seed 共享初始权重。评估固定 last checkpoint，在未参与训练的 seeds 201–205 上与 PFQ/RRQ 对齐比较。

注意：去除 attention 后仍有 GNN 跨小区消息传递；去除无量纲特征后仍有基础队列/时延信息；三组都保留 CTBS 和顺序解码。因此本实验不是“无图 vs 有图”，也不是独立的解码消融。

## 产物与进度

### 2026-09-11：暂停队列，先诊断 Full

首个完整结果（训练 seed701、测试 seed201、TTI 1001–4000）中，新 Full goodput 为 1285.622 Mbps、expiry 为 16.281%，PFQ 对应 1501.294 Mbps、0%；二进制、offered stream、拓扑及动作/CTBS 审计通过。该结果提示当前随机重训 Full 尚未复现冻结版能力，但单一训练/测试 seed 不能确定具体原因，不能据此断言模块无效。

按用户指示，2026-09-11 10:14 左右对派发进程 `1071869` 发送 SIGSTOP，已确认进程为 stopped。正在运行的 `eval/201/701/no_dimensionless` 允许自然收尾，保留完整产物；不再启动后续 ablation 任务。`status.json` 标记为 `paused`。已完成的 9 个训练 checkpoint 和评估数据不删除、不覆盖。

两个最小诊断使用既有评估入口、同二进制、argmax、CTBS-v1、guard off、4000 连续 TTI（主窗口 1001–4000）：

1. **旧冻结 DPA → seed201**：使用原 iter8 checkpoint，与已完成的本次 Full701/PFQ/RRQ 的 seed201 结果严格配对，判断该测试场景是否同样使旧策略失败。
2. **本次 Full701 → seed42**：测试训练场景上的 argmax 行为，并补跑同条件 PFQ42 控制；判断失败是否也发生在训练场景，而不仅是未见场景泛化。

这两项诊断不重新训练、不修改解码、不选择新 best，也不能单独识别 BC、KL 控制、预算、checkpoint 选择的因果贡献。若旧模型也失败，则应先考虑场景风险/跨版本复现；若旧模型良好而新 Full 在训练场景也差，则优先检查当前训练与部署行为。不能预先指定哪项假设必然成立。

- [诊断实时状态](../output/pfq_independent_reuse/formal/dpa_full_minimal_diagnostics_v1/status.json)，独立后台 PID `1074266`。
- [诊断 manifest](../output/pfq_independent_reuse/formal/dpa_full_minimal_diagnostics_v1/manifest.json)，记录 checkpoint/hash、来源与固定协议。
- 诊断根目录：`output/pfq_independent_reuse/formal/dpa_full_minimal_diagnostics_v1/`；每项完成后更新 `partial_report.json`，全部审计完成后生成 `final_report.json` 与 `report.md`。
- 原派发队列不会自动恢复。恢复前需用户决定，并核对派发 PID、当前子任务结束状态及脚本超时状态；不要直接重复启动原 campaign 命令。

### 原 campaign 产物

两项诊断的[最终结果](../output/pfq_independent_reuse/formal/dpa_full_minimal_diagnostics_v1/report.md)确认：旧冻结 Full 在 seed201 为 1514.524 Mbps、0 expiry；新 Full701 在训练场景 seed42 仍为 1364.505 Mbps、11.414% expiry，PFQ42 为 1521.884 Mbps、0 expiry。失败不只发生于未见场景，但具体训练原因尚未因果分离。用户已确认不再启动随机初始化 Full，改走上述 checkpoint 条件下的最小局部对照；两类实验不能混合聚合。

- 运行器：[dpa_ablation_experiment.py](../training/stageb_hgraph/dpa_ablation_experiment.py)
- [烟测最终 JSON](../output/pfq_independent_reuse/formal/dpa_ablation_equal_budget_smoke_v3/final_report.json)：3 次短训练、5 次评估，配对与动作/CTBS 审计通过；不作为性能结果。
- 正式根目录：`output/pfq_independent_reuse/formal/dpa_ablation_equal_budget_v1/`
- [正式 status.json](../output/pfq_independent_reuse/formal/dpa_ablation_equal_budget_v1/status.json)：当前任务、已完成任务与失败信息；容器后台 PID `1071869`。
- `train/<training_seed>/<variant>/online_ppo_metrics.csv`：逐 iteration 学习曲线与实际更新次数。
- `train/<training_seed>/<variant>/budget_audit.json`：有效训练预算、随机初始化与 checkpoint hash。
- `eval/<test_seed>/<training_seed>/<variant>/`：逐 TTI、service/outcome/offered stream 与在线评估。
- `eval/<test_seed>/pfq/`、`rrq/`：同一测试 seed 的原生基线。
- 全部训练/评估/对齐审计完成后自动写入 `final_report.json` 与 `report.md`；仅这些产物出现并通过检查后，才能形成正式消融结论。

验证记录：74 项相关测试通过。三组 smoke 均为 32 warmup + 64 PPO transitions、4 次 minibatch 更新，共享 actor/critic 初始 tensor hash；96 TTI 的 argmax 评估三组均全 Blank，说明短训练尚未产生有效部署策略，而非已经支持消融性能假设。正式实验使用已确认的 48 iterations，不据此短测试调整参数。

烟测修正记录保留：v1 审计 CSV 布尔值读取错误；v2 在核对出 RRQ 需要原生 `-b 3` 后由版本保护停止；v3 两个基线默认从 TTI 602 采集 service trace，改为显式 TTI 1 后仅补跑这两个基线，未改三组训练及 actor 评估。原基线目录移动至 `pfq_incomplete_trace602` / `rrq_incomplete_trace602`，可完整恢复；详情见 v3 的 `smoke_repair.json` 和 `status.json`。

按烟测耗时粗估，串行完整 campaign 约需 3–4 天，实际以逐 iteration 日志为准；固定 4 epochs 比历史 KL 提前停止运行更耗时。作业在容器后台运行，成功后自行评估、审计并生成原始报告；本页面不是实时进度源，请查看正式根目录的 `status.json`。

## 启动命令（容器内）

```bash
cd /opt/nvidia/cuBB/.worktrees/pfq-independent-reuse-actor
/opt/nvidia/cuBB/.venv/bin/python -m training.stageb_hgraph.dpa_ablation_experiment \
  --sim-bin build.reuse_actor/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection \
  --out-root output/pfq_independent_reuse/formal/dpa_ablation_equal_budget_v1 \
  --gpu 1
```

运行器拒绝覆盖已有根目录；请勿重复启动正式命令。停止/失败的任务保留日志，不自动把部分训练视为完成。运行期间不要修改 Python 源码或仿真二进制，版本变化会使后续任务中止以防混用证据。

## 解释边界

报告给出每个 `(training_seed, test_seed)` 单元和配对差值，不把 15 个单元当成 15 个独立训练重复。三训练重复的区间条件于五个测试拓扑，统计功效有限。若 Full 在固定预算下没有学出有效策略，应报告训练预算下结论不充分，而非推断 attention/deadline 机制没有价值。

本 campaign 的随机重训 Full 不是历史冻结 iter8 checkpoint；结果不得替换或混入 [冻结 DPA-v1/PFQ/RRQ 综合报告](stageB_boundary_intelligent_algorithm_experiment_report.md) 的原有数值。

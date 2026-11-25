# Stage-B 当前项目进展

更新至 **2026-11-25**。最新研究交付是 **DA-PLS：M-LWDF-PF 权重＋容量上界剪枝的精确局部搜索**。已完成十 seed 主场景、五 seed 的 G1–G4 泛化，以及三 seed 的七小区和邻区输入简化研究。二阶段报告最小源码与结果包于 11 月 23 日整理，本次归入[证据索引](evidence/2026-11-25/README.md)。

**注册默认入口仍是 10 月 27 日的 PFQ＋CTBS＋缺口权重、ρ=0.90 版本**，本次没有修改[算法注册表](../training/stageb_hgraph/current_algorithm.json)。报告实验从原始 PFQ 出发、不做 CTBS 裁剪；报告中的 DPLS/DA-PLS 不能与旧默认入口混同。它们没有模型训练，精确性指与规范完整局部搜索等价，不指全局最优。

## 1. 十 seed 主场景与验收结果

主实验共 220 点（210 新跑、10 条经核验的 PFQ 复用）；最终报告选入两场景、六算法、seed41–50 的 120 条结果。每条轨迹为 4000 TTI 共同 PFQ 预热＋4000 稳定＋20000 测量＋400 随访，TTI 为 0.5 ms，测量窗口为 10 秒。seed44–50 是权重选择之外的稳健性子集，不称完全未见盲测。

下表为逐 seed 等权均值，吞吐只计测量期到达且完整成功交付的包；时延只计成功包。

| 场景 | 算法 | 完整包吞吐 Mbps | 平均时延 ms | P95 ms | 过期率 % |
|---|---|---:|---:|---:|---:|
| Boundary / rate0.8 | PFQ | 749.497 | 121.577 | 200.00 | 31.344 |
| Boundary / rate0.8 | DPLS（缺口） | 1232.912 | 12.656 | 90.90 | 0.721 |
| Boundary / rate0.8 | DA-PLS（M-LWDF-PF） | 1257.499 | 2.624 | 5.15 | 0.000 |
| Uniform / rate1.0 | PFQ | 1516.871 | 4.340 | 25.05 | 0.662 |
| Uniform / rate1.0 | DPLS（缺口） | 1605.527 | 0.620 | 1.05 | 0.000 |
| Uniform / rate1.0 | DA-PLS（M-LWDF-PF） | 1603.460 | 0.613 | 1.10 | 0.000 |

Boundary 中 DA-PLS 相对 PFQ 完整包吞吐 **+67.78%**、成功包平均时延 **−97.84%**；Uniform 吞吐 **+5.71%**。DA-PLS 在 Uniform 的吞吐略低于 DPLS，不能写成所有指标、所有场景最优。旧 11 月 4 日三 seed、2 秒的 +68.96% 只保留在[历史进展](archive/progress/stageB_progress_20261104.md)，不再作为最新验收数字。

来源：[结果速览](evidence/2026-11-25/minimal_delivery/结果速览.csv)、[逐 seed](evidence/2026-11-25/minimal_delivery/results/main/per_seed.csv)、[完整 220 点验收](evidence/2026-11-25/stage2_closure/完整交付核验.json)。报告扩充了 outcome、队列和逐小区指标；outcome 包含 CRC 损坏的 MAC 完成包与自然过期包，不能与成功包时延互换。

## 2. 泛化与七小区：收益和局限分别保留

G1–G4 使用 seed51–55，分别覆盖混合布点、异构包业务、小区负载不平衡及其组合；三算法共 60 条轨迹。DA-PLS 完整包吞吐均值为 1270.242–1275.143 Mbps，成功包平均时延为 0.607–0.701 ms，20 条 DA-PLS 轨迹均零过期。缺口公式按每 UE 配置到达字节扩展，不能称业务参数完全未变。见[冻结协议](evidence/2026-11-25/stage2_generalization/protocol.json)与[泛化逐 seed](evidence/2026-11-25/minimal_delivery/results/generalization/per_seed.csv)。

七小区是独立的 84 UE、seed41–43 扩展检查。原 DA-PLS 相对 DPLS 完整包吞吐 **−3.14%**，成功包平均时延 **138.601 vs 87.292 ms**，过期率 **24.977% vs 23.470%**，两者 P95 均为 200 ms。三分支重启改善原 DA，却仍比独立 DPLS 轨迹平均时延高 41.843 ms。新有界紧急度与服务保护方案（λ=0.1、β=0.98）相对 DPLS 吞吐 +1.423%、平均时延 −2.886 ms，但 P95 仍为 200 ms，最差 UE 也没有全面改善。

这表明三小区结论不能直接外推；同状态目标或服务下界不保证另一条闭环轨迹的长期指标。保留[原七小区报告](evidence/2026-11-25/seven_cell/analysis/report.txt)、[三分支报告](evidence/2026-11-25/seven_cell_multistart/analysis/report.txt)、[时延诊断](evidence/2026-11-25/seven_cell_delay/report.txt)及[保护方案报告](evidence/2026-11-25/seven_cell_guarded/analysis/report.txt)。

## 3. 邻区输入简化与成本

RSRP 研究只替换搜索中的邻区干扰反事实模型，原生执行仍用真实信道。代理是单参考端口、跨 PRG/接收支路的线性功率，100 ms 因果 EMA、10 ms 上报、0.5 ms 传输延迟；不是标准化空口 RSRP，也未模拟测量噪声、量化或波束失配。

研究链包含[纯标量 RSRP](evidence/2026-11-25/rsrp/report.html)、[六级有序消融](evidence/2026-11-25/rsrp_ablation/report.html)和[归一化空间协方差更新](evidence/2026-11-25/rsrp_covariance/report.html)。分别汇总 24、72、144 条轨迹，其中包含复用端点；不能相加当作独立新增次数。协方差研究有 120 条新增轨迹。

Boundary 的 DA-PLS 三 seed 长窗对照中，精确输入完整包吞吐为 1254.737 Mbps、平均时延 2.807 ms；纯 RSRP 为 1065.423 Mbps、128.037 ms。每 PRG、0.5 ms 更新协方差恢复至 1166.927 Mbps，但时延仍为 113.874 ms。输入信息减少尚不能视为无损替代，也不能按反馈量比例推断运行加速。完整数字见[输入方案逐 seed](evidence/2026-11-25/minimal_delivery/results/input_simplification/per_seed.csv)。

精确剪枝的最终成本审计覆盖三个 Boundary 负载共 90 个保存状态、2477 轮、3923174 个候选，最终串行计时在本批质量仿真结束后进行。见[审计完成记录](evidence/2026-11-25/minimal_delivery/results/search_audit/completion.json)。保存状态的单核搜索成本、并发仿真墙钟与端到端调度时延分别解释；不沿用旧 κ0 成本曲线作为 DA-PLS 相对 PFQ 的公平加速比。

## 4. 复核入口与实现边界

最小交付包包含冻结代码、原生源码身份、配置、逐 seed/UE/小区表、报告数值核查器和一个离线状态。运行：

```bash
python3 Doc/evidence/2026-11-25/minimal_delivery/verify_results.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python Doc/evidence/2026-11-25/minimal_delivery/replay_sample.py
```

第一条只需标准库；第二条需 NumPy、g++ 和 Eigen3，临时编译后比较 Python 完整枚举、C++ 完整枚举与剪枝结果。它们不重跑全量 GPU 轨迹。完整仿真仍需冻结工作树、CUDA 容器和本地大工件；详见[交付说明](evidence/2026-11-25/minimal_delivery/README.md)。

本次同时纳入 `training/stage2_*`、RSRP 与七小区研究源码，以及此前留在主工作区的可选动作后干扰 CSV 插桩、配对分析和 teacher passthrough 工具。此插桩按实际干扰 UE 的预编码/层数统计；冻结报告的 activity-only PHY 是另一口径，不能混用。主工作区 C++ 改动不参与上述冻结实验。

## 5. 尚未解决的问题与历史去向

七小区长尾与过期、邻区输入简化的质量损失、端到端实时预算仍未解决；不因归档或单元检查通过而提升注册默认版本。后续若继续，应分别冻结质量、反馈与计算成本协议，并采用相匹配的初始化、业务和统计窗口。

神经路径的质量失败、预算门槛、条件树负结果及旧 SoA 逐位等价失败全部保留在[截至 11 月 4 日的进展快照](archive/progress/stageB_progress_20261104.md)。旧分支中的计划和“下一步”仅作为历史记录，不再与当前交付状态并列。旧证据目录、算法注册身份与失败门槛保持原样。

# 2026-10-28 场景验证与神经路径证据

当前结论统一见[项目进展](../../stageB_current_progress.md)，两场景及研究路径的配置边界见[配置说明](../../current_stageB_effective_configuration.md)。本目录的原报告保持冻结，不另维护一套“当前进展”。

| 子目录 | 协议与内容 | 结果边界 |
|---|---|---|
| [uniform_dpls_rho090](uniform_dpls_rho090/report.txt) | 同一ρ0.90算法对PFQ，seeds41–45，10秒测量；跨场景源码身份、逐seed指标、配对区间与稳定性 | 完整包吞吐+5.07%；平均时延及P99配对区间跨零；与boundary不混池 |
| `tb_deadline/` | TTL风险特征训练/验证协议、输入来源、门槛与计数 | 风险直接神经验证门槛失败；混合分支通过其当轮门槛，不代表后续所有质量门槛通过 |
| [tb_direct_risk](tb_direct_risk/final_report.txt) | 冻结模型，seeds104–106，五臂各5秒测量；本次补收最终报告和逐seed/UE统计 | 神经比同风险直接物理少96.789%物理候选，平均成功时延却增加19.171ms |
| [tb_neural_cost](tb_neural_cost/report.txt) | 冻结87维模型，18保存状态串行回放；seed48/50两条新增原生等价轨迹 | 搜索成本降低36.108%；仅证明相对原神经自身不退化 |
| [tb_neural_choice](tb_neural_choice/report.txt) | 104–106既有轨迹的PHY、同包时延、积压及同状态选择分析 | 没有新仿真、训练或测试seed调参，不单独归因网络权重 |
| [tb_prefix](tb_prefix/report.txt) | 冻结actor、有限精确前缀验证及覆盖消融；开发seed48/50短窗闭环 | 平均时延下降，但逐seed payload/P5保护未全部通过；不更换有效算法 |

`tb_direct_risk/manifest.json`记录本次从本地`deliverables/tb_direct_risk_20261028_native_v1/`复制的八份证据来源、大小与SHA-256。其它各轮的`manifest.json`为原始来源清单；可能同时列出未复制的模型、完整轨迹或脚本，路径相对原始实验目录解释。比如`tb_neural_cost`未收入模型及两份交付脚本，uniform清单保留原生二进制等本地工件身份。这些缺省不代表仅凭快照即可重放全实验。

`tb_deadline/plan.json`和`plan_v2.json`分别保留原协议及修订，`validation_means.csv`和`residual_offline_validation.csv`属于不同验证，不按文件名相似删除。源报告记录的历史检查与本次文档/统计复核分开解释。

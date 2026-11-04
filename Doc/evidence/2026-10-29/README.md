# 2026-10-29 DPLS 消融、PPO 重训与神经搜索诊断证据

当前结论见[项目进展](../../stageB_current_progress.md)，配置见[配置说明](../../current_stageB_effective_configuration.md)。本轮未替换有效算法，未运行311–315新seed闭环。

| 子目录 | 内容与结论 |
|---|---|
| [dpls_component_ablation](dpls_component_ablation/conclusions.txt) | boundary 五 seed 八组消融，30条新原生轨迹＋10条参照；本小区物理优化与全局协调均有实质收益，CTBS未显示明确额外吞吐收益；缺口权重小幅增加吞吐、降低TTL，同时增加平均时延。未更换默认算法 |
| [dpls_deficit_weight_analysis](dpls_deficit_weight_analysis/report.txt) | 仅读取冻结消融，详查全指标、匹配UE及十条成熟包cohort；最差UE整包成功率均值62.43%→77.67%，但5ms内交付比例80.41%→76.60%，共同成功包平均多等2.82ms；保留为保障选项有价值，不能称全面改善；无新轨迹或默认修改 |
| [ppo_no_ctbs_equal](ppo_no_ctbs_equal/report.txt) | 无 CTBS、无业务/缺口加权的联合自回归 PPO；41–43 训练，44–45 各两条 BC/PPO 原生验证。12 轮、6144 个 PPO 决策完成；两 seed 平均整包吞吐 812.07 Mbps，较 BC +1.85%，较同窗口等权 DPLS −34.40%，过期率 34.14%，质量门槛失败，未晋级 |
| [tb_process_value](tb_process_value/report.txt) | 58快照/1392块的离线价值蒸馏；强近似PHY及同深度对照；技术完整性通过，名义10%预算下95%高收益集合召回门槛失败 |
| [tb_residual](tb_residual/report.txt) | 全合法扩展、UE物理残差、集合外正例、C++局部缓存与完整搜索回放；缓存独立降本37.83%，冻结池收益未稳定转化为搜索优势，召回门槛仍失败；无新原生包时延结论 |
| [tb_search_gain_analysis](tb_search_gain_analysis/report.txt) | 仅读取冻结候选树的覆盖/排序损失与继续收益翻转诊断；评分混排、活动模式分解和继续价值均为待验证建议，未实施新策略 |
| [tb_joint_repair_phy_supplement](tb_joint_repair_phy_supplement/protocol_and_checks.json) | 本次补收10-27联合修复既有seeds46–48的PFQ/纯神经PHY、资源与成功率补表；不是新轨迹，也不是对当前物理搜索的独立胜出证据 |

完整候选、模型、原生轨迹、源码和profiler仍在本地`deliverables/tb_process_value_20261029_v1/`、`tb_residual_20261029_v1/`、`tb_search_gain_analysis_20261029_v1/`及`tb_joint_repair_phy_supplement_20261029_v1/`。快照脚本依赖本地研究包、模型、原状态和容器，不能独立运行。

各轮`manifest.json`保留原始来源SHA-256；补表使用本次复制清单。两份`tb_process_value/progress.txt`、`tb_residual/progress.txt`临时摘要已被相邻`report.txt`覆盖，本次去重删除；原来源清单仍保留其历史哈希，缺失原因是有意不收录冗余摘要。独有研究笔记、协议修订、负结果和核验文件保留，合并去向见[归档记录](../../archive/README.md)。

[此前文档复核](documentation_review.json)记录原神经研究证据的链接、哈希、统计重算及排除项；它不代表重新执行原生核验器，也不改变原报告的质量门槛判定。新增 DPLS 消融有独立的[原生与cohort核验](dpls_component_ablation/validation.json)及[归档清单](dpls_component_ablation/archive_manifest.json)，完整原生轨迹和包表保存在本地 `deliverables/dpls_component_ablation_20261029_v1/`。

# 2026-10-29 过程价值、物理残差与搜索诊断证据

当前结论见[项目进展](../../stageB_current_progress.md)，配置见[配置说明](../../current_stageB_effective_configuration.md)。本轮未替换有效算法，未运行311–315新seed闭环。

| 子目录 | 内容与结论 |
|---|---|
| [tb_process_value](tb_process_value/report.txt) | 58快照/1392块的离线价值蒸馏；强近似PHY及同深度对照；技术完整性通过，名义10%预算下95%高收益集合召回门槛失败 |
| [tb_residual](tb_residual/report.txt) | 全合法扩展、UE物理残差、集合外正例、C++局部缓存与完整搜索回放；缓存独立降本37.83%，冻结池收益未稳定转化为搜索优势，召回门槛仍失败；无新原生包时延结论 |
| [tb_search_gain_analysis](tb_search_gain_analysis/report.txt) | 仅读取冻结候选树的覆盖/排序损失与继续收益翻转诊断；评分混排、活动模式分解和继续价值均为待验证建议，未实施新策略 |
| [tb_joint_repair_phy_supplement](tb_joint_repair_phy_supplement/protocol_and_checks.json) | 本次补收10-27联合修复既有seeds46–48的PFQ/纯神经PHY、资源与成功率补表；不是新轨迹，也不是对当前物理搜索的独立胜出证据 |

完整候选、模型、原生轨迹、源码和profiler仍在本地`deliverables/tb_process_value_20261029_v1/`、`tb_residual_20261029_v1/`、`tb_search_gain_analysis_20261029_v1/`及`tb_joint_repair_phy_supplement_20261029_v1/`。快照脚本依赖本地研究包、模型、原状态和容器，不能独立运行。

各轮`manifest.json`保留原始来源SHA-256；补表使用本次复制清单。两份`tb_process_value/progress.txt`、`tb_residual/progress.txt`临时摘要已被相邻`report.txt`覆盖，本次去重删除；原来源清单仍保留其历史哈希，缺失原因是有意不收录冗余摘要。独有研究笔记、协议修订、负结果和核验文件保留，合并去向见[归档记录](../../archive/README.md)。

[本次文档复核](documentation_review.json)记录链接、已收录文件哈希、统计重算及排除项；它不代表重新执行原生核验器，也不改变原报告的质量门槛判定。

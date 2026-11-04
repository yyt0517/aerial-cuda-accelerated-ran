# 2026-11-03 研究证据

最新为[零新增PHY的可证容量界与规范惰性精评离线诊断](certified_bounds/report.txt)：同12状态378轮，基础界在floor可行候选中UB≥最大值的比例首扫1.31–1.33%、轮内3.80–3.83%；确认轮6.34%仍有UB>阈值。上界排序的条件评分次数上限23738（现有cache的12.97%），并非已运行的惰性轨迹或墙钟比例。原生干扰单调性精确算术成立，收紧版浮点证书仍缺；优先容量界。修正了1单位规范平局的停止边界，7项零PHY测试通过，未修改在线核、未训练/新闭环。[分组数据](certified_bounds/summary.csv)、[工作量界](certified_bounds/UB_order_work_bound.json)、[下一门协议](certified_bounds/next_protocol.json)、[复现](../../../training/stageb_lns/certified_bounds_README.rst)。

此前[full服务门校准、持久整数精确缓存与守卫cap4闭环](persistent_cache_reconstruction/report.txt)：新规范full/cache逐候选等价通过；公平串行墙钟23.48→10.95ms，缓存为同架构full的46.62%。对旧Python full的6.02%另列，其中包含编译/候选架构收益。守卫cap4为11.02ms，虽过12/12经验包络，却在两seed2秒原生闭环中因seed44 UE24过期+2.77pp未过长期服务门。M/学习暂停，默认不变。

- [校准及去重](persistent_cache_reconstruction/calibration_interpretation.json)、[描述性经验包络](persistent_cache_reconstruction/envelope/protocol.json)：冷/热full旧门6/24；不同服务模式互比0/22。随机平局终态重复，去重留一仅4/19；不是可靠概率分布，包络只作开发筛选。
- [整数等价门](persistent_cache_reconstruction/integer_gate/completion.json)、[串行墙钟](persistent_cache_reconstruction/wall/summary.csv)、[图](persistent_cache_reconstruction/wall_and_work.png)、[首扫与极宽松排序上限](persistent_cache_reconstruction/ranking_upper_bound.json)。首扫已占cache墙钟25.56%。
- [原生配对](persistent_cache_reconstruction/native_analysis/paired.csv)、[质量与成本门](persistent_cache_reconstruction/native_analysis/gates.json)、[工作账本](persistent_cache_reconstruction/work_ledger.json)、[20项测试与SHA核验](persistent_cache_reconstruction/final_audit.json)、[复现入口](../../../training/stageb_lns/persistent_README.rst)。4正式轨迹49600步，另32步smoke；44/45均为已暴露开发seed，没有最终新seed确认。源码、逐UE/成熟cohort、小型证据及全部原始大文件SHA已归档；原生大轨迹留在`deliverables/persistent_native_development_20261103_v1`。

此前[无审计墙钟、cap4与M16/32/64实际轨迹](cached_cost_budget/report.txt)：旧架构精确缓存墙钟67.31%-full；cap4为33.82%，相对full单一终态的快照J/服务仅4/12通过；M三档为62.85%/64.94%/67.34%，只过1/4/1。所有分支确认局部停止，但服务差异本身不能认定长期损失。该历史轮保存161700条精确脏候选标签，未训练、未实施整数增量、无新闭环；后续实现见上述最新报告。[曲线](cached_cost_budget/dirty_budget_curve.png)、[复现](../../../training/stageb_lns/cached_budget_README.rst)。

此前[依赖并集及旧精确ΔJ排序](dpls_joint_cache_diagnostics/report.txt)与[规范平局缓存等价门](canonical_cached_dpls/report.txt)：前者零新增PHY，冷/热full联合候选工作量局部估算18.37%/22.88%，不能乘成7.91%/12.01%；轮内旧ΔJ top16召回约80%，跨TTI首扫0/62。后者新增付费完整/缓存624690/183079次候选评分，12保存输入378轮、624670个候选全部PHY/选择逐位一致，尚无公平墙钟或闭环质量结论。[复现](../../../training/stageb_lns/cached_dpls_README.rst)。

此前[既有日志四项诊断](dpls_move_diagnostics/report.txt)无新增PHY/原生运行。冷/热full公共候选ΔJ变化23.30%/26.59%，依赖刷新量28.74%/34.02%；第5轮后仍有7.14/5.38个结构独立正Δ移动。廉价SIR top16召回1/251。热启动单独3–8倍预期撤回，先做跨轮精确依赖缓存与守共同floor的多移动，排序提前做诊断。[复现](../../../training/stageb_lns/dpls_move_diagnostics_README.rst)。

此前[第0步插桩与因果热启动物理对照](anytime_dpls_step0/report.txt)已实现逐候选ΔJ、真实相邻TTI重合率、成本分解，以及冷/热启动截断和总PHY预算对照。49项测试及短原生联调通过；完整2秒采集、开发质量/弱UE与公平成本门槛尚未评估。[复现入口](../../../training/stageb_lns/anytime_dpls_README.rst)、[原阶段门槛](anytime_dpls_step0/next_protocol.json)。服务PPO冻结，最新执行顺序以上述四项诊断为准。

此前[补充可修复失败状态与定向修复监督](repair_expansion/report.txt)保留为条件学习证据：学习选择修复关键验证失败，历史21状态J/服务均通过；验证成本和跨训练seed覆盖仍未通过。下文各轮结论按当时版本保留，已有实现身份见[路线梳理](ai_scheduling_review/report.txt)。

- [现有池定向2槽、服务风险终态选择与同根缓存](terminal_service/report.txt)：六臂、378次正式决策。缓存与原查询/物理值等价，验证93.66→84.96ms、开发106.04→94.19ms；验证仍缺服务与2.40ms成本，学习选择器没有独立收益。
- [验证](terminal_service/validation_summary.csv)、[开发](terminal_service/evaluation_summary.csv)、[结果图](terminal_service/quality_cost.png)、[矢量图](terminal_service/quality_cost.pdf)。
- [冻结协议](terminal_service/protocol.json)、[训练统计](terminal_service/training_summary.json)、[训练候选](terminal_service/training_candidates.npz)、[冻结模型](terminal_service/risk_model.npz)、[验证选型](terminal_service/selection.json)。仅41–43训练，44–48为历史评估状态；教师服务集合/J仅作离线标签。
- [关键终态诊断](terminal_service/validation_terminal_diagnostics.csv)、[逐UE诊断](terminal_service/validation_ue_diagnostics.csv)、[原始候选诊断](terminal_service/validation_candidate_predictions.npz)、[列定义](terminal_service/validation_diagnostic_schema.json)。关键状态窗口内11个合格候选，模型状态内排序AUC=0；不以总体Brier掩盖失败。
- [工作账本](terminal_service/total_work_ledger.json)、[74项测试及185组独立核验](terminal_service/final_audit.json)、[门槛和下一步](terminal_service/gate_results.json)。下一步逐UE因果表征、状态内配对风险排序和置信不足回退，尚未实施。
- [复现说明](../../../training/stageb_lns/terminal_service_README.rst)。本地全量来源为 `deliverables/terminal_service_20261103_v1`；归档保存原始计时、全部付费档案、训练数据/模型、冻结源码、代码/文件SHA及负结果。原始大轨迹由各输入manifest定位，不复制进归档。

条件仍依赖另付教师资源数/最小层/活动模式；结果不代表独立在线AI推理时延。未读最终seed311–315，无新原生轨迹，默认算法未变。

逐UE配对排序与低置信度回退：

- [完整报告](ue_pair/report.txt)：六臂378次正式决策；所有带回退方法21/21终态等于缓存最大J，没有新增损害或AI独立收益。验证服务仍1例失败，成本距门槛3.15ms。
- [验证结果](ue_pair/validation_summary.csv)、[开发结果](ue_pair/evaluation_summary.csv)、[配对结果](ue_pair/evaluation_paired_summary.csv)、[图](ue_pair/quality_cost.png)。
- [训练复用/去重](ue_pair/training_reuse.json)、[模型训练](ue_pair/training_summary.json)、[训练失败覆盖](ue_pair/training_repair_coverage.csv)：15独立状态、3165候选、29569对；只有1个可修复的最大J失败状态，直接修复vs最大J只有2对。
- [关键排序诊断](ue_pair/validation_ranking_diagnostics.csv)、[逐UE预测](ue_pair/validation_ue_diagnostics.csv)、[候选分数](ue_pair/validation_candidate_scores.npz)、[列定义及oracle范围](ue_pair/validation_diagnostic_schema.json)。替换辅助服务预测头的oracle仅用于事后空间诊断，不是在线策略。
- [门槛与下一步](ue_pair/gate_results.json)：先补充训练41–43其它TTI内可修复失败覆盖，再针对修复候选与最大J监督；尚未实施。44–48不回流训练。
- [83项测试及159组独立核验](ue_pair/final_audit.json)、[评分账本](ue_pair/total_work_ledger.json)、[复现说明](../../../training/stageb_lns/ue_pair_README.rst)。本地来源 `deliverables/ue_pair_20261103_v1`，完整冻结源码、模型、训练特征/配对、原始计时与诊断已归档。

训练状态扩充与定向修复：

- [完整报告](repair_expansion/report.txt)：六臂378次正式决策。训练源状态18→36、输入/档案去重15→30种，可修复失败1→3；41/42/43分别2/1/0，跨seed覆盖失败。按事前协议只完成一次诊断训练。
- [训练覆盖](repair_expansion/training_state_coverage.csv)、[覆盖门槛](repair_expansion/coverage_gate.json)、[训练排序对照](repair_expansion/training_ranking_summary.json)：普通扩充版仍不修复验证失败；定向配对＋状态/UE权重版成功。定向模型训练内修2/3、留seed仅1/3，不能认定泛化充分。
- [验证](repair_expansion/validation_summary.csv)、[开发](repair_expansion/evaluation_summary.csv)、[图](repair_expansion/quality_cost.png)：repair_guard为86.32/95.68ms，J均值对full为+0.2214%/+0.1418%；J/服务12/12及9/9通过。验证仍超82.17ms门槛4.15ms。
- [关键候选排序](repair_expansion/validation_ranking_diagnostics.csv)、[逐UE诊断](repair_expansion/validation_ue_diagnostics.csv)、[终态配对](repair_expansion/validation_terminal_diagnostics.csv)：关键raw44 t18000从最大J档案125改选178，P5与弱服务比恢复1.0，J仍高于full0.1463%；21状态只修复该1例，其余20终态不变。
- [模型](repair_expansion/repair_guard/ue_pair_model.npz)、[冻结清单](repair_expansion/models_frozen.json)、[训练协议](repair_expansion/protocol.json)、[验证选型](repair_expansion/selection.json)。上轮冻结模型和同数据普通配对模型均保留，不把组合训练改动归因于单一因素。
- [89项测试与195组独立核验](repair_expansion/final_audit.json)、[6167892次新评分账本](repair_expansion/total_work_ledger.json)、[下一步与停止条件](repair_expansion/gate_results.json)、[归档SHA](repair_expansion/archive_manifest.json)。下一步冻结模型做查询等价降本；尚未实施。
- 本地完整来源 `deliverables/repair_expansion_20261103_v1`；[复现说明](../../../training/stageb_lns/repair_expansion_README.rst)。教师资源/层/活动条件及其另付成本仍是重要边界，未做新原生轨迹、未用最终seed、未改默认。

历史路线与保留实现：

- [梳理报告](ai_scheduling_review/report.txt)、[25条路线对照](ai_scheduling_review/attempts.csv)、[实现及检查点清单](ai_scheduling_review/recommended_implementation.json)、[来源核验](ai_scheduling_review/source_evidence.json)。
- 已有原生闭环且有同候选池学习归因的AI研究参考：`stageb_ppo_v2`小批量服务选择PPO终版；payload/过期优于同池贪心，但平均/P95时延更差且不加速。不能称全面最优或通过晋级。
- 近期条件学习参考：本轮`repair_guard`；当前有效默认：原物理DPLS。三者身份分开保存，PPO参数及冻结源码已核验，可加载，未重跑历史原生实验。

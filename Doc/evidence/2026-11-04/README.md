# 2026-11-04 研究证据

新增[M-LWDF-PF＋精确剪枝DPLS与PFQ同窗口对比](mlwdf_pfq_boundary_comparison/report.txt)：Boundary/rate0.8、开发seed41–43、各2秒，同345471个成熟到达包。完整包吞吐742.516→1254.580Mbps（+68.96%），成功包平均时延122.502→2.743ms（−97.76%），P95 200→5.5ms，过期31.754%→0%；108/108个匹配UE吞吐与平均成功时延改善。零过期仍有9.21%非过期未交付。此为既有轨迹重算，无新增仿真/PHY/训练；不是五seed10秒或高负载的普遍最优结论。

- [完整指标](mlwdf_pfq_boundary_comparison/comparison.csv)、[逐seed](mlwdf_pfq_boundary_comparison/per_seed.csv)、[逐UE](mlwdf_pfq_boundary_comparison/per_ue.csv)、[共同成功包](mlwdf_pfq_boundary_comparison/common_success_pairing.csv)、[输入SHA](mlwdf_pfq_boundary_comparison/input_manifest.json)、[源码与复现](mlwdf_pfq_boundary_comparison/README.rst)。
- [推理计时对照](mlwdf_pfq_boundary_comparison/engineering_timing.csv)：相对完整DPLS，旧κ0保存输入的冻结剪枝内层加速2.78/2.30/2.06倍；M-LWDF-PF并发质量实验的9.757ms均时只作描述，缺少其与PFQ的公平串行对照。容量界＋最佳优先搜索可归为经典AI启发式搜索，没有模型训练或已证明的学习新增贡献。

最新为[新跨负载κ网格、文献权重与条件浅树](crossload_weights_v2/report.txt)。κ={0,1,2,4,8}×三负载×三seed的45点，加18文献点；入口通过后运行9条学习轨迹，共72点，新增731600原生步，另48步smoke。所有科学臂用同一个冻结精确内层，未追改旧网格。

| 负载 | 三seed描述性κ oracle | 仅训练41/42 κ oracle | 主要判断 |
|---|---:|---:|---|
| ×1 | 8 | 8 | κ8聚合支配其余κ，但M-LWDF-PF以1254.58Mbps、P95 5.5ms更强，过期均为0 |
| ×1.4 | 1 | 2 | 存在payload/总体过期/最差UE取舍；文献两形式被κ族强点压过 |
| ×1.6 | 0 | 0 | κ0聚合支配其余κ，仍有18.84%过期、最差UE63.78%，不承诺学习可消除 |

- [完整指标/全部Pareto](crossload_weights_v2/analysis/aggregate.csv)、[κ曲线](crossload_weights_v2/analysis/crossload_grid.png)、[文献/模型对照图](crossload_weights_v2/baseline_comparison.png)、[冻结协议](crossload_weights_v2/campaign/protocol.json)。oracle先保留均值payload≥最佳99.7%，再优先最差UE过期/总体过期/P95；两端的支配关系不只依赖此筛选。各条负载固定，没有负载切换实验。
- [条件浅树](crossload_weights_v2/learning/policy.txt)：仅41/42的80000条稳定期记录、一次fit；未输入真实负载/未来标签/seed。8/9条完全复现训练固定oracle，唯一额外收益源于×1.6 seed43稳定期14次κ2扰动，测量期所有点均不切κ。该点payload+3.50%、过期−2.675pp，但没有匹配非学习扰动控制，不能归为持续动态调权收益。[阶段选择](crossload_weights_v2/coefficient_choice_phases.csv)、[逐seed门](crossload_weights_v2/learning_analysis/paired.csv)。
- 对κ2和训练oracle全负载逐seed不劣，对三seed描述性oracle和M-LWDF未全过，对EXP-PF全过；学习新增贡献仍未稳健证明。经典权重只移植形式到相同联合内层，PF分母与κ族不同，不称原论文调度器完整复现。[来源/公式/复现](../../../training/stageb_lns/crossload_README.rst)。
- [最后SoA门](crossload_weights_v2/SoA_final/completion.json)：252状态全未过Γ逐位门，最大绝对差3.439e−12，无FMA，未接入求解器。按用户要求停止追40%，保留既有κ0输入33.90%/41.23%/46.38%精确成本曲线，原连续TTI尾部另存旧报告。
- [236状态独立full核验](crossload_weights_v2/final_audit.json)、[17项测试](crossload_weights_v2/tests.txt)、[成本账本](crossload_weights_v2/work_ledger.json)、[归档SHA](crossload_weights_v2/archive_manifest.json)。失败日志与同模型恢复记录保留；无默认、旧标签或旧cap4失败的修改，无最终确认seed。

前一阶段为[B与部署成本解耦：冻结TTL前沿及独立等价工程](B_decoupled_and_tournament/report.txt)。用户授权后，B沿原冻结方案完成κ=0/.25/.5/1/2、seed41–43共15点，新增13条161200原生步，复用两条κ0。科学臂始终使用同一冻结精确内层，工程库未替换中途求解器；旧40%部署墙钟失败保留。

| κ | payload/Mbps，三seed均值 | 合池过期率/% | 成功包P95/ms | 最差(seed,UE)过期率/% | 逐seed B门 |
|---|---:|---:|---:|---:|---|
| 0 | 1232.372 | 0.670100 | 125.5 | 24.144144 | 控制 |
| .25 | 1239.244 | 0.129389 | 73.5 | 10.180180 | seed41 P95 +1.5ms失败 |
| .5 | 1241.352 | 0 | 50.5 | 0 | 全过 |
| 1 | 1241.412 | 0 | 35.5 | 0 | 全过 |
| 2 | 1241.620 | 0 | 26.0 | 0 | 全过 |

- [全前沿](B_decoupled_and_tournament/frontier/ttl_frontier.png)、[逐seed四指标](B_decoupled_and_tournament/frontier/ttl_per_seed.png)、[配对门](B_decoupled_and_tournament/frontier/paired.csv)。κ2为冻结网格唯一四目标非支配点，payload+0.7504%；这是固定TTL权重收益，没有学习或泛化结论，后续模型需比较固定κ2。52个加权状态/1638547候选独立full核验通过。
- [MMSE覆盖](B_decoupled_and_tournament/activity/coverage_by_load.csv)：每TTI最多2448键，×1.6 full/bounded各覆盖bank87.52%/71.40%，现有日志不能恢复逐(UE,PRG)分布。零新增PHY，核已是固定4×4。
- [独立工程门](B_decoupled_and_tournament/engineering/audit_completion.json)252状态/6905轮/10006941候选逐位通过；[公平计时](B_decoupled_and_tournament/engineering/wall_summary.csv)3780次。锦标赛无净收益，native输入＋直接Γ＋树在×1.4/×1.6仍为full的41.23%/46.38%，40%门未过；未新增工程版连续尾部。
- [冻结协议](B_decoupled_and_tournament/B_campaign/protocol.json)、[工作账本](B_decoupled_and_tournament/work_ledger.json)、[13项测试](B_decoupled_and_tournament/tests.txt)、[最终核验](B_decoupled_and_tournament/final_audit.json)、[复现](../../../training/stageb_lns/decoupled_B_README.rst)。默认和旧标签保持，cap4原失败不改。

此前[基础容量界精确惰性内层、必评集合与高负载成本门](bounded_inner_load/report.txt)记录当时未启动B的历史判定，冻结报告不追写。12状态工程门通过，UB序精评/必评=1.000，full/cache/bounded为23.46/10.97/4.46ms。连续负载240状态等价通过，旧×1.4/×1.6墙钟43.49%/48.57%失败与连续尾部数据保留如下。

| 负载，rate0.8为基准 | 连续8000测量TTI墙钟P95/ms | 最大/ms | 同输入bounded/full均时 |
|---|---:|---:|---:|
| ×1 | 7.040 | 21.377 | 35.939% |
| ×1.4 | 9.735 | 27.942 | 43.494% |
| ×1.6 | 11.646 | 32.065 | 48.570% |

- [连续成本](bounded_inner_load/cost_by_load.csv)、[逐seed与紧度变化](bounded_inner_load/stress_analysis/native_costs.csv)、[公平串行对照](bounded_inner_load/stress_analysis/wall_summary.csv)、[尾分布与子工作量图](bounded_inner_load/cost_tail_and_work.png)。上界随负载变松，计时离群点保留，连续尾部与稀疏输入中位数分开报告。
- [PHY子账本](bounded_inner_load/PHY_accounting_by_load.csv)：×1.6候选评分仅14.47%，独立MMSE miss却保留81.59%；[单独剖析](bounded_inner_load/load_profile/summary.csv)解释墙钟缺口，评分数不等于逆矩阵数。
- [已完成的小样本工程门](bounded_inner_gate/report.txt)、[逐轮所需集合](bounded_inner_gate/rounds.csv)、[原始计时](bounded_inner_gate/timing_raw.csv)、[冻结源码](bounded_inner_load/source_manifest.json)、[11项测试](bounded_inner_load/tests.txt)。
- [原B网格](bounded_inner_load/B_plan/protocol.json)保存当时仅准备的方案；本轮只解耦入口后执行，κ/seed/因果HOL权重和服务门不变，成熟前沿见上方新报告。旧16步κ1接口smoke不作质量点。
- [工作账本](bounded_inner_load/work_ledger.json)、[归档核验](bounded_inner_load/final_audit.json)、[归档SHA](bounded_inner_load/archive_manifest.json)、[完整复现](../../../training/stageb_lns/bounded_README.rst)。六条正式原生轨迹74400步、24000测量TTI，另48步工程smoke；旧标签、默认与cap4原失败保持。

原生大日志及240个保存状态保留于`deliverables/bounded_load_stress_20261104_v1`，原始文件SHA随归档定位。当前版本仍是开发证据，没有最终新seed确认或默认替换。

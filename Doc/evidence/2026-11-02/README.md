# 2026-11-02 研究证据

- [结构化联合动作、继续价值与自身前缀教师纠错](structured_continuation/report.txt)：冻结 79 维缓存基线；小块分解与穷举一致，两个教师纠错轮次及 27 状态九臂串行回放完成，质量与成本联合门槛未通过。
- [完整质量与成本表](structured_continuation/summary.csv)、[逐 seed 表](structured_continuation/per_seed.csv)、[结果图](structured_continuation/quality_cost.png)。
- [冻结协议](structured_continuation/protocol.json)、[小块正确性](structured_continuation/decomposition_checks.json)、[事后预算分布诊断](structured_continuation/work_allocation.csv)。
- [实现与复现说明](../../../training/stageb_lns/structured_README.rst)。

新增项目 AI 加速主线的第一轮：

- [无 CTBS 候选排序加速 DPLS](no_ctbs_ai_ranked_dpls/report.txt)：训练/教师/推理/全部对照均去 CTBS；21 个纯 PFQ 来源开发状态、三臂三次串行计时完成。所选 AI 退回零残差，质量与成本联合门槛及学习独立贡献门槛均失败。
- [完整结果](no_ctbs_ai_ranked_dpls/summary.csv)、[AI 与同 K 启发式归因](no_ctbs_ai_ranked_dpls/ai_vs_heuristic.csv)、[耗时拆分](no_ctbs_ai_ranked_dpls/cost_breakdown.csv)、[补充诊断](no_ctbs_ai_ranked_dpls/analysis_notes.txt)。
- [冻结协议](no_ctbs_ai_ranked_dpls/protocol.json)、[模型选型](no_ctbs_ai_ranked_dpls/selection.json)、[实现与复现](../../../training/stageb_lns/ranked_dpls_README.rst)。

三个规模受控实验与路径选择：

- [正式 v2 报告](three_bounded_ai_probes/report.txt)：输入诊断、协调块/上界剪枝、独立残差生成器三个实验完成；9 状态×9 臂×3 次计时，另有固定模型成本归因。
- [完整质量/成本](three_bounded_ai_probes/summary.csv)、[输入逐 seed 结果](three_bounded_ai_probes/ranking_per_seed.csv)、[判定与路径选择](three_bounded_ai_probes/gate_results.json)。
- 物理上界剪枝最终动作及逐轮编辑与完整 DPLS 一致，均时 145.40→122.01 ms（−16.08%），未达 20% 降时门槛；不计为 AI 收益。
- 标量模型相对同预算启发式生成器平均 J +5.04%，7 胜 2 负，但对完整 DPLS 仍有约 16% 缺口；空间输入与学习块排序未提供稳定的质量/成本优势。下一步选择更深的标量单轨迹修复与自身访问状态纠错，暂不接 PPO。
- [冻结协议](three_bounded_ai_probes/protocol.json)、[完整路径核验](three_bounded_ai_probes/trace_audit.json)、[固定模型归因](three_bounded_ai_probes/scalar_attribution/summary.csv)、[准备批次与数值修正](three_bounded_ai_probes/preparation_note.json)、[复现说明](../../../training/stageb_lns/bounded_probes_README.rst)。

浅束/深轨迹与学生实际访问根训练：

- [完整报告](own_trajectory/report.txt)：完成“7条×8次尝试”与“1条×32次尝试”的验证对照、两轮学生真实访问根训练及冻结模型开发评估，共369次正式串行决策。
- [开发质量/成本/覆盖](own_trajectory/evaluation_summary.csv)、[逐状态结果](own_trajectory/evaluation_per_state.csv)、[几何选择](own_trajectory/geometry_selection.json)、[模型选型](own_trajectory/selection.json)、[判定及下一步](own_trajectory/gate_results.json)。
- 验证集选择浅束；两轮采集288组，去重并加入锚点后共295个训练根。续训浅束在9个开发状态8胜1负、平均J提高0.834%，但seed46均值略退化，对完整DPLS仍有15.56%质量缺口，服务覆盖未过门槛。
- 单轨迹32次尝试平均仅9.33次有效编辑，其余反复停滞；[停滞诊断](own_trajectory/diagnostics/stalls.csv)确认9/9开发失败根存在遗漏改善。下一阶段选择top2/top4物理短名单与根内失败去重，暂缓简单增深、同标签扩训和PPO。
- 全UE即时服务P5在本轮参考状态中为0，原P5比值没有非平凡保护意义；[事后有效UE服务诊断](own_trajectory/diagnostics/positive_coverage.csv)单独保留，不追改选型门槛。[复现说明](../../../training/stageb_lns/own_trajectory_README.rst)。

单轨迹top2/top4精验与根内失败去重：

- [完整报告](shortlist_trajectory/report.txt)：完成共享特征缓存的启发式/AI、去重/重复短名单对照，一轮实际根与失败掩码状态训练，以及558次正式串行决策。
- [机制验证表](shortlist_trajectory/phase_a_summary.csv)、[开发质量/成本/覆盖](shortlist_trajectory/evaluation_summary.csv)、[机制选择](shortlist_trajectory/mechanism_selection.json)、[模型选型](shortlist_trajectory/selection.json)、[门槛与下一步](shortlist_trajectory/gate_results.json)。
- 验证选择top4；采集144状态（139物理根），与锚点去重后179组，选epoch10。开发初始top2/top4均时83.85/65.08 ms，平均J相对full为−8.74%/−9.55%；续训top4为50.41 ms/−9.94%，学习没有稳定迁移。top2续训迁移对照为68.81 ms/−7.41%，不能据此回选本轮K。
- 所有去重臂根内重复评分为0；所有新臂成本通过，但质量/有效服务覆盖未过。[终态诊断](shortlist_trajectory/terminal_diagnostics/terminal_opportunities.csv)中，初始/续训两种模型的top2/top4共36个开发终态仍有>16 B改善。
- 下一阶段冻结模型，验证106/160/212次PHY预算曲线，保持56特征根上限与原质量/覆盖/成本门槛，暂不追加训练/PPO。[复现说明](../../../training/stageb_lns/shortlist_README.rst)。

冻结模型的106/160/212评分预算曲线：

- [完整报告](budget_curve/report.txt)、[质量—成本曲线PNG](budget_curve/quality_cost_curves.png)、[矢量PDF](budget_curve/quality_cost_curves.pdf)：两个冻结模型与启发式×top2/top4×三预算，共18点＋full，完成1197次串行决策，未追加训练。
- [验证曲线](budget_curve/validation_summary.csv)、[开发曲线](budget_curve/evaluation_summary.csv)、[逐状态预算增量](budget_curve/evaluation_budget_deltas.csv)、[冻结选型](budget_curve/selection.json)、[判定及下一步](budget_curve/gate_results.json)。
- 验证选择训练后top4/212；其开发106→160→212的均时为50.06→64.70→79.28 ms，平均J缺口9.94%→7.78%→6.29%；新增未分配UE均值1.33→1.78→2.44，有效UE P5比0.461→0.403→0.367，未过质量/覆盖门槛。
- 所有开发预算增量的J均改善，无特征根上限触发，但全部18点仍未达标。没有观察到J饱和，也未排除更大预算；下一步优先验证整TB协调宏动作及有限继续收益，更大单编辑预算保留作备选对照，暂不扩训/PPO。
- 46项测试、399组独立NumPy终态核验通过；模型、求解器、注册表均未改。[复现说明](../../../training/stageb_lns/budget_curve_README.rst)。

整TB联合候选与付费有限继续搜索：

- [完整报告](tb_macro_continuation/report.txt)、[质量/服务—成本图](tb_macro_continuation/quality_cost.png)：9臂、21状态、567次正式串行决策，比较212/424评分上限下的单编辑、联合及继续搜索；本轮不训练新模型。
- [验证表](tb_macro_continuation/validation_summary.csv)、[开发表](tb_macro_continuation/evaluation_summary.csv)、[配对收益](tb_macro_continuation/evaluation_paired_summary.csv)、[冻结选型](tb_macro_continuation/selection.json)、[判定与下一步](tb_macro_continuation/gate_results.json)。
- 开发212联合对同机制单编辑4胜5负、平均J差−501.62B；联合继续对联合1胜7负1平、差−187.53B。424联合继续仅1胜8平，均时增加13.98ms，覆盖仍失败。所有新路径成本也未过；计算覆盖17/17PRG不代表UE服务覆盖通过。
- [付费事后验证诊断](tb_macro_continuation/posthoc_headroom/summary.json)另用6219次物理评分核对48个固定终态块：联合20块有正收益、单编辑13块；16块只有联合改善，但10个联合正收益块被代理top2遗漏。保护当前根服务后仍有互补机会，不能将该离线候选上界当作在线方法表现。
- 下一步选择廉价增量TB描述符与服务保护的选择性联合提议/排序，保留单编辑和原根回退；先验证特征成本，再仅在训练seed采集标签做小规模蒸馏。暂缓继续价值、固定域扩张、同标量目标扩训和PPO；该下一步尚未实施。
- 53项测试、189组正式终态及184组事后候选独立NumPy核验通过；[实际工作](tb_macro_continuation/evaluation_work.json)、[继续域事件](tb_macro_continuation/evaluation_continuations.json)、[复现说明](../../../training/stageb_lns/tb_macro_README.rst)。

廉价增量TB特征、服务保护与选择性联合排序：

- [完整报告](selective_tb/report.txt)、[质量/服务—成本图](selective_tb/quality_cost.png)：10臂、630次正式串行决策，一轮真实访问状态蒸馏；[完整开发表](selective_tb/evaluation_summary.csv)、[配对收益](selective_tb/evaluation_paired_summary.csv)、[判定及下一步](selective_tb/gate_results.json)。
- 工程成功：21个状态增量固定联合与旧实现最终动作完全相同，开发完整均时168.40→71.76ms（−57.39%），是逻辑核优化收益。[工程一致性](selective_tb/engineering_checks.json)、[微基准](selective_tb/engineering_features.csv)。
- 训练seed41–43采集72组真实根/PRG/失败掩码、9417候选标签，教师实际9507评分，20epoch选epoch5；[访问与付费工作](selective_tb/training_data.json)、[训练记录](selective_tb/training_history.json)、[选型](selective_tb/model_selection.json)。
- 学习模型开发对同机制启发式4胜5负、平均J差−371.54B；P5均比0.662→0.774，但总J与覆盖仍未达标。受保护路径新增未分配UE为0，来自显式约束，不计为学习独立收益。
- [事后保护范围核对](selective_tb/posthoc_anchor_scope.csv)：full在21/21状态均不满足永久保留全部初始PFQ分配的约束；开发PFQ平均35.78个分配UE、full15.89个。本轮锁定集合比原目标更强，不能把扩大约束下的退化归因于特征近似，也不据此证明近full解不可行。
- 下一步保留增量引擎，先验证与终态目标对齐的因果服务集合提议与有限终态修复；完整DPLS集合仅作离线标签，不作在线oracle掩码。暂停扩训当前锁定集合学生、PPO和继续价值；该下一步尚未实施。
- 63项测试、210组正式终态、16批工程及420个训练候选独立核验通过；[完整核验](selective_tb/final_audit.json)、[复现说明](../../../training/stageb_lns/selective_tb_README.rst)。

因果服务集合提议与预算内终态修复：

- [完整报告](service_repair/report.txt)、[在线质量/服务—成本图](service_repair/quality_cost.png)：11条在线路径、693次正式串行决策；另有24次验证集教师集合条件探针，后者不计作在线性能。
- [开发结果](service_repair/evaluation_summary.csv)、[集合校准](service_repair/set_calibration_grid.csv)、[模型冻结](service_repair/model_selection.json)、[路径选型](service_repair/selection.json)、[判定及下一步](service_repair/gate_results.json)。
- 18训练状态621合法UE行，一轮60epoch选epoch10/阈值0.3；训练/验证完整教师分别1011932/617686次评分，显式离线计费。开发学习分配召回87.96%、平均21.22UE，最差75%，未过逐状态集合门槛。
- 同一main180前缀后，学习修复32相对普通搜索212平均J−882.27B、多39.49ms；新增未分配UE3.44→1.44、P5均比0.111→0.510，但质量/覆盖未过。repair64为150.21ms，成本也失败。[配对表](service_repair/evaluation_paired_summary.csv)、[逐seed](service_repair/evaluation_paired_per_seed.csv)。
- 相对集合大小匹配对照，学习身份选择7胜2负、平均J+30.09B，但seed48退化，未形成稳定学习贡献。真实教师集合锁定修复验证平均J仍低5.16%、P5比0.809，[条件探针](service_repair/conditional_summary.csv)不是完备上界，排除教师准备的耗时不作在线成本声称。
- [条件终态资源差异](service_repair/conditional_resource_gaps.csv)仍涉及资源数/最小层/活动模式；下一步分别验证这两类条件及联合条件，再决定因果联合提议器训练。暂停扩大二值分类、硬锁定和晚修复预算作为主线；该下一步尚未实施。
- 72项测试、231组正式终态＋24组条件终态＋30组教师终态独立NumPy核验通过；[付费工作](service_repair/training_teacher_work.json)、[修复工作](service_repair/evaluation_work.json)、[复现说明](../../../training/stageb_lns/service_repair_README.rst)。

资源数/最小层、活动模式及联合条件消融：

- [完整报告](resource_conditions/report.txt)、[条件质量—执行成本图](resource_conditions/quality_cost.png)、[工程等价降本图](resource_conditions/backend_cost.png)：三条件＋同解码器无条件＋普通212＋full，共6臂；两版各378次正式决策，合计756次。没有训练新模型。
- 条件只含每UE资源数/精确最小层和/或小区-PRG活动bit，不提供教师UE到PRG位置。匹配用最小层见证槽保证严格可行；[条件清单](resource_conditions/evaluation_condition_manifest.json)、[原协议](resource_conditions/protocol.json)、[V2后端说明](resource_conditions/backend_protocol.json)。
- V2所有21状态×6臂终态动作/物理值/实际工作与V1一致；联合执行均时203.95→75.77ms，属于工程优化。[开发全部结果](resource_conditions/evaluation_summary.csv)、[配对比较](resource_conditions/evaluation_paired_summary.csv)。教师准备排除，不能称在线AI耗时。
- 开发资源/层、活动、联合条件平均J变化分别−1.44%/+1.03%/−2.34%，最差−6.02%/−3.44%/−7.70%；全部质量/覆盖失败。资源/层P5均比0.994仍不能替代逐状态门槛。验证选择联合，未据开发结果回选资源臂。
- [付费终态局部诊断](resource_conditions/terminal_audit/summary.json)另计1292次评分：12个联合验证根只有1个>1B、没有>16B局部改善；另有一根存在通过服务门槛的邻居但最大J解未保住P5。下一步先改更广多PRG重排及J/弱服务Pareto选择，条件解码稳定达标后再训练因果提议器；该下一步尚未实施。
- 84项测试；两版252组正式、46组教师、16组工程终态及20组事后候选独立NumPy核验。MMSE bank每决策2448次唯一求解及代理工作全部单列，[实际工作](resource_conditions/evaluation_work.json)、[门槛及下一步](resource_conditions/gate_results.json)、[复现说明](../../../training/stageb_lns/resource_conditions_README.rst)。

联合条件下的广UE—PRG重排与J—弱服务前沿：

- [完整报告](joint_reorder/report.txt)、[质量/服务—成本图](joint_reorder/quality_cost.png)：9臂、567次正式串行决策；所有条件路径212评分上限，未训练新模型。
- [开发全表](joint_reorder/evaluation_summary.csv)、[配对收益](joint_reorder/evaluation_paired_summary.csv)、[验证选型](joint_reorder/selection.json)、[门槛及下一步](joint_reorder/gate_results.json)。条件仍只含资源数/精确最小层/活动bit，不提供教师位置或服务集合。
- 稀疏窄池21状态动作/完整物理值与旧联合相同，开发均时76.27→67.44ms。广前沿maxJ对旧联合5胜0负4平，开发J平均缺口2.34%→0.22%、最差7.70%→1.64%，服务覆盖9/9通过；但质量2状态失败，136.82ms超过116.28ms成本门槛，验证服务仍有3状态失败。
- 三个广前沿最终选择共用同一搜索，0.5%/2%相对固定档案bestJ限制让步，不放宽最终99.5%-full门槛；扩大让步没有解决剩余失败。活动参考平均J+1.03%仍伴随新增未分配UE5.78/P5比0，继续仅作对照。
- [事后服务组诊断](joint_reorder/posthoc_group_diagnosis.json)揭示部分失败与初始化遗漏有效UE相伴，并非唯一原因证明。下一步保留广重排/前沿，消融连续弱服务风险并批量/缓存前沿计算；暂缓因果提议器训练，该下一步尚未实施。
- 94项测试；189组正式、23组教师、8组工程解码终态独立核验。[实际工作](joint_reorder/evaluation_work.json)、[完整核验](joint_reorder/final_audit.json)、[复现说明](joint_reorder/code/joint_reorder_README.rst)。MMSE bank和直接数组读取全部记账；条件执行排除教师准备，不是在线AI耗时。

连续弱服务指标与前沿计算降本：

- [完整报告](weak_service/report.txt)、[逐状态门槛—成本图](weak_service/quality_cost.png)：7臂、441次正式串行决策，保留广重排/212评分/56根与最终maxJ；没有新训练。
- 等价降本在21正式＋2工程状态的完整评分顺序、查询根、全访问物理值、动作、组与工作量均一致，开发136.78→100.76ms（−26.34%）。[等价记录](weak_service/evaluation_equivalence.csv)、[组件成本](weak_service/evaluation_work.json)。
- 全配额组消除原服务组遗漏；连续指标开发平均J缺口0.2231%→0.0455%，J失败2→1，覆盖9/9通过，97.48ms低于117.15ms成本门槛。验证J全通过但仍1状态服务失败，86.82ms高于81.83ms；仍没有路径全面达标。[验证表](weak_service/validation_summary.csv)、[开发表](weak_service/evaluation_summary.csv)、[配对差](weak_service/evaluation_paired_summary.csv)。
- log/exp最终动作在21状态全相同，访问序列大多不同；验证按耗时并列决胜冻结exp，不据微小时间差称指数指标更优。[选型](weak_service/selection.json)、[判定与下一步](weak_service/gate_results.json)。
- [已付费访问档案诊断](weak_service/observed_service_diagnosis.json)无新PHY：验证失败状态211个条件候选无覆盖通过解；开发失败状态159个覆盖通过候选中无J达标解。下一步保留降本/连续指标，验证预算内弱UE定向联合重排及付费代理召回诊断；暂停扫平滑参数、终态让步和上游因果提议器训练，该下一步尚未实施。
- 106项测试，147组正式、23组教师、10组工程终态独立核验。[完整核验](weak_service/final_audit.json)、[教师账本](weak_service/teacher_ledger.json)、[复现说明](weak_service/code/weak_service_README.rst)。条件执行排除教师准备，不能作为在线AI推理耗时。

预算内弱UE定向联合重排与付费召回诊断：

- [完整报告](directed_reorder/report.txt)、[质量/服务—成本图](directed_reorder/quality_cost.png)：6臂、378正式决策；212评分/56根保持，保留降本引擎与连续指标，没有新训练。
- 四条定向路径开发9/9同时通过质量/覆盖/成本，平均J−0.0455%→+0.1418%、最差−1.6391%→+0.0332%。验证选joint4后冻结，开发109.20ms低于116.76ms；验证仍1服务失败且95.58ms高于81.98ms。[验证表](directed_reorder/validation_summary.csv)、[开发表](directed_reorder/evaluation_summary.csv)、[选型](directed_reorder/selection.json)。
- 现有池2槽与联合池2槽的21状态J/覆盖相同，20/21动作及完整物理值相同；新增联合池未形成额外终态收益。稀疏交替开发较快但验证反而较慢。[配对表](directed_reorder/evaluation_paired_summary.csv)、[工作量](directed_reorder/evaluation_work.json)。
- [付费诊断](directed_reorder/recall_diagnosis/summary.json)4根另计2523评分/9792唯一MMSE，原baseline验证/开发根最佳J候选代理排36/287，显示漏排；只限冻结终态未访问的一跳池，不算在线额外免费收益。
- [已访问档案](directed_reorder/observed_feasibility_summary.csv)中新路径21/21存在J/覆盖通过解，但maxJ选择丢掉验证服务。下一步候选级因果服务风险校准/受约束终态选择，配合同根候选/代理缓存降本；仅训练seed收集标签，暂缓上游联合提议器，该下一步尚未实施。[判定及下一步](directed_reorder/gate_results.json)。
- 118项测试、126正式/23教师/12工程终态及21诊断候选/根独立核验；诊断JSON索引序列化修复后重复21次独立检查，无新增native评分。[核验](directed_reorder/final_audit.json)、[恢复记录](directed_reorder/serialization_recovery.json)、[复现说明](directed_reorder/code/directed_reorder_README.rst)。条件执行排除教师准备，未声称在线AI成功。

全部是历史开发状态的离线实验。未读取 311–315，未新增原生轨迹，未更换默认算法；不据此宣称场景泛化或整包时延改善。

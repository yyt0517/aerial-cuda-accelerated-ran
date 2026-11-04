from pathlib import Path
import csv,json
from training.stageb_lns.experiment import write_json
p=Path('deliverables/terminal_service_20261103_v1')
read=lambda n:list(csv.DictReader((p/n).open()))
val=read('validation_summary.csv');dev=read('evaluation_summary.csv');sel=json.loads((p/'selection.json').read_text())
nextpath=dict(selected='保留现有池2槽和同根缓存；验证逐UE因果服务表征与状态内成对风险排序，置信不足回退最大J；进一步定位代理阶段成本',
 implementation_status='仅选为下一轮，尚未实施',
 reason='关键验证状态2%窗口内11个合格候选仍被漏选；同状态服务排序AUC为0，概率阈值调整不能修正错误排序；缓存有效但代理读数减少未转为代理阶段时间下降',
 mechanism=['逐UE可服务性/服务损失特征与软有效服务集合；离线full集合仅作标签，不作为在线掩码',
 '学习同一状态候选相对风险和弱UE损失，按seed划分并对相同档案去重；仅41-43训练，不将44-48诊断回流',
 '低置信度或无可信改善时保留原最大J候选；回退防止新增伤害，不宣称可修复剩余覆盖失败',
 '对候选ID/代理缓存查找与选择计算做分项计时，再验证按索引复用的等价实现，保留有序查询/物理值核验'],
 gates='不改99.5%-full J、98%有效覆盖、原PFQ90%floor与80%完整成本门槛；新模型须优于同机制启发式',
 deferred=['当前聚合分类器扩容/继续同标签扩训','调整概率阈值以追当前失败样本','更大联合池','上游资源/活动提议器','PPO','新最终seed或原生闭环','默认替换'])
write_json(p/'gate_results.json',dict(selection=sel,validation=val,development=dev,cached_search_equivalence_states=21,
 training_done=True,learned_independent_gain=False,online_ai_success=False,next_path=nextpath))
write_json(p/'final_audit.json',dict(passed=True,tests=74,formal_decisions=378,formal_independent_finals=126,teacher_independent_finals=39,
 training_decoder_independent_finals=18,engineering_reference_independent_finals=2,total_independent_finals=185,
 cached_full_query_action_value_parity_states=21,extra_training_parity_states=2,all_terminal_arms_paid_archive_equal=True,
 training_candidates=3798,training_states=18,distinct_training_archives=15,gradient_steps=960,model_frozen_before_validation=True,
 source_model_registry_physics_hashes_valid=True,additional_posthoc_native_scores=0,final_seeds_read=False,native_trajectories=0,
 all_validation_and_development_gates_passed=False,selected_development_gate_passed=True,learned_independent_gain=False))
lines=['现有池定向2槽＋候选级服务风险校准与终态选择＋同根候选/代理缓存',
'日期：2026-11-03（Asia/Shanghai）',
'实验：terminal_service_20261103_v1；正式执行完成，无运行失败或搜索代码事后修补。','',
'结论',
'缓存可保留，当前学习终态选择不晋级。21状态最大J缓存版与原existing2的动作、有序根/查询、所有已付费物理值完全相同；验证均时93.66→84.96ms（−9.28%），开发106.04→94.19ms（−11.18%）。这是等价工程收益，不计为AI贡献。',
'验证冻结选择cached_maxj，开发9/9质量/覆盖通过且成本通过。验证仍有1/12覆盖失败，平均成本84.9645ms高于82.5604ms门槛2.4042ms。学习选择器开发终态相同、额外约3.63ms；验证降低J且把唯一失败状态P5从0.7852降到0，没有AI独立收益。','',
'1. 不变协议与六臂',
'3小区、36UE、17PRG；canonical native BLER，因果缺口权重J，原未剪枝rawPFQ90%出队floor。无CTBS/headroom/PPO。各条件臂212次整调度精验（包括初始PFQ），56特征根上限、16种子候选、8候选/批、4根前沿与95%探索保留；实际均212评分、2448次唯一MMSE。',
'条件仍来自另行付费的完整物理DPLS：每UE资源数、精确最小层、小区×PRG活动。教师UE位置/J/服务值/有效UE集合不传入解码或终态特征API。完整物理参考不是全局最优。',
'existing2为冻结旧实现；cached_maxj只缓存同一不可变决策的根候选池与根/候选代理值；continuous按quota-exp/平均可靠度/J选终态；attainable按自身档案各UE最大服务形成因果集合与P5/固定弱20%参考；calibrated以小MLP预测服务覆盖通过概率。最后三臂均只从已付费、满足条件/floor的档案挑选终态。',
'内部J风险窗冻结为自身最佳J的2%，依据已暴露的历史诊断空间，不称新盲测。外部每状态J≥99.5%full、无新增未分配UE、full有效集合P5与其固定最弱20%服务≥98%、seed等权完整均时≤80%full不变。',
'连续终态在2%窗内优先quota-exp；可达启发式优先分数≥.98中的最大J，否则最大分数；学习臂优先概率≥.9中的最大J，否则最高概率再J。无可信候选时仍按低概率排序，是本轮冻结策略，未在看到失败后改回退。','',
'2. 学习与冻结顺序',
'先仅用seed41–43、raw_global_deficit/PFQ两来源、TTI4001/8001/18000的18状态，跑同机制自身搜索。共3798条件/floor可行候选，2290个覆盖正标签；候选物理值全部已付费，不追加候选精验。另付18次完整教师，仅离线生成覆盖标签。',
'42维输入：当前候选J/自身最好J、quota/自身可达集合的连续可靠度统计、相对自身档案max/q90/mean/bestJ的服务统计、因果quota/权重汇总。模型42→32(tanh)→16(tanh)→1；固定Adam lr=.01、weight_decay=.001，每次240 epoch。三次留一seed拟合得到OOF logits，拟合单调Platt；再用全部训练集拟合一次，共960梯度步。无模型规模、epoch、阈值搜索。部署npz为17146B，NumPy推理。',
'每折归一化仅拟合该折训练seed，候选不随机拆分。18训练状态含3对完全相同的付费档案，实际15种；同一seed内重复没有跨折泄漏，但样本数不能解释为18个独立物理状态。训练均匀按状态加权，重复未去重，此局限保留。',
'OOF原/校准Brier为0.053649/0.054638，校准未改善Brier；OOF用于拟合校准，不能把该诊断当独立测试。留种子终态诊断有服务误判和质量失败；训练档案本身有1状态无J/服务联合达标方案。模型按事前协议冻结后才读44–45本轮验证；验证选型冻结后才读46–48开发。','',
'3. 正式结果',
'三次串行轮换，每状态取完整耗时中位数后seed等权。验证12状态有2对相同档案（10种），开发9状态9种。44–48此前多轮暴露，均是历史开发证据，不能宣称泛化。','']
for label,data in [('验证：12状态',val),('开发：9状态',dev)]:
 lines += [label,'方法 | 平均J对full% | 最差J对full% | 完整ms | 最差P5比 | 最差弱20%比 | J失败/覆盖失败 | 成本通过']
 for r in data:
  lines.append(f"{r['arm']} | {float(r['mean_relative_j_pct']):+.6f} | {float(r['worst_relative_j_pct']):+.6f} | {float(r['mean_ms']):.4f} | {float(r['worst_positive_p5']):.6f} | {float(r['worst_positive_weak']):.6f} | {r['quality_failures']}/{r['coverage_failures']} | {r['execution_cost_pass']}")
 lines.append('')
lines += ['所有条件臂均无新增未分配UE。验证选择cached_maxj为质量与成本路线，但execution_cost_feasible=false；开发没有回选。开发cached_maxj相对full均J+0.141814%，最差+0.033181%；成本为full的63.83%。',
'calibrated对cached_maxj验证2次J下降（>1B）、10次平，平均J−201.64B；开发9次平。两个改变发生在raw seed44/45 t18000，均未找到概率≥.9候选；seed45无必要地让出约1.6447%内部J，seed44服务进一步恶化。continuous开发平均J略降、弱服务略升，未修复验证失败；attainable也未修复。','',
'4. 为什么仍漏掉好终态（事后只读，不训练/调参）',
'关键状态raw_global_deficit:seed44:tti18000有211个条件候选，14个J/服务联合达标；2%内部窗129个候选中仍有11个联合达标。最好达标J为full的100.146343%，因此本轮失败不能归因于窗口完全排除好方案。所有验证/开发21状态在该窗内均存在联合通过候选；这是离线oracle空间，不是可部署选择器。',
'因果可达集合比full有效集合多UE5、25两个UE；其full服务为0，自身档案最大服务约2879/2880B。逐UE最大值形成的服务上包络不等于同一方案同时可达的参考。关键状态窗内所有候选的attainable最小服务比均为0，启发式失去区分力。',
'当前最大J终态已丢失full服务1689.49B的UE34。学习臂选memo索引27，除UE34仍为0，还把UE17从2059B降为0，UE12从3009B降为1058.87B；全局P5降0、固定弱20%比降0.571135。学习臂内部J损失1.5650%，对full J仍+0.280466%，不能以总J过线掩盖服务损失。',
'关键窗内：好候选最高预测概率0.074412696；坏候选最高0.075290446；最大J原终态0.074696082。所有都远低于.9；状态内服务正负候选排序AUC=0，所有好候选都被排在坏候选之后。固定单调概率校准不改变排序，单纯调概率阈值不能修复这处错误。没有输入归一化裁剪触发，不把此现象解释为裁剪故障。',
'总体候选Brier验证0.05964、开发0.02975也未揭示这一关键状态内排序失败。聚合分类准确度不是终态调度收益。逐UE身份/损失结构在聚合中被抹平、模型偏重跨状态服务难度，是据现象提出的解释，尚须下一轮逐UE/配对消融验证。','',
'5. 缓存收益与成本边界',
'候选池以完整根/wide/seed索引，代理以完整根/候选索引；仅同一不可变决策内存活。按需填代理，保留候选原序/重复去除/已评分过滤与周期seed_action全局随机数调用。所有终态臂搜索档案完全相同，2训练工程状态加21正式状态核验有序根/候选与所有物理值；最大J终态等价。代理计算次数、生成尝试与gamma读取本来就应下降，不要求这些工程工作量相同。',
'验证候选生成均时17.83→8.15ms，代理阶段23.38→23.92ms；开发生成21.03→9.27ms，代理29.40→29.14ms。代理直接gamma读取验证65809.7→32800.4、开发79435.1→36859.1；实际新代理行验证5414、开发6168.3，但代理完整阶段几乎未降时。现有分项含查询/索引、服务特征和选择，不能单凭读数推断其耗时归因。',
'每状态缓存数组与键载荷平均验证6.05MB/开发6.89MB，全体最大12.73MB；不含Python容器额外开销，不能当作进程峰值内存。最大J缓存版仍须再省验证约2.40ms才过80%门槛；学习版验证还差5.94ms。',
'整个终态阶段首重复均值：cached_maxj约0.18ms；attainable约3.69/3.73ms；calibrated约3.83/3.85ms（验证/开发）。完整均时学习对cached_maxj增加3.54/3.63ms。这包含全部终态特征、排序与网络，未单独测网络，不把其全部归为神经推理。各分项是首重复均值，不与完整耗时中位数强行相加。',
'完整计时含数值输入/rawPFQ、freshPHY/MMSE、压缩条件检查、全搜索/缓存/终态特征/推理及上下文清理；公共轻量观测复制也计入。排除教师准备、训练、模型文件IO、一次性库/TBS准备、归档和独立核验。teacher条件成本另列，因此94.19ms不是独立在线AI调度端到端时延。','',
'6. 工作量与核验',
'378次正式决策，3920388次精验；18训练＋2工程对照解码，4240次精验；39次另付完整教师，2296468次精验/4553.36ms；合计6221096次native整动作评分。3798个监督候选的值来自前述已付搜索，没有额外标签评分。事后归因无新增native评分。',
'74项针对PHY/因果输入/资源条件/联合重排/连续前沿/定向选择/缓存/终态边界的测试通过；126正式＋39教师＋20训练/工程=185组独立NumPy终态核验；三重复动作/物理值/查询一致。冻结源码、模型、旧物理内核和注册表SHA有效；无执行后搜索修补。未读取311–315，无新原生轨迹，默认PFQ＋CTBS＋DPLS与注册表未改。','',
'7. 下一步路径（选定，尚未实施）',
'保留existing2、quota-exp前沿与同根缓存作为对照；保留最大J作为可靠回退。下一轮验证逐UE因果可服务性/服务损失表征＋同状态候选成对风险排序，配合低置信度回退。参考有效集合只作离线标签，不能直接成为线上掩码。回退只能避免此次新增损害，不能声称已修复原有覆盖失败。',
'训练仍仅41–43，对重复档案去重、按状态内好坏候选构造比较，分seed验证；以状态内排序/最终覆盖收益而非总体Brier选型。不将本轮44–48错误样本用于训练，不直接扩容聚合分类器或追调概率阈值。',
'工程上先拆解代理索引/候选选择的实际成本，评估稳定候选ID和数组复用是否能再省约2.40ms；每次改动仍核对完整查询与物理值。上游资源/活动提议器继续暂缓，现阶段没有证据宣布AI智能调度目标达成。','',
'主要证据：protocol.json、risk_model.npz、training_candidates.npz、training_summary.json、selection.json、两阶段summary/per_state/measurements、equivalence.csv、risk/terminal/UE_diagnostics、candidate_predictions.npz、teacher_ledger.json、total_work_ledger.json、final_audit.json、gate_results.json、quality_cost.png/pdf。',
'本地全量来源：deliverables/terminal_service_20261103_v1；长期归档：Doc/evidence/2026-11-03/terminal_service。']
(p/'report.txt').write_text('\n'.join(lines)+'\n')

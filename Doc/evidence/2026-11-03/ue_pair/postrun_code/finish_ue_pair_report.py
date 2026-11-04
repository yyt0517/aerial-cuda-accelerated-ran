from pathlib import Path
import csv,json,numpy as np
from training.stageb_lns.experiment import write_json,sha
p=Path('deliverables/ue_pair_20261103_v1');prior=Path('deliverables/terminal_service_20261103_v1')
read=lambda n:list(csv.DictReader((p/n).open()))
val=read('validation_summary.csv');dev=read('evaluation_summary.csv');sel=json.loads((p/'selection.json').read_text())
for phase in ('validation','evaluation'):
 aa=np.load(p/f'{phase}_actions.npz');vv=np.load(p/f'{phase}_values.npz');prev=np.load(prior/f'{phase}_observed_values.npz');cur=np.load(p/f'{phase}_observed_values.npz')
 for key in cur.files:
  if key.endswith('|cached_maxj'):np.testing.assert_array_equal(prev[key],cur[key])
 for key in aa.files:
  if key.endswith('|cached_maxj'):
   sid=key.split('|')[0]
   for arm in ('aggregate_guard','ue_heuristic_guard','ue_pair_guard'):
    np.testing.assert_array_equal(aa[key],aa[f'{sid}|{arm}']);np.testing.assert_array_equal(vv[key],vv[f'{sid}|{arm}'])
nextpath=dict(selected='保留缓存最大J与低置信度回退；先补充训练seed内可修复失败状态，再做修复候选对最大J的定向配对监督',
 implementation_status='下一轮建议，尚未实施',
 reason='15个独立训练状态只有1个可修复的最大J失败状态，直接修复vs最大J仅2对；关键验证UE34参考服务预测错误，辅助头oracle替换可恢复21/21候选质量覆盖',
 scope='先在现有41-43其它TTI做有上限的保存状态采样，另付教师/缓存搜索，按seed报告可修复失败覆盖；44-48不回流训练',
 learning='可修复失败状态和稀少服务UE均衡；优先合格修复vs最大J/高J坏候选，保留基线已通过状态的无害对照；验证每个训练seed的修复覆盖后才追加一轮小训练',
 retain='同根缓存/现有池2槽/212评分/连续前沿/2%内部窗/低置信回退；外部门槛不变',
 defer=['继续增大网络','针对44-48调阈值','扩大候选池作为主线','上游资源活动提议器','PPO','最终seed与原生闭环','默认替换'],
 cost='本轮缓存验证距80%成本线3.15ms；未实施新缓存优化，模型额外约2.12ms须计入后续成本')
write_json(p/'gate_results.json',dict(selection=sel,validation=val,development=dev,learned_independent_gain=False,
 online_ai_success=False,all_guarded_finals_equal_maxj_states=21,unguarded_changed_states=3,
 oracle_auxiliary_heads_passing_states=21,oracle_labels_posthoc_only=True,next_path=nextpath))
write_json(p/'final_audit.json',dict(passed=True,tests=83,formal_decisions=378,formal_independent_finals=126,
 new_teacher_independent_finals=21,engineering_independent_finals=12,total_independent_finals=159,
 training_unique_states=15,training_candidates=3165,training_pairs=29569,gradient_steps=1440,
 models_frozen_before_validation=True,source_model_registry_physics_valid=True,
 previous_cached_paid_values_equal_states=21,all_terminal_query_archive_equivalent=True,
 all_guarded_actions_values_equal_maxj_states=21,posthoc_native_scores=0,training_native_scores=0,
 all_validation_development_gates_passed=False,selected_development_passed=True,learned_independent_gain=False,
 final_seeds_read=False,native_trajectories=0))
lines=['逐UE服务损失＋状态内配对排序＋低置信度回退最大J',
'日期：2026-11-03（Asia/Shanghai）',
'实验：ue_pair_20261103_v1；正式执行完成，冻结搜索/训练源码无事后修补。','',
'结论',
'保留缓存版最大J与回退机制，当前配对模型不晋级。所有带回退终态在21状态的动作/物理值均与cached_maxj完全相同，质量/覆盖没有新增损失，但没有AI独立收益。开发9/9质量和覆盖、平均成本通过；验证仍1/12覆盖失败，cached_maxj均时85.1397ms高于81.9903ms门槛3.1494ms。',
'配对模型带回退完整均时验证87.2553ms、开发96.8021ms，比cached_maxj多2.1156/2.1366ms。关闭回退仍不修复关键失败状态，说明不是单纯门槛过严。下一步先补充训练seed内可修复失败状态，针对“修复候选vs最大J”监督，而非扩大网络或追调当前验证阈值。','',
'1. 冻结实验与边界',
'3小区/36UE/17PRG；canonical native BLER、因果缺口加权J、原未剪枝rawPFQ90%出队floor。保留现有池定向2槽、quota-exp前沿和同根候选/代理缓存，212整动作评分含初始PFQ，56根上限，16种子、8批大小。条件臂实际均212评分、2448唯一MMSE。',
'条件仍为另付完整物理DPLS的每UE资源数、精确最小层和小区×PRG活动；教师位置、J、服务值与有效集合不进入运行时特征/终态API。参考DPLS不是全局最优。无CTBS/headroom/PPO、无新原生轨迹或最终seed。',
'最后选择只在已评分且条件/floor可行的档案内进行，自身最大J的2%风险窗不变；外部每状态J≥99.5%full、无新未分配UE、full有效集合P5与固定最弱20%服务≥98%、seed等权完整均时≤80%full不变。风险窗及历史44–48均已暴露，结果是开发证据。',
'六臂：full；cached_maxj；aggregate_guard（上轮冻结聚合模型，只改无可信候选时回退）；ue_heuristic_guard（同逐UE损失形式的启发式＋回退）；ue_pair（关闭回退的诊断对照，不参与选型）；ue_pair_guard（主实验）。','',
'2. 特征、模型、配对与置信门',
'每UE20维共享上下文，包括因果资源/最小层/MCS/quota/queue/权重/活动/合法性及自身档案max/q90/mean/median/std/positive频率/bestJ服务；不使用UE编号嵌入。候选有7个单调服务损失基，另加预测参考服务缺口，共8个损失项。',
'共享20→24(tanh)→10网络输出8个非负权重、软有效服务概率、参考E/D；候选分数为逐UE损失的负加权均值。在固定档案和上下文中，提高候选服务不会使该分数变差。模型学习概率不直接变成教师服务掩码。',
'训练风险标签=10*max(.995-J/fullJ,0)+max(.98-P5比,0)+max(.98-固定弱20%比,0)。仅同状态、2%窗口内配对：风险差>.005作为严格偏好，差≤.0001作为平局，其间忽略；各最多1024对，优先包含最大J比较。严禁跨状态配对或候选随机拆分。辅助监督为0.2*(参考有效服务BCE+参考E/D均方误差)，full信息仅在离线标签中。',
'固定Adam lr=.01、weight_decay=.001、240epoch，配对logit为20倍分差。3次留seed拟合给单调零截距概率尺度，3次全部训练不同初始化形成运行时集成，共1440梯度步；没有epoch/网络规模/阈值搜索。模型npz22342B。',
'候选提议取窗口内三模型平均分最高者，J破平。只有三个成员相对同一最大J的校准胜率均≥.9且分差均为正，才替换；否则精确回退最大J。置信度是相对模型偏好，不是覆盖或质量保证。启发式使用相同损失基、均匀权重、自身positive频率/q90参考，训练集拟合其尺度并采用同.9门槛。','',
'3. 训练样本与先后顺序',
'搜索完全未改，终态不影响访问路径，故复用上轮已付费41–43的18个自身访问档案/教师标签。按完整不可变输入＋档案＋参考字节SHA去重，去掉各seed的raw/PFQ t4001重复，剩15状态、3165候选。训练没有新增native评分；继承成本由training_reuse.json指向上轮账本，不解释为免费教师。',
'共29569对，其中严格偏好14638、平局14931；按状态均匀加权。每折归一化仅用其训练seed的有效quota UE。模型和概率尺度冻结后先做两训练状态×六终态工程对照，再读44–45本轮验证、冻结选型，最后读46–48开发。44–48没有训练回流。',
'尺度alpha=42.963483，启发式alpha=60.425118。留seed校准使用其OOF分数，不是独立测试；三个初始化共享同一训练数据，不能当三份独立统计证据。','',
'4. 正式结果',
'验证：44–45，两来源，TTI4001/8001/18000，共12状态（10种档案）；开发：46–48 PFQ，TTI3001/4001/5000，共9状态。三次串行轮换，每状态耗时取中位数后seed等权。所有条件臂无新增未分配UE。','']
for label,rs in [('验证',val),('开发',dev)]:
 lines += [label,'方法 | 平均J对full% | 最差J对full% | 完整ms | 最差P5比 | 最差弱20%比 | J/覆盖失败 | 成本通过']
 for r in rs:
  lines.append(f"{r['arm']} | {float(r['mean_relative_j_pct']):+.6f} | {float(r['worst_relative_j_pct']):+.6f} | {float(r['mean_ms']):.4f} | {float(r['worst_positive_p5']):.6f} | {float(r['worst_positive_weak']):.6f} | {r['quality_failures']}/{r['coverage_failures']} | {r['execution_cost_pass']}")
 lines.append('')
lines += ['验证选型仍是cached_maxj；没有成本可行验证路线，开发未回选。带回退的三条终态臂均21/21动作/物理值与cached_maxj相同。开发所有方法通过质量/覆盖/成本门槛，不抵消验证仍失败，也不能将相同结果计作学习收益。',
'关闭回退共改变3状态：验证raw seed45 t8001，开发seed46 t3001、seed48 t5000；置信度分别约.500044/.500105/.511550。变化均仍满足质量/覆盖，提供小幅弱服务交换而非关键失败修复；开发1次>1B的J下降，平均J−3.2308B，弱20%最差比从.995626到1。主实验拒绝这3次低置信替换，其余18状态排序器本就把最大J列第一。',
'上轮聚合模型的guard对照在两个raw t18000低概率状态回退，避免了此前seed44 P5降0的改动；其它19状态最大J本身满足其置信线。回退在这些样本防止新增损害，但没有证明新增服务改善。','',
'5. 关键失败与训练覆盖诊断（事后只读）',
'raw_global_deficit:seed44:tti18000的129个窗口内候选有11个J/覆盖合格者；当前最大J比full高1.87484%，P5比.785229、固定弱20%比.806705。无回退/有回退模型均直接选择该最大J，未被门槛拦截。',
'窗口内好坏候选排序AUC=.076271，最好合格候选排第110；所有合格候选相对最大J的最高最小成员胜率仅.024830，分差最高仍−.085389。没有高置信合格候选；降低.9阈值不能把一个未被提议的候选变成首选。',
'UE34在full中E=1689.49B、E/D=.995575，但模型预测有效概率仅.02236、参考E/D=.06685，平均损失权重.29593；它在最大J终态中为0。相反full服务为0的UE5，模型有效概率.96550、参考E/D=.92453、权重1.83760。模型把“自身经常容易服务”误当成“参考中应保护”，至少在此状态出现明确错误。',
'仅在事后诊断中，将逐UE有效服务和参考E/D两个预测头替换为full标签，其余已学权重、候选、2%窗均不变：该状态选memo178，J/full=1.00146343、P5比1，并且21/21状态终态质量/覆盖通过。此oracle不参与决策、计时、训练或选型，不能称为因果AI效果；它说明该损失表示在已访问档案上存在表达空间，预测头错误值得优先排查。',
'训练15状态中，最大J有2个联合失败；只有raw seed41 t18000在窗口内可修复，仅2个合格候选。另一个raw seed43 t8001档案内没有联合合格候选。全部29569训练对中，直接“合格修复vs失败最大J”只有2对；可修复状态内虽有57对好坏候选比较，仍被普通排序比较淹没。',
'该唯一可修复状态即使训练集内也没学会：最好合格候选排42、好坏AUC=.493827，留seedAUC=.333333；当前模型和guard均选失败最大J。留出seed41时，训练其它seed没有可修复的最大J失败状态。因此更多候选数/配对数不等于覆盖了更多修复情形。数据覆盖与定向监督不足是据此提出的下一步假设，尚未证明补数据就能泛化。','',
'6. 完整耗时与工作量',
'本轮没有再改缓存引擎。带回退配对模型比cached_maxj增加验证2.1156ms、开发2.1366ms。首重复终态均时约2.32/2.37ms，其中逐UE特征.99/1.00ms、三成员计算及损失聚合1.04/1.07ms、选择.084/.086ms；完整中位数与这些首重复分项不可强行相加。',
'旧聚合guard约3.75/3.81ms终态，主要是3.39/3.45ms聚合特征；逐UE表示降低了终态计算，但整体并未改善质量或修复缓存版验证成本缺口。本轮full验证102.4878ms，80%线81.9903ms，缓存仍差3.1494ms，配对guard差5.2650ms。该差距与上轮2.40ms随参考完整计时变化，不将其当成确定性算法退化。',
'完整计时包含数值输入/PFQ、freshPHY/MMSE、条件压缩、搜索/缓存、终态特征/三模型计算/门控和上下文清理，共同轻量观测复制也计入；排除教师准备、训练/模型文件IO、一次性库/TBS准备、归档/独立审计。仍是教师条件下的执行耗时，不是独立在线AI端到端推理时延。',
'378次正式决策3920388评分，21次新离线教师1284536评分/2556.49ms，12工程解码2544评分，本轮合计5207468次native整动作评分。复用训练监督的既有成本另有来源清单，事后诊断新增native评分为0。',
'83项测试，126正式＋21教师＋12工程=159组独立NumPy终态核验；三重复动作/物理值/查询一致、所有终态臂搜索档案相同。缓存版21档案物理值与上轮相同，所有guard最终动作/物理值等于maxJ。冻结源码/模型/注册表SHA有效；未读311–315、未新增原生轨迹、默认算法未改。','',
'7. 下一步选择（尚未实施）',
'保留缓存版最大J为主基线，保留低置信度回退作为候选学习策略的保护。先在现有训练seed41–43其它TTI做规模受控的保存状态采样，计费完整教师与同预算缓存搜索；按seed统计“最大J失败且档案可修复”的覆盖。不能用44–48本轮失败样本补训练，也不读取最终seed。',
'覆盖达到预先冻结的跨训练seed要求后，再做一轮小训练：对可修复失败状态与少见的参考有效UE均衡，优先“合格修复vs最大J/高J坏候选”，保留最大J已通过状态的无害对照。核心验收是失败修复率、误改率及逐状态J/服务，而不是普通候选AUC或总pair数。',
'暂不扩大模型、不针对当前验证调门槛、不扩大候选池或训练上游资源/活动提议器。原212评分、2%内部窗、外部J/服务/成本门槛保持；AI独立收益和验证成本未过前，不进入新原生闭环或默认替换。','',
'证据：protocol.json、training_reuse.json、training_data.npz、training_pairs.csv、training_summary.json、training_repair_coverage.csv/json、models_frozen.json、ue_pair_model.npz、selection.json、两阶段原始计时/终态/排序/逐UE诊断、total_work_ledger.json、final_audit.json、gate_results.json、quality_cost.png/pdf。',
'本地来源：deliverables/ue_pair_20261103_v1；长期归档：Doc/evidence/2026-11-03/ue_pair。']
(p/'report.txt').write_text('\n'.join(lines)+'\n')

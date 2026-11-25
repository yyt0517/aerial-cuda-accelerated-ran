"""Figures, supporting aggregates and objective-change proposal for frozen runs."""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import OUT,OLD,NEW,ARMS

LABELS={'deficit':'DPLS','mlwdf_pf':'Original DA','da_three_branch':'Three-branch DA'}
COLORS={'deficit':'#2879b4','mlwdf_pf':'#dc8535','da_three_branch':'#28916d'}

def main():
    assert json.loads((OUT/'validation.json').read_text())['passed']
    trace=pd.read_csv(OUT/'trajectory_summary.csv');packet=pd.read_csv(OUT/'packet_delay_summary.csv')
    means=trace.groupby('arm').mean(numeric_only=True);pm=packet.groupby('arm').mean(numeric_only=True)
    ue=pd.read_csv(OUT/'per_ue_diagnosis.csv');base=ue[ue.arm.eq('deficit')].set_index(['seed','ue'])
    new=ue[ue.arm.eq('da_three_branch')].set_index(['seed','ue'])
    paired=new.add_prefix('new_').join(base.add_prefix('base_'))
    paired['delay_delta_ms']=paired.new_success_delay_mean_ms-paired.base_success_delay_mean_ms
    paired['queue_delta_kb']=paired.new_queue_mean_kb-paired.base_queue_mean_kb
    paired['base_expiry_group']=pd.cut(paired.base_expiry_pct,[-1,5,50,101],labels=['<5%','5-50%','>=50%'])
    paired.to_csv(OUT/'per_ue_paired.csv')
    groups=[]
    for group,z in paired.groupby('base_expiry_group',observed=True):
        groups.append(dict(group=group,ue_seed_pairs=len(z),additional_cluster_queue_mb=z.queue_delta_kb.sum()/3/1000,
            mean_ue_delay_dpls_ms=z.base_success_delay_mean_ms.mean(),mean_ue_delay_new_ms=z.new_success_delay_mean_ms.mean(),
            success_packet_weighted_delay_dpls_ms=np.average(z.base_success_delay_mean_ms,weights=z.base_payload_mbps),
            success_packet_weighted_delay_new_ms=np.average(z.new_success_delay_mean_ms,weights=z.new_payload_mbps),
            expiry_dpls_pct=z.base_expiry_pct.mean(),expiry_new_pct=z.new_expiry_pct.mean()))
    groups=pd.DataFrame(groups);groups.to_csv(OUT/'ue_group_effects.csv',index=False)
    x=trace.merge(packet,on=['arm','seed'])
    x['queue_area_implied_outcome_ms']=x.queue_packets_mean/(x.offered_mbps*1e6/24000)*1000
    x[['arm','seed','queue_area_implied_outcome_ms','mean_outcome_delay_ms']].to_csv(OUT/'queue_area_delay_check.csv',index=False)
    time=pd.read_csv(OUT/'queue_time_series.csv');time['period']=pd.cut(time.time_s,[0,2,4,6,8,10,12,14])
    time.groupby(['arm','period'],observed=True)[['queue_mb','hol_mean_ms']].mean().to_csv(OUT/'queue_phase_summary.csv')
    transfer=pd.read_csv(OUT/'saved_service_transfers.csv');transfers=[]
    for direction,mask in [('gains',transfer.service_delta>0),('losses',transfer.service_delta<0)]:
        z=transfer[mask];w=z.service_delta.abs()
        transfers.append(dict(direction=direction,expected_bytes_per_state=w.sum()/9,hol_ms=np.average(z.hol_ms,weights=w),
            ema=np.average(z.ema,weights=w),da_weight=np.average(z.da_weight,weights=w)))
    pd.DataFrame(transfers).to_csv(OUT/'service_transfer_summary.csv',index=False)
    rerank=pd.read_csv(OUT/'saved_candidate_reweight.csv')
    rr=rerank.groupby('objective').agg(dpls_selected_pct=('chosen',lambda x:100*(x=='dpls').mean()),
        expected_service_vs_dpls_pct=('expected_service_vs_dpls_pct','mean'),weight_ratio_mean=('weight_ratio','mean'))
    rr.to_csv(OUT/'candidate_reweight_summary.csv')
    guards=pd.read_csv(OUT/'same_state_service_guards.csv').groupby('min_expected_service_ratio').mean(numeric_only=True)
    guards.to_csv(OUT/'service_guard_summary.csv')
    matched=pd.read_csv(OUT/'matched_packets.csv')
    fig,ax=plt.subplots(2,2,figsize=(12.2,8.2),layout='constrained')
    for arm in ARMS:
        s=time[time.arm.eq(arm)]
        for _,z in s.groupby('seed'):ax[0,0].plot(z.time_s,z.queue_mb,color=COLORS[arm],alpha=.18,lw=.8)
        z=s.groupby('time_s').mean(numeric_only=True)
        ax[0,0].plot(z.index,z.queue_mb,color=COLORS[arm],label=LABELS[arm],lw=2)
        ax[0,1].plot(z.index,z.hol_ge180ms_pct,color=COLORS[arm],label=LABELS[arm],lw=2)
    for a in ax[0]:
        a.axvline(2,color='gray',ls='--',lw=1);a.axvline(4,color='gray',ls=':',lw=1)
        a.set_xlabel('Simulation time (s); policy starts at 2 s, measurement at 4 s')
    ax[0,0].set_title('A  Persistent backlog separates after the common warmup')
    ax[0,0].set_ylabel('Cluster queue (MB)');ax[0,0].legend(fontsize=9)
    ax[0,1].set_title('B  Near-deadline queues remain prevalent')
    ax[0,1].set_ylabel('Active UE-times with HOL >= 180 ms (%)')
    cdf=pd.read_csv(OUT/'delay_cdf.csv')
    for arm in ARMS:
        z=cdf[cdf.arm.eq(arm)].groupby('delay_ms').success_cdf.mean()
        ax[1,0].plot(z.index,100*z,color=COLORS[arm],label=LABELS[arm],lw=2)
    ax[1,0].set_title('C  Most of the lost advantage is at short delays')
    ax[1,0].set_xlabel('Successful-packet delay (ms)');ax[1,0].set_ylabel('CDF (%)');ax[1,0].legend(fontsize=9)
    for group,color in [('<5%','#2879b4'),('5-50%','#dc8535'),('>=50%','#b74363')]:
        z=paired[paired.base_expiry_group.eq(group)]
        ax[1,1].scatter(z.base_success_delay_mean_ms,z.new_success_delay_mean_ms,s=19,color=color,alpha=.65,label='DPLS expiry '+group)
    ax[1,1].plot([0,200],[0,200],ls='--',color='gray');ax[1,1].set_xlim(-3,203);ax[1,1].set_ylim(-3,203)
    ax[1,1].set_title('D  Low-expiry users acquire additional waiting')
    ax[1,1].set_xlabel('DPLS mean successful delay per UE (ms)')
    ax[1,1].set_ylabel('Three-branch DA mean successful delay per UE (ms)');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Seven-cell Boundary08: delay mechanism / seeds 41, 42, 43\nRetrospective analysis of frozen trajectories; no modified-objective rollout',fontsize=13)
    fig.savefig(OUT/'delay_mechanism.png',dpi=180);fig.savefig(OUT/'delay_mechanism.pdf');plt.close(fig)
    b=means.loc['deficit'];n=means.loc['da_three_branch']
    m=matched[matched.arm.eq('da_three_branch')]
    lines=[
        '七小区DA目标、队列演化与时延诊断（2026-11-06）','='*50,'',
        '结论：额外时延主要来自原本能及时服务的用户积压增多。DA仍保留DPLS候选，但当前age/EMA目标会用年轻队列的及时服务换取老队列的加权服务。三分支改善了搜索终点，未改变这种取舍。',
        '本次只读取冻结的9条闭环轨迹（DPLS、原DA、三分支DA，各seeds41/42/43），不修改原策略、不重跑仿真。以下跨seed统计均按seed等权，用户分组另有说明。','',
        '1. 实际权重与关键差别',
        '原目标 J(A)=sum_u w_u*S_u(A)，S_u=min(Q_u,TBS_u(A))*(1-BLER_u(A))。',
        'DPLS：b_u=clip(2400/max(EMA_u,1e-9),1,4)，再按均值归一化。',
        'DA：w_u正比于max(1,HOL年龄[TTI])/max(EMA_u,1e-9)，再按均值归一化。',
        'DA没有保留DPLS的基础服务权重，也没有保留1–4截断。年轻队列获得的年龄因子很小；0.5ms与约200ms的队首，在相同EMA下可相差约400倍。',
        '所有UE共用a=-log(0.01)/400。该a在均值归一化时完全抵消，单独调整共同a不会改变权重或选择。',
        'EMA更新为0.995*EMA+0.005*本时隙成功字节；时间常数约99.75ms、半衰期约69.14ms。历史成功服务倒数能强化低服务用户的优先级，但数据不支持把它视为这次唯一或全局主导原因。',
        f'三分支DA活跃UE权重P5均值{n.weight_p05:.4f}；DPLS为{b.weight_p05:.4f}。全时域逐TTI的log(DA权重)与log(HOL年龄)相关系数均值{n.da_logweight_logage_corr_mean:.3f}，与-log(EMA)为{n.da_logweight_loginverseema_corr_mean:.3f}；这些相关性是描述性诊断。',
        '在180–200ms的临期队列子集中，年龄只能再放大约11%，因而该子集内EMA更容易决定相对权重；不能据此推断全部用户的权重都由EMA主导。','',
        '2. 队列与包时延证据','指标 | DPLS | 原DA | 三分支DA',
    ]
    for col,label in [('queue_mean_mb','平均积压MB'),('queue_packets_mean','平均在队包数'),('hol_mean_ms','活跃UE平均HOL ms'),('hol_ge180ms_pct','活跃UE时隙HOL>=180ms %'),('occupancy_pct','PRG占用率%'),('goodput_bytes_per_prg','成功字节/已占用PRG')]:
        lines.append(label+' | '+' | '.join(f'{means.loc[a,col]:.4f}' for a in ARMS))
    for col,label in [('mean_success_delay_ms','成功包平均时延ms'),('mean_outcome_delay_ms','全部包结局平均时延ms'),('success_le10ms_pct','成功包10ms内交付比例%'),('success_le50ms_pct','成功包50ms内交付比例%')]:
        lines.append(label+' | '+' | '.join(f'{pm.loc[a,col]:.4f}' for a in ARMS))
    lines += [
        f'三分支额外积压{n.queue_mean_mb-b.queue_mean_mb:.4f}MB（+{100*(n.queue_mean_mb/b.queue_mean_mb-1):.2f}%），约多{n.queue_packets_mean-b.queue_packets_mean:.1f}个在队包。',
        '共同PFQ预热结束后，DPLS迅速消化积压；正式测量后段DPLS仍约45MB、三分支DA约58MB。差异不是只发生在预热切换瞬间。',
        '队列面积除以包到达率，三分支相对DPLS多约31.83ms；全部包结局平均时延实际增加31.86ms。有限窗口及离散TTI下不要求二者精确相等。',
        '输入业务约3.224Gbps；三分支计划出队约2.605Gbps，剩余约0.619Gbps通过过期移除。队列维持平台不意味着业务被及时服务，TTL剔除参与了流量平衡。',
        f'两种方案都成功交付的相同包占新方案成功包的{m.both_fraction_of_new_success_pct.min():.2f}%–{m.both_fraction_of_new_success_pct.max():.2f}%；其平均时延差依次为'+', '.join(f'{v:+.2f}ms' for v in m.same_packet_delay_delta_ms)+'。因此不能用“只是救回更多慢包”解释这次均值差。',
        '按DPLS下每UE过期率进行事后描述性分组，252个UE-seed组合中有118个低于5%。这一组用户的平均成功包时延从18.22ms变为86.88ms；其贡献的额外全网积压为12.844MB，几乎覆盖全部增加。该分组未用于调度，也不是独立随机试验。',
        '原本过期率5–50%及>=50%的用户，平均成功包时延分别仍约194.5ms与198.7ms，改善不足0.1ms（UE等权口径）。较低平均时延的用户被拖慢，临期严重用户的时延没有同等幅度改善。',
        '同状态9个测量期保存点中，相对DPLS获增服务的用户，其增量加权HOL约160.67ms；被减服务用户约68.32ms。预期服务增量约45.93kB/状态、减量54.87kB/状态，存在净损失。',
        '全60,000个测量状态中，按DA择优的无权预期成功服务量相对DPLS候选平均低约6–8%。闭环中PRG占用率增加约4.76个百分点、每PRG成功字节减少约6.01%。这与更高资源占用及更低服务效率一致，但不能单凭关联把全部时延差归因于干扰。','',
        '3. 为什么保留搜索路径仍不能保留低时延',
        'DPLS候选在场，并且其服务更高，依旧可能因年轻UE年龄小而在DA目标上输掉。选择后年轻队列积压、HOL增大，直到足以竞争服务；EMA和后续链路状态也沿新轨迹演化。',
        '在均衡到达且持续积压时，队长与HOL近似同向。简单把HOL改成队长、继续增大HOL系数或使用更陡的紧急度函数，都可能保留这种等积压增大才服务的机制；不能直接当成平均时延修复。',
        '目标中的预期成功字节也不是完整包成功交付数：没有利用FIFO内每包剩余字节、已发生CRC损坏与完成状态。这是进一步改进的方向，但本次证据首先指向年轻队列的普遍推迟，而非CRC成为主因。','',
        '4. 建议先验证的最小修改',
        '定义 h_u=clip(HOL_age_u/TTL,0,1)，b_u=clip(2400/max(EMA_u,epsilon),1,4)。',
        '新权重：w_new,u = b_u*(1+lambda*h_u)，然后按均值归一化。',
        '新搜索目标：J_new(A)=sum_u w_new,u*S_u(A)。基础1使年轻队列保持服务权重；有界紧急度只作温和加成。lambda=0恢复DPLS目标，但多起点候选选择不必复现DPLS独立轨迹的同一动作。',
        '建议第一轮小范围消融lambda={0,0.05,0.1,0.25}，不是继续增大a。lambda=0.1时所有UE的权重比至多4.4；该值只是待闭环验证的起点，不是已证实最优参数。',
        '最终候选增加成功服务量保护：sum S_u(A) >= beta*sum S_u(A_DPLS)，A_DPLS必须在同一当前状态重新计算。保留原始PFQ的90%计划出队约束。',
        'beta=1保证同状态总预期成功服务不低于DPLS；beta=0.98允许最多2%折让。此保护不能保证长期时延不退化；同样的总服务仍可能分配给不同用户。',
        '第一版可把保护线放在最终候选过滤层，保证DPLS始终可选；新权重应同时用于搜索和最终排序。以后可保留搜索路径上满足保护线的中间解；若把新保护线放入每轮搜索可行性，需同步更新检查并重新核验剪枝。','',
        '已有候选离线重排（9个保存点，不是新目标搜索或闭环结果）：',
    ]
    for name in ['da','capped_pf_times_age','bounded_hol_0.05','bounded_hol_0.1','bounded_hol_0.25','bounded_queue_hol_0.25']:
        r=rr.loc[name];lines.append(f'{name}：DPLS候选选择率{r.dpls_selected_pct:.2f}%，无权预期成功服务相对DPLS均值{r.expected_service_vs_dpls_pct:+.3f}%。')
    lines += ['仅给PF项截断没有改变这9点选择；恢复基础服务项才明显改变排序。队长加成没有在这9点提供同样的保护。',
        '保持原DA排序，仅加服务保护线，对全部60,000个已记录测量状态进行离线重排：']
    for beta in [.0,.95,.98,1.]:
        r=guards.loc[beta];lines.append(f'beta={beta:.2f}：预期成功服务相对DPLS均值{r.expected_service_vs_dpls_pct:+.3f}%，DPLS选择率{r.dpls_selected_pct:.3f}%。')
    lines += ['这些重排使用原轨迹状态，不能把它们当作新策略将会达到的吞吐/时延。特别是beta=1几乎总选DPLS，说明当前三个终点的DA提升主要通过服务取舍获得，也说明只改最终择优未必能找到更好的折中解。','',
        '5. 若要进一步直接对齐包时延',
        '以在队包数的时间积分作为所有包停留时间成本，同时奖励完整成功交付并惩罚过期和失败，避免只缩短队列却制造丢包。',
        '一个待研究的有限步目标为：max E[sum_t (alpha*完整成功交付字节 - beta_q*在队包数 - gamma*过期字节 - eta*失败完成字节)]，并保留必要的有界公平性与服务保护。',
        '包完成项可从当前FIFO剩余字节、已损坏标记及本次TB成功概率构建；不能用未来实际CRC、未来到达或未来交付标签给在线动作评分。',
        '队列面积对应所有包停留时间，不等于条件成功包平均时延。完整包吞吐、过期、失败和成功包时延仍需联合评估。',
        '这类包级/多步目标不再是固定正权重乘预期字节，不能直接套用当前剪枝证明；需要新的上界与一致性审计。','',
        '核验与边界',
        '9条轨迹的队列流量恒等式通过；用FIFO终态回溯得到的HOL、按在线float32反馈运算顺序重建的EMA及权重，与30个保存状态一致。该回溯仅用于离线诊断，在线策略不得读取未来终态。',
        '相同包比较逐一匹配packet_id与UE，所有测量cohort已完成随访。统计独立单位仍只有3个seed，252个UE-seed组合与60,000时隙不是额外独立seed。',
        '同状态重排仅用已保存候选，不代表全空间最优，也没有验证修改目标后的闭环收益。原实验文件保持不变。',
        '主要文件：delay_mechanism.png/pdf；trajectory_summary.csv；matched_packets.csv；ue_group_effects.csv；service_guard_summary.csv；candidate_reweight_summary.csv。',
    ]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'report_source.py').write_bytes(Path(__file__).read_bytes())
    # Verify previously delivered artifacts and execution source hashes remain frozen.
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for root in [OLD,NEW]:
        for path,h in json.loads((root/'delivery_manifest.json').read_text())['artifact_sha256'].items():assert sha(root/path)==h,path
    for path,h in json.loads((NEW/'campaign/protocol.json').read_text())['source_sha256'].items():assert sha(NEW.parents[1]/path)==h,path
    files=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['delivery_manifest.json','delay_diagnosis.zip']]
    manifest=dict(completed=True,changed_objective_rollouts=0,baseline_artifacts_unchanged=True,
        original_three_branch_sources_unchanged=True,native_trace_runs=9,saved_states_reconstructed=30,
        same_state_guard_measurement_decisions=60000,reweight_saved_measurement_states=9,
        artifact_sha256={p.name:sha(p) for p in sorted(files)})
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    files.append(OUT/'delivery_manifest.json')
    with zipfile.ZipFile(OUT/'delay_diagnosis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.name)
    with zipfile.ZipFile(OUT/'delay_diagnosis.zip') as z:assert z.testzip() is None
    print(OUT/'report.txt',flush=True)

if __name__=='__main__':main()

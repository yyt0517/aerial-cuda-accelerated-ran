import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
AN=ROOT/'deliverables/stage2_generalization_20261106/analysis'
d=pd.read_csv(AN/'per_seed.csv');val=json.loads((AN/'validation.json').read_text());assert val['complete'] and val['passed'] and len(d)==60
m=d.groupby(['scenario','arm']).mean(numeric_only=True)
a=m.xs('mlwdf_pf',level='arm');p=m.xs('pfq',level='arm');z=m.xs('deficit',level='arm')
u=pd.read_csv(AN/'per_ue_all.csv');assert u[u.arm.ne('pfq')].expiry_pct.eq(0).all()
assert d[d.arm.ne('pfq')].expiry_pct.eq(0).all()
paired=pd.read_csv(AN/'paired_differences.csv')
q=paired[(paired.baseline=='pfq') & paired.metric.isin(['payload_mbps','expiry_pct','delay_mean_ms'])]
assert q[q.metric.eq('payload_mbps')].positive.eq(5).all()
assert q[q.metric.ne('payload_mbps')].negative.eq(5).all()
gain=100*(a.payload_mbps/p.payload_mbps-1);against_dpls=100*(a.payload_mbps/z.payload_mbps-1)
assert against_dpls.abs().max()<.1
max_dt=(a.delay_mean_ms-z.delay_mean_ms).abs().max();assert max_dt<.017
assert m.loc[pd.IndexSlice[:,['deficit','mlwdf_pf']], 'queue_mean_mb'].max()<.02
g=pd.read_csv(AN/'groups_mean.csv').set_index(['scenario','arm','dimension','group'])
get=lambda s,arm,dim,group,col:float(g.loc[(s,arm,dim,group),col])
risk3=get('G3','pfq','cell_load','heavy_cell','expiry_pct');risk4=get('G4','pfq','cell_load','heavy_cell','expiry_pct')
risklarge=get('G4','pfq','heavy_cell_traffic','large_low','expiry_pct');delaylarge=get('G4','mlwdf_pf','heavy_cell_traffic','large_low','delay_mean_ms');successlarge=get('G4','mlwdf_pf','heavy_cell_traffic','large_low','success_pct')
assert get('G3','pfq','cell_load','light_cells','expiry_pct')==0 and get('G4','pfq','cell_load','light_cells','expiry_pct')==0
# Monotonic deterioration across topology is not claimed: this paired seed is a counterexample.
p53=d[(d.arm=='pfq') & (d.seed==53)].set_index('scenario');assert p53.loc['G3','expiry_pct']<p53.loc['G1','expiry_pct']
gainrange=f'{gain.min():.2f}%–{gain.max():.2f}%'
words={
'core_paragraphs':[
 f'相对PFQ，协同收益在四类场景中保持稳定。DA-PLS的完整包吞吐提高{gainrange}，成功包平均时延由{p.delay_mean_ms.min():.3f}–{p.delay_mean_ms.max():.3f} ms降至{a.delay_mean_ms.min():.3f}–{a.delay_mean_ms.max():.3f} ms。20个场景—seed配对点均体现吞吐提高、平均时延和过期率下降；在测量期到达包集合中，DA-PLS的20次运行、每次36个UE均未观测到过期，按包数和按字节计算的过期率都为0。',
 f'与DPLS的差异很小：两者均无过期，四场景吞吐均值变化仅为{against_dpls.min():+.2f}%至{against_dpls.max():+.2f}%，均时延差小于0.017 ms，平均簇队列均低于0.02 MB。本组配置未显示DA-PLS对DPLS的稳定额外优势；时延权重的价值仍应结合9.4节Boundary08的持续积压条件理解。'
],
'group_paragraphs':[
 f'PFQ的风险主要集中在重载小区：G3、G4分别为{risk3:.3f}%和{risk4:.3f}%，轻载小区均为0；G4重载区大包为{risklarge:.3f}%。DA-PLS各分组均无过期，重载区大包均时延{delaylarge:.3f} ms。个别拓扑下负载重分配也可改善PFQ：seed 53从G1的{p53.loc["G1","expiry_pct"]:.3f}%降至G3的{p53.loc["G3","expiry_pct"]:.3f}%，因此场景均值需结合逐seed结果解读。',
 f'无过期不代表无损交付：G4中DA-PLS重载区大包的完整成功率为{successlarge:.3f}%，仍受CRC失败影响。不同包型由不同UE承载，组间差异不作单一包长的因果解释。'
],
'validation_paragraph':f'60次运行均通过动作与字节守恒核验，20组场景—seed配对的拓扑、逐包到达和共同预热一致。{val["packets"]:,}条测量期到达包记录全部终止；搜索类运行首、中、末保存的{val["saved_states_verified"]}个输入经独立完整枚举对照，{val["exact_search_rounds"]}轮检查全部通过。逐seed、逐UE、逐小区和分组数据随报告提供。',
'summary_paragraph':f'四类复杂场景进一步表明，混合布点、大小包混合及2∶1∶1小区负载下，DA-PLS相对PFQ的完整包吞吐提高{gainrange}，测量期各UE均无过期。DPLS在这些配置下也已无过期，二者核心指标基本接近；本组结果验证了适应性，同时明确了时延紧迫度额外收益取决于排队压力。',
'conclusion_paragraph':f'新增四类场景、seed 51–55共60次正式仿真中，DA-PLS相对PFQ的完整包吞吐提高{gainrange}，平均成功包时延为{a.delay_mean_ms.min():.3f}–{a.delay_mean_ms.max():.3f} ms，测量期全部UE未观测到过期。DPLS同样无过期，四场景吞吐均值与DA-PLS相差不足0.1%。复杂业务下的协同收益保持稳定，但时延权重并非在所有配置中都带来额外提升；CRC导致的未完整交付也仍然存在。',
'abstract_paragraph':f'新增混合布点、大小包混合及小区负载不平衡的四类场景（seed 51–55，60次仿真）中，DA-PLS完整包吞吐较PFQ提高{gainrange}，测量期各UE均未观测到过期；DPLS同样无过期，两者核心指标基本接近。这表明协同收益能延续至本组复杂业务配置，而时延紧迫度的额外收益与排队压力有关。'
}
(OUT/'source/result_narrative.json').write_text(json.dumps(words,ensure_ascii=False,indent=2)+'\n')
(OUT/'核验/正文数值核验.json').write_text(json.dumps(dict(passed=True,runs=60,gain_vs_pfq_pct=gain.to_dict(),gain_vs_dpls_pct=against_dpls.to_dict(),max_absolute_mean_delay_difference_vs_dpls_ms=max_dt,da_all_720_ue_run_expiry_values_zero=True,dpls_all_720_ue_run_expiry_values_zero=True,all_20_paired_comparisons_improved_payload_delay_and_expiry_vs_pfq=True),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(words,ensure_ascii=False,indent=2))

"""Auditable spreadsheet and compact evidence bundle; raw runs stay alongside."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,'/tmp/stage2_report_python')
import pandas as pd
import xlsxwriter

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(args):
    root=Path(args.root).resolve();tag=getattr(args,'tag',None) or ('核心对比' if args.core else '完整结果')
    target=root/f'仿真结果汇总_{tag}.xlsx'
    files=[('逐seed指标','analysis/per_seed.csv'),('等权均值与区间','analysis/aggregate.csv'),
      ('对PFQ配对差值','analysis/paired.csv'),('DA对缺口权重差值','analysis/weight_paired.csv'),
      ('DA对各权重配对差值','analysis/ablation_paired.csv'),('共同预热核验','analysis/prefix_validation.csv'),
      ('连续计算成本','analysis/continuous_cost.csv'),('串行成本均值','audit/timing_summary.csv'),
      ('串行成本逐seed','audit/timing_per_seed.csv'),('串行计时原始重复','audit/timing_raw.csv'),
      ('阴影采样验证','e0/shadow_statistics.csv'),('传统基线配对差值','analysis/baseline_paired.csv'),
      ('小区逐seed指标','analysis/cell_per_seed.csv'),('小区等权均值','analysis/cell_aggregate.csv')]
    with xlsxwriter.Workbook(target,{'strings_to_formulas':False,'strings_to_urls':False,'nan_inf_to_errors':True}) as book:
        title=book.add_format({'bold':True,'font_color':'#FFFFFF','bg_color':'#165D86','text_wrap':True})
        wrap=book.add_format({'text_wrap':True,'valign':'top'})
        readme=book.add_worksheet('阅读说明');readme.set_column(0,0,24);readme.set_column(1,1,105)
        notes=[('交付范围',tag+'；仅使用已完成并通过原生验证的轨迹。核心版的扩展实验尚未齐全。' if args.core else '完整220点；质量与等价性验证全部通过后生成。'),
          ('统计单位','先计算每个seed指标，再等权平均。P95及最差UE也先在单seed内计算。'),
          ('两类时延','成功包与outcome均报告平均值和P95。outcome采用同一成熟到达集合，包含全部MAC完成包（含CRC损坏）与自然过期包；不等同于成功交付时延。'),
          ('配对区间','同seed算法减PFQ；Student t 95%区间，未做多重比较校正。'),
          ('种子','41–50共10个；44–50未参与本次权重选择，但不是项目历史完全未见种子。'),
          ('实验时序','4000共同PFQ预热＋4000各策略稳定＋20000测量＋400随访；TTI为0.5 ms。'),
          ('整包吞吐','测量窗口到达、所有分段成功的整包payload/10 s；与分段goodput区分。'),
          ('零过期','不等于零丢包；other_nondelivery_pct保留CRC损坏等其它未交付。'),
          ('串行成本','保存状态同输入、固定单核、轮换顺序；不含原生PFQ计算、跨进程通信和GPU仿真。'),
          ('连续成本','并发轨迹中的每TTI描述统计；含观测、HOL及三个保存点的归档I/O。P95为各seed P95的均值，max为全部观测最大值。'),
          ('原始记录',str(root/'campaign')),('字段','scene: boundary/uniform；rate: 包/UE/TTI；arm: pfq、deficit、mlwdf_pf等。')]
        for i,(a,b) in enumerate(notes):readme.write(i,0,a,title);readme.write(i,1,b,wrap);readme.set_row(i,34)
        for name,rel in files:
            path=root/rel
            if not path.exists() or path.stat().st_size<=1:continue
            frame=pd.read_csv(path)
            if args.core and 'scene' in frame and 'arm' in frame:
                frame=frame[(frame.rate<=1)&frame.arm.isin(['pfq','deficit','mlwdf_pf'])]
            if args.core and 'baseline' in frame:
                frame=frame[frame.baseline.isin(['pfq','deficit'])]
            if args.core and 'rate' in frame and 'scene' not in frame and name.startswith('串行'):
                frame=frame[frame.rate==.8]
            sheet=book.add_worksheet(name);sheet.freeze_panes(1,0);sheet.set_row(0,30)
            for j,c in enumerate(frame.columns):sheet.write(0,j,c,title);sheet.set_column(j,j,min(42,max(14,len(c)+2)))
            for i,row in enumerate(frame.itertuples(index=False,name=None),1):
                for j,v in enumerate(row):
                    if pd.isna(v):sheet.write_blank(i,j,None)
                    elif isinstance(v,(str,bool,int,float)):sheet.write(i,j,v)
                    else:sheet.write(i,j,str(v))
            if len(frame):sheet.autofilter(0,0,len(frame),len(frame.columns)-1)
    selected=[target,root/'campaign/protocol.json',root/'analysis/completion.json',root/'audit/completion.json',root/'tests.log',root/'early_native_prefix_check.json',root/'baseline_feasibility.json',root/'source_documents/manifest.json']
    selected.extend(root/rel for _,rel in files)
    selected.extend(root.glob('figures/*.pdf'))
    selected.extend(root.glob('audit/*tti*.json'))
    selected.extend(root.glob('audit/concurrent_timing_archive/*.json'))
    selected.extend(root.glob('analysis/*/input_manifest.json'))
    selected.extend(root.glob('analysis/*/validation.json'))
    selected.extend(root.glob('analysis/*/physical_validation.json'))
    selected.extend(root.glob('analysis/*/source_validation.json'))
    selected.extend(root.glob('analysis/*/per_ue.csv'))
    selected.extend(root.glob('analysis/*/per_cell.csv'))
    selected.extend(root.glob('analysis/*/revision_validation.json'))
    selected.extend(root.glob('analysis/*/*_hist.csv'))
    selected.extend(root.glob('指标补充核验.json'))
    selected.extend(root.glob('小区关联核验.json'))
    selected.extend(root.glob('e0/*.cpp'))
    selected.extend(root.glob('environment_*.json'))
    selected.extend(root.glob('execution_priority.json'))
    selected=[p for p in selected if p.exists()]
    code=sorted(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('README.rst')]
    repo=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'campaign/protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        assert sha(repo/name)==digest, f'Frozen execution source changed: {name}'
    algorithm_names=set(protocol['sources'])|{
        'training/stageb_lns/bounded_kernel.cpp','training/stageb_lns/persistent_kernel.cpp',
        'training/stageb_lns/build.py','training/stageb_lns/build_bounded.py','training/stageb_lns/build_persistent.py',
        'training/stageb_lns/physics_kernel.cpp','training/stageb_lns/_physics.so','training/stageb_lns/dpls_operations.py',
        'training/stageb_lns/bounded_campaign.py','training/stageb_lns/ranked_dpls.py',
        'training/stageb_lns/archive.py','training/stageb_ppo/native.py','training/stageb_ppo/reward.py'}
    algorithm={name:sha(repo/name) for name in sorted(algorithm_names)}
    manifest={'scope':tag,'files':{str(p.relative_to(root)):sha(p) for p in selected},
      'reproduction_sources':{p.name:sha(p) for p in code},
      'algorithm_snapshot':algorithm,
      'snapshot_scope':'Selected algorithm and validation modules; a full project checkout, frozen simulator and CUDA environment remain required.',
      'raw_runs_excluded_from_zip':True,'raw_directory':str(root/'campaign')}
    m=root/f'交付文件校验_{tag}.json';m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(root/f'仿真证据包_{tag}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in selected+[m]:z.write(p,str(p.relative_to(root)))
        for p in code:z.write(p,'reproduction/'+p.name)
        for name in algorithm:z.write(repo/name,'algorithm_snapshot/'+name)
    print(target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--core',action='store_true');p.add_argument('--tag');main(p.parse_args())

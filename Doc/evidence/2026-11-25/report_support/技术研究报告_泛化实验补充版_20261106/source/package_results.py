"""Package compact reproducible evidence; leave raw simulation logs in deliverables."""
import json,shutil,hashlib,csv
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
EXP=ROOT/'deliverables/stage2_generalization_20261106';AN=EXP/'analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    validation=json.loads((AN/'validation.json').read_text());assert validation['passed'] and validation['complete'] and validation['runs']==60
    pairs=[('per_seed.csv','60次实验_逐seed指标.csv'),('aggregate.csv','四场景_均值与范围.csv'),('paired_differences.csv','DA-PLS相对基线_配对差值.csv'),('groups_per_seed.csv','重点分组_逐seed指标.csv'),('groups_mean.csv','重点分组_均值.csv'),('per_ue_all.csv','逐UE指标.csv'),('per_cell_all.csv','逐小区指标.csv')]
    for src,dest in pairs:shutil.copy2(AN/src,OUT/'data'/dest)
    plans=[]
    for p in sorted((EXP/'plans').glob('*.csv')):
        d=pd.read_csv(p);d.insert(0,'scenario',p.stem.split('_')[0]);d.insert(1,'seed',int(p.stem.split('_s')[1]));plans.append(d)
    pd.concat(plans,ignore_index=True).to_csv(OUT/'data/20组场景_逐UE配置.csv',index=False,encoding='utf-8-sig')
    for src,name in [(EXP/'protocol.json','正式实验协议.json'),(EXP/'native/build_manifest.json','场景扩展构建记录.json'),(AN/'validation.json','60次实验核验.json'),(EXP/'smoke_analysis/validation.json','短程试跑核验.json'),(EXP/'parity_disabled_seed41/parity.json','原场景一致性核验.json')]:shutil.copy2(src,OUT/'核验'/name)
    for src,name in [(EXP/'completion.json','正式运行完成记录.json'),(EXP/'parallel_tail/execution_plan.json','并行执行计划.json'),(EXP/'parallel_tail/completion.json','并行运行完成记录.json')]:shutil.copy2(src,OUT/'核验'/name)
    each=OUT/'核验/逐次运行';each.mkdir(exist_ok=True)
    for folder in sorted(AN.glob('*')):
        if not folder.is_dir():continue
        for filename,label in [('validation.json','核验'),('input_manifest.json','输入文件校验')]:
            src=folder/filename
            if src.exists():shutil.copy2(src,each/f'{folder.name}_{label}.json')
    sources=OUT/'source/experiment';sources.mkdir(exist_ok=True)
    for p in (ROOT/'training/stage2_generalization').glob('*.py'):shutil.copy2(p,sources/p.name)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.zip','.pyc'] and not any(x in p.parts for x in ['预览','__pycache__']) and p.name!='交付文件校验.json']
    manifest={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    (OUT/'核验/交付文件校验.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=OUT/'泛化实验_报告与结果数据.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in files+[OUT/'核验/交付文件校验.json']:z.write(p,p.relative_to(OUT))
    print(archive,archive.stat().st_size)
if __name__=='__main__':main()

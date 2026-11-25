"""Finish analysis and staged documents as this single authorized campaign runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main(args):
    root=Path(args.root).resolve();campaign=root/'campaign'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/stage2_mpl')
    def run(module,*extra):
        print('RUN',module,*extra,flush=True)
        subprocess.run([sys.executable,'-u','-m','training.stage2_closure.'+module,*map(str,extra)],env=env,check=True)
    previous=-1;audited=-1;core=(root/'core_ready.json').exists()
    while True:
        done=list(campaign.glob('*/done.json'));states=list(campaign.glob('boundary_r*_mlwdf_pf_s*/states/tti*.pt'))
        progress=campaign/'progress_campaign.json'
        if progress.exists():
            p=json.loads(progress.read_text())
            if p['failed']:raise RuntimeError(json.dumps(p['failed'],ensure_ascii=False))
        if len(done)>0 and len(done)!=previous:
            run('analyze','--source',campaign,'--output',root/'analysis','--workers',4,'--partial');previous=len(done)
        if len(states)>0 and len(states)!=audited:
            run('audit','--source',campaign,'--output',root/'audit','--partial','--cpu',128,'--repeats',3)
            audited=json.loads((root/'audit/completion.json').read_text())['states']
        keys={json.loads(p.read_text())['key'] for p in done}
        needed={f'{scene}_r{rate:g}_{arm}_s{seed}' for scene,rate in [('boundary',.8),('uniform',1.)]
          for arm in ['pfq','deficit','mlwdf_pf'] for seed in range(41,51)}
        if not core and needed<=keys:
            run('plot_results','--root',root,'--core')
            run('build_report','--root',root,'--core')
            run('export_results','--root',root,'--core')
            path=root/'技术研究报告_二阶段同频收尾_核心对比版.docx'
            subprocess.run(['/usr/bin/python3','training/stage2_closure/render_doc.py',str(path)],env=env,check=True)
            (root/'core_ready.json').write_text(json.dumps({'ready':True,'points':60,'report':str(path)},ensure_ascii=False,indent=2)+'\n')
            print('CORE_READY',path,flush=True);core=True
        if len(done)==220:
            run('analyze','--source',campaign,'--output',root/'analysis','--workers',4)
            run('audit','--source',campaign,'--output',root/'audit','--cpu',128,'--repeats',3)
            run('plot_results','--root',root)
            run('build_report','--root',root)
            run('export_results','--root',root)
            path=root/'技术研究报告_二阶段同频收尾_DA-PLS.docx'
            subprocess.run(['/usr/bin/python3','training/stage2_closure/render_doc.py',str(path)],env=env,check=True)
            print('ALL_READY',path,flush=True);break
        tracks=[json.loads(p.read_text()) for p in campaign.glob('*/progress.json')]
        print('PROGRESS',len(done),'completed;',sum(t['tti'] for t in tracks),'native TTIs;',len(states),'saved ML states',flush=True)
        time.sleep(45)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())

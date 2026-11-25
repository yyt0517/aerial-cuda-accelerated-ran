"""Compact progress for the already running experiment."""
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',type=int,default=0)
    p.add_argument('--root',default='deliverables/seven_cell_boundary08_guarded_l010_b098_20261106_v1');a=p.parse_args()
    assert 0<=a.wait<=55
    time.sleep(a.wait);root=Path(a.root);rows=[]
    for seed in [41,42,43]:
        f=root/f'campaign/da_guarded_l010_b098_s{seed}/progress.json'
        if f.exists():
            d=json.loads(f.read_text());rows.append(dict(seed=seed,tti=d['tti'],elapsed_min=round(d['elapsed_s']/60,1),done=(f.parent/'done.json').exists()))
    print(json.dumps(rows,separators=(',',':')))
    p=root/'campaign/progress_campaign.json'
    if p.exists():
        d=json.loads(p.read_text())
        if d.get('failed'):print('FAILED',json.dumps(d['failed']))
    for f,label in [('analysis/completion.json','ANALYSIS COMPLETE'),('analysis/search_audit.json','AUDIT COMPLETE'),
                    ('analysis/report.txt','REPORT READY'),('seven_cell_guarded_da_results.zip','PACKAGE READY')]:
        if (root/f).exists():print(label)

if __name__=='__main__':main()

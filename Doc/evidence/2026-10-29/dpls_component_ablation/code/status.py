"""Compact progress view; never reads provisional quality outcomes."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
PROTOCOL=json.loads((OUT/'protocol.json').read_text())
ARMS,SEEDS=PROTOCOL['arms'],PROTOCOL['seeds']


def main():
    rows=[]
    for arm,config in ARMS.items():
        if config['reused']:continue
        for seed in SEEDS:
            path=OUT/'native'/arm/f'seed{seed}_current_20000'
            progress=path/'progress.json'
            state=json.loads(progress.read_text()) if progress.exists() else dict(tti=0,elapsed_s=0)
            tti=state['tti']
            complete=(path/'component_validation.json').exists()
            phase='complete' if complete else ('followup' if tti>=28000 else 'measurement' if tti>=8001 else 'settling' if tti>=4001 else 'warmup')
            log=OUT/'logs'/f'{arm}_seed{seed}.log'
            error=log.exists() and 'Traceback' in log.read_text()
            rows.append(dict(arm=arm,seed=seed,tti=tti,phase=phase,elapsed_s=state['elapsed_s'],error=error))
    phases={phase:sum(row['phase']==phase for row in rows) for phase in ['warmup','settling','measurement','followup','complete']}
    result=dict(runs=len(rows),phases=phases,min_tti=min(row['tti'] for row in rows),
        max_tti=max(row['tti'] for row in rows),completed_tti_fraction=sum(row['tti'] for row in rows)/(30*28400),
        errors=sum(row['error'] for row in rows),rows=rows)
    temporary=OUT/'progress.json.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(OUT/'progress.json')
    print(json.dumps({key:value for key,value in result.items() if key!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

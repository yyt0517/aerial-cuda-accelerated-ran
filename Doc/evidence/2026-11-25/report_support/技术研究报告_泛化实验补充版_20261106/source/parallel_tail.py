"""Use spare capacity for the last seed without altering frozen experiment code.
Each run uses the existing campaign.one function in a separate output root.
Only fully closed and validated runs are linked into the main completion index.
If the primary pool has already started a run, its output wins regardless of value.
"""
import os,sys,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from training.stage2_generalization import campaign
MAIN=campaign.OUT;EXTRA=MAIN/'parallel_tail'
def run(job):
    campaign.OUT=EXTRA
    return campaign.one(job,smoke=False)
def main():
    protocol=json.loads((MAIN/'protocol.json').read_text())
    for p,h in protocol['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    EXTRA.mkdir(exist_ok=True);(EXTRA/'campaign').mkdir(exist_ok=True)
    if not (EXTRA/'plans').exists():(EXTRA/'plans').symlink_to(MAIN/'plans',target_is_directory=True)
    jobs=[dict(scenario=s,arm=a,seed=55) for s in campaign.SCENARIOS for a in campaign.ARMS]
    (EXTRA/'execution_plan.json').write_text(json.dumps(dict(jobs=jobs,script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((MAIN/'protocol.json').read_bytes()).hexdigest(),publication='Completed validated runs only; primary existing directories take precedence independent of result'),indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(run,j):j for j in jobs}
        for f in as_completed(pending):
            result=f.result();source=Path(result['directory']);target=MAIN/'campaign'/result['key']
            assert json.loads((source/'validation.json').read_text())['passed']
            try:target.symlink_to(source,target_is_directory=True);published=True
            except FileExistsError:published=False
            results.append(dict(**result,published=published));print('TAIL DONE',result['key'],'published',published,flush=True)
            (EXTRA/'completion.json').write_text(json.dumps(dict(completed=len(results)==len(jobs),runs=results),indent=2))
if __name__=='__main__':main()

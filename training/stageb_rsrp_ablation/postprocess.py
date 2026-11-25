"""Local sequential postprocessing; never overlaps fair timing with trajectories."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from training.stageb_ppo.native import ROOT,write


def main(args):
    root=Path(args.root).resolve();completion=root/'campaign/completion.json'
    status=root/'postprocessing_status.json'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR='/tmp/rsrp_ablation_mpl')
    try:
        write(status,dict(stage='waiting_for_campaign',completed=False))
        while not completion.exists():time.sleep(30)
        assert json.loads(completion.read_text())['completed']
        commands=[('analysis','analyze',['--source',str(root/'campaign'),'--output',str(root/'analysis'),'--workers',str(args.workers)]),
                  ('replay','replay',['--source',str(root/'campaign'),'--output',str(root/'replay'),'--repeats','6']),
                  ('channel_information','channel_diagnostics',['--root',str(root)]),
                  ('report','report',['--root',str(root)])]
        for name,module,options in commands:
            write(status,dict(stage=name,completed=False))
            with (root/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-u','-m','training.stageb_rsrp_ablation.'+module,*options],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(status,dict(stage='complete',completed=True))
    except BaseException:
        write(status,dict(stage='failed',completed=False,error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args())

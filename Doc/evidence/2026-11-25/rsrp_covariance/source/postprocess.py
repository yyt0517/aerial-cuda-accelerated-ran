"""Wait for native completion, then perform isolated serial replay and reporting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from training.stageb_ppo.native import ROOT, write


def main(args):
    root=Path(args.root).resolve();status=root/'postprocessing_status.json'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR='/tmp/rsrp_covariance_mpl')
    try:
        write(status,dict(stage='waiting_for_campaign',completed=False))
        while not (root/'campaign/completion.json').exists():time.sleep(30)
        assert json.loads((root/'campaign/completion.json').read_text())['completed']
        commands=[('analysis','analyze',['--source',str(root/'campaign'),'--output',str(root/'analysis'),'--workers','4']),
                  ('replay','replay',['--source',str(root/'campaign'),'--output',str(root/'replay'),'--repeats','12']),
                  ('report','report',['--root',str(root)])]
        for name,module,options in commands:
            write(status,dict(stage=name,completed=False))
            with (root/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-u','-m','training.stageb_rsrp_covariance.'+module,*options],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(status,dict(stage='complete',completed=True))
    except BaseException:
        write(status,dict(stage='failed',completed=False,error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args())

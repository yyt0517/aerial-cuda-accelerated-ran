#!/usr/bin/env python3
"""Build the supplied CPU search code in a temporary directory and replay one NPZ.
Requires Python >=3.10, NumPy, g++, Eigen3; no CUDA, PyTorch or model weights.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
 import numpy as np
 with tempfile.TemporaryDirectory(prefix='stage2_offline_replay_') as folder:
  work=Path(folder);shutil.copytree(ROOT/'code',work/'code')
  code=work/'code';kernel=code/'training/stageb_lns'
  for filename in ['build.py','build_persistent.py','build_bounded.py']:
   print('Building '+filename,flush=True);subprocess.run([sys.executable,str(kernel/filename)],check=True,cwd=code)
  sys.path.insert(0,str(code));sys.dont_write_bytecode=True
  from training.stageb_lns.physics import Problem,ReferencePhy
  from training.stageb_lns.persistent_dpls import solve_integer_reference,solve_persistent,PersistentConfig
  from training.stageb_lns.bounded_dpls import solve_bounded,certificate_sets
  with np.load(ROOT/'sample/boundary_s43_tti018001.npz',allow_pickle=False) as z:
   p=Problem(**{k:z[k] for k in ['h','w','noise','queue','serving','mcs','caps','tbs','weights','legal']})
   draft=z['raw_pfq'];expected_action=z['expected_action'];expected_values=z['expected_values']
  trace=[]
  def capture(i):trace.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in i.items()})
  ref=solve_integer_reference(p,draft,observer=capture)
  verified=[]
  def audit(i):
   r=trace[i['iteration']-1]
   for k in ['root','root_values','keys','delta_d','selected_keys']:np.testing.assert_array_equal(i[k],r[k],err_msg=k)
   assert i['score_q']==r['score_q'] and i['floor']==r['floor']
   assert np.all(i['upper']>=r['delta_q'])
   valid=(i['status']&1)!=0;np.testing.assert_array_equal(i['delta_q'][valid],r['delta_q'][valid])
   required,fresh=certificate_sets(i,r);np.testing.assert_array_equal(required,fresh)
   verified.append(len(i['keys']))
  bounded=solve_bounded(p,draft,observer=audit)
  full=solve_persistent(p,draft,PersistentConfig('full'))
  for name,result in [('Python full',ref),('C++ full',full),('C++ bounded',bounded)]:
   np.testing.assert_array_equal(result[0],expected_action,err_msg=name)
   np.testing.assert_array_equal(result[1],expected_values,err_msg=name)
   assert result[2]['converged']
  np.testing.assert_allclose(bounded[1],ReferencePhy(p).evaluate(bounded[0])[0],rtol=1e-8,atol=1e-6)
  assert len(verified)==len(trace)
  result=dict(passed=True,sample='Boundary08, seed43, TTI18001',python_full_cpp_full_bounded_equal_to_archived_action_and_values=True,independent_numpy_phy_passed=True,exact_rounds=len(verified),audited_candidates=sum(verified),floor=bounded[2]['floor'],chosen_dequeue=bounded[2]['chosen_dequeue'],compiled_from_packaged_source=True,scope='One-state offline correctness check; not a reproduction of the full-campaign timing or end-to-end GPU experiment.')
 if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

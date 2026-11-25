#!/usr/bin/env python3
"""Verify delivery hashes and recompute the report's tables from per-seed CSVs.
Python 3 standard library only. Does not rerun the GPU simulator.
"""
from pathlib import Path
from collections import Counter
from functools import lru_cache
import argparse,csv,hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parent
@lru_cache(None)
def read(name):
 with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
 if (ROOT/'MANIFEST.sha256').exists():
  files=0
  for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
   expected,name=line.split('  ',1);actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
   if actual!=expected:raise ValueError('File hash mismatch: '+name)
   files+=1
 else: files=0
 matrices=[('results/main/per_seed.csv',120,10,['scene','arm']),('results/generalization/per_seed.csv',60,5,['scenario','arm']),('results/input_simplification/per_seed.csv',36,3,['scene','stage'])]
 for file,n,seeds,keys in matrices:
  rows=read(file);assert len(rows)==n,(file,len(rows));counts=Counter(tuple(r[k] for k in keys) for r in rows)
  assert all(v==seeds for v in counts.values()),(file,counts)
  assert len({tuple(r[k] for k in keys+['seed']) for r in rows})==n
 checks=json.loads((ROOT/'provenance/report_numeric_expectations.json').read_text())
 failures=[]
 for c in checks:
  rows=[r for r in read(c['dataset']) if all(r[k]==v for k,v in c['selector'].items())]
  assert len(rows)==c['n'],c
  got=statistics.mean(float(r[c['metric']]) for r in rows)
  tolerance=.5*10**(-c['decimals'])+1e-9
  if not math.isfinite(got) or abs(got-c['report_value'])>tolerance:failures.append(dict(**c,actual=got))
 # Independently cross-check the included full precision aggregates.
 aggregate_checks=0
 for c in read('results/main/aggregate.csv'):
  if c['subset']!='41-50':continue
  rows=[r for r in read('results/main/per_seed.csv') if all(r[k]==c[k] for k in ['scene','rate','arm'])]
  got=statistics.mean(float(r[c['metric']]) for r in rows)
  assert math.isclose(got,float(c['mean']),rel_tol=1e-11,abs_tol=1e-10),c
  aggregate_checks+=1
 for c in read('results/generalization/aggregate.csv'):
  rows=[r for r in read('results/generalization/per_seed.csv') if r['scenario']==c['scenario'] and r['arm']==c['arm']]
  for metric in ['payload_mbps','expiry_pct','expiry_byte_pct','delay_mean_ms','worst_ue_expiry_pct']:
   assert math.isclose(statistics.mean(float(r[metric]) for r in rows),float(c[metric+'__mean']),rel_tol=1e-11,abs_tol=1e-10)
   aggregate_checks+=1
 for c in read('results/input_simplification/means.csv'):
  rows=[r for r in read('results/input_simplification/per_seed.csv') if r['scene']==c['scene'] and r['stage']==c['stage']]
  for metric in ['payload_mbps','expiry_pct','delay_mean_ms','delay_p95_ms']:
   assert math.isclose(statistics.mean(float(r[metric]) for r in rows),float(c[metric]),rel_tol=1e-11,abs_tol=1e-10)
   aggregate_checks+=1
 audit=json.loads((ROOT/'results/search_audit/completion.json').read_text());assert audit['passed'] and audit['states']==90 and not audit['concurrent_quality_campaign']
 case=json.loads((ROOT/'results/search_audit/boundary_r0.8_mlwdf_pf_s43_tti018001.json').read_text())['case']
 assert (case['gain_q'],case['dequeue_before'],case['dequeue_after'],case['floor'])==(13136147515,87291,85656,78562)
 zero_checks=0
 for file in ['results/main/per_seed.csv','results/generalization/per_seed.csv']:
  for r in read(file):
   if r['arm']=='mlwdf_pf':assert float(r['expiry_pct'])==float(r['worst_ue_expiry_pct'])==0;zero_checks+=1
 result=dict(passed=not failures,hash_verified_files=files,report_numeric_checks=len(checks),aggregate_checks=aggregate_checks,zero_expiry_run_checks=zero_checks,archived_search_audit_states=90,table_9_4_case_verified=True,failures=failures,scope='Recompute equal-seed means from archived CSVs and verify report rounding. No full trajectory/GPU rerun.')
 if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if failures:raise SystemExit(1)
if __name__=='__main__':main()

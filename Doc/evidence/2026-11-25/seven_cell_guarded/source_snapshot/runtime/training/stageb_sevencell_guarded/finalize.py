"""Verify frozen provenance and package completed experiment artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch
from training.stageb_lns.no_ctbs_inputs import problem_from_raw_row
from .policy import ARM,make_weights
from .provenance import verify_sources

def main(root,baseline,multistart):
    root=Path(root);baseline=Path(baseline);multistart=Path(multistart)
    read=lambda p:json.loads((root/p).read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    complete=read('campaign/completion.json');analysis=read('analysis/completion.json');audit=read('analysis/search_audit.json')
    assert complete['completed'] and not complete['failed'] and complete['total']==3
    assert analysis['completed'] and analysis['new_runs']==3 and analysis['reused_runs']==9
    assert audit['passed'] and audit['states']==12 and audit['branch_searches']==36
    assert read('analysis/paired_validation.json')['passed']
    verify_sources(read('campaign/protocol.json')['source_sha256'])
    for p,h in read('source_snapshot/sha256.json').items():assert sha(root/'source_snapshot'/p)==h,p
    for ref in [baseline,multistart]:
        for p,h in json.loads((ref/'delivery_manifest.json').read_text())['artifact_sha256'].items():assert sha(ref/p)==h,p
    native=0;decisions=0;physics=[]
    for seed in [41,42,43]:
        p=root/f'campaign/{ARM}_s{seed}'
        v=json.loads((p/'validation.json').read_text());a=json.loads((p/'analysis_validation.json').read_text())
        q=json.loads((p/'policy_validation.json').read_text());s=json.loads((p/'summary.json').read_text())
        assert v['passed'] and v['sources_unchanged'] and a['passed'] and a['cohort_mature'] and a['fifo_crc_native_terminals_exact'] and s['censored']==0
        assert q['passed'] and q['sources_unchanged'] and q['service_guard_every_decision'] and q['selected_new_score_max_over_eligible']
        assert v['horizon']==v['exact_native_actions']==v['fresh_input_checks']==v['exact_mcs_layer_tbs_dequeue']==28400
        assert q['decision_ttis']==24400
        launch=json.loads((p/'launch.json').read_text());old=json.loads((baseline/f'campaign/pfq_s{seed}/launch.json').read_text())
        assert launch['source_sha256']==old['source_sha256']
        for f in (p/'states').glob('*.pt'):
            r=torch.load(f,map_location='cpu',weights_only=False);problem=problem_from_raw_row(r)
            np.testing.assert_array_equal(problem.weights,r['deficit_weights'])
            np.testing.assert_array_equal(make_weights(problem.weights,r['hol_urgency']),r['objective_weights'])
        native+=v['horizon'];decisions+=q['decision_ttis'];physics.extend(a['selected_physics'])
    assert native==85200 and decisions==73200 and len(physics)==12
    (root/'RESULTS_README.rst').write_text('七小区有界紧急度DA结果包\n========================\n\n报告：analysis/report.txt；图表：analysis/comparison.png/pdf。\n逐seed、汇总、配对差值、分支选择和验证结果均在analysis/。\n新权重lambda=0.1，最终同状态预期成功服务保护beta=0.98。\n新跑3条闭环轨迹，复用原DPLS、原DA、原三分支DA共9条对照。\nsource_snapshot/保存本轮代码与依赖哈希，仿真器及原始对照仍使用工作区冻结路径。\n完整原始轨迹与包队列保留在本目录campaign/，未加入轻量ZIP。\n',encoding='utf-8')
    reference=root/'baseline_reference';reference.mkdir(exist_ok=True)
    for ref,label in [(baseline,'original'),(multistart,'three_branch')]:
        for relative,name in [('analysis/per_seed.csv','per_seed.csv'),('campaign/protocol.json','protocol.json'),('delivery_manifest.json','delivery_manifest.json')]:
            shutil.copy2(ref/relative,reference/(label+'_'+name))
    # The live pipeline log is still receiving this finalizer's stdout. Seal a
    # copy before hashing so the packaged provenance cannot change afterwards.
    shutil.copy2(root/'pipeline.log',root/'execution_log.txt')
    files=[]
    for directory in ['analysis','source_snapshot','baseline_reference']:files.extend(p for p in (root/directory).rglob('*') if p.is_file())
    for name in ['RESULTS_README.rst','preflight.json','policy_tests.log','execution_log.txt','campaign/protocol.json','campaign/completion.json',
                 'smoke/analysis/paired_validation.json','smoke/analysis/search_audit.json']:
        p=root/name
        if p.exists():files.append(p)
    for seed in [41,42,43]:
        p=root/f'campaign/{ARM}_s{seed}'
        for name in ['summary.json','launch.json','done.json','validation.json','analysis_validation.json','policy_validation.json',
                     'per_ue.csv','per_cell.csv','ue_topology.csv','serving.npy','saved_states.json']:files.append(p/name)
    manifest=dict(completed=True,cells=7,users=84,seeds=[41,42,43],lambda_value=.1,beta=.98,new_runs=3,reused_runs=9,
        native_ttis_validated=native,strategy_decisions_validated=decisions,measurement_seconds_per_seed=10,
        all_cohorts_uncensored=True,paired_inputs_identical=True,full_warmup_and_first_policy_inputs_identical=True,
        every_decision_service_guard_and_max_eligible_new_score=True,physics_states_checked=12,
        full_neighborhood_branch_searches=36,audited_search_rounds=sum(x['rounds'] for x in audit['checks']),
        audited_candidates=sum(x['candidates'] for x in audit['checks']),
        max_saved_sinr_error_db=max(x['sinr_max_error_db'] for x in physics),max_saved_bler_error=max(x['bler_max_error'] for x in physics),
        execution_sources_unchanged=True,baseline_artifacts_unchanged=True,frozen_native_dependencies_match_baseline=True,
        artifact_sha256={str(p.relative_to(root)):sha(p) for p in sorted(set(files))})
    (root/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');files.append(root/'delivery_manifest.json')
    archive=root/'seven_cell_guarded_da_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(root)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({k:v for k,v in manifest.items() if k!='artifact_sha256'},indent=2),flush=True)
    print('PACKAGED',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--multistart',required=True)
    a=p.parse_args();main(a.root,a.baseline,a.multistart)

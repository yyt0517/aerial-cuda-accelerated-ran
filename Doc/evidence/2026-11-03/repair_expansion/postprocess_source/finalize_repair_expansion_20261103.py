import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path('/data/aerial-cuda-accelerated-ran')
out=ROOT/'deliverables/repair_expansion_20261103_v1'
review=ROOT/'Doc/evidence/2026-11-03/ai_scheduling_review'
dest=ROOT/'Doc/evidence/2026-11-03/repair_expansion'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read(p):return list(csv.DictReader(Path(p).open()))
def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path),bytes=Path(path).stat().st_size)

val={r['arm']:r for r in read(out/'validation_summary.csv')}
dev={r['arm']:r for r in read(out/'evaluation_summary.csv')}
gap=float(val['repair_guard']['mean_ms'])-.8*float(val['full']['mean_ms'])
write(out/'gate_results.json',dict(
    selected_quality='repair_guard',selected_cost_fallback='cached_maxj',validation_cost_feasible=False,
    validation_quality_service_pass=True,development_quality_service_cost_pass=True,
    all_stages_joint_pass=False,training_cross_seed_coverage_pass=False,independent_online_ai_success=False,
    learned_same_archive_service_repair=dict(states_changed=1,failures_repaired=1,new_joint_failures=0,exposed_states=21,
        scope='historically exposed frozen states with paid teacher resource/layer/activity conditions; not native packet-level gain'),
    cost_gap_ms=gap,validation_time_ratio=float(val['repair_guard']['execution_time_ratio']),
    training_repairable_by_seed={'41':2,'42':1,'43':0},focused_oof_repairs=1,focused_repairable_states=3,
    next_path='freeze repair_guard and verify query-equivalent proxy/frontier/terminal cost reduction; repair coverage and time-held-out generalization prerequisite before causal resource/activity proposals',
    next_phase_implemented=False,no_more_threshold_or_network_sweep=True,default_changed=False,
    stop_rule='if bounded held-time repair audit remains unstable or equivalent cost reduction has insufficient room, stop expanding this conditional terminal branch; retain native service PPO research reference and physical default',
    caveat='combined pair sampling/state weighting/UE auxiliary weighting intervention; no individual factor attribution'))
post=out/'postprocess_source';post.mkdir(exist_ok=True)
shutil.copyfile(ROOT/'training/stageb_lns/report_repair_expansion.py',post/'report_repair_expansion.py')
shutil.copyfile(__file__,post/'finalize_repair_expansion_20261103.py')
write(out/'code_manifest.json',dict(frozen_experiment=json.loads((out/'source_manifest.json').read_text()),
    postprocess={p.name:sha(p) for p in post.iterdir()},postprocess_changes_training=False))

ppo=ROOT/'deliverables/ppo_diagnosis_continue_20261030_v1/minibatch_continuation'
checks=[];source_audit=[]
for name in ('source_manifest.json','base_source_manifest.json','runtime_supplement_manifest.json'):
    d=json.loads((ppo/name).read_text())
    for path,digest in d.items():assert sha(ROOT/path)==digest,path
    source_audit.append(dict(manifest=record(ppo/name),entries=len(d),all_current_files_match=True))
import torch
from training.stageb_ppo_v2.model import ServiceActor
checkpoint=torch.load(ppo/'ppo_final.pt',map_location='cpu',weights_only=False)
actor=ServiceActor();actor.load_state_dict(checkpoint['actor']);actor.eval()
assert all(torch.isfinite(p).all() for p in actor.parameters())
assert sha(ppo/'ppo_final.pt')=='1167d28b1141dc186fbb3d1e6828384c74177228c6f10969b09ba1bc1e6e3a8a'
assert sha(out/'repair_guard/ue_pair_model.npz')=='ad4995666c113bca382004fd10c63992946f7a02015d3243be0db8361f4fc177'

recommend=dict(date='2026-11-03',default_changed=False,all_objectives_ai_winner=None,
    selection_criterion='for a complete native AI research implementation: actual learned decisions, matched candidate-pool attribution, payload/expiry prioritized; not a joint dominance claim',
    native_ai_research=dict(id='service_choice_ppo_minibatch_round12_20261030',
        checkpoint=record(ppo/'ppo_final.pt'),frozen_neural_repair=record(ROOT/'deliverables/tb_joint_repair_20261027_train_v1/repair.npz'),
        actor='training/stageb_ppo_v2/model.py::ServiceActor',catalog='training/stageb_ppo_v2/catalog.py::Catalog',
        training_entry='training.stageb_ppo_v2.refine',evaluation_entry='training.stageb_ppo_v2.evaluate',
        usage='training/stageb_ppo_v2/README.md',evidence='Doc/evidence/2026-10-30/ppo_service_choice/report.txt',
        protocol='seeds44-45 exposed; 4000 warmup +4000 stabilization +4000 measured arrival TTIs +400 follow-up; noCTBS/equalUE',
        quality=dict(payload_mbps=1240.686,expiry_pct=.663140775786466,success_mean_ms=7.012133541058531,success_p95_ms=24.5,ue_payload_p5_mbps=33.414),
        matched_physical=dict(payload_mbps=1237.848,expiry_pct=.8508401794546321,success_mean_ms=5.653562699273025,success_p95_ms=10.,ue_payload_p5_mbps=33.3765),
        limitations=['average/P95 successful packet latency fails','full DPLS every TTI; no online speedup','two exposed development seeds; no generalization claim'],
        runtime=dict(container='c_aerial_oai2',container_repo='/opt/nvidia/cuBB',python='/opt/nvidia/cuBB/.venv/bin/python',
            native_adapter='training/stageb_ppo/native.py',native_simulator_sha256='a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615',
            required_worktree='.worktrees/pfq-independent-reuse-actor',requires_cuda_native_simulator=True),
        cpu_checkpoint_load_verified=True,current_source_manifests_verified=source_audit,new_native_run_this_review=False),
    conditional_learning_research=dict(id='repair_guard_expanded_focused_20261103',checkpoint=record(out/'repair_guard/ue_pair_model.npz'),
        solver='training/stageb_lns/ue_pair_solver.py::solve_ue_pair',selector='training/stageb_lns/ue_pair_risk.py::select_terminal',
        train='training/stageb_lns/repair_pair_train.py',campaign='training.stageb_lns.repair_expansion',
        runtime_mode='ue_pair_guard',budget=212,confidence=.9,own_j_window=.02,
        validation=val['repair_guard'],development=dev['repair_guard'],
        teacher_condition_required=True,not_a_complete_online_ai_scheduler=True,
        limitations=['validation complete execution above 80% full','seed43 has no repairable training failure','1/3 leave-seed-out repairs','teacher condition generation absent from runtime cost'],
        evidence='Doc/evidence/2026-11-03/repair_expansion/report.txt'),
    physical_default=dict(registry=record(ROOT/'training/stageb_hgraph/current_algorithm.json'),
        id='pfq_ctbs_deficit_search_rho090_20261027',launcher='cuMAC/scripts/run_stageB_current_algorithm.sh',
        status='unchanged research default; no neural actor required',config='Doc/current_stageB_effective_configuration.md'),
    next_decision='conditional repair has a local positive signal; freeze it for equivalent cost reduction; do not immediately train a larger proposer or PPO')
write(review/'recommended_implementation.json',recommend)

attempts=[
('physical_components','10-29','native 5 seeds,10s,deficit/equal ablations','no','full1236.070Mbps vsPFQ720.236; full expiry0.492%','retain physical default; objectives have delay tradeoffs','Doc/evidence/2026-10-29/dpls_component_ablation/report.txt'),
('neural_joint_hybrid','10-27','native3seeds,own short-window protocol','not isolated superiority','hybrid payload+0.337%; physical block+0.636%','joint TB structure useful; no neural dominance','Doc/evidence/2026-10-27/tb_joint_repair/report.txt'),
('direct_risk','10-28','native seeds104-106,5s','neural proposed actions','less PHY but worse delay','do not promote','Doc/evidence/2026-10-28/tb_direct_risk/final_report.txt'),
('neural_cache','10-28','same-state serial and native equivalence','engineering only','same outcomes lower feature/decode cost','retain cache','Doc/evidence/2026-10-28/tb_neural_cost/report.txt'),
('verified_prefix','10-28','native48/50,1s; offline36states','mixed neural/extra physical verification','verified323.099ms vsdirect548.569; seed50 payload decline','no strict quality pass','Doc/evidence/2026-10-28/tb_prefix/report.txt'),
('service_choice_ppo','10-30','native44/45,2s,equal,noCTBS','yes,matched pool payload/expiry gain','1240.686Mbps/0.663%/7.012ms/P9524.5ms','best retained complete AI research entry for payload/expiry; latency fails','Doc/evidence/2026-10-30/ppo_service_choice/report.txt'),
('tail_wait_ppo','10-30','same native44/45,2s','learned tradeoff','1235.202Mbps/1.205%/4.341ms/P956.75ms','delay better; payload/expiry fail','Doc/evidence/2026-10-30/ppo_tail_wait/report.txt'),
('tail_balance_ppo','10-30','same native44/45,2s','learned tradeoff','1235.232Mbps/1.306%/4.951ms/P957.5ms','joint gates fail','Doc/evidence/2026-10-30/ppo_tail_balance/report.txt'),
('fast_student_ppo','10-30','native44/45,2s; policy-only timing','learned decode/select','4/8 candidates935/945Mbps; ~5x policy speedup','quality insufficient','Doc/evidence/2026-10-30/ppo_search_teacher/report.txt'),
('masked_structured','10-30 to11-02','finite prefix/candidate studies','no stable independent full-search gain','coverage and continuation errors remain','retain feasibility and exact small-block analysis','Doc/evidence/2026-11-02/structured_continuation/report.txt'),
('ranked_dpls','11-02','12validation+9development snapshots','no,zero residual selected','AI same21actions as heuristic,slower','no learning gain','Doc/evidence/2026-11-02/no_ctbs_ai_ranked_dpls/report.txt'),
('three_bounded_probes','11-02','same frozen-state family','scalar limited positive gain','scalar improves matched heuristic; far belowfull','not full quality success','Doc/evidence/2026-11-02/three_bounded_ai_probes/report.txt'),
('own_trajectory','11-02','shallow7x8/deep1x32; training41-43','unstable','shallow validation better; repeated deep failures','do not only increase depth','Doc/evidence/2026-11-02/own_trajectory/report.txt'),
('topk_dedup','11-02','top2/top4,106 budget','engineering plus mixed learning','removes duplicate failed queries; quality still insufficient','retain dedup','Doc/evidence/2026-11-02/shortlist_trajectory/report.txt'),
('budget_curve','11-02','106/160/212,21snapshots','not sufficient','selectedtop4 J gaps9.944/7.781/6.285%; coverage worsens','do not choose by meanJ only','Doc/evidence/2026-11-02/budget_curve/report.txt'),
('tb_macro','11-02','joint TB/finite continuation','mostly candidate mechanism','joint coupling benefit exists','needs service protection','Doc/evidence/2026-11-02/tb_macro_continuation/report.txt'),
('selective_tb','11-02','cheap increment/selective joint','no stable protected learning gain','J improves with coverage tradeoff','retain cheap features','Doc/evidence/2026-11-02/selective_tb/report.txt'),
('causal_service_repair','11-02','causal group and budgeted terminal repair','small mixed gain','quality/service still fail','group prediction not sufficient','Doc/evidence/2026-11-02/service_repair/report.txt'),
('resource_conditions','11-02','teacher quota/layer/activity ablation','no,teacher conditioned','activity meanJ+1.027% but newunserved5.778','conditional space only','Doc/evidence/2026-11-02/resource_conditions/report.txt'),
('wide_reorder_frontier','11-02','teacher joint conditions,212scores','physical search','joint decoding and service Pareto','retain wide pool/frontier','Doc/evidence/2026-11-02/joint_reorder/report.txt'),
('continuous_weak_service','11-02','same conditional21states','engineering/metric','frontier equivalent136.78to100.76ms','retain continuous E/D and fast frontier','Doc/evidence/2026-11-02/weak_service/report.txt'),
('directed_reorder','11-02','budgeted2slots plus paid recall diagnosis','physical search','development quality repaired; validation archive has good candidates','retain existingpool2slots','Doc/evidence/2026-11-02/directed_reorder/report.txt'),
('terminal_cached_aggregate','11-03','same conditional21states','cache benefit; learned no gain','same queries9-11%faster; aggregate misranks failure','retain cache','Doc/evidence/2026-11-03/terminal_service/report.txt'),
('ue_pair_guard','11-03','15unique train states,21historical eval','no terminal gain','all guarded finals equalmaxJ','retain UE losses and fallback; improve labels','Doc/evidence/2026-11-03/ue_pair/report.txt'),
('repair_expansion','11-03','30unique train states,21historical eval','one matched-archive failure repaired','validation12/12anddev9/9J/service;86.32/95.68ms','retain conditional learning reference; cost/coverage incomplete','Doc/evidence/2026-11-03/repair_expansion/report.txt')]
with (review/'attempts.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['branch','date','protocol','learning_contribution','result','decision','evidence']);w.writerows(attempts)
evidence=[]
for row in attempts:
    path=ROOT/row[-1]
    if row[0]=='repair_expansion':path=out/'report.txt'
    assert path.is_file(),path
    evidence.append(record(path))
write(review/'source_evidence.json',dict(reports=evidence,ppo_source_checks=source_audit,checkpoint_cpu_load_verified=True,
    native_tests_reused_not_rerun=True,no_new_training=True,no_default_change=True))

assert not dest.exists(), 'Do not overwrite a previously frozen evidence bundle'
shutil.copytree(out,dest)
paths=[p for p in out.rglob('*') if p.is_file()]
for p in paths:
    q=dest/p.relative_to(out);assert p.read_bytes()==q.read_bytes()
    if p.suffix=='.json':json.loads(q.read_text())
    elif p.suffix=='.npz':
        with np.load(q) as z:
            for key in z.files:z[key]
audit=dict(passed=True,copied_files=len(paths),all_copied_bytes_equal=True,all_json_npz_load=True,
    source_model_registry_hashes_valid=True,postprocessing_did_not_retrain=True)
write(out/'bundle_audit.json',audit);write(dest/'bundle_audit.json',audit)
manifest=dict(source=str(out),files={str(p.relative_to(dest)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*')) if p.is_file()})
write(dest/'archive_manifest.json',manifest)
write(review/'archive_manifest.json',dict(files={str(p.relative_to(review)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(review.rglob('*')) if p.is_file()}))
print(json.dumps(dict(bundle_files=len(manifest['files'])+1,bundle_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
    reviewed_branches=len(attempts),validation_gap_ms=gap,ppo_cpu_load=True,registry_sha256=sha(ROOT/'training/stageb_hgraph/current_algorithm.json')),indent=2))

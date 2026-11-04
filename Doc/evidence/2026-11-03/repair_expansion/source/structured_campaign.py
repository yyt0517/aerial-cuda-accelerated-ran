"""Isolated J-only pilot: exact structure, continuation, own-prefix correction.

Run once into a NEW output directory. Historical development replay only;
there is no simulator launch, final-seed access, or active-registry update.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .archive import ROOT, problem_from_row, initial_draft, tbs_table
from .cached_repair import CachedJointRepair
from .continuation_repair import ContinuationRepair, ContinuationValue, label_own_prefix, EXTRA_FEATURES
from .experiment import sha, write_json
from .physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from .quality import JOINT, FULL
from .search import solve, ProposalGraph
from .structured_repair import StructuredRepair, factorized_block, FactorizationLimit


FROZEN79 = ROOT/'deliverables/tb_joint_repair_20261027_train_v1/repair.npz'
FROZEN79_SHA = 'b53b7193f5b0b9e962a948664b689fb6b6a77351646f0e4bb48adde4d394d409'
REGISTRY = ROOT/'training/stageb_hgraph/current_algorithm.json'


def identity(row):
    return f"{row['family']}:seed{row['seed']}:tti{row['tti']}"


def split_rows(seeds):
    from .train_masked_joint import rows_for_split
    # Avoid adjacent snapshots and declare the sample before reading outcomes.
    return [r for r in rows_for_split(seeds) if r['tti'] in (4001, 8001, 18000)]


class CombinedRepair:
    deterministic = True

    def __init__(self, actor, structure):
        self.actor, self.structure = actor, structure

    def repair(self, graph, base, block, evaluate):
        self.actor.repair(graph, base, block, evaluate)
        self.structure.repair(graph, base, block, evaluate)


def run_policy(row, arm, value_path=None, *, recorder=None, cap=4096):
    # Weights are loaded before the timed interval. Every decision constructs a
    # fresh PHY, graph and decoder cache, including during model selection.
    if arm == 'cached79':
        repair = CachedJointRepair(FROZEN79)
    elif arm == 'structure':
        repair = StructuredRepair(0, max_blocks=8)
    elif arm in ('likelihood', 'current', 'rollout', 'learned', 'combined'):
        value = ContinuationValue(value_path) if arm in ('learned', 'combined') else None
        repair = ContinuationRepair(FROZEN79, 0,
            mode='learned' if value is not None else arm, value=value,
            recorder=recorder, shortlist=32, horizon=2, rollout_width=2)
        if arm == 'combined':
            repair = CombinedRepair(repair, StructuredRepair(0, max_blocks=4))
    elif arm in ('greedy', 'physical4', 'physical16'):
        repair = None
    else:
        raise ValueError(arm)
    started = time.perf_counter()
    p = problem_from_row(row)
    draft = initial_draft(p, row['obs'])
    # Shared causal-floor preparation is timed and counted for EVERY arm. It
    # costs one extra scored draft; the solve cap is reduced by exactly one.
    prep = NativePhy(p)
    try:
        v = prep.evaluate(draft).values[0]
        floor = (90*int(v[:, DEQUEUE].sum())+99)//100
        prep_stats = prep.stats
    finally:
        prep.close()
    if isinstance(repair, CombinedRepair):
        repair.actor.floor = repair.structure.floor = floor
    elif repair is not None and hasattr(repair, 'floor'):
        repair.floor = floor
    config = replace(FULL if arm == 'greedy' else JOINT, max_candidates=cap-1)
    if arm in ('physical4', 'physical16'):
        config = replace(config, repair_width=16, ue_topk=p.U,
                         block_size=16 if arm == 'physical16' else 4)
    action, values, record = solve(p, draft, config, repairer=repair)
    elapsed = (time.perf_counter()-started)*1000
    record['e2e_ms'] = elapsed
    record['arm'] = arm
    record['state_id'] = identity(row)
    record['seed'] = row['seed']
    for key, value in prep_stats.items():
        record[key] += value
    if isinstance(repair, CombinedRepair):
        record['structured_calls'] = repair.structure.calls
        record['structured_completed'] = repair.structure.completed
    elif isinstance(repair, StructuredRepair):
        record['structured_calls'] = repair.calls
        record['structured_completed'] = repair.completed
        record['structured_limits'] = repair.limits
    if record['floor'] != floor or record['candidate_evaluations'] > cap:
        raise AssertionError('Floor or actual total candidate budget mismatch')
    return action, values, record


def collect(rows, out, round_index, value_path):
    groups, costs, decisions = [], [], []
    for row in rows:
        visited = []
        arm = 'likelihood' if value_path is None else 'learned'
        action, values, record = run_policy(row, arm, value_path, recorder=visited.append)
        # The student decision is FINISHED before a separate teacher context
        # is constructed. Teacher actions never replace student rollout states.
        student_hash = hashlib.sha256(action.tobytes()).hexdigest()
        p = problem_from_row(row)
        np.testing.assert_allclose(values, ReferencePhy(p).evaluate(action)[0], rtol=1.e-8, atol=1.e-6)
        for observation in visited:
            label = label_own_prefix(p, observation, horizon=2, width=4, max_candidates=128)
            group = dict(label, state_id=identity(row), seed=row['seed'], round=round_index,
                         depth=observation['depth'], block=observation['block'],
                         base_action=observation['base_action'], partial=observation['partial'],
                         choices=observation['choices'], floor=observation['floor'],
                         student_action_sha256=student_hash)
            groups.append(group)
            costs.append({k:label[k] for k in ('candidate_evaluations','ue_evaluations','mmse_requests','mmse_misses')})
        if student_hash != hashlib.sha256(action.tobytes()).hexdigest():
            raise AssertionError('Teacher changed an already executed student decision')
        decisions.append(dict(record, student_action_sha256=student_hash, prefix_groups=len(visited)))
        print('COLLECT', round_index, identity(row), len(visited), flush=True)
    import torch
    torch.save(groups, out/f'own_prefix_round{round_index}.pt')
    write_json(out/f'collection_round{round_index}.json', dict(
        round=round_index, student='frozen79' if value_path is None else str(value_path),
        groups=len(groups), independently_visited_states=len(rows),
        complete_option_targets=sum(int((g['known'] & g['complete']).sum()) for g in groups),
        missing_options=sum(int((~g['known']).sum()) for g in groups),
        truncated_options=sum(int((~g['complete']).sum()) for g in groups),
        teacher_cost={k:sum(c[k] for c in costs) for k in costs[0]}, decisions=decisions,
        note='Correlated prefixes; observed finite-beam continuation values, not optimal suffix labels'))
    return groups


def export_value(path, model, mean, scale):
    np.savez(path, version=1, mean=mean, scale=scale,
        **{name:layer.detach().cpu().numpy() for name,layer in (
            ('w1',model[0].weight),('b1',model[0].bias),('w2',model[2].weight),
            ('b2',model[2].bias),('w3',model[4].weight),('b3',model[4].bias))})


def seed_mean(records, key):
    by_seed = {}
    for record in records:
        by_seed.setdefault(record['seed'], []).append(record[key])
    return float(np.mean([np.mean(values) for values in by_seed.values()]))


def train_value(groups, validation, out, round_index, epochs=20):
    import torch
    torch.manual_seed(7301+round_index)
    groups = [g for g in groups if (g['known'] & g['complete']).any()]
    if not groups:
        raise RuntimeError('No complete observed teacher targets')
    max_options = max(len(g['x']) for g in groups)
    d = groups[0]['x'].shape[-1]
    x = np.zeros((len(groups), max_options, d), np.float32)
    y = np.zeros((len(groups), max_options), np.float32)
    mask = np.zeros(y.shape, bool)
    for i, g in enumerate(groups):
        n = len(g['x'])
        x[i, :n] = g['x']
        valid = g['known'] & g['complete']
        mask[i, :n] = valid
        y[i, :n] = np.where(valid, g['target'], 0)
    mean = x[mask].mean(0)
    scale = np.maximum(x[mask].std(0), .1)
    tx = torch.from_numpy(np.clip((x-mean)/scale, -10, 10))
    ty, tm = torch.from_numpy(y), torch.from_numpy(mask)
    model = torch.nn.Sequential(torch.nn.Linear(d,32),torch.nn.ReLU(),
        torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
    # Epoch zero has exactly zero value, hence uses the frozen likelihood tie
    # break. It is an explicit no-learning candidate, never relabelled a win.
    torch.nn.init.zeros_(model[4].weight)
    torch.nn.init.zeros_(model[4].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(7301+round_index)
    best, selected, history = -np.inf, None, []
    checkpoints = {0, 5, 10, epochs}
    for epoch in range(epochs+1):
        loss_value = None
        if epoch:
            losses = []
            for start in range(0, len(groups), 32):
                # A new permutation per epoch is generated once below.
                indices = permutation[start:start+32]
                pred = model(tx[indices])[...,0]
                m, target = tm[indices], ty[indices]
                target_logits = (target/.3).masked_fill(~m, -1.e9)
                q = torch.softmax(target_logits, -1)
                ranking = -(q*torch.log_softmax(pred.masked_fill(~m,-1.e9),-1)).sum(-1).mean()
                regression = torch.nn.functional.smooth_l1_loss(pred[m], target[m])
                loss = ranking+.2*regression
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); losses.append(float(loss.detach()))
            loss_value = float(np.mean(losses))
        permutation = rng.permutation(len(groups))
        item = dict(epoch=epoch,loss=loss_value)
        if epoch in checkpoints:
            path = out/f'value_r{round_index}_e{epoch:02d}.npz'
            export_value(path, model, mean, scale)
            # Check deployed NumPy head independently of the Torch forward.
            with torch.no_grad():
                reference = model(tx[:8])[...,0].numpy()
            np.testing.assert_allclose(ContinuationValue(path).predict(x[:8]), reference, rtol=2.e-5, atol=2.e-6)
            records = [run_policy(row,'learned',path)[2] for row in validation]
            metric = seed_mean(records,'score')
            item.update(validation_j=metric,validation_ms=seed_mean(records,'e2e_ms'),
                        validation_records=records,checkpoint=str(path))
            if metric > best:
                best, selected = metric, path
            print('VALUE',round_index,epoch,metric,flush=True)
        history.append(item)
    write_json(out/f'training_round{round_index}.json',dict(
        parameters=sum(p.numel() for p in model.parameters()),features=d,
        groups=len(groups),epochs=epochs,selection='highest seed-equal validation J; earliest tie',
        selected=str(selected),best_validation_j=best,history=history))
    return selected


def audit_decomposition(rows, out):
    records = []
    for row in rows:
        p = problem_from_row(row); draft = initial_draft(p,row['obs'])
        phy = NativePhy(p)
        try:
            base, graph = phy.evaluate(draft), ProposalGraph(phy)
            floor = int(np.ceil(.9*base.values[0,:,DEQUEUE].sum()))
            # Prescribed single-PRG blocks; full all-legal Cartesian comparison.
            for prg in (0,8,16):
                block = tuple((c,prg) for c in range(p.C))
                before = phy.stats
                result = factorized_block(graph,base,block,floor,phy.evaluate)
                work = {k:phy.stats[k]-before[k] for k in before}
                domains = [np.r_[-1,np.flatnonzero(p.legal[t])] for t in block]
                combinations = list(itertools.product(*domains))
                actions = np.repeat(draft[None],len(combinations),axis=0)
                for a,choice in zip(actions,combinations):
                    for token,value in zip(block,choice):a[token]=value
                full = phy.evaluate(actions,base)
                j = full.values[:,:,EXPECTED]@p.weights
                feasible = full.values[:,:,DEQUEUE].sum(1)>=floor
                oracle = float(np.max(np.where(feasible,j,-np.inf)))
                if not result.complete or abs(result.predicted_j-oracle)>1.e-6:
                    raise AssertionError('Native all-combination oracle disagrees with decomposition')
                independent = ReferencePhy(p).evaluate(result.action)[0]
                np.testing.assert_allclose(independent,phy.evaluate(result.action).values[0],rtol=1.e-8,atol=1.e-6)
                records.append(dict(state_id=identity(row),prg=prg,oracle_j=oracle,
                    decomposed_j=result.predicted_j,cartesian_candidates=len(actions),
                    factored_work=work,patterns=result.patterns,local_assignments=result.local_assignments))
        finally:phy.close()
    write_json(out/'decomposition_checks.json',dict(passed=True,blocks=records,
        scope='Prescribed single-PRG blocks; independent final NumPy check plus native all-legal Cartesian oracle'))


def evaluation_rows():
    from .eval_masked_joint import evaluation_rows as load
    rows, files = load(dict(evaluation_seeds=[46,47,48],evaluation_sources=['pfq0','neural','greedy'],
                           evaluation_ttis=[3001,4001,5000]))
    return rows, files


def evaluate(rows, out, selected, repeats=3):
    import pandas as pd
    arms = ('cached79','greedy','physical4','physical16','structure','current','rollout','learned','combined')
    rng = np.random.default_rng(7307)
    records, outputs, audits = [], {}, []
    # Immutable TBS construction and one warm run per arm are outside timing.
    for arm in arms:run_policy(rows[0],arm,selected,cap=1000000 if arm in ('greedy','physical4','physical16') else 4096)
    for row in rows:
        for repetition in range(repeats):
            for arm in rng.permutation(arms):
                cap = 1000000 if arm in ('greedy','physical4','physical16') else 4096
                action,values,record = run_policy(row,str(arm),selected,cap=cap)
                key = (identity(row),str(arm))
                if key in outputs:
                    np.testing.assert_array_equal(action,outputs[key][0])
                    np.testing.assert_array_equal(values,outputs[key][1])
                    if record['candidate_evaluations']!=outputs[key][2]:
                        raise AssertionError('Repeated decisions changed actual work')
                else:outputs[key]=(action.copy(),values.copy(),record['candidate_evaluations'])
                records.append(dict(record,repetition=repetition))
        print('EVAL',identity(row),flush=True)
    # Independent audits run AFTER serial timing, never inside a timed repeat.
    for row in rows:
        p = problem_from_row(row)
        reference = ReferencePhy(p)
        for arm in arms:
            action,values,_ = outputs[(identity(row),arm)]
            independent = reference.evaluate(action)[0]
            np.testing.assert_allclose(values,independent,rtol=1.e-8,atol=1.e-6)
            audits.append(dict(state_id=identity(row),arm=arm,max_error=float(np.abs(values-independent).max())))
    flat = [{k:v for k,v in r.items() if isinstance(v,(int,float,str,bool))} for r in records]
    f = pd.DataFrame(flat); f.to_csv(out/'measurements.csv',index=False)
    per = f.groupby(['state_id','seed','arm']).median(numeric_only=True).reset_index()
    per.to_csv(out/'per_state.csv',index=False)
    reference = per[per.arm=='physical4'].set_index('state_id')
    baseline = per[per.arm=='cached79'].set_index('state_id')
    summary=[]
    for arm, group in per.groupby('arm'):
        group=group.set_index('state_id'); ref=reference.loc[group.index]; old=baseline.loc[group.index]
        delta=group.score-ref.score
        relative=group.score/ref.score-1
        summary.append(dict(arm=arm,score=seed_mean(group.reset_index().to_dict('records'),'score'),
            mean_ms=seed_mean(group.reset_index().to_dict('records'),'e2e_ms'),
            candidates=float(group.candidate_evaluations.mean()),
            relative_j_pct=float(relative.groupby(group.seed).mean().mean()*100),
            worst_relative_j_pct=float(relative.min()*100),wins=int((delta>1).sum()),losses=int((delta<-1).sum()),
            all_states_quality=bool((group.score>=.995*ref.score).all()),
            no_slower_than_cached=bool(group.e2e_ms.groupby(group.seed).mean().mean()<=old.e2e_ms.groupby(old.seed).mean().mean())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    np.savez_compressed(out/'actions.npz',**{f'{i}_{arm}':outputs[(identity(row),arm)][0]
                         for i,row in enumerate(rows) for arm in arms})
    write_json(out/'evaluation_checks.json',dict(passed=True,repeats=repeats,independent_audits=audits,
        action_and_value_repeatability=True,actual_work_repeatability=True,
        scientific_gate_passed=any(x['all_states_quality'] and x['no_slower_than_cached'] for x in summary
                                   if x['arm'] in ('structure','learned','combined','rollout')),
        caveat='Historical development replay; no new-seed, native packet, or deployment-latency conclusion'))


def main(args):
    import torch
    torch.set_num_threads(1)
    if sha(FROZEN79)!=FROZEN79_SHA:
        raise RuntimeError('Frozen79 checkpoint hash mismatch')
    if args.epochs<10 or args.repeats<1:
        raise ValueError('Pilot requires at least 10 epochs and one timing repetition')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    registry_sha=sha(REGISTRY)
    # Protocol is committed to disk before data collection or outcome inspection.
    protocol=dict(version=1,date='2026-11-02',training_seeds=[41,42,43],validation_seeds=[44,45],
        training_and_validation_sources=['current','pfq0'],training_and_validation_ttis=[4001,8001,18000],
        evaluation_seeds=[46,47,48],evaluation_ttis=[3001,4001,5000],
        evaluation_sources=['pfq0','neural','greedy'],historical_development_states=True,
        untouched_final_seeds=[311,312,313,314,315],frozen79=str(FROZEN79),frozen79_sha256=FROZEN79_SHA,
        registry_sha256=registry_sha,objective='unchanged weighted expected-byte J; own PFQ+CTBS 90% dequeue floor',
        extra_features=list(EXTRA_FEATURES),student_prefix_sampling='first retained parent, each depth, first four visited blocks',
        teacher=dict(horizon=2,width=4,all_legal=True,max_candidates_per_option=128,
                     missing_value='masked; not zero',query_after_student_decision=True),
        self_rounds=2,epochs_per_round=args.epochs,selection='validation seed-equal complete-search J, no evaluation read',
        candidate_cap=4096,physical_reference_cap=1000000,structure_blocks=8,combined_structure_blocks=4,
        prefix_shortlist=32,rollout_horizon=2,rollout_width=2,repeats=args.repeats,
        quality_gate='every development state J >= 99.5% of physical4; mean decision cost <= cached79',
        timing='serial decoded snapshot to action; fresh contexts; common floor probe counted; excludes weights/archive IO/immutable TBS setup',
        native_gate='not entered in this pilot; existing payload/TTL/mean/P95/UE-P5 and degraded-UE gates remain required')
    write_json(out/'protocol.json',protocol)
    paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.cpp'))
    paths += [ROOT/'cuMAC/src/cumac.h',REGISTRY,FROZEN79]
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);hashes[str(rel)]=sha(path)
    write_json(out/'source_manifest.json',hashes)
    train,validation=split_rows((41,42,43)),split_rows((44,45))
    write_json(out/'input_manifest.json',{str(Path(r['source'])):sha(Path(r['source'])) for r in train+validation})
    tbs_table(16,12,17)
    audit_decomposition([next(r for r in train if r['seed']==seed) for seed in (41,42,43)],out)
    groups=collect(train,out,0,None)
    first=train_value(groups,validation,out,0,args.epochs)
    groups+=collect(train,out,1,first)
    second=train_value(groups,validation,out,1,args.epochs)
    choices=[]
    for path in (first,second):
        records=[run_policy(r,'learned',path)[2] for r in validation]
        choices.append((seed_mean(records,'score'),path))
    selected=max(choices,key=lambda x:x[0])[1]
    write_json(out/'selection.json',dict(selected=str(selected),sha256=sha(selected),
        round_candidates=[dict(validation_j=j,path=str(p)) for j,p in choices],evaluation_unread_at_selection=True))
    rows,files=evaluation_rows()
    write_json(out/'evaluation_input_manifest.json',files)
    evaluate(rows,out,selected,args.repeats)
    if sha(REGISTRY)!=registry_sha or sha(FROZEN79)!=FROZEN79_SHA:
        raise RuntimeError('Protected baseline or active registry changed')
    for rel,digest in hashes.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError(f'Source changed during experiment: {rel}')
    write_json(out/'completion.json',dict(completed=True,protected_files_unchanged=True,
        final_seeds_not_read=True,new_native_trajectories=0))
    print('COMPLETED',str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=3)
    main(parser.parse_args())

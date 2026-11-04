"""Independent first-version masked-joint collection, training and self-improvement.

Data splits and search budgets are fixed before collection. Teacher labels keep
the complement of each repair mask fixed. Validation is never backpropagated.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import time
import numpy as np

from .archive import ROOT, load_rows, enable_legacy_types, problem_from_row, initial_draft
from .experiment import write_json, sha
from .physics import NativePhy, EXPECTED, DEQUEUE
from .search import ProposalGraph, SearchConfig, repaired_candidates, solve
from .quality import FULL, JOINT
from .masked_joint import (JointContext, make_torch_model, MaskedJointRepair,
                           prefix_features, torch_data, export_model)
from .masked_search import masks, solve_masked


PERMUTATIONS = np.array(list(itertools.permutations(range(4))), dtype=np.int64)


def rows_for_split(seeds):
    import torch
    rows = [dict(r, family='current') for r in load_rows(seeds)]
    enable_legacy_types()
    for seed in seeds:
        path = ROOT/f'deliverables/boundary08_rho090_five_seed_20261027/raw_runs/seed{seed}_pfq0/states.pt'
        for r in torch.load(path, map_location='cpu', weights_only=False):
            rows.append(dict(r, seed=seed, source=str(path), family='pfq0'))
    return rows


def identity(row):
    return f"{row['family']}:seed{row['seed']}:tti{row['tti']}"


def package(context, base, floor):
    result = context.dynamic(base, floor)
    result.update(edges=context.edges, edge_slots=context.slots, edge_users=context.users,
                  legal=context.legal, C=context.problem.C, P=context.problem.P)
    return result


def tensor_package(data):
    import torch
    return {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in data.items()}


def teacher_pool(phy, graph, base, block, floor, rng, extras=(), candidate_limit=8192):
    p = phy.problem
    block = np.array(block, np.int64)
    score = float(base.values[0,:,EXPECTED]@p.weights)
    pool = {base.actions[0].tobytes(): (score, base.actions[0].copy())}
    calls_before = phy.stats.copy()
    evaluated = 0
    fixed = np.ones(p.C*p.P, bool); fixed[block] = False

    def evaluate(actions, source):
        nonlocal evaluated
        if evaluated >= candidate_limit:
            return None
        actions = actions[:candidate_limit-evaluated]
        if not np.all(actions.reshape(len(actions),-1)[:,fixed] == base.actions[0].ravel()[fixed]):
            raise AssertionError('Teacher changed the fixed complement of a mask')
        result = phy.evaluate(actions, source); evaluated += len(actions)
        scores = result.values[:,:,EXPECTED]@p.weights
        eligible = result.values[:,:,DEQUEUE].sum(1) >= floor
        # A label is an acceptable complete repair or the unchanged baseline.
        for i in np.flatnonzero(eligible & (scores >= score-1.e-7)):
            pool[result.actions[i].tobytes()] = (float(scores[i]), result.actions[i].copy())
        return result

    valid_extras = [a for a in extras if np.array_equal(a.ravel()[fixed],base.actions[0].ravel()[fixed])]
    if valid_extras:
        evaluate(np.asarray(valid_extras),base)
    cfg = SearchConfig(budget_ms=None,block_size=len(block),repair_width=16,ue_topk=p.U)
    coordinates = tuple(divmod(int(i),p.P) for i in block)
    for _ in repaired_candidates(graph,base,coordinates,cfg,rng,evaluate):
        pass
    ranked = sorted(pool.values(),key=lambda x:-x[0])[:4]
    return ranked, {k:phy.stats[k]-calls_before[k] for k in calls_before}


def make_case(context, root, block, ranked, base_score, rng):
    block = np.asarray(block,np.int64).copy()
    # Randomize the generation order while preserving the physical assignment.
    block = rng.permutation(block)
    while len(ranked)<4:
        ranked = ranked+[ranked[0]]
    targets = np.stack([a.ravel()[block] for _,a in ranked])
    phi = np.stack([prefix_features(context,root,block,t) for t in targets])
    scores = np.array([s for s,_ in ranked])
    q = np.exp((scores-scores.max())/3000.);q /= q.sum()
    gain = float(scores.max()-base_score)
    return dict(block=block,labels=targets+1,phi=phi,quality=q.astype(np.float32),
                gain=gain,weight=float(np.clip(gain/3000.,.15,4.)),scores=scores)


def collect(rows, blocks_per_root=8, student=None):
    groups=[]; audit=[]
    for row_index,row in enumerate(rows):
        started=time.perf_counter()
        p=problem_from_row(row); draft=initial_draft(p,row['obs'])
        greedy,_,greedy_record=solve(p,draft,FULL)
        if not greedy_record['converged']:raise RuntimeError('Greedy teacher did not converge')
        strong,_,strong_record=solve(p,draft,replace(JOINT,repair_width=16,ue_topk=p.U),initial_action=greedy)
        if student is None:
            roots=[('draft',draft),('greedy',greedy)]
        else:
            own,_,_=solve_masked(p,draft,student)
            roots=[('student',own)]
        phy=NativePhy(p)
        try:
            original=phy.evaluate(draft)
            floor=(90*int(original.values[0,:,DEQUEUE].sum())+99)//100
            graph=ProposalGraph(phy);context=JointContext(phy,graph)
            rng=np.random.default_rng(7711+row['seed']*100000+row['tti'])
            row_audit=dict(id=identity(row),teacher_greedy_candidates=greedy_record['candidate_evaluations'],
                           teacher_strong_candidates=strong_record['candidate_evaluations'])
            for name,root in roots:
                base=phy.evaluate(root);score=float(base.values[0,:,EXPECTED]@p.weights)
                local=masks(graph,base,blocks_per_root,sizes=(4,8,16,4,8,16,8,16))
                selected=[np.arange(p.C*p.P)]+local
                cases=[]
                for b in selected:
                    ranked,cost=teacher_pool(phy,graph,base,b,floor,rng,extras=(greedy,strong))
                    cases.append(make_case(context,root,b,ranked,score,rng))
                groups.append(dict(id=identity(row),root_kind=name,root=root.copy(),floor=floor,
                                   data=package(context,base,floor),cases=cases))
            row_audit.update(teacher_context_stats=phy.stats,collection_seconds=time.perf_counter()-started)
            audit.append(row_audit)
        finally:phy.close()
        print('COLLECT',row_index+1,len(rows),identity(row),'groups',len(groups),
              'seconds',round(time.perf_counter()-started,2),flush=True)
    return groups,audit


def joint_loss(model,data,encoded,case):
    import torch
    block=torch.from_numpy(case['block'])
    logits=model.logits(data,encoded,block)
    phi=torch.from_numpy(case['phi'])
    scores=logits[:,None]+torch.einsum('jlad,hd->hjla',phi,model.beta)
    logp=scores.log_softmax(-1)
    labels=torch.from_numpy(case['labels'])[None,:,:,None].expand(4,-1,-1,-1)
    cost=-logp.gather(-1,labels).squeeze(-1).mean(-1)
    quality=torch.from_numpy(case['quality'])
    assignment_cost=cost.detach().numpy()[np.arange(4)[None],PERMUTATIONS]
    assignment_cost=(assignment_cost*case['quality'][PERMUTATIONS]).sum(1)
    assignment=torch.from_numpy(PERMUTATIONS[int(assignment_cost.argmin())])
    return (cost[torch.arange(4),assignment]*quality[assignment]).sum()*case['weight']


def validate(model,rows,path):
    export_model(path,model)
    decoder=MaskedJointRepair(path)
    scores=[];gains=[];times=[]
    for row in rows:
        p=problem_from_row(row);draft=initial_draft(p,row['obs'])
        _,_,record=solve_masked(p,draft,decoder)
        scores.append(record['score']);gains.append(record['improvement']);times.append(record['core_ms'])
    return dict(mean_score=float(np.mean(scores)),mean_gain=float(np.mean(gains)),
                mean_core_ms=float(np.mean(times)),states=len(rows))


def replay_log_prob(model,data,encoded,context,base,trace):
    import torch
    block=trace['block']
    logits=model.logits(data,encoded,torch.from_numpy(block))
    values=[]
    for action,head in zip(trace['actions'],trace['modes']):
        targets=action.ravel()[block]
        phi=torch.from_numpy(prefix_features(context,base.actions[0],block,targets))
        scores=logits[int(head)]+phi@model.beta[int(head)]
        labels=torch.from_numpy(targets+1)
        values.append(scores.log_softmax(-1).gather(-1,labels[:,None]).sum())
    return torch.stack(values)


def best_of_k_updates(model,optimizer,rows,updates,seed):
    import torch
    rng=np.random.default_rng(seed);records=[]
    for index in range(updates):
        row=rows[int(rng.integers(len(rows)))];p=problem_from_row(row);draft=initial_draft(p,row['obs'])
        decoder=MaskedJointRepair(state={k:v.detach().numpy() for k,v in model.state_dict().items()})
        own,_,_=solve_masked(p,draft,decoder)
        phy=NativePhy(p)
        try:
            original=phy.evaluate(draft);floor=(90*int(original.values[0,:,DEQUEUE].sum())+99)//100
            # Include both draft and the policy's own visited endpoint.
            base=phy.evaluate(own if index%2 else draft)
            graph=ProposalGraph(phy);context=JointContext(phy,graph)
            available=[np.arange(p.C*p.P)]+masks(graph,base)
            block=available[index%len(available)]
            actions,traces=decoder.propose_many(context,base,floor,[block],rng,stochastic_all=True,return_trace=True)
            evaluated=phy.evaluate(actions,base)
            base_score=float(base.values[0,:,EXPECTED]@p.weights)
            reward=(evaluated.values[:,:,EXPECTED]@p.weights-base_score)/3000.
            reward[evaluated.values[:,:,DEQUEUE].sum(1)<floor]=-1.
            best=max(0.,float(reward.max()))
            credit=np.array([best-max(0.,float(np.delete(reward,k).max())) for k in range(len(reward))],np.float32)
            data=torch_data(context,context.dynamic(base,floor))
            logp=replay_log_prob(model,data,model.encode(data),context,base,traces[0])
            loss=-(torch.from_numpy(credit)*logp).sum()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            records.append(dict(update=index,state=identity(row),mask_size=len(block),best_gain=best*3000,
                                nonzero_credit=int((credit>0).sum()),loss=float(loss.detach()),
                                exact_candidates=phy.stats['candidate_evaluations']))
        finally:phy.close()
    return records


def main(args):
    import torch
    torch.set_num_threads(1);torch.manual_seed(3717)
    out=Path(args.output).resolve()
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version=1,training_seeds=[41,42,43],validation_seeds=[44,45],
                  evaluation_seeds=[46,47,48],training_sources=['current','pfq0'],
                  evaluation_sources=['pfq0','neural','greedy'],evaluation_ttis=[3001,4001,5000],
                  historical_test_states=True,seed=3717,dim=32,heads=4,proposals_per_mask=8,
                  global_reconstruction=True,local_rounds=2,blocks_per_round=4,local_sizes=[4,8,16,8],
                  candidate_cap_direct=73,objective='J weighted expected bytes, original draft 90% floor',
                  teacher=dict(width=16,all_legal_ues=True,mask_candidate_limit=8192,blocks_per_root=args.blocks),
                  epochs=args.epochs,self_rounds=args.self_rounds,self_epochs=args.self_epochs,best_k_updates=args.best_k_updates,
                  selection='highest mean frozen-state validation J at fixed 73-candidate envelope',
                  timing='CPU single-thread complete decoded-snapshot decision; model/library loading excluded',
                  frozen79=str(ROOT/'deliverables/tb_joint_repair_20261027_train_v1/repair.npz'),
                  frozen79_sha256=sha(ROOT/'deliverables/tb_joint_repair_20261027_train_v1/repair.npz'))
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol differs')
    else:write_json(out/'protocol.json',protocol)
    rows=rows_for_split((41,42,43));valid=rows_for_split((44,45))
    if not (out/'input_manifest.json').exists():
        paths=sorted(set(r['source'] for r in rows+valid))
        write_json(out/'input_manifest.json',dict(states=[identity(r) for r in rows+valid],
                   files={p:sha(p) for p in paths}))
    dataset=out/'dataset.pt'
    if dataset.exists():
        stored=torch.load(dataset,weights_only=False,map_location='cpu');groups=stored['groups']
    else:
        groups,audit=collect(rows,args.blocks)
        torch.save(dict(groups=groups,audit=audit),dataset)
        write_json(out/'collection.json',audit)
    model=make_torch_model(groups[0]['data']['edges'].shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(3717)
    history=[];best=-np.inf;best_epoch=None;epoch_number=0

    def train_epochs(count,phase):
        nonlocal best,best_epoch,epoch_number
        tensor_groups=[tensor_package(g['data']) for g in groups]
        for local_epoch in range(count):
            started=time.perf_counter();losses=[]
            for i in rng.permutation(len(groups)):
                data=tensor_groups[i];encoded=model.encode(data)
                cases=groups[i]['cases']
                loss=sum(joint_loss(model,data,encoded,c) for c in cases)/sum(c['weight'] for c in cases)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
            epoch_number+=1
            record=dict(epoch=epoch_number,phase=phase,loss=float(np.mean(losses)),seconds=time.perf_counter()-started)
            if local_epoch==0 or (local_epoch+1)%5==0 or local_epoch+1==count:
                metrics=validate(model,valid,out/f'checkpoint_{epoch_number:03d}.npz')
                record.update(metrics)
                if metrics['mean_score']>best:
                    best=metrics['mean_score'];best_epoch=epoch_number
                    export_model(out/'repair.npz',model)
            history.append(record)
            print('TRAIN',json.dumps(record),flush=True)

    train_epochs(args.epochs,'masked_reconstruction')
    reinforcement=[]
    for iteration in range(args.self_rounds):
        # Keep the current learner, including its own discoveries; teacher search
        # stays entirely inside the training split and its cost is recorded.
        student=MaskedJointRepair(state={k:v.detach().numpy() for k,v in model.state_dict().items()})
        extra,audit=collect(rows,min(4,args.blocks),student)
        torch.save(dict(groups=extra,audit=audit),out/f'self_data_{iteration}.pt')
        groups.extend(extra)
        for param in optimizer.param_groups:param['lr']=.0003
        train_epochs(args.self_epochs,f'self_improvement_{iteration}')
        reinforcement.extend(best_of_k_updates(model,optimizer,rows,args.best_k_updates,8817+iteration))
        metrics=validate(model,valid,out/f'checkpoint_bok_{iteration}.npz')
        history.append(dict(epoch=epoch_number,phase=f'best_of_8_{iteration}',**metrics))
        if metrics['mean_score']>best:
            best=metrics['mean_score'];best_epoch=f'best_of_8_{iteration}';export_model(out/'repair.npz',model)
    write_json(out/'training.json',dict(history=history,best_epoch=best_epoch,best_validation_score=best,
               parameters=sum(p.numel() for p in model.parameters()),groups=len(groups),
               cases=sum(len(g['cases']) for g in groups),training_states=len(rows),validation_states=len(valid),
               reinforcement=reinforcement,checkpoint_sha256=sha(out/'repair.npz')))
    print('TRAINING_COMPLETE',str(out/'repair.npz'),best_epoch,best,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--blocks',type=int,default=8);parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--self-rounds',type=int,default=2);parser.add_argument('--self-epochs',type=int,default=10)
    parser.add_argument('--best-k-updates',type=int,default=36)
    main(parser.parse_args())

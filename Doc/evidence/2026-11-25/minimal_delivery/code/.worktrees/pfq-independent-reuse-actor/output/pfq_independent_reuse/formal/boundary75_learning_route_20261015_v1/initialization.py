"""Fixed-state, equal-candidate-budget initialization ablation; no native claims."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training.stageb_hgraph.directed_reuse import DirectedReuseCoordinator,build_directed_features,choose_blank
from output.pfq_independent_reuse.formal.boundary75_load_sweep_20261014_v1.policy import two
from output.pfq_independent_reuse.formal.boundary75_service_capacity_20261014_v1.parity_phy import VerifiedPhy

OUT=Path(__file__).resolve().parent
DIST=OUT.parent/'boundary75_distillation_20261014_v1'
old,p,c,prior=two.old,two.p,two.c,two.prior

def search(root,phy,obs,weights,floor,budget=8192):
    chosen=root.copy();pred=phy.evaluate(chosen)
    score=float((pred['expected'][0]*weights).sum())
    assert float(pred['dequeue'][0].sum())+1e-8>=floor
    expected=float(pred['expected'][0].sum());dequeue=float(pred['dequeue'][0].sum())
    used=1;history=[dict(evaluations=used,score=score,expected=expected,round=0)]
    for number in range(1,257):
        if used>=budget:break
        actions,_=two.candidates(chosen,chosen,obs,phy.snapshot['queue'],True)
        # Identical native candidate order; last partial neighborhood is explicit.
        full=len(actions);actions=actions[:budget-used]
        pred=two.evaluate(phy,actions);used+=len(actions)
        scores=(pred['expected']*weights[None]).sum(1)
        eligible=pred['dequeue'].sum(1)>=floor
        best=int(np.argmax(np.where(eligible,scores,-np.inf)))
        gain=float(scores[best]-score);accept=gain>(1. if number<=2 else 16.)
        if accept:
            chosen=actions[best];score=float(scores[best])
            expected=float(pred['expected'][best].sum());dequeue=float(pred['dequeue'][best].sum())
        assert dequeue+1e-8>=floor
        history.append(dict(evaluations=used,score=score,expected=expected,round=number,
                            accepted=accept,partial=len(actions)<full))
        if len(actions)<full or (number>=3 and not accept):break
    assert used<=budget and all(b['score']>=a['score'] for a,b in zip(history,history[1:]))
    return chosen,history

def main():
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    out=OUT/'initialization';out.mkdir(exist_ok=False)
    frozen=json.loads((OUT.parent/'boundary75_teacher_weight_20261014_v1/frozen_teacher.json').read_text())
    for path,digest in frozen['sha256'].items():assert c.sha(OUT.parents[3]/path)==digest,path
    actor,_=old._load_actor_checkpoint(old.CHECKPOINT,torch.device('cpu'));actor.eval()
    coordinator=DirectedReuseCoordinator();coordinator.load_state_dict(torch.load(prior.CHECKPOINT,map_location='cpu',weights_only=False)['model']);coordinator.eval()
    rows=torch.load(DIST/'labelled_states.pt',map_location='cpu',weights_only=False)
    rows=sorted([r for r in rows if r['seed']==306 and r['load']==1.2],key=lambda r:r['tti'])[::2]
    assert len(rows)==32
    table=[];histories=[];started=time.monotonic()
    for index,row in enumerate(rows):
        phy=VerifiedPhy(row['ctbs'].config);phy.prepare(row['snapshot'],row['ctbs']);phy.ctbs=row['ctbs']
        with torch.no_grad():
            _,d=old.select_independent_action(actor,row['graph'],row['packed'],mode='greedy',ctbs_headroom_inputs=row['ctbs'])
        draft=p.decode_type0_teacher_action(d.action_ue_select.numpy(),d.action_prg_alloc.numpy(),n_cell=3,n_prg=17).reshape(3,17)
        features=build_directed_features(draft,row['ctbs'],row['snapshot']['queue'],phy.bank,phy.received,row['snapshot']['noise'],row['obs']['obs_ue_features'],phy.tbs)
        with torch.no_grad():scores=coordinator(**{k:torch.from_numpy(v)[None] for k,v in features.items()})[0].numpy()
        learned,_=choose_blank(draft,scores,features['valid'])
        obs=row['obs'];pfq=p.decode_type0_teacher_action(obs['teacher_action_ue_select'],obs['teacher_action_prg_alloc'],n_cell=3,n_prg=17).reshape(3,17)
        pred=phy.evaluate(np.stack([learned,pfq]));floor=float(pred['dequeue'].sum(1).min())
        for name,root in [('learned',learned),('pfq',pfq)]:
            prior.encode(root,obs)
            unchanged,h0=search(root,phy,obs,row['teacher_weights'],floor,1)
            np.testing.assert_array_equal(root,unchanged);assert len(h0)==1
            chosen,h=search(root,phy,obs,row['teacher_weights'],floor)
            prior.encode(chosen,obs)
            table.append(dict(tti=row['tti'],arm=name,floor=floor,initial_score=h[0]['score'],final_score=h[-1]['score'],
                              final_expected=h[-1]['expected'],evaluations=h[-1]['evaluations'],rounds=h[-1]['round'],
                              blank_pct=float((chosen<0).mean()*100)))
            histories.extend(dict(tti=row['tti'],arm=name,**x) for x in h)
        print(json.dumps(dict(stage='initialization',state=index+1,total=32,elapsed_s=time.monotonic()-started)),flush=True)
    df=pd.DataFrame(table);hs=pd.DataFrame(histories)
    df.to_csv(out/'paired.csv',index=False);hs.to_csv(out/'search_history.csv',index=False)
    wide=df.pivot(index='tti',columns='arm',values='final_score');delta=wide.learned-wide.pfq
    target_rows=[]
    for tti in wide.index:
        target=float(wide.loc[tti].min())
        counts={arm:int(hs[(hs.tti==tti)&(hs.arm==arm)&(hs.score>=target-1e-8)].evaluations.min()) for arm in ['learned','pfq']}
        target_rows.append(dict(tti=tti,target=target,**counts))
    targets=pd.DataFrame(target_rows);targets.to_csv(out/'common_target.csv',index=False)
    result=dict(passed_validity=True,states=32,budget=8192,mean_scores=df.groupby('arm').final_score.mean().to_dict(),
                learned_mean_delta_bytes=float(delta.mean()),learned_relative_mean_delta_pct=float(100*delta.mean()/wide.pfq.mean()),
                learned_win_states=int((delta>1).sum()),pfq_win_states=int((delta<-1).sum()),
                common_target_mean_evaluations=targets[['learned','pfq']].mean().to_dict(),
                limits='Paired correlated states on one archived trajectory, shared min-root dequeue floor; model-objective evidence only. Root inference cost excluded from candidate-count comparison.',
                source_labels_sha256=c.sha(DIST/'labelled_states.pt'),source_sha256=c.sha(__file__))
    two.write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

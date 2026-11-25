"""Bounded same-PRG search. Inputs precede execution; no realized outcomes."""
import itertools
import numpy as np

def choose_candidates(base,phy,obs,tti):
    base=np.asarray(base).reshape(3,17)
    base_value=float(phy.evaluate(base)['expected'].sum())
    best=base.copy();best_value=base_value;best_prg=-1;rows=[]
    legal=np.asarray(obs['action_mask_cell_ue'],bool)&np.asarray(obs['action_mask_ue'],bool)[None,:]
    queue=phy.snapshot['queue']
    # Rank each position with a whole-TB calculation, not additive PRG rates.
    for prg in range(17):
        choices=[]
        for cell in range(3):
            if base[cell,prg]<0:
                choices.append([-1]);continue
            ids=np.flatnonzero(legal[cell]&(queue>0)).tolist()
            options=np.repeat(base[None],len(ids),axis=0)
            options[:,cell,prg]=ids
            scores=phy.evaluate(options)['expected'].sum(1)
            order=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))[:3]
            shortlist=[int(base[cell,prg])]+[ids[i] for i in order]
            choices.append([-1]+sorted(set(shortlist)))
        combos=np.asarray(list(itertools.product(*choices)),dtype=np.int64)
        actions=np.repeat(base[None],len(combos),axis=0);actions[:,:,prg]=combos
        pred=phy.evaluate(actions);values=pred['expected'].sum(1)
        index=int(np.argmax(values));value=float(values[index])
        for i in range(len(values)):
            rows.append(dict(prg=prg,u0=int(combos[i,0]),u1=int(combos[i,1]),u2=int(combos[i,2]),expected_bytes=float(values[i]),gain_vs_old=float(values[i]-base_value)))
        if value>best_value+1.e-8:
            best=actions[index].copy();best_value=value;best_prg=prg
    fill=best.copy();muted=(base>=0)&(best<0)
    fill[muted]=base[muted]
    shuffle=fill.copy();rng=np.random.default_rng(20261010+tti)
    for cell in range(3):
        count=int(muted[cell].sum())
        if count:
            eligible=np.flatnonzero((fill[cell]>=0)&~muted[cell])
            assert len(eligible)>=count
            shuffle[cell,rng.choice(eligible,count,replace=False)]=-1
    assert np.array_equal((shuffle<0).sum(1),(best<0).sum(1))
    assert np.all((best>=0)<= (base>=0))
    arms=dict(old=base,joint=best,fill=fill,shuffle=shuffle)
    summary=dict(tti=tti,prg=best_prg,candidate_count=len(rows),changed_positions=int((best!=base).sum()),new_mutes=int(muted.sum()),approximate_best_gain_bytes=best_value-base_value)
    predictions={}
    for name,a in arms.items():
        for cell in range(3):
            ids=a[cell,a[cell]>=0];assert legal[cell,ids].all()
        predictions[name]={k:v[0] for k,v in phy.evaluate(a).items()}
    diff=predictions['joint']['expected']-predictions['fill']['expected']
    source_cells=np.flatnonzero(muted.any(1));source=np.isin(phy.serving,source_cells)
    summary.update(muted_source_cells=source_cells.tolist(),blank_source_expected_delta=float(diff[source].sum()),blank_neighbor_expected_delta=float(diff[~source].sum()),blank_net_expected_delta=float(diff.sum()),joint_ue_swaps=int(((best>=0)&(base>=0)&(best!=base)).sum()))
    return arms,predictions,rows,summary

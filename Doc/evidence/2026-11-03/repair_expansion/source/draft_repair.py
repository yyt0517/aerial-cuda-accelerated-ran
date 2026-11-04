"""Experimental sparse-draft search with exact verification of a fixed quota.

Unverified prefixes may remain in the beam; only exact evaluator results can
update the accepted schedule. This changes the search tree and is not lossless.
"""
import numpy as np
from .cached_repair import CachedDeadlineRepair
from .neural_repair import destroy
from .prefix_repair import shortlist
from .residual import DraftCache,ResidualValue
from .deadline import packet_objective


class RebasedDraft:
    """Reuse graph geometry and refresh root-dependent ratios after acceptance."""
    def __init__(self):self.cache=None

    def for_base(self,graph,base):
        old=self.cache
        if old is None or old.graph is not graph or (old.base is not base and graph.problem.sinr_hint is None):
            if old is not None:old.close()
            self.cache=DraftCache(graph,base)
        elif old.base is not base:
            if base.owner is not old.base.owner:raise ValueError('Foreign rebase')
            if old.lib.draft_root(old.context,base.actions,base.values):raise RuntimeError('Draft rebase failed')
            old.base=base
        return self.cache

    def close(self):
        if self.cache is not None:self.cache.close();self.cache=None


class DraftRepair(CachedDeadlineRepair):
    def __init__(self,checkpoint,*,mode='residual',residual_checkpoint=None,width=16,prefix_budget=48,exact_quota=8):
        super().__init__(checkpoint,width)
        if mode not in ('trace','spatial','residual'):raise ValueError('Unknown draft mode')
        if not 1<=exact_quota<=prefix_budget or prefix_budget<width:raise ValueError('Invalid verification quota')
        if mode=='residual' and residual_checkpoint is None:raise ValueError('Residual weights required')
        self.mode=mode;self.residual=ResidualValue(residual_checkpoint) if mode=='residual' else None
        self.prefix_budget=prefix_budget;self.exact_quota=exact_quota
        self.drafts=RebasedDraft()
        self.draft_candidates=self.verified_candidates=self.prefix_batches=0

    def propose(self,*args):raise RuntimeError('DraftRepair needs search verification callbacks')

    def repair(self,graph,base,block,evaluate):
        p=graph.problem;block=tuple(block)
        if not block or len(set(block))!=len(block):raise ValueError('Distinct block tokens required')
        cache=self.cache_for(graph,base);draft=self.drafts.for_base(graph,base)
        root=destroy(base.actions[0],block)
        assigned=root[p.serving,:]==np.arange(p.U)[:,None]
        count=assigned.sum(-1)[None];layer=np.where(assigned,p.caps,5).min(-1)[None]
        layer=np.where(count>0,layer,4);active=(root>=0)[None]
        edits=np.full((1,len(block)),-1,np.int64);logp=np.zeros(1)
        objective=packet_objective(p);root_r=float(objective(base.values)[0])
        self.calls+=1;self.prefix_initializations+=1
        for depth,(cell,prg) in enumerate(block):
            choices=cache.node(cell,prg)[0]
            logits=self.logits(cache.features(block,depth,count,layer,active))
            self.option_scores+=logits.size;self.logit_calls+=1
            logits-=logits.max(-1,keepdims=True);logits-=np.log(np.exp(logits).sum(-1,keepdims=True))
            joint=logp[:,None]+logits
            indices=shortlist(joint,choices,int(base.actions[0,cell,prg]),self.width,self.prefix_budget,2)
            parent,value=np.divmod(indices,len(choices));candidate_logp=joint.ravel()[indices]
            prefix=edits[parent].copy();prefix[:,depth]=choices[value]
            complete=np.broadcast_to(base.actions[0],(len(prefix),p.C,p.P)).copy()
            for step,(c,pr) in enumerate(block[:depth+1]):complete[:,c,pr]=prefix[:,step]
            approximate=draft.evaluate(complete);self.draft_candidates+=len(complete)
            score=(self.residual.score(approximate)[0] if self.mode=='residual'
                   else approximate['scores'][:,int(self.mode=='spatial')]).copy()
            order=np.lexsort((np.arange(len(score)),-candidate_logp,-score))
            verify=order[:self.exact_quota]
            result=evaluate(complete[verify],base)
            if result is None:break
            size=len(result.actions);verify=verify[:size]
            self.proposals+=size;self.verified_candidates+=size;self.prefix_batches+=1
            score[verify]=objective(result.values)-root_r
            keep=np.lexsort((np.arange(len(score)),-candidate_logp,-score))[:self.width]
            previous=parent[keep];pick=choices[value[keep]]
            edits=prefix[keep];count,layer,active=count[previous],layer[previous],active[previous]
            active[:,cell,prg]=pick>=0;rows=np.flatnonzero(pick>=0);users=pick[rows]
            count[rows,users]+=1;layer[rows,users]=np.minimum(layer[rows,users],p.caps[users,prg])
            logp=candidate_logp[keep]

    def close(self):self.drafts.close()

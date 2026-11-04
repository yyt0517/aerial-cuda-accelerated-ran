"""Decision-local equivalent pool/proxy cache and paid-archive terminal choice."""
import time
import numpy as np
from .directed_reorder import DirectedConfig,DirectedFeatures,target_ues,choose_batch
from .weak_service import ServiceMetric,FrontierCache
from .joint_reorder import JointFeatures
from .resource_conditions import ConditionalPhy,satisfies,seed_action,DecodeInfeasible
from .resource_conditions_fast import ordered_unique
from .physics import DEQUEUE,EXPECTED
from .search import ProposalGraph
from .tb_macro import ExactMemo
from .no_ctbs_inputs import raw_pfq_floor
from .terminal_risk import select_terminal,RISK_CAP


class CachedFeatures(JointFeatures):
    """Lifetime is one immutable decision. Proxy key includes the full root.

    Pool uses only its local root-hash RNG; periodic seed_action RNG calls
    remain in the solver. Cache lazy proxy rows so unused pool rows stay free.
    """
    def __init__(self,*args):
        super().__init__(*args);self.pools={};self.proxies={}
        self.pool_hits=self.pool_misses=self.proxy_hits=self.proxy_computed=0

    def pool(self,base,wide,seed):
        key=(base.actions[0].tobytes(),wide,seed)
        if key not in self.pools:
            self.pool_misses+=1;self.pools[key]=super().pool(base,wide,seed)
        else:self.pool_hits+=1
        return self.pools[key]

    def proxy(self,base,actions):
        root=base.actions[0].tobytes();keys=[a.tobytes() for a in actions]
        if root not in self.proxies:
            self.proxies[root]=({},np.empty(0),np.empty((0,self.p.U)),np.empty(0,bool))
        ids,ss,ee,vv=self.proxies[root]
        missing=[];newkeys=set()
        for i,k in enumerate(keys):
            if k not in ids and k not in newkeys:missing.append(i);newkeys.add(k)
        if missing:
            s,e,v=super().proxy(base,actions[missing]);start=len(ids)
            for j,i in enumerate(missing):ids[keys[i]]=start+j
            ss=np.concatenate([ss,s]);ee=np.concatenate([ee,e]);vv=np.concatenate([vv,v])
            self.proxies[root]=(ids,ss,ee,vv)
        self.proxy_computed+=len(missing);self.proxy_hits+=len(actions)-len(missing)
        indexes=np.fromiter((ids[k] for k in keys),np.int64,count=len(keys))
        return ss[indexes],ee[indexes],vv[indexes]

    def cache_stats(self):
        # ndarray storage plus action-key payload; Python container overhead excluded.
        size=sum(v.nbytes+len(k[0]) for k,v in self.pools.items())
        size+=sum(sum(len(k) for k in ids)+s.nbytes+e.nbytes+v.nbytes+len(root)
            for root,(ids,s,e,v) in self.proxies.items())
        return dict(pool_cache_hits=self.pool_hits,pool_cache_misses=self.pool_misses,
            proxy_cache_hits=self.proxy_hits,proxy_computed=self.proxy_computed,cache_payload_bytes=size)


def solve_cached(p,draft,condition,config=DirectedConfig(joint_candidates=False),*,mode="maxj",model=None,offline_diagnostic=False):
    if not offline_diagnostic:raise ValueError('Compressed teacher conditions require offline diagnostic mode')
    if config.joint_candidates or config.alternate_sparse or config.directed_slots!=2:raise ValueError("Existing-pool two-slot search only")
    condition=condition.validate(p)
    if condition.counts is None or condition.activity is None:raise ValueError('Joint conditions required')
    phy=ConditionalPhy(p)
    try:
        memo=ExactMemo(phy,draft,config.max_candidates);original=memo.root
        floor=raw_pfq_floor(original.values[0]);graph=ProposalGraph(phy)
        stamp=time.perf_counter();bank=phy.bank();bank_ms=(time.perf_counter()-stamp)*1000
        features=CachedFeatures(p,bank,condition);directed_features=DirectedFeatures(features);rng=np.random.default_rng(config.seed)
        archive={};expanded={};roots=set();pending=[];dequeues={};objectives={}
        cache=FrontierCache(config.max_candidates)
        matching_ms=proxy_ms=generation_ms=service_ms=0.
        proxy_candidates=decode_failures=seed_calls=0;stop='round_limit'
        directed_queries=directed_pool_queries=multicell_queries=wide_rounds=sparse_rounds=0;directed_generation_ms=0.

        def store(exact):
            if exact is None:return
            for j in np.flatnonzero(satisfies(p,exact.actions,condition)):
                key=exact.actions[j].tobytes()
                if key not in archive:
                    archive[key]=phy.take(exact,int(j));pending.append(key)
                    dequeues[key]=float(exact.values[j,:,DEQUEUE].sum())

        store(original);seeds=[];seen=set(memo.rows);stamp=time.perf_counter()
        for k in range(config.seed_candidates):
            try:a=seed_action(p,original,bank,graph,condition,rng,k);seed_calls+=1
            except DecodeInfeasible:decode_failures+=1;continue
            if a.tobytes() not in seen:seeds.append(a);seen.add(a.tobytes())
        matching_ms+=(time.perf_counter()-stamp)*1000
        if seeds:exact,_=memo.evaluate(np.stack(seeds),original);store(exact)
        bootstrap=np.zeros(p.U,bool)
        for value in archive.values():bootstrap|=value.values[0,:,EXPECTED]>1.
        metric=ServiceMetric(p,condition,bootstrap,config);metric.weights=p.weights;group=metric.group

        def refresh():
            nonlocal service_ms
            if not pending:return
            stamp=time.perf_counter()
            expected=np.stack([archive[k].values[0,:,EXPECTED] for k in pending])
            services=metric.expected(expected)
            for k,s in zip(pending,services):
                # Retain the original 1-D dot operation: batched BLAS changes
                # last-bit ties that can alter exact frontier decisions.
                j=float(archive[k].values[0,:,EXPECTED]@p.weights)
                objectives[k]=np.r_[j,s]
            cache.add(pending,objectives);pending.clear()
            service_ms+=(time.perf_counter()-stamp)*1000

        refresh();iterations=0;frontier_peak=0
        for iteration in range(config.rounds):
            if memo.remaining<=0:stop='physical_budget';break
            candidates=sorted(archive,key=lambda k:(int(dequeues[k]>=floor),objectives[k][0],dequeues[k]),reverse=True)
            if not candidates:stop='no_conditional_seed';break
            available=[k for k in candidates if expanded.get(k,0)>=0]
            stamp=time.perf_counter()
            if config.pareto:
                usable,peak=cache.roots(available,objectives,config.beam_width);frontier_peak=max(frontier_peak,peak)
            else:usable=available[:config.beam_width]
            service_ms+=(time.perf_counter()-stamp)*1000
            if not usable:stop='finite_pool_stalled';break
            key=usable[0] if iteration%2==0 else min(usable,key=lambda k:expanded.get(k,0));base=archive[key]
            if key not in roots and len(roots)>=config.max_feature_roots:stop='feature_budget';break
            roots.add(key);expanded[key]=expanded.get(key,0)+1;iterations+=1
            stamp=time.perf_counter();targets=target_ues(p,base.values[0],group,config.target_count)
            macro=directed_features.pool(base,targets,config.chain_depth) if config.joint_candidates else np.empty((0,p.C,p.P),np.int64)
            macro_keys={a.tobytes() for a in macro};directed_generation_ms+=(time.perf_counter()-stamp)*1000
            stamp=time.perf_counter()
            if config.alternate_sparse and iteration%2 and len(macro):actions=macro;sparse_rounds+=1
            else:
                actions=features.pool(base,config.wide,config.seed);wide_rounds+=1
                if len(macro):actions=np.concatenate([actions,macro])
            generation_ms+=(time.perf_counter()-stamp)*1000
            regenerated=[];stamp=time.perf_counter()
            if iteration%4==0:
                for j in range(2):
                    try:regenerated.append(seed_action(p,base,bank,graph,condition,rng,3+(iteration+j)%9));seed_calls+=1
                    except DecodeInfeasible:decode_failures+=1
            matching_ms+=(time.perf_counter()-stamp)*1000
            if regenerated:actions=np.concatenate([actions,np.stack(regenerated)])
            if len(actions):actions=ordered_unique(actions);actions=actions[memo.unseen(actions)]
            if not len(actions):expanded[key]=-1;continue
            stamp=time.perf_counter();scores,predicted,valid=features.proxy(base,actions);proxy_candidates+=len(actions)
            selected,dedicated=choose_batch(scores,predicted,valid,base,metric,targets,memo.remaining,config,iteration)
            directed_queries+=len(dedicated)
            directed_pool_queries+=sum(actions[j].tobytes() in macro_keys for j in selected)
            multicell_queries+=sum(int(np.any(actions[j]!=base.actions[0],axis=1).sum()>1) for j in selected)
            proxy_ms+=(time.perf_counter()-stamp)*1000
            if not selected:expanded[key]=-1;continue
            exact,_=memo.evaluate(actions[np.asarray(selected,np.int64)],base);store(exact);refresh()
        terminal_start=time.perf_counter();terminal_stats={}
        feasible=[k for k in archive if dequeues[k]>=floor]
        if feasible:
            selected_key,terminal_stats=select_terminal(p,condition,archive,feasible,objectives,mode,model)
            best=archive[selected_key]
            jbest=max(objectives[k][0] for k in feasible)
        else:best=original;jbest=float(original.values[0,:,EXPECTED]@p.weights)
        v=best.values[0];score=float(v[:,EXPECTED]@p.weights)
        assert phy.stats['candidate_evaluations']<=config.max_candidates and len(roots)<=config.max_feature_roots
        assert v[:,DEQUEUE].sum()>=floor and score+1.e-8>=(1-(0. if mode=="maxj" else RISK_CAP))*jbest
        if feasible:assert satisfies(p,best.actions,condition)[0]
        terminal_ms=(time.perf_counter()-terminal_start)*1000
        r=dict(terminal_ms=terminal_ms,terminal_mode=mode,**terminal_stats,**features.cache_stats(),score=score,floor=floor,best_observed_j=jbest,selected_j_loss_fraction=(jbest-score)/max(jbest,1),
            condition_feasible=bool(feasible),condition_satisfied=bool(satisfies(p,best.actions,condition)[0]),
            fallback_to_raw=not bool(feasible),feature_roots=len(roots),iterations=iterations,proxy_candidates=proxy_candidates,
            proposal_attempts=features.attempts,proxy_gamma_array_reads=features.gamma_reads,seed_gamma_array_reads=seed_calls*p.U*p.P,
            service_group_size=int(group.sum()),bootstrap_group_size=int(bootstrap.sum()),frontier_peak=frontier_peak,
            bank_ms=bank_ms,matching_ms=matching_ms,generation_ms=generation_ms,proxy_ms=proxy_ms,service_ms=service_ms,
            directed_queries=directed_queries,directed_pool_queries=directed_pool_queries,multicell_queries=multicell_queries,
            directed_generation_ms=directed_generation_ms,directed_candidates=directed_features.candidates,
            directed_attempts=directed_features.attempts,directed_gamma_reads=directed_features.gamma_reads,
            directed_swap_steps=directed_features.swap_steps,directed_combinations=directed_features.combinations,
            directed_multicell_candidates=directed_features.multicell,wide_rounds=wide_rounds,sparse_rounds=sparse_rounds,
            service_metric=config.metric,service_rows=metric.rows,dominance_pairs=cache.comparisons,frontier_queries=cache.queries,
            decode_failures=decode_failures,stop_reason=stop,offline_diagnostic=True,full_search_calls=0,**phy.stats)
        return best.actions[0].copy(),v.copy(),r,group.copy()
    finally:phy.close()


"""MMSE-free integer bounds and a tie-safe lazy canonical selector.

The capacity bound is valid for the frozen finite-arithmetic PHY: its BLER
lookup stays in [0,1], and positive floating multiplication/nearest-even
rounding are monotone. The interference-added tightening is a separate
exact-arithmetic model diagnostic, not an enabled online floating certificate.
No PHY evaluator is used here.
"""
import numpy as np
from .physics import COUNT,LAYER,DEQUEUE,EXPECTED,curves,native_bler
from .dpls_operations import edit_metadata

SCALE=1_000_000


def activity(root):
    return np.sum((root>=0).astype(np.uint64)*np.left_shift(np.uint64(1),
        np.arange(root.shape[0],dtype=np.uint64))[:,None],axis=0,dtype=np.uint64)


def logical_bounds(p,root,values,prg,assignment):
    """Dense offline implementation; online integration should use sparse edits.

    Full-UE arrays here calculate cheap count/layer/TBS, never MMSE or BLER of
    a counterfactual. Root values must already be exact, charged observations.
    ``capacity_q`` and ``theory_added_q`` are bounds on integer *delta* J.
    """
    root=np.asarray(root);prg=np.asarray(prg);assignment=np.asarray(assignment)
    u=np.arange(p.U)[:,None];selected=root[p.serving]==u
    counts=selected.sum(1)
    np.testing.assert_array_equal(counts,values[:,COUNT])
    layers=np.where(selected,p.caps,5)
    prefix=np.minimum.accumulate(np.pad(layers,((0,0),(1,0)),constant_values=5),axis=1)[:,:p.P]
    suffix=np.minimum.accumulate(np.pad(layers,((0,0),(0,1)),constant_values=5)[:,::-1],axis=1)[:,::-1][:,1:]
    excluding=np.minimum(prefix,suffix)
    was=root[p.serving[:,None],prg[None,:]]==u
    now=assignment[:,p.serving].T==u
    new_count=counts[:,None]-was+now
    new_layer=np.minimum(excluding[:,prg],np.where(now,p.caps[:,prg],5))
    new_layer=np.where(new_count>0,new_layer,0)
    capacity=p.tbs[u,new_count,new_layer]
    dequeue=np.minimum(p.queue[:,None],capacity)
    root_q=np.rint((values[:,EXPECTED]*p.weights)*SCALE).astype(np.int64)
    cap_q=np.rint((dequeue*p.weights[:,None])*SCALE).astype(np.int64)
    keys,affected,footprint=edit_metadata(root,prg,assignment,p.U)
    support=((affected[None,:]>>np.arange(p.U,dtype=np.uint64)[:,None])&1).astype(bool)
    cap_q=np.where(support,cap_q,root_q[:,None])
    upper=(cap_q-root_q[:,None]).sum(0,dtype=np.int64)
    oldmask=activity(root)[prg]
    newmask=activity(assignment.T)
    # One-PRG edits: was==now means this victim's entire assignment/layers
    # remain fixed. Superset activity can only add PSD interference.
    fixed=(was==now)&support
    add_only=(oldmask & ~newmask)==0
    tighter=np.where(fixed&add_only[None,:],root_q[:,None],cap_q)
    return dict(keys=keys,affected=affected,footprint=footprint,
        capacity_q=upper,theory_added_q=(tighter-root_q[:,None]).sum(0,dtype=np.int64),
        delta_d=dequeue.sum(0).astype(np.int64)-int(values[:,DEQUEUE].sum()),
        monotone_victims=(fixed&add_only[None,:]).sum(0))


def curve_structure():
    """Audit the native descending-scan branches, not sorted interpolation.

    A reachable branch j has domain (max(previous cutoffs,x[-1]),x[j-1]].
    Its endpoints lie within [x[j],x[j-1]], so it is interpolation, not unsafe
    extrapolation. Ordered y values make slopes <=0 and jumps <=0 even for
    MCS3's out-of-order SNR entries. These are exact real-arithmetic checks on
    stored finite coefficients; MMSE rounding is a separate issue.
    """
    records=[]
    for m,(x,y) in enumerate(curves()):
        low=float(x[-1]);segments=[]
        for j in range(len(x)-1,0,-1):
            high=float(min(x[j-1],x[0]))
            if high>low:
                assert x[j-1]>x[j] and low>=x[j]
                slope=float((y[j-1]-y[j])/(x[j-1]-x[j]))
                assert slope<=0
                left=float(y[j]+(low-x[j])/(x[j-1]-x[j])*(y[j-1]-y[j]))
                right=float(y[j]+(high-x[j])/(x[j-1]-x[j])*(y[j-1]-y[j]))
                segments.append((low,high,left,right))
            low=max(low,float(x[j-1]))
        previous=1.
        for low,high,left,right in segments:
            assert 0<=right<=left<=previous<=1
            previous=right
        assert np.all((y>=0)&(y<=1))
        assert np.all(np.diff(y)>=0)
        # Exhaust all cutoff intervals plus one-sided floating neighbours as
        # a lookup-only check against the actual Python native-scan replica.
        points=np.unique(np.r_[x,-4.75,x.max()+1])
        probe=np.unique(np.r_[points,np.nextafter(points,-np.inf),np.nextafter(points,np.inf),
            (points[:-1]+points[1:])/2])
        observed=native_bler(probe,np.full(probe.shape,m))
        assert np.all((observed>=0)&(observed<=1))
        assert np.max(np.diff(observed),initial=0)<=1.e-12
        records.append(dict(mcs=m,SNR_axis_strictly_descending=bool(np.all(np.diff(x)<0)),
            reachable_linear_segments=len(segments),BLER_real_monotone=True,BLER_range_0_1=True,
            lookup_probes=len(probe)))
    return records


def lazy_canonical(keys,upper,eligible,threshold,evaluate,*,priority=None,known=None):
    """Canonical argmax with max-anchored one-integer-unit ties.

    ``evaluate(i)`` returns an exact integer delta or raises. ``known`` contains
    only CURRENT exact values (stale values may be priority, never known).
    Metadata scanning is deliberate: this is a correctness reference, not a
    production heap implementation or a speed claim.
    """
    keys=np.asarray(keys);upper=np.asarray(upper);eligible=np.asarray(eligible,dtype=bool)
    if not np.issubdtype(upper.dtype,np.integer):raise TypeError('Integer upper bounds required')
    if len(np.unique(keys))!=len(keys):raise ValueError('Duplicate canonical keys')
    if not (keys.shape==upper.shape==eligible.shape):raise ValueError('Mismatched metadata')
    order=upper if priority is None else np.asarray(priority)
    if order.shape!=keys.shape or not np.isfinite(order).all():raise ValueError('Invalid priority')
    exact={} if known is None else dict(known)
    def check(i,v):
        if not isinstance(i,(int,np.integer)) or not 0<=i<len(keys):raise IndexError('Invalid exact-label index')
        if not isinstance(v,(int,np.integer)):raise TypeError('Exact integer label required')
        if v>int(upper[i]):raise AssertionError('Invalid upper bound; refuse certificate')
    for i,v in exact.items():check(i,v)
    paid=[]
    while True:
        pool=[i for i,v in exact.items() if eligible[i] and v>threshold]
        unseen=np.array([i for i in range(len(keys)) if i not in exact and eligible[i] and upper[i]>threshold],dtype=int)
        winner=-1
        if pool:
            maximum=max(exact[i] for i in pool)
            winner=min((i for i in pool if exact[i]>=maximum-1),key=lambda i:keys[i])
            # UB==maximum-1 still matters when its canonical key is smaller.
            unseen=unseen[(upper[unseen]>=maximum)|((upper[unseen]==maximum-1)&(keys[unseen]<keys[winner]))]
        if not len(unseen):return winner,paid
        i=int(unseen[np.lexsort((keys[unseen],-order[unseen]))[0]])
        v=evaluate(i);check(i,v);exact[i]=int(v);paid.append(i)

"""Research-only depth sweep, preserving the frozen first and second rounds."""
import numpy as np
from output.pfq_independent_reuse.formal.boundary75_two_round_search_20261014_v1 import run as two


def select(draft,keep,phy,obs,ctbs,depth):
    limit=256 if depth=='converged16' else int(depth)
    threshold=16. if depth=='converged16' else 1.
    stop_audit={}
    rounds=[]
    if limit==1:
        d=two.materialize(draft,keep,phy,obs,ctbs)
        chosen=d['base_action']
        i=next(i for i,a in enumerate(d['actions']) if np.array_equal(a,chosen))
        prediction={k:v[i] for k,v in d['pred'].items()}
        score=float(prediction['expected'].sum())
        record=dict(keep_dequeue=float(d['pred']['dequeue'][0].sum()),
            keep_expected=float(d['pred']['expected'][0].sum()),first_expected=score,
            second_expected=score,second_gain=0.,second_accepted=False,
            first_candidates=d['full_candidates'],second_candidates=0)
    else:
        _,chosen,prediction,record=two.two_round(draft,keep,phy,obs,ctbs)
        score=record['second_expected']
        rounds.append(dict(round=2,expected=score,gain=record['second_gain'],
            accepted=record['second_accepted'],candidates=record['second_candidates']))
    rounds.insert(0,dict(round=1,expected=record['first_expected'],
        gain=record['first_expected']-record['keep_expected'],accepted=True,
        candidates=record['first_candidates']))
    converged=limit==2 and not record['second_accepted']
    for number in range(3,limit+1):
        if converged:break
        actions,meta=two.candidates(chosen,chosen,obs,phy.snapshot['queue'],True)
        pred=two.evaluate(phy,actions)
        scores=pred['expected'].sum(1);dequeues=pred['dequeue'].sum(1)
        assert np.isfinite(scores).all()
        for k in ['count','layer','mcs','capacity','dequeue']:
            np.testing.assert_array_equal(pred[k][0],prediction[k])
        np.testing.assert_allclose(pred['sinr_db'][0],prediction['sinr_db'],rtol=0,atol=1e-10)
        delta=pred['bler'][0]-prediction['bler'];mcs3=pred['mcs'][0]==3
        assert np.abs(delta[~mcs3]).max(initial=0)<1e-10
        assert np.abs(delta[mcs3]).max(initial=0)<8.1e-5
        eligible=dequeues>=record['keep_dequeue'];assert eligible[0]
        best=int(np.argmax(np.where(eligible,scores,-np.inf)))
        gain=float(scores[best]-score);accepted=gain>threshold
        rounds.append(dict(round=number,expected=float(scores[best]) if accepted else score,
            gain=gain if accepted else 0.,accepted=accepted,candidates=len(actions)))
        if accepted:
            chosen=actions[best];prediction={k:v[best] for k,v in pred.items()};score=float(scores[best])
        else:
            converged=True
            active=pred['count']>0
            native_bler=np.zeros_like(pred['bler'])
            native_bler[active]=two.native_bler(pred['sinr_db'][active],pred['mcs'][active])
            native_scores=(pred['dequeue']*(1-native_bler)).sum(1)
            stop_audit=dict(stop_model_gain_bytes=gain,
                stop_native_curve_gain_bytes=float(np.max(native_scores[eligible])-native_scores[0]),
                stop_model_curve_max_error_bytes=float(np.abs(scores-native_scores).max()))
    record.update(chosen_expected=score,chosen_dequeue=float(prediction['dequeue'].sum()),
        predicted_failed=float((prediction['dequeue']*prediction['bler']).sum()),
        rounds_evaluated=len(rounds),converged=converged,
        cap_reached=depth=='converged16' and not converged,**stop_audit,
        extra_blanks=int(((draft>=0)&(chosen<0)).sum()),changed_from_draft=int((chosen!=draft).sum()))
    assert record['chosen_dequeue']>=record['keep_dequeue']
    assert record['extra_blanks']<=limit and record['changed_from_draft']<=2*limit
    return chosen,prediction,record,rounds

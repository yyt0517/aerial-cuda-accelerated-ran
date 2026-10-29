def select(draft,keep,phy,obs,ctbs,weights):
    assert weights.shape==(36,) and np.isfinite(weights).all() and (weights>0).all()
    chosen=keep.copy()
    prediction=None
    score=None
    keep_dequeue=None
    rows=[]
    for number in range(1,257):
        actions,_=two.candidates(draft if number==1 else chosen,keep if number==1 else chosen,
            obs,phy.snapshot['queue'],True)
        pred=two.evaluate(phy,actions)
        scores=(pred['expected']*weights[None]).sum(1)
        dequeues=pred['dequeue'].sum(1)
        assert np.isfinite(scores).all()
        if number==1:
            score=float(scores[0])
            keep_score=score
            draft_dequeue=int(dequeues[0])
            keep_dequeue=float((RATIO_NUM*draft_dequeue+99)//100)
            prediction={k:v[0] for k,v in pred.items()}
        eligible=dequeues>=keep_dequeue
        assert eligible[0]
        best=int(np.argmax(np.where(eligible,scores,-np.inf)))
        gain=float(scores[best]-score)
        accepted=gain>(1. if number<=2 else 16.)
        if accepted:
            chosen=actions[best]
            prediction={k:v[best] for k,v in pred.items()}
            score=float(scores[best])
        rows.append(dict(round=number,weighted_expected=score,gain=gain if accepted else 0.,
            accepted=accepted,candidates=len(actions)))
        # Preserve frozen behavior: even when round 2 rejects, round 3 performs
        # the 16-byte stopping check, using the carried incumbent score.
        if number>=3 and not accepted:
            active=pred['count']>0
            bler=np.zeros_like(pred['bler'])
            bler[active]=two.native_bler(pred['sinr_db'][active],pred['mcs'][active])
            native=(pred['dequeue']*(1-bler)*weights[None]).sum(1)
            break
    assert not rows[-1]['accepted'],'256-round cap hit'
    record=dict(keep_dequeue=keep_dequeue,keep_weighted_expected=keep_score,
        chosen_dequeue=float(prediction['dequeue'].sum()),chosen_expected=float(prediction['expected'].sum()),
        chosen_weighted_expected=score,rounds_evaluated=len(rows),converged=True,cap_reached=False,
        stop_weighted_model_gain=gain,stop_weighted_native_curve_gain=float(native[eligible].max()-native[0]),
        weight_min=float(weights.min()),weight_max=float(weights.max()),weight_ue20=float(weights[20]))
    assert record['chosen_dequeue']>=keep_dequeue
    record.update(draft_dequeue=draft_dequeue, floor_ratio_num=RATIO_NUM)
    return chosen,prediction,record,rows

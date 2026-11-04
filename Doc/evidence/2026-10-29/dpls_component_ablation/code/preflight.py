"""Validate interventions and exact promoted-policy parity before the campaign."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch

from common import ARMS, OUT, PROMOTED, REFERENCE, ROOT, baseline, load_file, optimized, path_for, sha, write
from policy import Ablation


def smoke(arm):
    with (OUT / f'smoke_{arm}.log').open('x') as log:
        subprocess.run([sys.executable, '-u', str(OUT/'run.py'), '--arm', arm, '--seed', '41', '--smoke'],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    path = path_for(arm, 41, True)
    data = np.load(path/'prefix.npz')
    ref = np.load(REFERENCE/'seed41_pfq0_20000/prefix.npz')
    for key in ['actions','ue','prg','goodput','input_hashes','ctbs_hashes']:
        np.testing.assert_array_equal(data[key][:128], ref[key][:128])
    for key in ['channel_fingerprints','precoder_fingerprints','fresh_input_tti']:
        np.testing.assert_array_equal(data[key],ref[key][:136])
    if arm == 'full':
        assert (path/'metrics.bin').read_bytes() == (PROMOTED/'smoke_seed41_current_136/metrics.bin').read_bytes()
        for key in ['actions','goodput','ue','prg']:
            old = np.load(PROMOTED/'smoke_seed41_current_136/prefix.npz')
            np.testing.assert_array_equal(data[key], old[key])
    return arm


def main():
    results=[]
    arms=[arm for arm in ARMS if arm != 'pfq']
    with ThreadPoolExecutor(max_workers=7) as pool:
        for arm in pool.map(smoke,arms):
            results.append(arm)
            print('SMOKE_PASSED',arm,flush=True)
    # Non-smoke, saved production states cover different MCS/queues and phases.
    from output.pfq_independent_reuse.formal.boundary08_causal_floor_ratio_seed41_20261027_v1 import policy as current
    for seed in [41,45]:
        for state in torch.load(PROMOTED/f'seed{seed}_current_20000/states.pt',map_location='cpu',weights_only=False):
            obs,ctbs,snapshot=state['obs'],state['ctbs'],state['snapshot']
            builder=baseline.r.p.FeatureSnapshotBuilder(baseline.r.p.SnapshotBuilderConfig(tb_err_ema_alpha=.1))
            edges=baseline.r.p.SparseEdgeGenerator(baseline.r.p.EdgeGeneratorConfig(ue_prg_topk=4,ue_prg_diversity_extra=0,candidate_mode='all_demand_ues',include_pfq_score_anchor=False,include_candidate_risk_features=False))
            phy=optimized.OptimizedPhy(ctbs.config)
            arm=Ablation('full')
            draft,keep,*_=arm.policy(obs,ctbs,snapshot,builder,edges,None,None,phy)
            weights=arm.weights(state['ema'],np.full(36,2400.))
            chosen,prediction,record,rounds=arm.select(draft,keep,phy,obs,ctbs,weights)
            current.RATIO_NUM=90
            current.AUDIT.clear()
            current.DRAFTS.clear()
            original_phy=optimized.OptimizedPhy(ctbs.config)
            original_draft,original_keep,*_=current.policy(obs,ctbs,snapshot,builder,edges,None,None,original_phy)
            original,op,orecord,orounds=current.select(original_draft,original_keep,original_phy,obs,ctbs,weights)
            np.testing.assert_array_equal(chosen,original)
            assert rounds==orounds
            for key in prediction:
                np.testing.assert_array_equal(prediction[key],op[key])
            for key in ['keep_dequeue','chosen_dequeue','chosen_weighted_expected','rounds_evaluated','candidate_evaluations']:
                assert record[key]==orecord[key],key
            print('FROZEN_STATE_EQUIVALENT',seed,state['tti'],flush=True)
    write(OUT/'preflight.json',dict(passed=True,smoke_arms=results,smoke_full_native_byte_exact=True,
        promoted_saved_state_exact_comparisons=12,full_candidate_rounds_and_predictions_exact=True,
        protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':main()

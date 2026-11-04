"""Solo, rotated-order policy timing on identical live states for every arm."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import ROOT, NativeEnv, teacher, packets, write, sha
from training.stageb_ppo_v2.catalog import Catalog as OldCatalog
from training.stageb_ppo_v2.model import ServiceActor as OldActor
from .catalog import FastCatalog
from .full_catalog import FullCatalog
from .model import ServiceActor


def main(root, samples=128):
    root = Path(root).resolve(); torch.set_num_threads(1); torch.backends.mkldnn.enabled = False
    actor = ServiceActor(); actor.load_state_dict(torch.load(root/'ppo/ppo_final.pt',weights_only=False)['actor']); actor.eval()
    original_path = ROOT/'deliverables/ppo_diagnosis_continue_20261030_v1/minibatch_continuation/ppo_final.pt'
    original = OldActor(); original.load_state_dict(torch.load(original_path,weights_only=False)['actor']); original.eval()
    rows = []
    for seed in [41,42]:
        env = NativeEnv(root/f'timing_seed{seed}', seed, 4000+512+samples)
        study = teacher(); catalogs = {'fast4':FastCatalog(root/'ppo/generator.npz',4),
            'fast8':FastCatalog(root/'ppo/generator.npz',8), 'full_shared':FullCatalog(), 'full_original':OldCatalog(.98)}
        try:
            for _ in range(4000): env.step(env.raw_pfq())
            for _ in range(512):
                if seed == 41:
                    row = catalogs['fast8'].build(env); env.step(row['actions'][actor.sample(row,greedy=True)[0]])
                else: env.step(env.expert(study))
            expert_method = env.expert
            for index in range(samples):
                order = ['dpls','full_original','full_shared','fast4','fast8']
                offset = index % len(order); order = order[offset:]+order[:offset]
                actions = {}
                for position, arm in enumerate(order):
                    env.expert = expert_method
                    if arm.startswith('fast'):
                        def forbid(*args, **kwargs): raise AssertionError('Search in fast benchmark')
                        env.expert = forbid
                    start = time.perf_counter()
                    if arm == 'dpls':
                        action = env.expert(study); built = done = time.perf_counter(); count = None
                    else:
                        row = catalogs[arm].build(env); built = time.perf_counter()
                        policy = original if arm == 'full_original' else actor
                        choice = policy.sample(row,greedy=True)[0]; action = row['actions'][choice]; done = time.perf_counter()
                        count = row['audit']['candidate_evaluations']
                    actions[arm] = action
                    rows.append(dict(seed=seed, tti=env.tti+1, state_distribution='student8' if seed==41 else 'dpls',
                        arm=arm, order=position, policy_ms=(done-start)*1000, build_ms=(built-start)*1000,
                        actor_ms=(done-built)*1000, counted_catalog_evaluations=count))
                env.expert = expert_method
                env.step(actions['fast8' if seed == 41 else 'dpls'])
            env.close(); packets(env.out)
        except BaseException:
            if not env.log.closed: env.close(validate=False)
            raise
    table = pd.DataFrame(rows); table.to_csv(root/'paired_timing.csv', index=False)
    timing = {arm:dict(mean_ms=float(g.policy_ms.mean()),p50_ms=float(g.policy_ms.median()),p95_ms=float(g.policy_ms.quantile(.95)),
        mean_actor_ms=float(g.actor_ms.mean())) for arm,g in table.groupby('arm')}
    for arm in ['fast4','fast8']:
        timing[arm]['speedup_over_dpls'] = timing['dpls']['mean_ms']/timing[arm]['mean_ms']
        timing[arm]['efficiency_target_passed'] = bool(timing[arm]['mean_ms'] <= timing['dpls']['mean_ms']/3 and timing[arm]['p95_ms'] <= timing['dpls']['p95_ms']/2)
    result = dict(paired_states=2*samples, order_rotated=True, solo_benchmark=True, timing=timing,
        student_checkpoint_sha256=sha(root/'ppo/ppo_final.pt'), generator_sha256=sha(root/'ppo/generator.npz'),
        original_checkpoint_sha256=sha(original_path),
        included='Ready observation to complete selected action: feature construction, proposal generation, all validation, policy selection. Native environment advancement excluded.',
        limits='Research Python wall time; no production deadline claim. Two training-seed state distributions; teacher CPU time excluded only for fast paths which never invoke teacher.')
    write(root/'inference_timing.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--samples',type=int,default=128)
    args=parser.parse_args(); main(args.root,args.samples)

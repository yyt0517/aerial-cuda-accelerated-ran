"""Predeclared bounded DAgger -> frozen-generator PPO experiment."""
import argparse
from pathlib import Path
import json
import shutil
import time
import traceback
from training.stageb_ppo.native import ROOT, write, sha
from training.stageb_ppo_v2.catalog import REPAIR
from . import dagger, ppo


def config(smoke=False):
    cfg = dict(training_seeds=[41,42,43], validation_seeds=[44,45], warmup=4000,
        dagger_rounds=6, dagger_rollout=1536, teacher_stride=6, generator_epochs=5, generator_lr=.001,
        settling=512, max_iterations=12, rollout=512, gamma=2**(-1/400), gae_lambda=.999,
        reward_scale=.01, holding=1/400, age_threshold_tti=20, age_ramp_tti=20, age_multiplier=.5, expiry_penalty=2.,
        actor_lr=3.e-4, critic_lr=1.e-3, ppo_epochs=8, critic_epochs=8, actor_minibatch=128,
        target_kl=.02, entropy_coef=.001, ctbs=False, business_weights=False,
        validation_warmup=4000, validation_settling=4000, validation_measured=4000, validation_followup=400,
        candidate_budgets=[4,8], candidate_budget_definition='includes one unmodified raw PFQ fallback; no refill of duplicates',
        teacher='equal global DPLS, queried only on student-executed training trajectories',
        generator='frozen 79-feature cached neural repair plus trainable 93-feature residual; complete autoregressive reconstruction',
        teacher_objective='single-TTI physical improvement; not a multi-TTI return oracle',
        ppo_initialization='new zero-residual selector and critic because deployable state/reference and proposal distribution changed',
        ppo_training='one shared selector; six persistent streams for all seed x K pairs; frozen generator throughout PPO',
        rewards='same previous balanced whole-packet wait/age reward; uniform coefficients across all UEs',
        quality_target='each development seed: payload >= original service PPO, expiry <= original service PPO, P95 <= equal DPLS',
        efficiency_target='zero complete DPLS calls in fast inference; mean policy time <= one third of pure DPLS, P95 <= one half',
        checkpoint_selection='fixed final DAgger and PPO iterations; no development checkpoint selection',
        test_seeds_unused=[311,312,313,314,315],
        limits=['development seeds 44/45 already exposed; no generalization claim',
                'raw-PFQ dequeue floor 90% remains; 98% of unknown DPLS is not an online guarantee',
                'candidate count reduction is not a wall-clock speed claim', 'no automatic promotion'])
    if smoke: cfg.update(training_seeds=[41], warmup=512, dagger_rounds=2, dagger_rollout=64,
        teacher_stride=8, generator_epochs=2, settling=16, max_iterations=2, rollout=64, ppo_epochs=2, critic_epochs=2)
    return cfg


def main(root, smoke=False):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); cfg = config(smoke)
    write(root/'protocol.json', cfg)
    packages = ['stageb_student','stageb_ppo','stageb_ppo_v2','stageb_ppo_tail_balance','stageb_ppo_tail','stageb_lns']
    sources = {str(p.relative_to(ROOT)): sha(p) for package in packages for p in (ROOT/'training'/package).rglob('*.py')}
    sources[str(REPAIR.relative_to(ROOT))] = sha(REPAIR)
    write(root/'source_manifest.json', sources)
    for package in packages:
        shutil.copytree(ROOT/'training'/package, root/'source_snapshot'/package, ignore=shutil.ignore_patterns('__pycache__'))
    runtime = ROOT/'deliverables/ppo_tail_wait_20261030_v1/balanced/runtime_supplement_manifest.json'
    shutil.copyfile(runtime, root/'runtime_supplement_manifest.json')
    started = time.monotonic()
    try:
        write(root/'status.json', dict(stage='teacher_guided_training'))
        dagger.run(root/'dagger', cfg)
        for path, digest in sources.items(): assert sha(ROOT/path) == digest, path
        write(root/'status.json', dict(stage='ppo_training', elapsed_s=time.monotonic()-started))
        ppo.run(root/'ppo', root/'dagger/generator_final.npz', cfg)
        for path, digest in sources.items(): assert sha(ROOT/path) == digest, path
        write(root/'status.json', dict(stage='training_complete', sources_unchanged=True,
            elapsed_s=time.monotonic()-started, convergence_proven=False, promoted=False))
    except BaseException:
        write(root/'status.json', dict(stage='failed', traceback=traceback.format_exc())); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); main(args.out, args.smoke)

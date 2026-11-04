"""Freeze the distilled generator; train one selector on both K budgets."""
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from training.stageb_ppo.native import NativeEnv, write, sha
from training.stageb_ppo_tail_balance.worker import finish
from training.stageb_ppo_tail_balance.reward import reward
from training.stageb_ppo_tail_balance.campaign import receive_all
from training.stageb_ppo_v2.minibatch_learn import learn, KEYS
from .catalog import FastCatalog
from .model import ServiceActor, Value


def worker(connection, root, seed, k, cfg):
    def stop(*args): raise SystemExit('student PPO worker terminated')
    signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(1); torch.backends.mkldnn.enabled = False
    torch.manual_seed(20261031+seed+k*100)
    root = Path(root).resolve(); env = None
    try:
        env = NativeEnv(root/f'train_k{k}_seed{seed}', seed, cfg['warmup']+cfg['settling']+cfg['max_iterations']*cfg['rollout'])
        # Failing hard here makes hidden online teacher calls detectable.
        def forbid(*args, **kwargs): raise AssertionError('Teacher search in student PPO')
        env.expert = forbid
        actor = ServiceActor(); catalog = FastCatalog(root/'generator.npz', k)
        sampler = torch.Generator().manual_seed(20261031+seed+k*100)
        for _ in range(cfg['warmup']): env.step(env.raw_pfq())
        for _ in range(cfg['settling']):
            row = catalog.build(env); choice = int(np.where(row['mask'], row['prior'], -1.e9).argmax())
            env.step(row['actions'][choice])
        pending = catalog.build(env); audits = []; rewards = []
        connection.send(dict(type='ready', seed=seed, k=k))
        for iteration in range(1, cfg['max_iterations']+1):
            command = connection.recv(); assert command['iteration'] == iteration
            actor.load_state_dict(command['actor'])
            data = {key: [] for key in KEYS+['choice','distribution','reward','actions','catalog_actions']}
            initial_rng = sampler.get_state()
            for _ in range(cfg['rollout']):
                row = pending; choice, lp, distribution = actor.sample(row, generator=sampler)
                actual = env.step(row['actions'][choice])
                if env.tti == env.horizon: env.refresh()
                pending = catalog.build(env)
                r = reward(actual, env.fifo, row['potential'], pending['potential'], cfg)
                for key in KEYS: data[key].append(row[key])
                for key, value in [('choice', choice), ('distribution', distribution.numpy()), ('reward', r['reward']),
                    ('actions', row['actions'][choice]), ('catalog_actions', row['actions'])]: data[key].append(value)
                audits.append(dict(row['audit'], selected=choice, sampled_logprob=lp))
                rewards.append(r)
            packed = {key: np.asarray(values) for key, values in data.items()}
            packed.update(seed=seed, k=k, iteration=iteration, first_tti=env.tti-cfg['rollout']+1, last_tti=env.tti,
                next={key: pending[key] for key in KEYS}, reward_components=rewards[-cfg['rollout']:],
                sampler_before=initial_rng, sampler_after=sampler.get_state(), generator_sha256=sha(root/'generator.npz'))
            torch.save(packed, root/f'rollout_{iteration:03d}_k{k}_seed{seed}.pt')
            connection.send(dict(type='rollout', data=packed))
        finish(env, audits, rewards, cfg)
        connection.send(dict(type='finished', seed=seed, k=k))
    except BaseException:
        error = traceback.format_exc()
        if env is not None and not env.log.closed: env.close(validate=False)
        connection.send(dict(type='error', seed=seed, error=error)); raise
    finally: connection.close()


def run(root, generator, cfg):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(generator, root/'generator.npz'); frozen_sha = sha(root/'generator.npz')
    torch.set_num_threads(1); torch.backends.mkldnn.enabled = False; torch.manual_seed(20261031)
    actor, critic = ServiceActor(), Value()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
        actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict()), root/'initial_policy.pt')
    context = mp.get_context('spawn'); processes = []; connections = []; history = []; started = time.monotonic()
    try:
        for k in [4, 8]:
            for seed in cfg['training_seeds']:
                parent, child = context.Pipe(); process = context.Process(target=worker, args=(child, str(root), seed, k, cfg))
                process.start(); child.close(); connections.append(parent); processes.append(process)
        receive_all(connections, 'ready', root, processes)
        for iteration in range(1, cfg['max_iterations']+1):
            write(root/'status.json', dict(stage='rollout', iteration=iteration))
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict()), root/f'behavior_{iteration:03d}.pt')
            for connection in connections: connection.send(dict(iteration=iteration, actor=actor.state_dict()))
            segments = sorted([m['data'] for m in receive_all(connections, 'rollout', root, processes)], key=lambda s:(s['k'],s['seed']))
            report, credit = learn(actor, critic, actor_opt, critic_opt, segments, cfg, iteration)
            report['streams'] = [dict(seed=s['seed'], k=s['k'], reward=float(np.mean(s['reward']))) for s in segments]
            report['elapsed_s'] = time.monotonic()-started
            history.append(report); write(root/'learning_history.json', history)
            torch.save(credit, root/f'credit_{iteration:03d}.pt')
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), actor_optimizer=actor_opt.state_dict(),
                critic_optimizer=critic_opt.state_dict(), generator_sha256=frozen_sha, iteration=iteration, protocol=cfg,
                torch_rng=torch.get_rng_state()), root/f'checkpoint_{iteration:03d}.pt')
            print('PPO', iteration, report['reward_by_seed'], report['actor_steps'], flush=True)
        receive_all(connections, 'finished', root, processes)
        for process in processes: process.join(timeout=30); assert process.exitcode == 0
        assert sha(root/'generator.npz') == frozen_sha
        shutil.copyfile(root/f'checkpoint_{cfg["max_iterations"]:03d}.pt', root/'ppo_final.pt')
        write(root/'status.json', dict(stage='complete', samples=sum(h['samples'] for h in history),
            actor_updates=sum(h['actor_steps'] for h in history), generator_unchanged=True,
            full_search_calls=0, checkpoint_sha256=sha(root/'ppo_final.pt'), elapsed_s=time.monotonic()-started,
            convergence_proven=False, promoted=False))
    except BaseException:
        write(root/'status.json', dict(stage='failed', traceback=traceback.format_exc()))
        for process in processes:
            if process.is_alive(): process.terminate()
        for process in processes: process.join(timeout=15)
        raise
    finally:
        for connection in connections: connection.close()

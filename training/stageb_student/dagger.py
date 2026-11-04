"""Teacher labels are queried on trajectories executed entirely by students."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import pandas as pd
import torch
from training.stageb_ppo.native import ROOT, NativeEnv, teacher, packets, write, sha
from training.stageb_ppo_tail_balance.campaign import receive_all
from .generator import GeneratorModel
from .catalog import FastCatalog


def collect(connection, root, seed, cfg):
    torch.set_num_threads(1)
    root = Path(root).resolve(); env = None
    try:
        horizon = cfg['warmup']+cfg['dagger_rounds']*cfg['dagger_rollout']
        env = NativeEnv(root/f'collect_seed{seed}', seed, horizon)
        study = teacher(); audits = []
        for _ in range(cfg['warmup']): env.step(env.raw_pfq())
        connection.send(dict(type='ready', seed=seed))
        for iteration in range(1, cfg['dagger_rounds']+1):
            command = connection.recv(); assert command['iteration'] == iteration
            k = 4 if (iteration+seed) % 2 else 8
            catalog = FastCatalog(command['generator'], k)
            samples = []; labels = []; first = env.tti+1
            for step in range(cfg['dagger_rollout']):
                target = None; cost = 0.
                if step % cfg['teacher_stride'] == 0:
                    start = time.perf_counter(); target = env.expert(study); cost = time.perf_counter()-start
                row = catalog.build(env, teacher_action=target)
                selected = int(np.where(row['mask'], row['prior'], -1.e9).argmax())
                if target is not None:
                    samples.append(row['label'])
                    label = row['teacher']; label.update(tti=env.tti+1, seed=seed, k=k,
                        teacher_seconds=cost, student_choice=selected, student_action=row['actions'][selected].copy())
                    labels.append(label)
                audits.append(dict(row['audit'], selected=selected, teacher_queried=target is not None,
                    teacher_seconds=cost, iteration=iteration))
                env.step(row['actions'][selected])
            path = root/f'labels_{iteration:02d}_seed{seed}.npz'
            np.savez_compressed(path, **{key: np.concatenate([s[key] for s in samples]) for key in ['features','mask','target','option_ue']})
            torch.save(labels, root/f'teacher_{iteration:02d}_seed{seed}.pt')
            report = dict(seed=seed, iteration=iteration, k=k, first_tti=first, last_tti=env.tti,
                teacher_queries=len(labels), teacher_seconds=sum(s['teacher_seconds'] for s in labels),
                best_student_over_teacher_mean=float(np.mean([s['best_student_over_teacher'] for s in labels])),
                best_student_over_teacher_p5=float(np.quantile([s['best_student_over_teacher'] for s in labels], .05)),
                assignment_match=float(np.mean([s['assignment_match'] for s in labels])),
                student_acted_every_step=True, teacher_actions_executed_as_override=0)
            write(root/f'collection_{iteration:02d}_seed{seed}.json', report)
            connection.send(dict(type='collected', path=str(path), report=report))
        env.close(); packets(env.out)
        pd.DataFrame(audits).to_csv(env.out/'decision_audit.csv', index=False)
        actual = np.load(env.out/'actions.npz')['actions']
        for iteration in range(1, cfg['dagger_rounds']+1):
            for label in torch.load(root/f'teacher_{iteration:02d}_seed{seed}.pt', weights_only=False):
                np.testing.assert_array_equal(actual[label['tti']-1], label['student_action'])
        write(env.out/'teacher_separation_validation.json', dict(passed=True,
            teacher_labels_only=True, student_actions_match_native=True,
            queries=sum(a['teacher_queried'] for a in audits)))
        connection.send(dict(type='finished', seed=seed))
    except BaseException:
        error = traceback.format_exc()
        if env is not None and not env.log.closed: env.close(validate=False)
        connection.send(dict(type='error', seed=seed, error=error)); raise
    finally: connection.close()


def fit(model, optimizer, paths, epochs, device, iteration):
    # CPU half precision storage; only the current minibatch goes to the GPU.
    arrays = {k: [] for k in ['features', 'mask', 'target']}
    for path in paths:
        with np.load(path) as data:
            for k in arrays: arrays[k].append(torch.from_numpy(data[k].copy()))
    data = {k: torch.cat(v) for k, v in arrays.items()}
    n = len(data['target']); history = []
    for epoch in range(epochs):
        loss_sum = correct = count = 0
        for indices in torch.randperm(n).split(1024):
            width = int(data['mask'][indices].sum(-1).max())
            x = data['features'][indices, :width].to(device=device, dtype=torch.float32)
            mask = data['mask'][indices, :width].to(device); target = data['target'][indices].to(device)
            prediction = model(x).masked_fill(~mask, -1.e9)
            loss = torch.nn.functional.cross_entropy(prediction, target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
            correct += int((prediction.argmax(-1) == target).sum()); count += len(indices)
        history.append(dict(epoch=epoch+1, loss=loss_sum/count, teacher_forced_accuracy=correct/count))
    return dict(iteration=iteration, prefix_examples=n, epochs=epochs, training=history,
                note='teacher-forced training fit, not independent validation or closed-loop quality')


def run(root, cfg):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(20261031)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeneratorModel().to(device)
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=cfg['generator_lr'])
    model.export(root/'generator_00.npz')
    torch.save(dict(model=model.cpu().state_dict(), optimizer=optimizer.state_dict()), root/'initial_generator.pt')
    model.to(device)
    context = mp.get_context('spawn'); connections = []; processes = []; paths = []; history = []; started = time.monotonic()
    try:
        for seed in cfg['training_seeds']:
            parent, child = context.Pipe(); process = context.Process(target=collect, args=(child, str(root), seed, cfg))
            process.start(); child.close(); connections.append(parent); processes.append(process)
        receive_all(connections, 'ready', root, processes)
        for iteration in range(1, cfg['dagger_rounds']+1):
            write(root/'status.json', dict(stage='student_collection', iteration=iteration))
            for connection in connections:
                connection.send(dict(iteration=iteration, generator=str(root/f'generator_{iteration-1:02d}.npz')))
            messages = receive_all(connections, 'collected', root, processes)
            paths.extend(m['path'] for m in messages)
            write(root/'status.json', dict(stage='supervised_learning', iteration=iteration))
            report = fit(model, optimizer, paths, cfg['generator_epochs'], device, iteration)
            report.update(collection=[m['report'] for m in messages], elapsed_s=time.monotonic()-started)
            model.export(root/f'generator_{iteration:02d}.npz')
            torch.save(dict(model={k: v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), iteration=iteration), root/f'generator_{iteration:02d}.pt')
            history.append(report); write(root/'learning_history.json', history)
            print(json.dumps(report), flush=True)
        receive_all(connections, 'finished', root, processes)
        for process in processes: process.join(timeout=30); assert process.exitcode == 0
        shutil.copyfile(root/f'generator_{cfg["dagger_rounds"]:02d}.npz', root/'generator_final.npz')
        write(root/'status.json', dict(stage='complete', iterations=len(history),
            student_decisions=len(cfg['training_seeds'])*cfg['dagger_rounds']*cfg['dagger_rollout'],
            teacher_queries=sum(m['teacher_queries'] for h in history for m in h['collection']),
            generator_sha256=sha(root/'generator_final.npz'), elapsed_s=time.monotonic()-started,
            no_teacher_action_overrides=True, convergence_proven=False))
    except BaseException:
        write(root/'status.json', dict(stage='failed', traceback=traceback.format_exc()))
        for process in processes:
            if process.is_alive(): process.terminate()
        for process in processes: process.join(timeout=15)
        raise
    finally:
        for connection in connections: connection.close()

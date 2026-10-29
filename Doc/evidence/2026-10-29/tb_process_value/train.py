"""Fit two fixed architectures; model/epoch selection sees only seed44 labels."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import curves, high_yield
from training.stageb_lns.experiment import write_json, sha
from training.stageb_lns.process_value import ProcessValue


def export(path, models, mean, scale, features):
    values = dict(version=1, features=features, mean=mean, scale=scale)
    for key, layer, attr in (('w1', 0, 'weight'), ('b1', 0, 'bias'),
                             ('w2', 2, 'weight'), ('b2', 2, 'bias'),
                             ('w3', 4, 'weight'), ('b3', 4, 'bias')):
        values[key] = np.stack([getattr(m[layer], attr).detach().numpy() for m in models])
    np.savez(path, **values)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((HERE/'plan.json').read_text())
    out = HERE/'training';out.mkdir(exist_ok=False)
    write_json(out/'protocol.json', dict(plan_sha256=sha(HERE/'plan.json'),
        candidates_sha256=sha(HERE/'data/candidates.npz'),
        loss='weighted smooth-L1(root_gain/2400)+.25*smooth-L1(parent_increment/2400); high-yield weight5 vs1; inverse group size normalization',
        checkpoints=list(range(5, 61, 5)), selection_labels='seed44 only',
        normalization='unique training candidates only', ensemble_seeds=[7301, 7302, 7303],
        calibration_and_evaluation_labels_used=False))
    with np.load(HERE/'data/candidates.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    groups = pd.read_csv(HERE/'data/groups.csv')
    tr = arrays['unique'] & np.isin(arrays['seed'], plan['train_seeds'])
    sel = arrays['unique'] & np.isin(arrays['seed'], plan['selection_seeds'])
    targets = np.stack((arrays['delta'], arrays['step_delta']), axis=1).astype(np.float32)/2400.
    weights = np.zeros(len(tr), dtype=np.float32)
    for g in groups[groups.split == 'train'].itertuples():
        index = np.flatnonzero((arrays['group'] == g.group) & tr)
        target = high_yield(arrays['delta'][index], arrays['eligible'][index])
        weights[index] = (1+4*target)/len(index)
    weights = weights[tr];weights /= weights.mean()
    y = torch.from_numpy(targets[tr]);w = torch.from_numpy(weights)
    selection_groups = groups[groups.split == 'selection']
    history = []
    best_key = None;best_path = None;best_identity = None
    started = time.perf_counter()
    for name, features in (('actor_features_mlp', 93), ('tb_features_mlp', 189)):
        raw = arrays['x'][tr, :features]
        mean = raw.mean(0).astype(np.float32)
        scale = np.maximum(raw.std(0), 1.e-3).astype(np.float32)
        x = torch.from_numpy(np.clip((raw-mean)/scale, -10, 10))
        sx = torch.from_numpy(np.clip((arrays['x'][sel, :features]-mean)/scale, -10, 10))
        models, optimizers, generators = [], [], []
        for seed in plan['mlp']['random_seeds']:
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(features, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 2))
            models.append(model)
            optimizers.append(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001))
            generators.append(torch.Generator().manual_seed(seed+100))
        loss_fn = nn.SmoothL1Loss(reduction='none')
        for epoch in range(1, 61):
            losses = []
            for model, optimizer, generator in zip(models, optimizers, generators):
                permutation = torch.randperm(len(x), generator=generator)
                total = 0.
                for ids in permutation.split(2048):
                    prediction = model(x[ids])
                    error = loss_fn(prediction, y[ids])
                    loss = ((error[:, 0]+.25*error[:, 1])*w[ids]).mean()
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    total += float(loss.detach())*len(ids)
                losses.append(total/len(x))
            if epoch % 5:
                continue
            with torch.no_grad():
                output = torch.stack([m(sx) for m in models]).mean(0).numpy()*2400.
            scores = np.zeros(len(tr));scores[sel] = output[:, 0]
            stats = curves(arrays, selection_groups, scores, [.1], name)
            recall = float(stats.recall.mean())
            regret = float(stats.regret.mean())
            path = out/f'{name}_epoch{epoch:02d}.npz'
            export(path, models, mean, scale, features)
            exported = ProcessValue(path)
            mean_pred, _ = exported.predict(arrays['x'][sel])
            np.testing.assert_allclose(mean_pred, output, rtol=2.e-5, atol=.01)
            history.append(dict(architecture=name, epoch=epoch, features=features,
                parameters_per_member=sum(v.numel() for v in models[0].parameters()),
                selection_recall_at10pct=recall, selection_regret=regret,
                train_loss=float(np.mean(losses)), checkpoint=str(path.name)))
            key = (recall, -regret, -epoch)
            if best_key is None or key > best_key:
                best_key, best_path = key, path
                best_identity = history[-1]
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            print('TRAIN', name, epoch, 'selection_recall', round(recall, 6), 'regret', round(regret, 3), flush=True)
            (HERE/'progress.txt').write_text(f'价值头蒸馏：{name}，epoch{epoch}/60；仅seed44用于选型。\n')
    import shutil
    shutil.copyfile(best_path, HERE/'value_frozen.npz')
    write_json(out/'selected.json', dict(**best_identity,
        checkpoint_sha256=sha(HERE/'value_frozen.npz'), selection_seed=44,
        data_sha256=sha(HERE/'data/candidates.npz'), wall_s=time.perf_counter()-started,
        exported_numpy_torch_parity=True, calibration_and_evaluation_labels_used=False))


if __name__ == '__main__':
    main()

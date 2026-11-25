"""Small block ranker trained on fixed-state joint-repair counterfactuals.

Train/validation are separated by native seed. Deployment uses NumPy matmuls,
not PyTorch. No exact candidate gain or post-action CRC is an input feature.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .archive import load_rows, problem_from_row, initial_draft
from .physics import NativePhy, COUNT, LAYER, CAPACITY, DEQUEUE, BLER, EXPECTED
from .search import SearchConfig, ProposalGraph, repaired_candidates, solve

FEATURE_VERSION = 3
BYTE_SCALE = 3000.


def token_features(graph, base, version=FEATURE_VERSION):
    """Causal token summaries, reusable while the verified incumbent is fixed."""
    p = graph.problem
    action, prediction = base.actions[0], base.values[0]
    u = np.maximum(action, 0)
    active = action >= 0
    prg = np.arange(p.P)[None, :]
    values = prediction[u]
    queue = p.queue[u]
    x = np.stack((
        np.log1p(queue/BYTE_SCALE), p.weights[u]/4.,
        values[:, :, COUNT]/p.P, np.maximum(values[:, :, LAYER], 0)/4.,
        p.mcs[u]/27., p.caps[u, prg]/4.,
        values[:, :, DEQUEUE]/np.maximum(queue, BYTE_SCALE),
        np.maximum(values[:, :, BLER], 0),
        values[:, :, EXPECTED]/np.maximum(queue, BYTE_SCALE),
        np.log1p(np.maximum(values[:, :, CAPACITY]-queue, 0)/BYTE_SCALE),
        np.log1p(graph.quality[u, prg]),
        (p.caps[u, prg] == values[:, :, LAYER]).astype(float),
    ), axis=-1)
    x *= active[:, :, None]
    x = np.concatenate((x, (~active)[:, :, None].astype(float)), axis=-1)
    if version >= 2:
        # Explicit source -> victim information is retained even when a source
        # is currently blank. Aggregate SINR alone cannot identify that source.
        received = graph.received[u, prg]  # [victim cell, PRG, source cell]
        own = np.take_along_axis(received, np.arange(p.C)[:, None, None], axis=-1)[:, :, 0]
        denominator = np.maximum(4*p.noise+received.sum(-1)-own, 1.e-30)
        fraction = received/denominator[:, :, None]
        fraction *= (np.arange(p.C)[:, None, None] != np.arange(p.C)[None, None, :])
        fraction *= active[:, :, None]
        risk = p.weights[u]*values[:, :, DEQUEUE]*np.maximum(values[:, :, BLER], 0)/BYTE_SCALE
        if version >= 3:
            risk *= active
        outgoing = (fraction*risk[:, :, None]).sum(axis=0).T
        strongest = fraction.max(axis=0).T
        x = np.concatenate((x, np.log1p(outgoing)[:, :, None],
            np.log1p(strongest)[:, :, None], np.log1p(risk)[:, :, None],
            np.clip(values[:, :, 4]/30., -1., 2.)[:, :, None]*active[:, :, None]), axis=-1)
    return x.reshape(p.C*p.P, -1)


def pool_block_features(tokens, blocks, n_prg):
    """Pool in the original float64 order before converting to model inputs."""
    width = max(map(len, blocks))
    indices = np.zeros((len(blocks), width), dtype=int)
    mask = np.zeros(indices.shape, dtype=bool)
    for i, block in enumerate(blocks):
        indices[i, :len(block)] = [c*n_prg+prg for c, prg in block]
        mask[i, :len(block)] = True
    gathered = tokens[indices]
    mean = (gathered*mask[:, :, None]).sum(axis=1)/mask.sum(axis=1)[:, None]
    maximum = np.where(mask[:, :, None], gathered, -np.inf).max(axis=1)
    root = gathered[:, 0]
    size = mask.sum(axis=1)[:, None]/6.
    result = np.concatenate((mean, maximum, root, size), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite block feature")
    return result


def block_features(graph, base, blocks, version=FEATURE_VERSION):
    """Pool causal token/TB summaries over the proposed block, without repair."""
    return pool_block_features(token_features(graph, base, version), blocks, graph.problem.P)


class BlockRanker:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            self.version = int(saved["feature_version"])
            if self.version not in (1, 2, FEATURE_VERSION):
                raise ValueError("Unknown block-feature version")
            for key in ("mean", "scale", "w1", "b1", "w2", "b2"):
                setattr(self, key, saved[key].copy())

    def predict(self, features):
        x = (features-self.mean)/self.scale
        return (np.maximum(x@self.w1.T+self.b1, 0)@self.w2.T+self.b2).ravel()

    def __call__(self, graph, base, blocks, remaining_ms):
        scores = self.predict(block_features(graph, base, blocks, self.version))
        return [blocks[i] for i in np.argsort(-scores, kind="stable")]


def label_blocks(row, root, config, root_index):
    p = problem_from_row(row)
    phy = NativePhy(p)
    try:
        base = phy.evaluate(root)
        graph = ProposalGraph(phy)
        rng = np.random.default_rng(row["seed"]*100000+row["tti"]+root_index)
        blocks = graph.blocks(root, base.values[0], config, rng)
        features = block_features(graph, base, blocks)
        draft = initial_draft(p, row["obs"])
        original = phy.evaluate(draft)
        draft_dequeue = int(original.values[0, :, DEQUEUE].sum())
        floor = (config.floor_numerator*draft_dequeue+config.floor_denominator-1)//config.floor_denominator
        initial_score = float(base.values[0, :, EXPECTED] @ p.weights)
        gains, costs = [], []
        for block in blocks:
            best = initial_score
            before = phy.stats["ue_evaluations"]

            def evaluate(actions, source):
                nonlocal best
                result = phy.evaluate(actions, source)
                scores = result.values[:, :, EXPECTED] @ p.weights
                eligible = result.values[:, :, DEQUEUE].sum(1) >= floor
                if eligible.any():
                    best = max(best, float(scores[eligible].max()))
                return result

            for _ in repaired_candidates(graph, base, block, config, rng, evaluate):
                pass
            gains.append((best-initial_score)/BYTE_SCALE)
            costs.append(phy.stats["ue_evaluations"]-before)
        return dict(seed=row["seed"], tti=row["tti"], root_index=root_index,
                    features=features, gain=np.array(gains, np.float32), cost=np.array(costs, np.int64))
    finally:
        phy.close()


def dataset(seeds, config):
    result = []
    for row in load_rows(seeds):
        problem = problem_from_row(row)
        draft = initial_draft(problem, row["obs"])
        roots = [draft]
        for cap in (512, 2048):
            action, _, _ = solve(problem, draft, replace(config, mode="random_lns", budget_ms=None,
                                                        max_candidates=cap, seed=row["tti"]+cap))
            roots.append(action)
        for i, root in enumerate(roots):
            result.append(label_blocks(row, root, config, i))
        print("LABELLED", row["seed"], row["tti"], flush=True)
    return result


def export_model(path, model, mean, scale):
    np.savez(path, feature_version=FEATURE_VERSION, mean=mean, scale=scale,
             w1=model[0].weight.detach().cpu().numpy(), b1=model[0].bias.detach().cpu().numpy(),
             w2=model[2].weight.detach().cpu().numpy(), b2=model[2].bias.detach().cpu().numpy())


def main(args):
    import torch
    from .experiment import write_json, sha, ROOT, snapshot_sources
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    train_seeds = tuple(map(int, args.train_seeds.split(",")))
    validation_seeds = tuple(map(int, args.validation_seeds.split(",")))
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("Training and validation seeds overlap")
    torch.set_num_threads(1); torch.manual_seed(1701)
    config = SearchConfig(budget_ms=None)
    source_hashes = snapshot_sources(output)
    protocol = dict(train_seeds=train_seeds, validation_seeds=validation_seeds,
        heldout_seeds=[43, 44, 45], feature_version=FEATURE_VERSION,
        roots=["PFQ_CTBS", "random_LNS_512_candidates", "random_LNS_2048_candidates"],
        supervision="Complete local-block repairs with the fixed original draft floor",
        config=config.__dict__, epochs=args.epochs, source_sha256=source_hashes,
        limitation="Pilot on few correlated archived radio states; no closed-loop claim")
    write_json(output/"protocol.json", protocol)
    training, validation = dataset(train_seeds, config), dataset(validation_seeds, config)
    torch.save(dict(training=training, validation=validation), output/"labels.pt")
    all_x = np.concatenate([g["features"] for g in training])
    mean, scale = all_x.mean(0), np.maximum(all_x.std(0), .1)
    model = torch.nn.Sequential(torch.nn.Linear(all_x.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    rng = np.random.default_rng(1701)
    best_metric, best_epoch, history = -np.inf, -1, []

    def forward(group):
        return model(torch.from_numpy((group["features"]-mean)/scale)).ravel()

    def evaluate(groups):
        values = []
        with torch.no_grad():
            for group in groups:
                index = int(forward(group).argmax())
                oracle = float(group["gain"].max())
                if oracle > 1.e-6:
                    values.append(float(group["gain"][index])/oracle)
        return float(np.mean(values)) if values else 1.

    for epoch in range(args.epochs):
        losses = []
        for i in rng.permutation(len(training)):
            group = training[i]
            prediction = forward(group)
            gains = torch.from_numpy(group["gain"])
            target = torch.softmax(gains/1.5, dim=0)
            ranking = -(target*torch.log_softmax(prediction/1.5, dim=0)).sum()
            regression = torch.nn.functional.smooth_l1_loss(prediction, gains)
            loss = ranking+.2*regression
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        metric = evaluate(validation)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_top1_gain_retained=metric))
        if metric > best_metric:
            best_metric, best_epoch = metric, epoch
            export_model(output/"block_ranker.npz", model, mean, scale)
        if epoch % 25 == 0:
            print("TRAIN", history[-1], flush=True)
    ranker = BlockRanker(output/"block_ranker.npz")
    ranker_checks = []
    for group in validation:
        predicted = ranker.predict(group["features"])
        gain = group["gain"]
        ranker_checks.append(dict(seed=group["seed"], tti=group["tti"], root_index=group["root_index"],
            selected_gain=float(gain[predicted.argmax()]), heuristic_first_gain=float(gain[0]),
            oracle_gain=float(gain.max()), random_mean_gain=float(gain.mean())))
    for relative, digest in source_hashes.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f"Source changed during training: {relative}")
    write_json(output/"training.json", dict(parameters=sum(p.numel() for p in model.parameters()),
        training_groups=len(training), validation_groups=len(validation), best_epoch=best_epoch,
        best_validation_top1_gain_retained=best_metric, history=history, validation=ranker_checks))
    print("TRAINING_COMPLETE", best_epoch, best_metric, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-seeds", default="41")
    parser.add_argument("--validation-seeds", default="42")
    parser.add_argument("--epochs", type=int, default=150)
    main(parser.parse_args())

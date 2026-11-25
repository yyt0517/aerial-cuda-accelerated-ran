"""Budget-conditioned, critic-free search-trajectory training with RLOO.

Each group shares a causal radio snapshot and a wall-clock budget. Actions are
the next local block or STOP. Physical candidates remain verified by the same
kernel. This is search RL, not an estimator of long-term packet-delivery return.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np

from .archive import load_rows, problem_from_row, initial_draft
from .learning import FEATURE_VERSION, BYTE_SCALE, block_features
from .search import SearchConfig, solve


def policy_features(graph, base, blocks, remaining_ms, mean):
    x = block_features(graph, base, blocks)
    # STOP has neutral physical features and its own indicator.
    x = np.vstack((x, mean))
    budget = 100. if remaining_ms is None else max(0., remaining_ms)
    extra = np.zeros((len(x), 2), dtype=np.float32)
    extra[:, 0] = np.log1p(budget/10.)
    extra[-1, 1] = 1.
    return np.concatenate((x, extra), axis=1)


class BudgetRanker:
    adaptive = True

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["feature_version"]) != FEATURE_VERSION or int(saved["budget_policy"]) != 1:
                raise ValueError("Unsupported budget-policy checkpoint")
            for key in ("mean", "scale", "w1", "b1", "w2", "b2"):
                setattr(self, key, saved[key].copy())

    def __call__(self, graph, base, blocks, remaining_ms):
        features = policy_features(graph, base, blocks, remaining_ms, self.mean[:-2])
        x = (features-self.mean)/self.scale
        logits = (np.maximum(x@self.w1.T+self.b1, 0)@self.w2.T+self.b2).ravel()
        logits[-1] -= 2.
        selected = int(logits.argmax())
        return [] if selected == len(blocks) else [blocks[selected]]


def main(args):
    import torch
    from .experiment import write_json, sha, snapshot_sources, ROOT
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(2701)
    with np.load(args.initial, allow_pickle=False) as initial:
        if int(initial["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Initial model feature mismatch")
        mean = np.r_[initial["mean"], 0., 0.].astype(np.float32)
        scale = np.r_[initial["scale"], 1., 1.].astype(np.float32)
        model = torch.nn.Sequential(torch.nn.Linear(len(mean), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(np.pad(initial["w1"], ((0, 0), (0, 2)))))
            model[0].bias.copy_(torch.from_numpy(initial["b1"]))
            model[2].weight.copy_(torch.from_numpy(initial["w2"]))
            model[2].bias.copy_(torch.from_numpy(initial["b2"]))
    import copy
    prior = copy.deepcopy(model).eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    train_seeds = tuple(map(int, args.train_seeds.split(",")))
    validation_seeds = tuple(map(int, args.validation_seeds.split(",")))
    budgets = tuple(map(float, args.budgets.split(",")))
    if set(train_seeds) & set(validation_seeds) or args.group_size < 2 or min(budgets) <= 0:
        raise ValueError("Invalid training partition/group/budget")
    sources = snapshot_sources(output)
    write_json(output/"protocol.json", dict(
        train_seeds=train_seeds, validation_seeds=validation_seeds, heldout_seeds=[43,44,45],
        budgets_ms=budgets, group_size=args.group_size, updates=args.updates,
        feature_version=FEATURE_VERSION, initial_sha256=sha(args.initial), source_sha256=sources,
        reward="(best_verified_J-initial_J)/3000 - 0.02*elapsed_ms/budget_ms",
        advantage="return minus leave-one-out mean; no group standard-deviation division",
        loss="on-policy REINFORCE + 0.01 categorical KL to initial policy - 0.002 entropy",
        actions="one next block or STOP, conditioned on current verified incumbent and remaining budget",
        limitations=["Same-state search objective, not long-term packet credit", "CPU wall-clock reward is noisy", "Small correlated archive pilot", "No per-TTI online weight update"]))
    training, validation = load_rows(train_seeds), load_rows(validation_seeds)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
    rng = np.random.default_rng(2701)
    config = SearchConfig(max_candidates=100000)
    history, validation_history = [], []
    best_metric = -np.inf

    def export(path):
        np.savez(path, budget_policy=1, feature_version=FEATURE_VERSION, mean=mean, scale=scale,
            w1=model[0].weight.detach().numpy(), b1=model[0].bias.detach().numpy(),
            w2=model[2].weight.detach().numpy(), b2=model[2].bias.detach().numpy())

    class Sampler:
        adaptive = True

        def __init__(self):
            self.logps, self.entropies, self.kls = [], [], []

        def __call__(self, graph, base, blocks, remaining_ms):
            raw = policy_features(graph, base, blocks, remaining_ms, mean[:-2])
            x = torch.from_numpy((raw-mean)/scale)
            logits = model(x).ravel()
            bias = torch.zeros_like(logits); bias[-1] = -2.
            distribution = torch.distributions.Categorical(logits=logits+bias)
            selected = distribution.sample()
            with torch.no_grad():
                reference = torch.distributions.Categorical(logits=prior(x).ravel()+bias)
            self.logps.append(distribution.log_prob(selected))
            self.entropies.append(distribution.entropy())
            self.kls.append(torch.distributions.kl_divergence(distribution, reference))
            index = int(selected)
            return [] if index == len(blocks) else [blocks[index]]

    def validate(step):
        nonlocal best_metric
        export(output/"candidate.npz")
        if step == -1:
            export(output/"budget_initial.npz")
        policy = BudgetRanker(output/"candidate.npz")
        values = []
        for row in validation:
            for budget in budgets:
                begin = time.perf_counter()
                p = problem_from_row(row); draft = initial_draft(p, row["obs"])
                _, _, result = solve(p, draft, replace(config, budget_ms=budget), block_orderer=policy, start_time=begin)
                values.append(result["improvement"]/BYTE_SCALE-.02*result["core_ms"]/budget)
        metric = float(np.mean(values))
        validation_history.append(dict(step=step, mean_reward=metric))
        if metric > best_metric:
            best_metric = metric
            export(output/"budget_ranker.npz")
        print("BUDGET_VALIDATION", step, metric, flush=True)

    validate(-1)
    for step in range(args.updates):
        row = training[step % len(training)]
        budget = budgets[int(rng.integers(len(budgets)))]
        samplers, rewards, costs = [], [], []
        for member in range(args.group_size):
            sampler = Sampler()
            begin = time.perf_counter()
            problem = problem_from_row(row); draft = initial_draft(problem, row["obs"])
            _, _, result = solve(problem, draft, replace(config, budget_ms=budget, seed=step*args.group_size+member), block_orderer=sampler, start_time=begin)
            samplers.append(sampler)
            rewards.append(result["improvement"]/BYTE_SCALE-.02*result["core_ms"]/budget)
            costs.append(result["core_ms"])
        rewards = np.array(rewards)
        advantages = rewards-(rewards.sum()-rewards)/(len(rewards)-1)
        terms = []
        for advantage, sampler in zip(advantages, samplers):
            if sampler.logps:
                terms.append(-float(advantage)*torch.stack(sampler.logps).sum()
                    +.01*torch.stack(sampler.kls).mean()-.002*torch.stack(sampler.entropies).mean())
        if terms:
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        else:
            loss = torch.tensor(0.)
        history.append(dict(step=step, seed=row["seed"], tti=row["tti"], budget_ms=budget,
            rewards=rewards.tolist(), elapsed_ms=costs, loss=float(loss.detach()),
            decisions=[len(s.logps) for s in samplers]))
        if (step+1) % 10 == 0:
            validate(step)
    if args.updates % 10:
        validate(args.updates-1)
    for relative, digest in sources.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f"Source changed during training: {relative}")
    write_json(output/"training.json", dict(history=history, validation=validation_history,
        best_validation_reward=best_metric, parameters=sum(p.numel() for p in model.parameters())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-seeds", default="41")
    parser.add_argument("--validation-seeds", default="42")
    parser.add_argument("--budgets", default="10,20,50")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=60)
    main(parser.parse_args())

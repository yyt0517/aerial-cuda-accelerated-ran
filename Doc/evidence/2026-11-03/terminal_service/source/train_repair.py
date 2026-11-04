"""Quality-first distillation of complete, physically verified joint repairs."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from .archive import load_rows, problem_from_row, initial_draft
from .physics import NativePhy, EXPECTED, DEQUEUE
from .search import SearchConfig, ProposalGraph, repaired_candidates, solve
from .neural_repair import features, options, destroy, export_model, BYTE_SCALE
from .learning import block_features


def collect(seeds, blocks_per_root=16):
    examples, cases = [], []
    teacher = SearchConfig(budget_ms=None, ue_topk=6, repair_width=16, max_candidates=1000000, max_rounds=256)
    for row in load_rows(seeds):
        p = problem_from_row(row); draft = initial_draft(p, row["obs"])
        middle, _, _ = solve(p, draft, replace(teacher, mode="greedy", max_candidates=4096))
        final, _, _ = solve(p, draft, replace(teacher, mode="greedy"))
        phy = NativePhy(p)
        try:
            graph = ProposalGraph(phy)
            original = phy.evaluate(draft)
            floor = (90*int(original.values[0, :, DEQUEUE].sum())+99)//100
            for root_index, root in enumerate((draft, middle, final)):
                base = phy.evaluate(root)
                score = float(base.values[0, :, EXPECTED] @ p.weights)
                rng = np.random.default_rng(row["seed"]*100000+row["tti"]+root_index)
                blocks = graph.blocks(root, base.values[0], teacher, rng)
                primary = min(len(blocks), blocks_per_root//2)
                selected = list(range(primary))
                remainder = min(blocks_per_root-primary, len(blocks)-primary)
                selected += rng.choice(np.arange(primary, len(blocks)), remainder, replace=False).tolist()
                for index in selected:
                    block = blocks[index]
                    pool = {tuple(int(root[c, prg]) for c, prg in block): (score, root.copy())}

                    def evaluate(actions, source):
                        result = phy.evaluate(actions, source)
                        scores = result.values[:, :, EXPECTED] @ p.weights
                        eligible = result.values[:, :, DEQUEUE].sum(1) >= floor
                        for i in np.flatnonzero(eligible):
                            key = tuple(int(actions[i, c, prg]) for c, prg in block)
                            pool[key] = (float(scores[i]), actions[i].copy())
                        return result

                    for _ in repaired_candidates(graph, base, block, teacher, rng, evaluate):
                        pass
                    ranked = sorted(pool.values(), key=lambda x: -x[0])
                    gain = ranked[0][0]-score
                    # Keep a small number of no-improvement examples; complete
                    # action alternatives retain coordinated choices and ties.
                    top = ranked[:4] if gain > 1 else ranked[:1]
                    q = np.exp((np.array([s for s, _ in top])-top[0][0])/BYTE_SCALE)
                    q /= q.sum()
                    context = block_features(graph, base, [block])[0]
                    case = dict(seed=row["seed"], tti=row["tti"], root_index=root_index,
                                root=root.copy(), block=block, floor=floor, base_score=score,
                                best_score=ranked[0][0], best_action=ranked[0][1].copy())
                    cases.append(case)
                    for probability, (_, target) in zip(q, top):
                        partial = destroy(root, block)
                        x, mask, labels = [], [], []
                        for depth, (c, prg) in enumerate(block):
                            choices = options(p, c, prg)
                            feature = features(graph, base, block, partial[None], depth, choices, context)[0]
                            legal = np.zeros(p.U+1, bool); legal[:len(choices)] = True
                            padded = np.zeros((p.U+1, feature.shape[-1]), np.float32); padded[:len(choices)] = feature
                            x.append(padded); mask.append(legal)
                            labels.append(int(np.flatnonzero(choices == target[c, prg])[0]))
                            partial[c, prg] = target[c, prg]
                        np.testing.assert_array_equal(partial, target)
                        while len(x) < teacher.block_size:
                            x.append(np.zeros_like(x[0])); inactive = np.zeros_like(mask[0]); inactive[0] = True
                            mask.append(inactive); labels.append(-100)
                        examples.append(dict(x=np.stack(x), mask=np.stack(mask), labels=np.array(labels),
                            weight=float(probability*max(.1, min(gain/BYTE_SCALE, 4.))), case_id=len(cases)-1))
                print("REPAIR_LABELS", row["seed"], row["tti"], root_index, len(cases), flush=True)
        finally:
            phy.close()
    return examples, cases


def main(args):
    import torch
    from .experiment import write_json, snapshot_sources, sha, ROOT
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(3711)
    train_seeds = tuple(map(int, args.train_seeds.split(",")))
    val_seeds = tuple(map(int, args.validation_seeds.split(",")))
    if set(train_seeds) & set(val_seeds):
        raise ValueError("Training/validation seeds overlap")
    sources = snapshot_sources(output)
    write_json(output/"protocol.json", dict(train_seeds=train_seeds, validation_seeds=val_seeds,
        final_native_test_seeds=[46,47,48], blocks_per_root=args.blocks_per_root,
        roots=["PFQ_CTBS", "greedy_4096_candidates", "converged_greedy"],
        teacher=dict(ue_topk=6, beam_width=16, max_alternatives=4),
        objective="weighted whole-sequence teacher-forced negative log likelihood; soft physical-score alternatives",
        feature_inputs="causal source/victim, whole-TB base state, partial block assignment, native per-PRG SINR observation",
        no_counterfactual_physics_before_neural_proposal=True, epochs=args.epochs,
        source_sha256=sources, device=args.device))
    train, train_cases = collect(train_seeds, args.blocks_per_root)
    valid, valid_cases = collect(val_seeds, args.blocks_per_root)
    torch.save(dict(training_cases=train_cases, validation_cases=valid_cases), output/"cases.pt")
    def pack(examples):
        return dict(x=np.stack([e['x'] for e in examples]), mask=np.stack([e['mask'] for e in examples]),
                    labels=np.stack([e['labels'] for e in examples]), weight=np.array([e['weight'] for e in examples], np.float32))
    a, b = pack(train), pack(valid)
    np.savez_compressed(output/"training.npz", **a); np.savez_compressed(output/"validation.npz", **b)
    flat = a['x'][a['mask'] & (a['labels'] != -100)[:, :, None]]
    mean = flat.mean(0); scale = np.maximum(flat.std(0), .1)
    device = torch.device(args.device)
    model = torch.nn.Sequential(torch.nn.Linear(len(mean),64),torch.nn.ReLU(),
        torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    def tensors(data):
        return (torch.from_numpy(np.clip((data['x']-mean)/scale,-10,10)).to(device),
                torch.from_numpy(data['mask']).to(device),torch.from_numpy(data['labels']).to(device),
                torch.from_numpy(data['weight']).to(device))
    train_t, val_t = tensors(a), tensors(b)
    def loss_batch(tensors, indices):
        x, mask, targets, weight = (t[indices] for t in tensors)
        logits = model(x).squeeze(-1).masked_fill(~mask, -1.e9)
        losses = torch.nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='none').reshape(targets.shape).sum(1)
        return (losses*weight).sum()/weight.sum().clamp_min(1.e-8)
    rng = np.random.default_rng(3711)
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        permutation = rng.permutation(len(a['x']))
        losses = []
        for begin in range(0,len(permutation),args.batch_size):
            ix = torch.as_tensor(permutation[begin:begin+args.batch_size],device=device)
            loss = loss_batch(train_t, ix)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            val_losses = [float(loss_batch(val_t, slice(i,i+args.batch_size))) for i in range(0,len(b['x']),args.batch_size)]
        val = float(np.mean(val_losses))
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_joint_nll=val))
        if val < best:
            best = val
            export_model(output/"repair.npz", model, mean, scale)
            best_epoch = epoch
        if epoch % 10 == 0:
            print("REPAIR_TRAIN", history[-1], flush=True)
    for relative, digest in sources.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f"Source changed during training: {relative}")
    write_json(output/"training.json", dict(training_cases=len(train_cases),validation_cases=len(valid_cases),
        training_sequences=len(train),validation_sequences=len(valid),parameters=sum(p.numel() for p in model.parameters()),
        feature_dim=len(mean),best_epoch=best_epoch,best_validation_joint_nll=best,history=history))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-seeds',default='41,42,43')
    parser.add_argument('--validation-seeds',default='44,45')
    parser.add_argument('--blocks-per-root',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--device',default='cpu')
    main(parser.parse_args())

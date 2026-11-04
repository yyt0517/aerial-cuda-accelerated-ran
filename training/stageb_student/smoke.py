"""Native contract checks before freezing the formal experiment."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from training.stageb_ppo.native import NativeEnv, teacher, packets, write
from training.stageb_lns.archive import problem_from_row
from training.stageb_lns.physics import ReferencePhy
from .generator import GeneratorModel, StudentRepair
from .catalog import FastCatalog
from .model import ServiceActor


def main(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(20261031)
    model = GeneratorModel(); model.export(root/'initial.npz')
    repair = StudentRepair(root/'initial.npz')
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3, 13, 93)).astype(np.float32)
    np.testing.assert_allclose(model(torch.from_numpy(x)).detach(), repair.logits(x), atol=2.e-5, rtol=1.e-5)
    actor = ServiceActor(); env = NativeEnv(root/'native', 41, 560); study = teacher()
    observations = []
    try:
        for _ in range(512): env.step(env.raw_pfq())
        expert_method = env.expert
        def forbid(*args, **kwargs): raise AssertionError('Fast path invoked teacher')
        for i in range(48):
            target = expert_method(study) if i % 12 == 0 else None
            env.expert = forbid
            a = FastCatalog(root/'initial.npz', 4).build(env)
            b = FastCatalog(root/'initial.npz', 8).build(env, teacher_action=target)
            np.testing.assert_array_equal(a['actions'][:4], b['actions'][:4])
            np.testing.assert_array_equal(a['mask'][:4], b['mask'][:4])
            assert a['audit']['candidate_evaluations'] == 4
            assert b['audit']['candidate_evaluations'] == 8
            if i == 0:
                p = problem_from_row(dict(ctbs=env.inputs, obs=env.obs, snapshot=env.snapshot, ema=np.full(36,2400.)))
                p.weights[:] = 1.; p.deadline = None
                values = ReferencePhy(p).evaluate(b['actions'])
                np.testing.assert_allclose(values[:, :, 6].sum(-1)/86400, b['summary'][:, 0], rtol=2.e-6, atol=1.e-7)
            if target is not None:
                label = b['label']
                logits = model(torch.from_numpy(label['features'].astype(np.float32))).masked_fill(~torch.from_numpy(label['mask']), -1.e9)
                loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(label['target']))
                loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.residual.parameters())
                model.zero_grad()
            choice, _, _ = actor.sample(b, greedy=True)
            observations.append(dict(**b['audit'], selected=choice))
            env.step(b['actions'][choice])
        env.close(); packets(env.out)
    except BaseException:
        if not env.log.closed: env.close(validate=False)
        raise
    result = dict(passed=True, native_steps=env.tti, fast_states=48,
        teacher_forbidden_in_fast_path=True, nested_4_8_candidates=True,
        exact_4_8_physical_evaluations=True, independent_phy_reference=True,
        torch_numpy_parity=True, supervised_gradients_finite=True,
        mean_initial_choices=float(np.mean([r['choices'] for r in observations])),
        mean_initial_policy_ms=1000*float(np.mean([r['wall_s'] for r in observations])))
    write(root/'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)

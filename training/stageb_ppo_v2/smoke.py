"""Exercise actual neural repair, PHY verification and complete native actions."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from training.stageb_ppo.native import NativeEnv,write,packets
from .catalog import Catalog
from .model import ServiceActor


def main(out):
    torch.set_num_threads(1);torch.manual_seed(20261030)
    out=Path(out).resolve()
    env=NativeEnv(out,41,160)
    policy=ServiceActor();catalog=Catalog();saved=[];records=[]
    generator=torch.Generator().manual_seed(42)
    try:
        for _ in range(128):env.step(env.raw_pfq())
        start=time.monotonic()
        for _ in range(32):
            row=catalog.build(env)
            action,lp,dist=policy.sample(row,generator=generator)
            actual=env.step(row['actions'][action])
            row.update(choice=action,logp=lp,distribution=dist,native=actual)
            saved.append(row);records.append(row['audit'])
        env.close();packets(out)
        torch.save(saved,out/'catalog_states.pt')
        write(out/'catalog_check.json',dict(passed=True,steps=32,
            elapsed_s=time.monotonic()-start,mean_wall_s=float(np.mean([r['wall_s'] for r in records])),
            mean_choices=float(np.mean([r['choices'] for r in records])),
            mean_neural_eligible=float(np.mean([r['neural_eligible'] for r in records])),
            mean_neural_catalog=float(np.mean([r['neural_catalog'] for r in records])),
            min_expected_ratio=float(min(r['min_catalog_expected_ratio'] for r in records)),
            repair_checkpoint_sha256=catalog.repair_sha256,records=records))
        print('NATIVE_CATALOG_SMOKE_PASSED',out,flush=True)
    except BaseException:
        if not env.log.closed:env.close(validate=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)

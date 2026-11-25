"""Check every UE against its intended adjacent shared boundary and Voronoi cell."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from training.stageb_ppo.native import write

def check(path):
    df=pd.read_csv(path);bs=df[df.node_type=='bs'].sort_values('cell_id');ue=df[df.node_type=='ue'];xy=bs[['x_m','y_m']].to_numpy()
    assert len(bs)==7 and len(ue)==84 and set(ue.placement)=={'coop_boundary'}
    assert np.allclose(df.site_spacing_m,1000) and np.allclose(df.cell_radius_m,500)
    counts=[]
    for c in range(7):
        adjacent=np.flatnonzero(np.abs(np.linalg.norm(xy-xy[c],axis=1)-1000)<.1)
        users=ue[ue.cell_id==c].sort_values('ue_local_id');assert len(users)==12
        for _,u in users.iterrows():
            target=adjacent[int(u.ue_local_id)%len(adjacent)];direction=(xy[target]-xy[c]);direction/=np.linalg.norm(direction)
            delta=np.array([u.x_m,u.y_m])-(xy[c]+xy[target])/2
            inset=-delta@direction;along=delta@np.array([-direction[1],direction[0]])
            assert -.02<=inset<=20.02 and abs(along)<=280.02,(c,u.ue_id,inset,along)
            distances=np.linalg.norm(xy-np.array([u.x_m,u.y_m]),axis=1)
            assert distances[c]<=distances.min()+.02
        counts.append(dict(cell=c,adjacent=adjacent.tolist(),nominal_users=12))
    return dict(passed=True,cells=7,users=84,geometry=counts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();write(a.output,check(Path(a.source)))

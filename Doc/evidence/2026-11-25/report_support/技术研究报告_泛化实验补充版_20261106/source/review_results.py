import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'deliverables/stage2_generalization_20261106';AN=EXP/'analysis'
def main():
    df=pd.read_csv(AN/'per_seed.csv');assert len(df)==60
    cols=['offered_mbps','payload_mbps','expiry_pct','expiry_byte_pct','delay_mean_ms','delay_p95_ms','worst_ue_expiry_pct','success_pct','queue_mean_mb','prg_occupancy_pct','sinr_tx_mean_db']
    means=df.groupby(['scenario','arm'])[cols].mean()
    print(means.round(5).to_string())
    print('\nPAIRED DIFFERENCES DA minus baseline')
    paired=pd.read_csv(AN/'paired_differences.csv');print(paired.round(6).to_string(index=False))
    print('\nDA individual seeds')
    print(df[df.arm=='mlwdf_pf'][['scenario','seed']+cols].round(6).to_string(index=False))
    groups=pd.read_csv(AN/'groups_mean.csv')
    print('\nGROUPS')
    print(groups[['scenario','arm','dimension','group','payload_mbps','expiry_pct','success_pct','delay_mean_ms','worst_ue_expiry_pct']].round(5).to_string(index=False))
    print('\nUE zero expiry')
    ue=pd.read_csv(AN/'per_ue_all.csv');print(ue.groupby(['scenario','arm']).expiry_pct.agg(['min','max',lambda x:int((x>0).sum())]).to_string())
    cells=pd.read_csv(AN/'per_cell_all.csv');c=cells[cells.scenario.isin(['G3','G4']) & cells.arm.eq('pfq')]
    print('\nObserved input by load role')
    print(c.groupby(['scenario','heavy']).offered_mbps.agg(['mean','min','max']).round(5).to_string())
if __name__=='__main__':main()

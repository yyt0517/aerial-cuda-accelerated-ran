"""Complete rate-distribution and identity tables from existing experiment artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .corrected_uniform_metrics import describe
from .dpa_ablation_experiment import sha, write_json

RATE_FIELDS=('goodput_mbps','served_mbps','mac_packet_service_rate_mbps',
             'mac_packet_expiry_aware_rate_mbps','successful_byte_outcome_time_rate_mbps',
             'pre_avg_rate_mbps_mean')


def complete_rate_tables(out):
    out=Path(out)
    ue=pd.read_csv(out/'per_ue_metrics.csv')
    rows=[]
    for window,data in ue.groupby('window'):
        assert len(data)==36 and set(data.ue_id)==set(range(36))
        for cell in [-1,0,1,2]:
            group=data if cell==-1 else data[data.cell_id==cell]
            for field in RATE_FIELDS:
                x=group[field].to_numpy(dtype=float)
                valid=x[np.isfinite(x)]
                rows.append(dict(seed=int(group.seed.iloc[0]),policy=group.policy.iloc[0],window=window,cell_id=cell,
                    metric=field,unit='Mbit/s',ue_population=len(group),unavailable_ue_count=int((~np.isfinite(x)).sum()),
                    jain=float(valid.sum()**2/(len(valid)*(valid**2).sum())) if len(valid) and np.any(valid) else None,
                    **describe(x,'ue_rate')))
    pd.DataFrame(rows).to_csv(out/'ue_rate_distributions.csv',index=False)
    catalog=pd.read_csv(out/'metric_catalog.csv').fillna('')
    extra=[]
    for field in RATE_FIELDS:
        for statistic in ['count','mean','min','p5','p10','p50','p90','p95','p99','max','jain']:
            extra.append(dict(metric=field+'.ue_distribution.'+statistic,legacy_definition='',
                source='per_ue_metrics.csv',window='full/measured; cluster/per-cell',
                semantics='Across all associated UEs, including zero valid rates; undefined packet rates excluded with explicit unavailable count.',
                availability='canonical CSV',canonical_source='ue_rate_distributions.csv',
                unit='ratio' if statistic=='jain' else ('count' if statistic=='count' else 'Mbit/s'),
                denominator='N*sum(rate^2) for Jain; otherwise UE sample population, NOT averaged packet quantiles',
                zero_sample='NA and count=0'))
    existing=set(catalog.metric)
    catalog=pd.concat([catalog,pd.DataFrame([r for r in extra if r['metric'] not in existing])],ignore_index=True)
    catalog.to_csv(out/'metric_catalog.csv',index=False)


def complete_dpa_manifests(root,baseline):
    root=Path(root);baseline=Path(baseline)
    protocol=json.loads((root/'protocol.json').read_text())
    for out in [p.parent for folder in ['validation','frozen'] for p in (root/folder).glob('*/validation_metrics.json')]:
        identity=json.loads((out/'validation_metrics.json').read_text())
        manifest=json.loads((baseline/f"seed{identity['seed']}_pfq/reference_manifest.json").read_text())
        command=json.loads((out/'command.json').read_text())
        manifest.update(reference_policy='dpa',command=command['command'],environment=command['env'],
                        checkpoint_sha256=identity['checkpoint_sha256'],checkpoint=identity['checkpoint'],
                        source_revision=protocol['source_revision'],source_sha256=protocol['source_sha256'],
                        binary_sha256=protocol['binary_sha256'],libcumac_sha256=protocol['libcumac_sha256'],
                        decode='argmax',guard='off',model_device=protocol['model_device'],
                        paired_inputs_verified=True,training_scene=True,
                        schema='corrected-uniform-v1-fifo-success',
                        lifecycle_semantics='MAC-completed, expired and pending distinct. Fully successful packets require zero failed dequeue bytes. Expiry terminal and event TTI separate; unused preparation excluded.')
        write_json(out/'manifest.json',manifest)
        complete_rate_tables(out)

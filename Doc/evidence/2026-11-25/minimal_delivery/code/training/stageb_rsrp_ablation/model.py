"""Controlled steps with explicit single-reference-port bridge.

Differences between adjacent stages are conditional on this order. The final
step includes periodic sample/hold AND transport delay, not transport alone.
"""
from dataclasses import replace
import numpy as np
from training.stageb_rsrp.model import approximate_problem

STAGES=('exact','spatial','frequency','pilot','filtered','reported')
LABELS={
    'exact':'原始瞬时协方差',
    'spatial':'逐PRG等功率各向同性',
    'frequency':'全端口宽带功率',
    'pilot':'单参考端口当前功率',
    'filtered':'单端口滤波功率（即时可用）',
    'reported':'滤波＋周期上报／延迟',
}
FACTORS=('空间协方差丢失','频域平均','单参考端口采样','时间滤波','周期上报与传输延迟')


def transform(p, stage, *, filtered=None, reported=None):
    if stage not in STAGES:raise ValueError(stage)
    if stage=='exact':return p
    if stage=='spatial':
        # trace(H H^H)/4 is the exact instantaneous power per RX branch.
        variance=np.square(np.abs(p.h)).sum(axis=(-1,-2))/4.0
        h=np.zeros_like(p.h);diagonal=np.arange(4)
        h[...,diagonal,diagonal]=np.sqrt(variance)[...,None]
        h[np.arange(p.U),:,p.serving]=p.h[np.arange(p.U),:,p.serving]
        return replace(p,h=h)
    if stage=='frequency':power=np.square(np.abs(p.h)).mean(axis=(1,3,4))
    elif stage=='pilot':power=np.square(np.abs(p.h[...,0])).mean(axis=(1,3))
    elif stage=='filtered':
        if filtered is None:raise ValueError('Missing current causal filtered measurement')
        power=filtered
    else:
        if reported is None:raise ValueError('Missing delayed report')
        power=reported
    return approximate_problem(p,power)


def from_row(p,stage,row):
    return transform(p,stage,filtered=row.get('rsrp_filtered'),reported=row.get('rsrp_report'))


def check_counters(info):
    # Exact disjoint partition over candidate occurrences across ALL rounds.
    total=sum(info[k] for k in ('scored_edits','known_exact_reuses','floor_pruned',
                                'threshold_pruned','certificate_heap_pruned'))
    if total!=info['sum_neighborhood_sizes']:raise AssertionError('Candidate accounting mismatch')
    if info['candidate_evaluations']!=info['scored_edits']+info['initialization_evaluations']:
        raise AssertionError('Initialization accounting mismatch')
    if info['heap_pushes']!=info['scored_edits']+info['certificate_heap_pruned']:
        raise AssertionError('Certificate heap accounting mismatch')

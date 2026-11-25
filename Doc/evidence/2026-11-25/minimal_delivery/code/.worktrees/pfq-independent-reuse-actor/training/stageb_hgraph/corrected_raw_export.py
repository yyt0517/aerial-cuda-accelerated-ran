"""Opt-in pre-action observables for the fixed corrected-environment experiment."""
import csv
import numpy as np
from .feature_snapshot import _UE_FEATURE_NAMES


class RawObservationExport:
    def __init__(self, path):
        self.stream = open(path, 'x', newline='')
        self.writer = None

    def write(self, obs, tti):
        features = np.asarray(obs['obs_ue_features'])
        sinr = np.asarray(obs['post_eq_sinr'])
        if features.shape != (36, 16) or sinr.shape[:2] != (36, 17):
            raise ValueError('corrected export requires full 36 UE / 17 PRG observations')
        fields = ['tti', 'ue_id'] + ['pre_' + k for k in _UE_FEATURE_NAMES]
        fields += [f'pre_post_eq_sinr_lin_prg{p}_layer{l}'
                   for p in range(17) for l in range(sinr.shape[2])]
        if self.writer is None:
            self.writer = csv.writer(self.stream)
            self.writer.writerow(fields)
        for u in range(36):
            self.writer.writerow([tti, u, *features[u].tolist(), *sinr[u].reshape(-1).tolist()])

    def close(self):
        self.stream.close()

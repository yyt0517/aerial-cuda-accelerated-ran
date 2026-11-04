DPLS component ablation (2026-10-29)
==================================

This directory contains a frozen, boundary-only attribution experiment. It
reuses the established five-seed, continuous-channel native simulator and
the complete packet-cohort metric definitions. The active algorithm registry
and all original experiment sources are unchanged.

Read ``protocol.json`` for the eight arms and interpretation limits. The six
distinct CTBS/global-search/weighting combinations implement the full logical
three-factor design; two independent-cell controls separate local physical
optimization from global coordination. Independent cells hold all neighbors
at the initial draft, optimize only their own users, enforce per-cell 90%
dequeue floors, and commit simultaneously. This is a particular independent
control; its contrast also includes joint edits and shared floor pooling.

``preflight.py`` checks native byte equality to the original promoted policy
and exact search/candidate/prediction parity on twelve saved production
states. ``campaign.py`` runs thirty new full trajectories; ten verified
PFQ/full trajectories are reused. ``analyze.py`` validates every trajectory,
reconstructs complete FIFO packet cohorts, and exports primary paired
contrasts, nominal confidence intervals, Holm-adjusted primary p-values,
interaction terms, and Shapley allocations over all six component orders.

Runtime
-------

Run inside the existing ``c_aerial_oai2`` container using
``/opt/nvidia/cuBB/.venv/bin/python``. Every script establishes its own import
path to the existing research worktree. The simulator and frozen shared
libraries must already exist. Output directories refuse overwrite. For
reproduction, copy only the source scripts into a new version directory, update
the reference paths only if necessary, and run ``setup.py`` before any
new trajectory. Then run ``preflight.py``, ``campaign.py``; the campaign
starts analysis automatically. Merely rerunning setup on this frozen
directory is intentionally rejected.

The first launch was interrupted during common PFQ warmup because thirty
separate GPU contexts caused contention. Its incomplete native files and
logs are preserved under ``attempts/01_separate_gpu_contexts`` and never
enter the analysis. The resource adjustment is recorded in
``resource_adjustment.json``. The second launch uses a dedicated CUDA MPS
server under ``runtime/``. Its environment is recorded in the campaign
launch file; exact native parity is rechecked under that environment.
The first MPS launch exceeded the server's 120 compute-connection limit
(eight connections per client). That startup failure and its partial
warmup-only files are preserved under ``attempts/02_mps_default_connections``.
The successful campaign sets ``CUDA_DEVICE_MAX_CONNECTIONS=2`` for thirty
clients. No startup-failed or interrupted trajectory enters the analysis.
The MPS daemon belongs only to this experiment and should be shut down
after all trajectories finish.

The frozen global score includes the historical BLER interpolation behavior
to isolate the established algorithm. All new arms use that same scorer;
this experiment does not claim it is a new canonical physical model.
Concurrent wall times are descriptive and must not be interpreted as fair
latency comparisons or real-time deployment measurements.

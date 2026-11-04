Paired Boundary M-LWDF-PF and PFQ comparison
=========================================

This archive recomputes existing traces only. There are no new native steps,
PHY evaluations, fitted parameters, or timing experiments. Read ``report.txt``
for the Chinese report, ``comparison.json`` for the complete statistics and
validation, and the CSV files for metric, seed, UE and common-packet results.
The three exposed development seeds and two-second window are separate from
the October27 five-seed ten-second registered-policy experiment.

The reproducible command, from the repository root, is::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.compare_mlwdf_pfq --output deliverables/mlwdf_pfq_comparison_NEW

It requires NumPy, pandas, and the original local artifacts identified in
``input_manifest.json``. It rejects an existing output directory. It checks
the original native field definition, per-TTI service identity, common packet
identities and uncensored cohorts, then reproduces archived proposed-arm
metrics with absolute tolerance1e-8. ``code/compare_mlwdf_pfq.py`` is an audit
snapshot of the repository module; invoke the repository module for replay.
Raw native traces and full baseline cohorts remain local; a fresh clone can
inspect this compact result but cannot reconstruct the raw experiment alone.

Successful-delay statistics are conditioned on complete delivery. Packet
fractions and delays pool seeds; throughputs, UE P5, Jain, PRG use and TB error
rates average seeds; worst-UE expiry maximizes over(seed,UE). The common-success
CSV supplies direct packet pairing separately. Zero expiry is not zero loss.

``engineering_timing.csv`` derives from the unchanged
``../B_decoupled_and_tournament/engineering/wall_summary.csv``. Those serial
saved-state measurements use the old deficit-weight k0 inputs. They do not
measure M-LWDF-PF speedup over PFQ, and they are not continuous deployed tails.
The M-LWDF-PF concurrent quality-run wall times are descriptive only.

Focused solver checks, completed during this documentation update::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_bounded_dpls training.stageb_lns.test_certified_bounds training.stageb_lns.test_tournament training.stageb_lns.test_crossload_weights

All17 tests passed. Zero-work counts describe report recomputation; synthetic
unit-test PHY calls are outside the formal experiment work totals.
Old frozen reports, manifests, failed gates, the active
algorithm registry and the native runner remain unchanged.

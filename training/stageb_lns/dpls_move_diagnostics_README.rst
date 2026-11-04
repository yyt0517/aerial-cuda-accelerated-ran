Offline diagnostics of existing DPLS moves
=========================================

This analysis reads completed step-0 NPZ traces and the twelve already saved
causal inputs. It runs no simulator, MMSE or candidate PHY evaluation. Both
physical scorer entrypoints are patched to raise if invoked during analysis.

It measures accepted edits per round, changes in matching edit operators'
root-relative gains, independent positive-move packing, and the accepted edit's
rank under fixed scalar-SIR proxies. Operator identities use PRG and overwritten
cell targets; full schedules and row numbers are not stable identities.

The dependency closure includes the edited PRG and affected UE intersections.
An activity change includes all active neighboring assignments on that PRG.
Packing requires distinct PRGs and disjoint affected UEs. Shared dequeue-floor
constraints are calculated from queue/count/min-layer/TBS on the saved-input
subset. Packing bounds are retained if the integer solver reaches its time
limit. Positive gains above 1e-6 bytes and the original 1/16-byte acceptance
threshold subset are reported separately.

One logged floating tie changes the unguarded cached choice. Logged fresh
scores simulate paid near-tie refreshes without new PHY calls. This is an
offline work/choice diagnostic, not an implemented online lazy scheduler or
a universal floating-error guarantee. Candidate refresh ratios are not speedup
measurements. The three arms' endogenous states differ, and all observations
are correlated transient samples from seed41.

Run from the repository root using fresh output directories::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_dpls_move_analysis -v

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_lns.analyze_dpls_moves --source deliverables/anytime_dpls_20261103_smoke_v3 --output deliverables/dpls_move_diagnostics_NEW

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.report_dpls_moves --source deliverables/dpls_move_diagnostics_NEW --output Doc/evidence/NEW/dpls_move_diagnostics

The November 3 report uses ``deliverables/dpls_move_diagnostics_20261103_v2``.
The first exploratory analysis (v1) counted only the native-threshold-qualified
pool. v2 also counts all numerically positive moves and includes the floating
tie diagnostic; neither experiment adds any physical scoring. Outputs and
source versions are retained separately.

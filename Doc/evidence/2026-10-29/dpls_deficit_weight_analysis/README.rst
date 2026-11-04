Deficit-weight follow-up
========================

This is a read-only analysis of the frozen boundary ablation. It adds no
native trajectories, model training, parameter tuning or default changes.
``analyze.py`` verifies the archived input hashes, reproduces original
paired contrasts, compares all metrics and matched users, and checks ten
complete packet cohorts before deriving delivery-by-deadline curves and
same-packet latency comparisons. Its outputs refuse overwrite.

Run the script from a fresh ``deliverables/<new_version>/`` directory with
the existing ``c_aerial_oai2`` container's Python. It needs the previous
archived CSV files and the local full/ctbs_global_equal packet cohorts.
The evidence copy of the script is a source snapshot; copy it back to a
fresh deliverables directory before running so repository paths resolve.

The primary payload comparison retains the prior predeclared Holm result.
New fairness, user-subset and deadline results are post-hoc exploration;
their paired intervals are nominal and use five seeds, not individual
users or packets as independent repetitions. Baseline-worst users and
both-success packets are explicitly conditional subsets.

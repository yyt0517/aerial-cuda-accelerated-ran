Independent masked joint-repair experiment
========================================

This is a first-version experiment combining MACSIM-inspired shared joint
scores and multiple complete-action heads with MaskCO-style conditional masked
reconstruction. It is not a reproduction of either paper, and is not imported
by an active scheduler or registered as the current algorithm.

Architecture and constraints
----------------------------

The 3-cell checkpoint uses 12 UE features, the existing 17 slot features and
89 candidate-edge features. Eight edge features contain causal channel hints;
81 contain normalized Hermitian interference covariance coordinates, desired
signal real/imaginary coordinates and noise. Static edge embeddings are cached
only within a decision. Two width-32 relation aggregation layers refresh the
verified incumbent state. Four heads produce a complete mask's scores; each
head generates one greedy and one sampled completion. A compiled lightweight
decoder updates PRG counts, minimum layers, TBS lookup and activity-dependent
rough SINR, without any candidate MMSE or BLER query.

UEs may occur on multiple PRGs. Only assigned slots are removed from the
sampling domain. Masks are distinct from blank choices. Complete candidates
retain the mask complement exactly and are validated by the unchanged exact
PHY. The original PFQ+CTBS 90% dequeue floor remains fixed throughout a decision.
The acceptance thresholds follow the original search's first-two-round 1-byte
and subsequent 16-byte weighted-objective improvement rules.

Training
--------

Seeds 41--43 provide 36 saved states from current and PFQ-only histories;
44--45 provide 24 validation states. Adjacent snapshots and masked examples are
correlated, not independent radio environments. Teacher searches keep each
mask's complement fixed. Global, 4-, 8- and 16-position maximum masks are used;
actual local masks can be smaller where the graph has fewer related positions.

Supervised training matches four complete teacher alternatives to the four
heads, uses exact-gain weights, randomizes the assignment order, and normalizes
sequence loss by mask length. Two self-improvement rounds collect the current
student's own endpoints and add physically searched alternatives. Best-of-eight
policy-gradient updates use fresh stochastic candidates and leave-one-out
maximum-reward credit. Checkpoint selection uses validation physical J, not
imitation loss. The final selected checkpoint can precede a harmful update.

The initial protocol uses 40 supervised epochs, two rounds of 10 additional
epochs and 36 Best-of-eight updates per round. All training and selection use
the weighted expected-byte J objective; there is no multi-TTI PPO or long-term
packet-return optimization in this experiment.

Inference and comparison
------------------------

From the PFQ+CTBS draft, generate eight global proposals, then perform two
rounds of four masks with eight proposals each. Each round uses one incumbent;
accept only the best complete candidate, never a combination of independently
accepted overlapping masks. The direct solver evaluates at most 73 complete
schedules including the initial draft (74 with a distinct warm-start schedule).
Duplicate and unchanged candidates are removed before scoring.

Independent test-state replay uses historical seeds 46--48, PFQ-only, neural
and greedy source trajectories, at TTIs 3001, 4001 and 5000: 27 states. These
are historically observed development states, not unseen deployment evidence.
No test outcomes update the model or search budget.

Primary controls include the frozen 79-input repair checkpoint with its
equivalent cached decoder, physical beam width 16 with every legal UE and
block size 4 or 16, and the original converged greedy search. Three serial
repetitions alternate method order. A decision timer includes decoded-snapshot
conversion, MCS/TBS handling, PHY construction, features, network, generation,
verification and destruction. Model/library loading, immutable TBS-table
creation, archive IO, IPC and the simulator are excluded. This is a CPU
saved-state decision measurement, not deployed 0.5-ms scheduling latency.

Escape testing separately prepares (a) original-neighborhood local optima with
an exhaustive eligible-neighbor certificate at tolerance 1e-7, and (b) stopped
physical width-16 block searches. The latter are bounded beam stop points,
not certified exact optima. Preparation is not hidden inside speedup claims.

The mask-reuse optimization computes identical interference priorities once
for all sizes. The pre-optimization code and timing remain in
``before_mask_reuse/`` and ``evaluation/``. The final timing is in
``evaluation_mask_reuse/``. Exact actions and physical work are checked against
the original run before and after optimization. No model weights change.

Reproduction
------------

Use an unused output directory. The existing virtual environment and Eigen PHY
library are sufficient; no native simulator or GPU is required::

    .venv/bin/python -m training.stageb_lns.build_masked
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_masked_joint training.stageb_lns.test_physics training.stageb_lns.test_repair training.stageb_lns.test_cached_repair -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.train_masked_joint --output deliverables/my_masked_joint_v1
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.eval_masked_joint --training deliverables/my_masked_joint_v1 --output deliverables/my_masked_joint_v1/evaluation_mask_reuse
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.diagnose_masked_joint --training deliverables/my_masked_joint_v1

Run timing after training, tests and other replay work have finished. Original
frozen sources are also copied with the experiment artifacts. No frozen
checkpoint, source trajectory or active algorithm registry is overwritten.

Design references
-----------------

* MACSIM, ICLR 2026: https://arxiv.org/abs/2510.12273
* MaskCO, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html

The report describes this adaptation's measured outcomes; results from those
papers are not evidence of scheduler improvement here.

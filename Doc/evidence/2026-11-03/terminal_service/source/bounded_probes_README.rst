Three bounded AI scheduling probes
=================================

Independent saved-state experiments, without CTBS, demand/headroom pruning,
PPO or native trajectories. Causal deficit weights and the current state's
unpruned PFQ 90% dequeue floor are shared with the canonical full DPLS control.
They do not use the equal-weight PPO baseline or the older BLER scorer.

The protocol is written before collection. Training/selection use seeds
41--43 / 44--45, two CTBS-free histories, three fixed TTIs per history and
full-search roots at iterations 1/5/13 if present. Models are selected only by
validation regret@8. Development seeds46--48 provide nine PFQ states; these
seeds have prior exposure and are not blind tests. Seeds311--315 are never
read. Three serial alternating repeats are summarized as per-state medians,
then equal-weight seed means.

1. Compare 48 causal scalar inputs with 120 inputs that add six affected-UE
   slots containing complete-TB resources and per-layer SINR. The 2448-entry
   current-state activity/MMSE bank, aggregation and all features are charged.
   They query no candidate BLER. This is an input-value probe, not free PHY.
2. Compare random/risk/learned ordering of bounded single-PRG assignment blocks
   with the same 4096 physical-candidate cap and four-sweep limit. Incumbent,
   blank and at most two alternatives per cell define the finite local domain.
   Separately, conservative expected-byte upper bounds certify skipped original
   DPLS candidates. A sparse score buffer preserves the full neighborhood's
   numerical reduction shape and original index tie rule.
3. Independently assess PFQ plus at most seven residual paths with eight edits
   each. Complete candidates receive exact verification and greedy selection;
   no search teacher or PPO runs online. Distinct actions are retained by J,
   without promising per-UE service diversity.

The saved-state quality gate requires every state J>=99.5% of full. The cost
gate requires complete mean time<=80% of full. Input gains, candidate counts,
engineering correctness and joint scheduler success are separate conclusions.
The certified search is a physical control, never attributed to learning.

Preparation v1 completed the three probes. Its later scalar-only attribution
exposed a floating tie: full and short strided NumPy reductions exchanged two
independent equal-score edits in one state, with the same final action. v1 is
retained. v2 changes reduction layout, adds its regression test, and reruns the
fixed protocol; no labels, hyperparameters or selected model weights change.
Both v1 and historical development exposure must be disclosed for v2.

Run from the repository root in the existing environment, using fresh output
and evidence directories (existing artifacts are never overwritten)::

    .venv/bin/python -m training.stageb_lns.bounded_probes --build
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_bounded_probes training.stageb_lns.test_no_ctbs_inputs training.stageb_lns.test_ranked_dpls -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.bounded_probes --output deliverables/new_three_probes
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.bounded_probe_followup --source deliverables/new_three_probes

The reporter performs full edit-index replay, records offline teacher work,
checks the tests log and exports small evidence. It does not train, select or
change the scheduler. The archived report contains experiment-specific counts
and interpretation; replay measurements are the source of truth. Full models,
data and original source snapshots stay in deliverables. This remains a
research package dependent on the existing historical artifacts and libraries.

No saved-state result establishes packet throughput, TTL, packet-delay gains,
cross-scene generalization or a 0.5ms deployment deadline. The active algorithm
registry remains untouched.

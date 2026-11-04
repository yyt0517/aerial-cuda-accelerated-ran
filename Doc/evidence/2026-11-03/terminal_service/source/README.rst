TB-aware local block search research
===================================

This package is a separate scheduling research implementation, with saved-state
experiments and a native closed-loop adapter. It does not replace
``training/stageb_hgraph/current_algorithm.json`` or the active native runner.
The experiment report is in
``Doc/evidence/2026-10-27/tb_lns/report.txt``. Full outputs, model checkpoints,
and frozen source copies are under ``deliverables/tb_lns_20261027_*``.

Implemented scope
-----------------

* Source-to-victim interference proposals plus same-UE, cross-PRG TB edges.
* Local joint repair with a bounded beam. Intermediate schedules can have lower
  scores, while the returned incumbent is always legal and verified.
* C++/Eigen affected-UE evaluation with on-demand MMSE caching. There is no
  global ``2**number_of_cells`` activity bank. The complete TB of every actually
  affected UE is recomputed, including PRGs outside the proposed block.
* Canonical native descending BLER lookup, including the nonmonotonic MCS3 row.
  Candidate scores do not depend on evaluation order or batch partition.
* The current causal floor: ceil(90% of this state's own PFQ+CTBS dequeue).
  No coordinator, external floor trajectory, or future CRC is required.
* A small supervised block ranker, with complete-block counterfactual targets.
* A budget-conditioned next-block/STOP policy trained with group RLOO returns.
  This optimizes fixed-state search return, not long-term packet delivery.
* Untrained adaptive control, random LNS, graph LNS, and original-neighborhood
  greedy controls. Greedy preserves the original candidate order, except for
  its two redundant incumbent rows, and uses batch candidate construction.

The second stage adds an autoregressive neural joint repair decoder. Each next
assignment conditions on the partial block; only complete proposals are passed
to the physical verifier. Outer, multi-TTI PPO is not implemented or claimed.
The learned block ranker is not a graph message-passing network; it pools causal
node/TB summaries and explicit directed interference information over blocks.

Physical assumptions
--------------------

The kernel implements the existing activity-only, 4T4R interference model:
changing the scheduled UE without changing a cell's activity does not alter
neighbor interference. Serving precoders and each UE's MCS are fixed during a
decision. Adding UE-dependent interfering precoders, power redistribution,
HARQ, or a different TB abstraction requires updating the dependency closure
and the independent reference. Activity masks currently support 1--63 cells;
larger cell counts and other antenna dimensions are rejected.

Build and validation
--------------------

Run from the repository root. The existing virtual environment supplies NumPy
and PyTorch; the native kernel needs g++ and Eigen headers::

    .venv/bin/python -m training.stageb_lns.build
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_physics training.stageb_lns.test_repair -v

The tests include independent full-UE NumPy scoring, affected-UE/full equality,
batch and permutation invariance, a cross-PRG layer bottleneck, physical
neighbors omitted from the proposal graph, a joint edit that crosses a BLER
valley, legal masks, anytime output, STOP, and rejection of late improvements.

Replay additionally checks the original archived candidate order and PFQ/CTBS
drafts. ``archive.py`` uses only repository-owned pickle files and resolves their
legacy CTBS definitions from ``.worktrees/pfq-independent-reuse-actor``. Run it
in a dedicated process, not inside a process running the main-tree bridge.

Reproduce a new experiment
-------------------------

Every output directory must be new. Completed experiment outputs are never
overwritten. Static TBS-table construction and library loading happen once,
outside measured decisions. Use a single BLAS thread::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.learning --output deliverables/my_lns_training --train-seeds 41 --validation-seeds 42
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.budget_learning --initial deliverables/my_lns_training/block_ranker.npz --output deliverables/my_lns_budget_training
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.experiment --output deliverables/my_lns_test --seeds 43,44,45 --ranker deliverables/my_lns_training/block_ranker.npz --budget-ranker deliverables/my_lns_budget_training/budget_ranker.npz --budget-initial deliverables/my_lns_budget_training/budget_initial.npz

The recorded first iteration used seed 41 for training, seed 42 for checkpoint
selection, and seeds 43--45 only for final reporting. Those test seeds are now
observed: further policy tuning needs new evaluation seeds. The heldout_v2 run
kept the same trained models and hyperparameters as heldout_v1; its only search
change removed unnecessary concatenation from the greedy control. Both runs
are retained rather than selecting whichever timing is more favorable.

Timing and interpretation
-------------------------

The timer starts at decoded causal snapshot input and includes input conversion,
MCS reconstruction, CTBS pruning, native context setup, graph construction,
model inference, candidate generation, physical verification, and cleanup.
It excludes production of the native PFQ observation, IPC, archive loading,
process startup, compilation, and one-time immutable TBS tables. It is not a
complete deployed scheduler latency measurement.

Budgets are checked between indivisible operations. A candidate whose
verification finishes after the deadline cannot replace the incumbent, but an
operation can still finish late. Elapsed time and overrun are reported. Even
baseline initialization may exceed a sub-millisecond budget; returning the
baseline then does not establish a real-time guarantee.

The reported retained gain is (J_method-J_initial)/(J_full_search-J_initial),
averaged per saved state/repetition. It is not throughput, capacity, or a ratio
of total mean gains. Both mean gain in weighted expected bytes and mean retained
gain are exported because their rankings can differ. Full local search is a
reference, not a global optimum. Closely spaced snapshots and timing repeats
are correlated and must not be treated as independent native rollout seeds.

Next gates
----------

1. Expand causal training states before interpreting small learned-policy gains.
2. Compare learned repair and physical repair across held-out topologies, loads,
   mobility and longer measurement windows, using equal candidate and wall costs.
3. Introduce a multi-TTI objective only after physical gain and native packet
   outcomes are both stable; keep a verified incumbent and the causal floor.
4. Profile graph/context reuse and compiled repair control. Production latency
   measurements remain a later gate, not the objective of the quality-first run.

Quality-first joint repair and native verification
-------------------------------------------------

``train_repair.py`` collects complete physical block repairs from PFQ+CTBS,
partially improved and converged search roots. Training seeds are 41--43;
validation seeds are 44--45; new native evaluation seeds are 46--48. These are
the second-stage splits: they intentionally differ from the first-stage split
described above. Once reported, 46--48 are no longer unseen test seeds.

The network has 7,233 parameters and 79 causal summary features. It minimizes
gain-weighted whole-sequence negative log likelihood, with soft targets over
verified alternative complete repairs. Validation NLL selects the checkpoint;
physical gain and packet delivery are evaluated separately. Lower imitation
loss does not by itself establish a better scheduler. ``ablate_repair.py``
trains 31-, 27- and 14-input controls on the same training data. Their first
layer is padded with zeros for a common inference interface: this tests input
information, not reduced feature-construction runtime.

``quality.py`` exposes four no-deadline search policies: converged original-
neighborhood greedy, neural repair from the PFQ+CTBS draft, greedy plus neural
repair, and greedy plus physical block repair. The last two optionally apply
greedy polishing. The hybrid keeps the original draft floor and cannot lower
the same-state physical objective below its converged greedy seed. This is not
a guarantee of higher long-term native goodput. Neural deltas and physical
polishing deltas are recorded separately. A deterministic full block scan with
no accepted improvement stops immediately; repeating it cannot change the
incumbent. Physical repair retains its stochastic diversity scans.

``native_repair.py`` connects these policies to the existing corrected 4T4R
simulator through fresh pre-action observations. It verifies every action,
layer, MCS, capacity and dequeue against native results; updates deficit EMA
only after actual goodput; checks sampled SINR/BLER; and reconstructs FIFO packet
outcomes with byte conservation. ``native_campaign.py`` freezes a paired
protocol with common PFQ warmup, separate policy queues, equal traffic/channel
seeds, and a 400-TTI TTL followup. Its concurrent runtime is not a real-time
latency benchmark. ``legacy_current`` retains the archived effective policy,
including its legacy BLER interpolation, as a distinct control.

Example commands from the repository root (use new output paths)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.train_repair --output deliverables/my_joint_training --device cpu
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.eval_repair --checkpoint deliverables/my_joint_training/repair.npz --output deliverables/my_joint_validation
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.ablate_repair --training deliverables/my_joint_training --output deliverables/my_joint_ablation

Native commands must run where the existing container mount, CUDA driver and
archived simulator are accessible. The current verified container is
``c_aerial_oai2``, repository mount ``/opt/nvidia/cuBB``, GPU 0. For example::

    docker exec -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -w /opt/nvidia/cuBB c_aerial_oai2 /opt/nvidia/cuBB/.venv/bin/python -m training.stageb_lns.native_campaign --checkpoint deliverables/my_joint_training/repair.npz --output deliverables/my_joint_native --workers 6

GPU access is available through this container. The earlier first-stage report
observed a restricted execution environment; it does not establish that the
host lacks a usable GPU. Historical results and their source snapshots remain
unchanged. Second-stage evidence is stored separately under
``Doc/evidence/2026-10-27/tb_joint_repair`` and
``deliverables/tb_joint_repair_20261027_*``.

Frozen direct risk-aware repair follow-up (2026-10-28)
---------------------------------------------------

The seeds101--103 experiment reduced physically evaluated candidates from
29187.51 per decision for the archived DPLS control to 1977.06 for the frozen
deadline-aware direct neural repair. This is a 93.23% reduction in physical
candidate evaluation, not a measured reduction in total computation. Concurrent
decision wall time was 168.08 ms versus 258.02 ms respectively. Neural repair
still scores beam continuations and verifies complete proposals with the PHY.

The next frozen comparison uses new seeds104--106 and five arms:
``legacy_current``, ``physical_lns``, ``risk_physical_lns``,
``risk_physical_direct`` and ``risk_neural``. The model weights remain those of
``deliverables/tb_deadline_20261028_training_v1/repair.npz``. The protocol is
2000 TTI shared PFQ warmup, 2000 settling, 10000 measured and 400 followup;
10000 TTI corresponds to five seconds. No deployment time limit is imposed.

``risk_physical_lns`` keeps the unshaped converged greedy anchor and adds the
same causal risk objective to physical beam ranking, block acceptance and
conditional polishing. The unshaped ``physical_lns`` branch remains unchanged.
``risk_physical_direct`` instead starts directly from PFQ+CTBS and uses all
legal UE options with a width-16 physical beam and at most three rounds. It
shares the risk objective, graph, floor and physical objective protection with
``risk_neural``. Physical prefix handling and stochastic beam diversity differ
from neural repair, so the comparison is not an exact FLOP or candidate match.

New counters separately record neural option scores and completed neural
proposals. They do not alter model inputs, weights, logits or returned actions.
The campaign automatically performs source/model, native, pairing, floor and
counter checks and then writes ``final_report.txt``. It retains every failure;
only completed, verified runs enter the report::

    docker exec -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -w /opt/nvidia/cuBB c_aerial_oai2 /opt/nvidia/cuBB/.venv/bin/python -u -m training.stageb_lns.direct_risk_campaign --plan deliverables/tb_direct_risk_20261028_plan_v1.json --output deliverables/tb_direct_risk_20261028_native_v1 --workers 15

Equivalent decoding with feature reuse (2026-10-28)
-------------------------------------------------

``cached_repair.py`` provides opt-in ``CachedJointRepair`` (79 inputs) and
``CachedDeadlineRepair`` (87 inputs). The reference decoders remain available.
The native runner accepts ``--decoder cached``; its default is ``reference``.
Model files and the effective algorithm registry are unchanged.

The cache belongs to a single graph/decision. Legal options and static option
features are computed once per visited token. Verified-incumbent token
summaries are rebuilt only when Evaluation identity changes; pooled block
contexts are invalidated with that incumbent. A new graph discards all cached
data, including queue, deadline, channel and legal-mask inputs. Problem and
Evaluation data must remain fixed during a solve, as in the existing search.

Beam prefixes carry UE resource counts, minimum layers, activity and sparse
edits. Previously destroyed, distinct block tokens are added incrementally;
full schedule tensors are materialized only at the end. Features retain the
reference float32 conversion order. Stable sorting, softmax, weights and
physical acceptance remain identical. Equivalence applies to searches without
a wall-clock cutoff; changing execution speed can change a timed search's stop.

``test_cached_repair.py`` covers byte-exact features/logits, ties, blank-only
options, different widths, cache reuse/invalidation and verified search work.
``benchmark_cached_repair.py`` audits complete feature/logit/proposal streams
and saved native actions on selection states, then times uninstrumented
alternating serial searches. Initialization and PHY verification are included.
``audit_cached_native.py`` compares all TTI inputs/actions/service, complete
packet tables, search work and mature-cohort means against a frozen reference.
For example (inside the existing container)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.benchmark_cached_repair --checkpoint deliverables/tb_deadline_20261028_training_v1/repair.npz --states deliverables/tb_deadline_20261028_validation_native_v1/seed48_risk_neural/states.pt deliverables/tb_deadline_20261028_validation_native_v1/seed50_risk_neural/states.pt --output deliverables/my_cached_replay --repeats 5
    .venv/bin/python -m training.stageb_lns.audit_cached_native --reference deliverables/tb_deadline_20261028_validation_native_v1/seed48_risk_neural --cached deliverables/my_cached_native/seed48_risk_neural --output deliverables/my_cached_native/seed48_audit.json

Preserving the frozen neural baseline's mean packet delay is the first gate.
It does not close its previously measured delay gap to direct physical repair.
Any later change to training, ranking, beam width or stopping needs separate
validation with mean successful-packet delay as a primary quality metric,
alongside all-outcome mean delay, complete payload, TTL and weak-UE delivery.
Selection seeds 48/50 establish implementation equivalence only. Final-test
seeds 101--106 must not be reused to select a new quality policy.
Results and evidence are maintained in
`Stage-B progress <../../Doc/stageB_current_progress.md>`_ and
`the cost report <../../Doc/evidence/2026-10-28/tb_neural_cost/report.txt>`_.

Bounded prefix verification and branch coverage (2026-10-28)
----------------------------------------------------------

``prefix_repair.py`` adds three experimental 87-input decoders without changing
actor weights or the risk objective. ``coverage`` keeps a likelihood beam with
reserved first-choice branches and verifies incumbent-suffix completions at
every depth. ``verified`` first verifies a capped neural shortlist, then ranks
prefixes by the exact risk score of those completions. ``verified_cover`` also
reserves branch coverage. This is a bounded physical process verifier guided
by a frozen neural proposal policy, not a newly trained value network.

Each depth keeps width 16. The default shortlist cap is 48: global likelihood
leaders, each parent's best two choices, then incumbent anchors and remaining
leaders. Four reserved slots cover previously unrepresented first UE/blank
choices; groups do not imply physical equivalence. The prefix value uses the
incumbent suffix as one completion, not an optimal suffix or long-term return.

All evaluations use the search's existing evaluator, original draft floor,
monotone physical objective and candidate/time budgets. Intermediate complete
actions can update the incumbent. Counters include every prefix verification;
no PHY calls are hidden inside the decoder. ``--decoder verified`` or
``--decoder verified_cover`` enables these modes for risk-neural policies;
``--prefix-budget 48 --prefix-reserve 4`` records their limits in launch.json.
The default decoder and current effective algorithm remain unchanged.

The research mapping and validation protocol are in
``deliverables/tb_prefix_20261028_v1/research.txt`` and ``plan.json``.
Tests cover shortlist coverage, stable branch reservation, incumbent completion,
budget truncation and independent physical verification. Quality changes must
pass native packet-delay/delivery checks; greater candidate recall alone does
not establish better closed-loop service.
The completed validation and its quality regressions are recorded in
`the prefix report <../../Doc/evidence/2026-10-28/tb_prefix/report.txt>`_; a lower
mean delay did not satisfy every frozen delivery-protection condition. The
effective algorithm remains unchanged.

Offline learned process value (2026-10-29)
----------------------------------------

``process_value.py`` provides causal full-TB/activity-trace features and a small
NumPy MLP ensemble. Its primary target is the exact completed-prefix risk-score
increment relative to a frozen root, with a parent-increment auxiliary head.
This is a one-step physical surrogate, not future-success process advantage.
The features do not call counterfactual PHY or consume future packet outcomes.
No online decoder, default or algorithm registry was changed to use this head.

The versioned pilot scripts are in
``deliverables/tb_process_value_20261029_v1/``. In the CUDA container, set
``PYTHONPATH=/opt/nvidia/cuBB`` and the three BLAS/OpenMP thread limits to 1;
run ``collect.py``, ``train.py``, ``evaluate.py``, ``controls.py``, ``audit.py``
and ``report.py`` in order. Scripts create output directories exclusively;
copy scripts/plan to a new version directory before reproducing the experiment.
The actor and value checkpoints, candidate arrays and sources remain local;
small audit artifacts are in
`the process-value report <../../Doc/evidence/2026-10-29/tb_process_value/report.txt>`_.

The pilot failed its predeclared 95-percent high-yield recall screen at a
10-percent unique-candidate budget. A stronger root-anchored approximate PHY
draft outperformed the learned head on recall. Empirical calibration also
failed to retain its calibration recall on development evaluation seeds.
One calibration seed cannot establish a nontrivial 2-percent distribution-free
seed-level guarantee. Feature construction still dominates head inference;
local incremental features and physics-residual learning are research next steps.
Teacher-generated-pool screening curves exclude the cost of generating that
pool and do not establish end-to-end search savings or closed-loop quality.

Approximate PHY residuals and affected-UE caching (2026-10-29)
-----------------------------------------------------------

Build the scalar approximation kernel outside measured search intervals::

    .venv/bin/python -m training.stageb_lns.build_draft

``draft_kernel.cpp`` and ``residual.py`` implement root-anchored trace and
directional-projection SINR drafts with native BLER tables. They reuse causal
H/W covariance and compute no candidate MMSE. Cache lifetime is one graph and
verified root; ``draft_repair.RebasedDraft`` reuses geometry while resetting
root-dependent ratios after acceptance. The current activity-mask cache
supports 1--8 cells; candidate work covers resource-changed UEs and victims
whose assigned PRGs see changed interfering-cell activity, including each
affected UE's whole-TB count and minimum layer.

``ResidualValue`` predicts normalized per-UE expected-byte corrections. Values
are clipped to candidate dequeue before rebuilding the original risk score.
``DraftRepair`` is an experimental search adapter: rank a bounded shortlist by
the draft/residual, verify a fixed quota through the search callback, and carry
potentially unverified prefixes. Such prefixes cannot directly update the
accepted schedule. ``FastDraftRepair`` preserves all mean scores/actions while
using validated private raw-pointer bindings and omitting unused uncertainty;
its mean-only score returns ``None`` for uncertainty, not a zero-risk claim.

Versioned scripts in ``deliverables/tb_residual_20261029_v1/`` regenerate full
legal next-step labels, train UE residuals and listwise coverage proposals,
evaluate frozen pools, replay complete searches and measure cache ablations.
Run all timing campaigns serially with one BLAS/OpenMP thread after training.
The original actor feature normalization is retained; outside-shortlist labels
from development evaluation seeds never enter training. Independent seed roles
in this round do not make previously exposed development seeds blind tests.

The cache independently reduces approximate scoring/features by 37.83 percent
against the same C++ full-UE recomputation. Frozen-pool neural gains did not
transfer reliably to complete search, and the 95-percent recall gate failed.
No decoder default, algorithm registry or native packet-delay claim changed.
Details are in `the residual/cache report
<../../Doc/evidence/2026-10-29/tb_residual/report.txt>`_ and the central progress
document. New seeds 311--315 remain unused until quality gates are satisfied.

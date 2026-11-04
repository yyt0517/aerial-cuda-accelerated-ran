Budgeted weak-UE joint proposals and paid finite-pool recall
=========================================================

Six frozen arms:full,previous fast quota-exp frontier,weak-target reservation
on existing pool(2 slots),new joint pool+2 slots,joint+4 slots,and alternating
sparse joint+2 slots. All keep212 exact complete-action scores includingPFQ,
56 roots,16 seeds,8-action batches,continuous quota-exp(.1) frontier,maxJ final.

Targets interleave low E/D and high weight*(D-E) among positive-dequeue quota
UEs. The separate compiled pool greedily acquires up to4 PRGs per target via
disjoint pair swaps,using donor penalties0/.25/1. Each step conserves resource
counts,exact min-layer witnesses and activity. First/last anchor plans compose
across disjoint serving cells,up to256 proposals per root. No native candidate
BLER/exactJ is used to generate or proxy rank them. Extra gamma reads and
pair/composition work are counted. Existing kernels stay frozen.

Reserve2/4 of the8 exact queries for rotating target-specific predicted gains
within95% root proxyJ;keep up to2 continuous-service proposals and fill withJ.
All actions still require canonical exact verification and the original floor.
Own exact archive/root fallback stays;budget reallocation cannot guarantee
parity with a separately run baseline. Alternating sparse rounds change search
coverage and are not labeled an equivalent engineering optimization.

Conditions remain paid offline teacher counts/exact layers/activity bits,
without teacher positions,J,service amounts or effective-service membership.
Two training states test engineering/disabled equivalence,then12 validation
states44--45 freeze selection before9 development states46--48. Three serial
rotating repeats. Baseline is eligible for selection;no development reselection.

After both phases,separately score the entire finite unseen wide+directed pool
at at most8 failure roots:up to2 worst failure states per phase,baseline plus
validation-selected quality method. Maximum8192 candidates/root;exclude all
already-paid main actions. Root replay/MMSE/labels are charged separately.
Diagnostics exclude seed rematching and deeper trajectories;teacher service
is an offline label only. Synthetic iteration0 shortlist recall is not actual
continuation performance. No policy tuning,training,native rollout or311--315.

Build new helper,run tests,then use single threads and a fresh output::

  .venv/bin/python -m training.stageb_lns.directed_reorder
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_directed_reorder -v
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.directed_reorder_campaign --output deliverables/directed_reorder_20261102_v1

Complete execution includes numeric input/PFQ,freshPHY/MMSE,condition checks,
generation/proxy/frontier/exact/final/context cleanup. Excludes teacher setup,
immutable firstTBS,compilation/library/archiveIO,digests and independent audit.
Common lightweight memo event copies stay inside timing. All physical/proxy
work is charged;teacher-excluded execution is not online AI inference time.
Gates remain eachJ>=99.5%full,no new unallocated UE,full-positive P5/fixed weak
20% service>=98%full,and execution mean<=80%full. Historical states are exposed.

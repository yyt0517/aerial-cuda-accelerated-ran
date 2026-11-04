Wider joint-conditioned UE--PRG reorder and J/service frontiers
=============================================================

2026-11-02 bounded follow-up. Activity-only remains an explicit reference:
its high mean J did not establish per-state quality or service coverage.
Nine paths: full; previous activity/joint212; narrow sparse-delta maxJ;
wide maxJ; narrow frontier+0.5% final J risk; wide frontier maxJ/0.5%/2%.
All candidates share212 exact complete-schedule scores (initial PFQ included),
56 roots, fixed seed,16 bootstrap candidates and canonical full PHY. No new
model, CTBS/headroom mask, PPO, native rollout or seeds311--315.

New paths use only compressed teacher counts/exact min-layer/activity bits,
never teacher UE-to-PRG positions, service amounts/groups or teacher J. Inputs
are explicit offline diagnostic conditions. Every teacher preparation is
separately charged; measured conditional execution is not online AI latency.

Narrow pool is the previous paired swaps and two orientations of3-cycles on
the first6 active PRGs. Wide pool covers all active-PRG triples plus cyclic
4-slot windows over8 fixed orders (sorted,reverse,root weak/strong gamma and4
root-keyed deterministic permutations). Every PRG participates; not all
possible4-cycles or global assignments are enumerated. Counts/activity stay
fixed, exact min-layer witnesses and legality are enforced before ranking.

Sparse proxy updates affected UE gamma sums only, equivalent in formula to
the previous root-anchored smooth reliability proxy. It never uses candidate
native BLER or exact J. All precomputed MMSE work remains inside timing;
direct gamma-array reads and seed matching reads are counted alongside native
gamma requests/misses. Low Context request counts do not mean free physical
work. Previous kernel/decoder files and their frozen evidence stay unchanged.

Service group freezes after paid bootstrap: union of UEs observed with>1B
expected service in those candidates, not the teacher-positive subset.
Archive retains exact-observed nondominated J/P5/bottom20% service candidates.
Up to4 roots include maxJ and service extremes within95% of the best currently
expandable root. Up to2/8 proposals explore projected service improvements;
all are exactly verified. Final choice either maximizes J or maximizes service
inside0.5%/2% of ONE fixed best-observed J, preventing compounding losses.
Final99.5%-full J/service/cost gates remain unchanged. The three wide frontier
selectors have identical search/query trajectories; only final selection
differs, verified in the campaign.

Engineering2 states41--43: sparse/dense proxy checks and8 solver runs, no
tuning. Validation44--45:12 historical states; select breadth/frontier/final
risk before development46--48:9 historical PFQ states. Three rotating serial
repeats, per-state medians then seed-equal means. Historical exposure is
retained. No conditional or engineering success is called learning success.

Complete execution includes numerical input/PFQ decode, fresh PHY/MMSE bank,
compressed condition validation, all pools/proxies/frontiers/exact queries,
final selection and cleanup. Compiler/library/archive IO and first immutable
TBS setup are outside; teachers and independent audits are separately charged.

Build once then use single BLAS/OMP/MKL threads and a fresh output directory::

  .venv/bin/python -m training.stageb_lns.joint_reorder
  .venv/bin/python -m unittest training.stageb_lns.test_joint_reorder -v
  .venv/bin/python -m training.stageb_lns.joint_reorder_campaign --output deliverables/joint_reorder_20261102_v1

Gates: each J>=99.5% full, no newly unallocated UE, full-positive UE P5 and
bottom20%>=98% reference, complete execution mean<=80% full. Teacher excluded
cost is conditional headroom only; upstream predictor errors/cost and packet
throughput/latency/fairness/native deployment remain untested.

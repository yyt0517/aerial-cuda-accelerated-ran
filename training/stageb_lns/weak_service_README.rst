Continuous weak-service metrics and equivalent frontier cost reduction
====================================================================

Frozen bounded follow-up,2026-11-02. Seven arms: full,activity reference,old
wide frontier,batched/cached equivalent frontier,all-quota raw-P5 frontier,
all-quota smooth-log frontier,all-quota exponential-tail frontier. All new
paths keep wide reorders,212 exact scores,56 roots,16 seed candidates,4 roots,
8-candidate batches/at most2 service proposals,and final maximum observed J.

Quota group: all UE with count>0 and D=min(queue,TBS[count,min-layer])>0.
No teacher expected service,positive membership or UE-to-PRG location enters
the solver. Conditions are still paid offline teacher counts/layers/activity.
For normalized expected ratio r=clip(E/D,0,1),new objectives are either
mean(log1p(r/.01))/log1p(100) or -.1*log(mean(exp(-r/.1))),each accompanied by
mean(r). Scales are fixed before validation. Raw all-quota P5 isolates group
expansion from smoothing/normalization. It does not isolate smoothing from
normalization separately. Final external J/service/cost gates remain fixed.

Engineering changes batch newly observed exact service,cache dequeue,increment
pairwise dominance/equality matrices,and query one frontier. Query-order ties
and dominated-candidate revival after an exhausted dominator are preserved.
Equivalence requires full ordered root/query hashes,all observed physical
values,final actions,groups,and work counters,not just aggregate J.

Two training states check engineering;12 historical validation states44--45
freeze selection before9 historical development states46--48. Three rotating
serial repeats. Teacher preparation/digests/archiveIO/independent audits are
outside complete execution timing; lightweight memo event copies are inside
old/new frontier timings equally. All MMSE/proxy work remains charged. No
training or new native trajectories;seeds311--315 remain unread. No online
AI latency or generalization claim follows from oracle-condition execution.

Existing shared libraries must already be built as in joint_reorder_README.
Use a fresh output directory and single threads::

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_weak_service -v
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.weak_service_campaign --output deliverables/weak_service_20261102_v1

Gates: every state J>=99.5%full,no newly unallocated UE,full-positive (>1B)
P5/bottom20%>=98%full,complete execution mean<=80%full. No new hard PFQ locks.

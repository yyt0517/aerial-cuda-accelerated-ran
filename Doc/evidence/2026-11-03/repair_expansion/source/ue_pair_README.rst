Per-UE service-loss features, within-state pairs and confidence fallback
======================================================================

Preserve cached existing-pool directed2 search, quota-exp frontier and212 exact
scores. New copied solver changes only terminal selector import; the old code
stays frozen. All terminal arms must have identical paid query/value archives.
Six arms:full,cached_maxj,old aggregate+fallback,perUE heuristic+fallback,
perUE pair ranker without fallback(diagnostic only),and pair ranker+fallback.

Reuse only previous seeds41-43 own-search archives and paid teacher labels;
terminal selection does not influence search. Deduplicate complete immutable
inputs plus archive/reference bytes, not just names. Runtime features have no
teacher arguments:20 shared perUE context inputs,7 monotone service loss bases
and a learned reference-shortfall basis. No UE identity embedding. Shared
20-24tanh-10 predicts8 nonnegative weights,soft reference relevance and service.
Candidate risk sums perUE losses; more service at fixed context cannot hurt.

Labels use offline full quality/service shortfall. Within-state own2%Jwindow,
up to1024 strict pairs(risk gap>.005) and1024 ties(gap<=.0001),prioritizing pairs
with maxJ; no cross-state pairs or random candidate split. Fixed240epochs Adam
.01/.001, pair logits20*score difference,0.2 auxiliary perUE effective-service
BCE and normalized-service MSE. Three leave-seed-out fits calibrate a positive
pair scale; three final initialization fits form the runtime ensemble. No
hyperparameter,epoch or confidence threshold search on validation/development.

Propose best mean score within2%ownbestJ. Accept only if all3 members give
calibrated pair-win probability>=.9 versus maxJ and positive score difference;
otherwise return exact maxJ. Confidence is relative model preference, not an
absolute coverage guarantee. Heuristic uses identical loss bases with uniform
weights,own positive frequency/q90 service,and training-only scale plus same
threshold. Old aggregate control falls back when no>=.9 terminal exists.

Historical validation44-45 selects route before development46-48;3serial
rotating repeats. External99.5%-fullJ,98%effectiveP5/fixedweak20,no newly
unallocatedUE,80%complete-cost gates stay. Teacher compressed conditions remain
paid offline; teacher preparation/modelIO/training are separately accounted.
No CTBS/PPO/final311-315/new native trajectories/default changes.

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_ue_pair -v
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.ue_pair_campaign --output deliverables/ue_pair_20261103_v1

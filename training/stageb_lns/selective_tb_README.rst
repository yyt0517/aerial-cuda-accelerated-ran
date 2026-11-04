Incremental TB descriptors and service-protected selective ranking
================================================================

2026-11-02 bounded follow-up. No CTBS/headroom masks, PPO, native rollout,
final seeds311--315 or registry change. Canonical BLER/deficit J and own raw
PFQ90% dequeue floor remain. All online candidates are exact PHY verified.

Logical kernel updates layer histograms on edited UEs, preserving full-TB
counts/min-layer/dequeue outside the edited PRGs. Only activity-changing PRGs
update the causal interference proxy. No candidate MMSE/SINR/BLER is queried
for features. Six semantic UE slots and twelve global features produce60
dimensions, including nominal weak-service risk from frozen root reliability.
Kernel proxy is checked against previous full-action formula, all logical
outputs against independent NumPy PHY. Microbenchmarks are separate from
complete decision timings; native implementation gains are engineering gains.

212 actual PHY scores including initial draft,56 distinct feature roots,5
sweeps. First sweep:4 singles/PRG, same heuristic across new guarded arms.
Then selective reserves2 single candidates and allocates remaining2 to either
family by score. Controls use single-only, fixed2+2, unprotected selective,
unprotected fixed, old wide joint and old frozen AI. Protected acceptance
requires no newly unallocated UE and effective-UE P5/bottom20% service>=98%
against BOTH immutable initial PFQ and current incumbent. Count prefilter is
logical; exact service guard is paid. Root protection does not guarantee full
DPLS coverage or packet performance. Failed actions are root-local; cached
unsafe PHY values may be reconsidered at a new root, never another decision.

One round of actual bootstrap student visits:18 training states41--43,<=4
post-first-sweep root/PRG/failed-mask visits each. Separate post-decision PHY
labels entire local pools and exact service feasibility. Teachers never affect
the student; replay equality is checked. Validation labels from the previous
headroom probe are not training inputs. The trained model's shifted visit
distribution is not recollected in this bounded round.

60->32->16->2 ReLU model, zero residual/feasibility output initially, training
normalization only. AdamW lr0.001,20epochs,checkpoints0/5/10/20. Score is causal
prior plus learned residual minus unsafe probability penalty. Training uses
safe near-optimal listwise ranking with retained-root option, feasibility BCE
and gain Huber loss. Smaller epoch wins exact validation ties. Model selection
uses validation trajectory quality/service; formal cost gate is assessed
afterward. Same fixed checkpoint gain-head-only inference is an ablation,
not a separately trained model. Changed weights alone are not learning success.

12 historical validation states44--45 freeze model and arm selection before9
historical PFQ development states46--48. Ten arms,3 serial rotating repeats,
per-state median then seed-equal means. Gates unchanged: each J>=99.5% full,
no newly unallocated UE, effective full-positive UE P5/bottom20%>=98% full,
complete mean<=80% full. Learning additionally needs>=6/9 state wins and every
seed mean improvement vs same protected selective heuristic. Historical
exposure remains; no claim of blind generalization or deployment latency.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1::

  .venv/bin/python -m training.stageb_lns.selective_tb
  .venv/bin/python -m unittest training.stageb_lns.test_selective_tb -v
  .venv/bin/python -m training.stageb_lns.selective_tb_campaign --output deliverables/selective_tb_20261102_v1

Compilation/model/archive IO and first immutable TBS setup are outside online
timing. Input/PFQ decode, fresh PHY/cache, candidate pools, descriptors/network,
exact validation, service checks, scoring and cleanup are inside. Teachers,
collection/replays, training, checkpoint validation, engineering and independent
audits are charged/saved separately. Frozen sources and binary hashes, labels,
actions/values, per-state work, negative results and selections are retained.

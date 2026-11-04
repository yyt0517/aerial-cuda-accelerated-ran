# PPO without CTBS or business weighting

This is a new retraining experiment motivated by the [DPLS component ablation](../../Doc/evidence/2026-10-29/dpls_component_ablation/conclusions.txt). It does not modify the historical HGraph experiments or the current algorithm registry.

The compact actor encodes all legal UE/PRG candidates and global context, then emits 51 autoregressive decisions in PRG-major/cell-major order. A 64-dimensional GRU conditions decisions on previously selected candidate identities, including other cells on the same PRG. Blank is an explicit choice. Masks restrict only cell association, PRG availability and initially nonempty queues. There is no CTBS pruning, sequential demand cap, deficit coefficient, service-class weight, rescue, or post-policy action repair. The frozen bridge's `ctbs` message remains solely a transport for fresh PHY, MCS and queue diagnostics; removing that transport would require a separate simulator ABI change.

This model starts with new parameters, learns the **raw_global_equal** DPLS teacher, then trains using native on-policy PPO. It is not a continuation of an old HGraph checkpoint. DPLS's own-state rho=0.90 floor applies to the teacher/comparator; the direct PPO policy does not apply a hard floor or search after sampling.

The equal-UE reward is `(fully successful completed packet bytes - full expired packet bytes - 0.01 * post-queue bytes) / 86400`. A packet is successful only when every segment succeeds. Expiry is charged on the causal removal step, one TTI after the last usable slot. All coefficients are common to all users; there is no business or service-deficit weighting. Online FIFO accounting is checked against the existing independent full-trajectory reconstruction.

The frozen campaign uses:

- Training seeds 41–43, each with 4000 PFQ warmup steps and 128 equal-DPLS teacher steps.
- 60 fixed BC epochs followed by 12 PPO iterations of 512 steps, rotating the three persistent environments. Both the BC checkpoint and final PPO checkpoint are retained.
- Full joint-action PPO likelihood ratios, actual sampled-prefix replay, GAE with next-state bootstrap at rollout truncations, and conditional-KL rejection with restoration of both actor and Adam state.
- Development evaluation on seeds 44–45: 4000 common PFQ warmup + 4000 stabilization + 4000 measurement + 400 followup steps. BC and PPO use deterministic autoregressive argmax.
- PFQ and raw equal-DPLS comparisons reuse frozen trajectories over **the same 8001–12000 arrival cohort**, rather than comparing a short PPO window with the full historical mean.

Seeds 41–45 have previous development exposure. This two-second measurement is a bounded retraining evaluation, not a replacement for the historical ten-second validation or a new-seed test. Seeds 311–315 are not used. The final checkpoint is selected by fixed budget, never by a peak training reward. Quality gates and a negative result are retained; no checkpoint automatically replaces `current_algorithm.json`.

Run from the repository root inside the existing `c_aerial_oai2` CUDA container:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s training/stageb_ppo/tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m training.stageb_ppo.campaign \
  --out deliverables/ppo_no_ctbs_equal_<new_run>/smoke --smoke

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m training.stageb_ppo.campaign \
  --out deliverables/ppo_no_ctbs_equal_<new_run>/formal
```

Every output directory refuses overwrite. The corrected native simulator, research worktree, ablation teacher and reference trajectories must already exist. Each run records source and binary hashes. Inspect `status.json`, `training_metrics.json`, `training_validation.json`, `evaluation.csv`, `evaluation_gate.json`, and `report.txt`. Large native traces and checkpoints stay under ignored `deliverables/`; small evidence is archived under `Doc/evidence/` after completion.

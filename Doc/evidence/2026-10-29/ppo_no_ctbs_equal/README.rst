PPO retraining without CTBS or business weighting
================================================

Read report.txt, evaluation.csv and evaluation_gate.json for the final
development results. Training integrity and scientific quality are separate:
training_validation.json and checkpoint_chain_validation.json verify the
actual training process; they do not establish superiority to DPLS.

The implementation is training/stageb_ppo/. This is a newly initialized,
compact joint autoregressive actor, not an old HGraph checkpoint. CTBS
pruning, headroom masks, demand caps, rescue and business/deficit weights
are absent from the PPO decision path. The old CTBS-named bridge message
only transports fresh physical diagnostics. The native PFQ baseline and
its existing zero-slack demand allocation are unchanged.

Training uses seeds 41--43, 384 equal-DPLS teacher decisions, 60 BC epochs
and 6144 on-policy steps in twelve PPO iterations. Both the BC endpoint
and fixed-budget final PPO endpoint are evaluated on development seeds
44--45. The two-second measurement window has 4000 common PFQ warmup,
4000 policy stabilization, 4000 measurement and 400 followup TTIs. PFQ
and raw_global_equal references use the exact same arrival cohort.
These are not unseen seeds or the historical ten-second validation.
Convergence is not established by the fixed training budget.

The full local experiment is
deliverables/ppo_no_ctbs_equal_20261029_v1/formal/.
It retains the BC dataset, all twelve sampled rollouts and optimizer
checkpoints, final models, native binary metrics, actions, packet cohorts
and launch records. The smoke is in the adjacent smoke_v1/ directory.
No current-algorithm registry or launcher is promoted by this campaign.

verify_training.py independently reconstructs the sampling RNG chain,
checks all 6144 actions against native execution, and checks all actor
and critic optimizer step counters. It requires the complete local run
and the existing container Python. The ordinary unit and native smoke
commands are documented in training/stageb_ppo/README.md.

This archive contains small evidence only. implementation_sha256.json
identifies source files; protocol.json records the frozen configuration
and original source hashes. archive_manifest.json covers archived files.
Existing research-worktree, frozen simulator and teacher dependencies
are required for reproduction and are not bundled here.

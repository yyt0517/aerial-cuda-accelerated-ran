Cached existing-pool two-slot search and candidate service-risk calibration
=======================================================================

Six arms: full, existing2, cached_maxj, continuous, attainable, calibrated.
All conditional arms use the exact same 212-score search and quota-exp frontier.
The new cache is confined to one immutable decision, keyed by full root; pool
local RNG and periodic seed-action global RNG semantics are preserved. Lazy
proxy records do not score unused pool candidates. Ordered root/query archives,
physical work and cached-maxJ terminal actions must equal the frozen existing2.
Generation/proxy work is reduced and reported, not required to remain equal.

Terminal choices use only paid, condition/floor-feasible candidates. Fixed 2%
internal best-J loss cap does not change the external .995 full-J gate. The cap
uses historical headroom, so this is explicitly an exposed development test.
Continuous selects quota-exp then mean reliability thenJ. Attainable uses own
perUE maximum expected service to form an effective group and reference P5 and
fixed weakest20%; it takes maxJ above .98 estimated coverage or best risk if no
candidate passes. These perUE maxima need not be jointly attainable.

The calibrated arm predicts actual offline coverage-pass labels with a fixed
42-32-16-1 tanh MLP. Inputs summarize causal quotas, weightedJ, candidate service
and own-archive max/q90/mean/bestJ references. Full teacher membership/service/J
never enters feature or runtime selector APIs. Train only seeds41-43,18 states;
three leave-seed-out fits supply monotone Platt calibration and one final fit
uses all18 states. Each fit240 epochs, Adam .01, weight_decay .001, fixed seed;
no hyperparameter/epoch selection. Threshold .9, maximumJ among predicted safe,
otherwise best probability thenJ inside the same cap. All labels arise from
already-paid own candidates and separately paid full reference actions.

Freeze model before12 validation states44-45, choose route before9 development
states46-48. Three serial rotating repeats, identical instrumentation. Timing
includes input/PFQ, freshPHY/MMSE, conditions, cache, all search, terminal features
and inference, final cleanup. Teacher, training/model file IO, immutable setup,
archive IO and independent audits excluded. Conditions are paid offline teacher
resource/layer/activity, so this is not end-to-end online AI inference timing.

Run single-threaded with a new output path::

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_terminal_service -v
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.terminal_service_campaign --output deliverables/terminal_service_20261103_v1

No CTBS/headroom/PPO/final311-315/new native trajectories/default changes.

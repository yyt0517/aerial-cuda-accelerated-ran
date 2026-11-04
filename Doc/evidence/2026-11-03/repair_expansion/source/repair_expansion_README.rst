Bounded repair-state expansion and directed repair supervision
===========================================================

Existing six training archives have only six saved TTIs each. Collect all18
previously unused source states at4000/8000/28000, seeds41-43 and both sources.
Pay fresh full teachers and unchanged cached212 search. Merge prior18 states,
deduplicate immutable inputs/archive/reference, and freeze coverage before any
validation:at leastone repairable maximumJ failure perseed,at leastthree total.
If unavailable,complete one diagnostic training comparison as requested,do not
claim coverage or launch another data search/native trajectory.

Two expanded-data controls share unchanged20-24-10 architecture,212search,
2%ownJwindow,3initialization ensemble and.9confidence fallback. Ordinary uses
previous pair/loss weighting. Focused retains all qualified repairs versus
failedmaxJ plus up to7other highestJbad candidates;good-baseline states teach
maxJvsbad and tiesvsothergood. Unrepairable states have no pair loss. Repairable
andgood-baseline cohorts share half mass,states equal withincohort. Auxiliary
UE targets balance referencepositive UEs underprotected bymaxJ against others.
Per-UE fullservice/membership remain offline labels. Runtime features unchanged.

Each variant has3leave-seed-out fits for calibration and3full-data initializations,
240epochs each;2880gradientsteps total. Include old frozen model,ordinary/focused
expanded models andfocused-no-fallback diagnostic. Validation44-45 freezes
selection before development46-48;3serial rotating repeats. Record all losses,
new harms,repair coverage and full execution,including feature/inference cost.
No threshold tuning,final311-315,new native trajectories/default replacement.

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_lns.test_repair_expansion -v
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.stageb_lns.repair_expansion --output deliverables/repair_expansion_20261103_v1

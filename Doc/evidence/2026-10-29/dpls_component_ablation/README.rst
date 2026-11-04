DPLS component ablation evidence
===============================

Read conclusions.txt and report.txt. Eight arms, five paired seeds,
thirty new full native trajectories and ten verified references.
This directory archives small results, input/physical checks, and
exact source snapshots; it does not contain native binaries, full
metrics streams, or the complete packet-cohort tables.

Canonical local experiment: deliverables/dpls_component_ablation_20261029_v1/.
The copied code is a source snapshot, not an independently runnable
package at this archive location. To reproduce, copy only code/*.py
into a fresh deliverables/<experiment_name>/ directory, retain the
existing research worktree, native simulator, frozen libraries and
reference trajectories, and run setup.py, preflight.py, campaign.py
inside the documented CUDA container. Do not copy protocol/output
files into the fresh directory before setup. See runtime_notes.rst.

manifest.json is the canonical local file manifest, so its paths
and hashes refer to that local experiment. archive_manifest.json
covers exactly the files archived here. Runtime notes below were
renamed to preserve this archive introduction.

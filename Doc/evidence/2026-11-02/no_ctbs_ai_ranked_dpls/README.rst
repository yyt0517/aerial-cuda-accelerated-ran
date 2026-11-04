CTBS-free AI ranking acceleration evidence
==========================================

Read report.txt, analysis_notes.txt, protocol.json, selection.json and the
paired tables together. Engineering checks passed, but the scientific quality
and speed gate and the independent learned-contribution gate both failed.
The selected zero-residual head follows the heuristic's actions exactly.

The canonical completed run is
deliverables/no_ctbs_ai_ranked_dpls_20261102_v1/. Large label datasets, model
checkpoints, complete original-source snapshots and action arrays remain there.
This directory contains small results and new-code snapshots, not a complete
standalone runtime. source_manifest.json names canonical local source files;
archive_manifest.json covers the files archived here. Models and libraries are
hashed but not copied into this small evidence directory.

All new decisions, teachers and controls use raw PFQ without CTBS clipping or
demand headroom masks. Historical pre-action PHY metadata is unpacked into a
neutral raw_phy mapping; ordinary CPU OLLA/MCS reconstruction remains necessary.
Training/evaluation source histories themselves are CTBS-free/PFQ-only.

Three seeds and correlated saved states are development evidence. No new
native packet trajectory or final seed311--315 was used. The registered older
effective algorithm and historical artifacts have not been overwritten.

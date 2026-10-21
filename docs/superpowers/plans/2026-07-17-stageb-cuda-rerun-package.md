# Stage-B CUDA Rerun Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, auditable archive that contains the complete tracked source at commit `62227dc`, retained Stage-B reference results, simple CUDA rerun instructions, an environment lock, and checksums.

**Architecture:** Export the committed source tree with `git archive`, then remove tracked training run/checkpoint/export artifacts and the unrelated SSH key pair so no `.git`, build directory, training artifact, or dirty-worktree change enters the package. Preserve ONNX files that belong to the released source/examples because removing them could make the source snapshot incomplete. Overlay the two retained baseline output roots, add three package metadata documents, generate a sorted SHA256 manifest, and produce a deterministic gzip archive.

**Tech Stack:** Git archive, GNU tar, gzip, SHA256, Docker/Aerial `25-3-cubb`, Bash/Markdown.

## Global Constraints

- Source snapshot is exactly `62227dc0b3bfbd644777e071370e4cae103632b0`.
- Package root is `deliverables/stageb_cuda_rerun_62227dc/`.
- Final archive is `deliverables/stageb_cuda_rerun_62227dc.tar.gz`.
- Include the tracked source tree from the source snapshot except the two unrelated root-level SSH key artifacts and tracked `training/stageb_hgraph/runs/` / `training/stageb_hgraph/exports/` experiment artifacts, and include exactly two retained reference output roots.
- Exclude `.git`, `build*`, local virtual environments, training checkpoints/runs/exports, all current dirty-worktree changes, and the tracked files `eval "$(ssh-agent -s)"` / `eval "$(ssh-agent -s)".pub` because the first is an OpenSSH private key unrelated to the simulation. Retain released-source ONNX assets outside the excluded training artifact directories.
- Record container digest `sha256:b8671dc35b245f4c49ae6c9364e01d11c5e64d60ddb3b32d45d96bc7b83e9396` and the observed A100/driver environment without claiming it is the historical run environment.
- State that historical runtime Git hash, dirty patch, container digest, compiler-options hash, and unified configuration hash remain unavailable.
- Do not run the full CUDA simulation while packaging.

---

### Task 1: Stage source, reference outputs, and rerun documentation

**Files:**
- Create: `deliverables/stageb_cuda_rerun_62227dc/REPRODUCE.md`
- Create: `deliverables/stageb_cuda_rerun_62227dc/SOURCE_VERSION.txt`
- Create: `deliverables/stageb_cuda_rerun_62227dc/ENVIRONMENT.lock`
- Copy: `output/stageB_baseline_eval_suite_fullkpi_seed41_50/`
- Copy: `output/stageB_rrq_pfq_multiseed_compare_coopboundary_assocstrongest_legacyshadow_r500_t4000_w0_s41_s50/`

- [x] **Step 1: Confirm target paths do not exist and sufficient free space is available**

Run `test ! -e deliverables/stageb_cuda_rerun_62227dc`, `test ! -e deliverables/stageb_cuda_rerun_62227dc.tar.gz`, and `df -h .`.

- [x] **Step 2: Export the exact committed source snapshot**

Create the package directory and run:

```bash
git archive --format=tar 62227dc | tar -xf - -C deliverables/stageb_cuda_rerun_62227dc
```

Expected: the package has `CMakeLists.txt`, `cuMAC/`, `cuPHY/`, `cuPHY-CP/`, and `testBenches/`, but no `.git` or `build*`.

- [x] **Step 3: Copy the two reference result roots under the package's `output/`**

Copy both roots recursively without changing the originals. Expected combined source size is about 13.5 MB.

- [x] **Step 4: Write concise build/run instructions**

`REPRODUCE.md` must cover host prerequisites, exact image/tag/digest, container entry, a seed-41/100-TTI smoke run, the formal seed-41--50/4000-TTI run, report figure regeneration, 14-test verification, expected output paths, and provenance limitations.

- [x] **Step 5: Write source and environment records**

`SOURCE_VERSION.txt` records the full source commit, Aerial version, architecture, and `17 PRG x 16 PRB`. `ENVIRONMENT.lock` records the image tag/digest, observed two A100 GPUs, driver 590.48.01, host CMake/Ninja/Python versions, and notes that CUDA compiler versions must be captured inside the container.

### Task 2: Generate checksums, archive, and verify contents

**Files:**
- Create: `deliverables/stageb_cuda_rerun_62227dc/SHA256SUMS`
- Create: `deliverables/stageb_cuda_rerun_62227dc.tar.gz`

- [x] **Step 1: Verify package scope before hashing**

Check required source, report, plotting script, test, two output roots, and metadata documents. Reject `.git`, `build*`, `*.pt`, the excluded training run/export roots, and package paths containing current untracked-only files. Do not reject released-source ONNX assets outside those training roots.

- [x] **Step 2: Generate a stable SHA256 manifest**

From the package root, hash every regular file except `SHA256SUMS` in null-delimited sorted path order, then save the manifest as `SHA256SUMS`.

- [x] **Step 3: Verify all manifest entries**

Run `sha256sum -c SHA256SUMS` from the package root. Expected: all entries report `OK`, exit 0.

- [x] **Step 4: Create the deterministic archive**

Use GNU tar with sorted names, UTC `2026-07-17` mtime, numeric owner/group 0, and gzip `-n` so the gzip header does not embed the current time.

- [x] **Step 5: Verify the archive independently**

List the archive, verify required paths and exclusions, extract into a fresh temporary directory, rerun `sha256sum -c`, confirm the extracted source commit record, and record archive size and SHA256.

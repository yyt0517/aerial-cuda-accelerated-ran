# Stage-B Dual-Package Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Windows-friendly source browsing ZIP and a top-level Chinese `README_FIRST.txt` while preserving the existing Linux/CUDA rerun archive byte-for-byte.

**Architecture:** Treat `deliverables/stageb_cuda_rerun_62227dc/` as the already-verified source of truth. Copy only regular files into a temporary `stageB_source_62227dc/` tree while excluding reference outputs and the Linux-package checksum manifest, create the ZIP from that tree, then independently inspect and extract it. Recheck the original archive checksum before and after all work.

**Tech Stack:** Bash, GNU `find`/`cp`, Info-ZIP `zip`/`unzip`, SHA256, Markdown/plain text.

## Global Constraints

- Rename the existing `deliverables/stageb_cuda_rerun_62227dc.tar.gz` to the approved delivery name `deliverables/stageB_cuda_rerun_62227dc.tar.gz`, but do not modify or regenerate its archive bytes.
- Its SHA256 must remain `1321254fe9b941536466f2c69033e59a1a764b1b3672b49f4f43b96cebded372`.
- Create `deliverables/stageB_source_62227dc.zip` with the single top-level directory `stageB_source_62227dc/`.
- Exclude `output/`, `SHA256SUMS`, all symbolic links, build/cache/training artifacts already excluded from the source package, and current dirty-worktree-only files.
- Retain source, configuration, scripts, documentation, local document images, and released-source assets.
- Create `deliverables/README_FIRST.txt` with the exact delivered filenames and Windows/Linux usage distinction.
- Do not commit or push binary deliverables.

---

### Task 1: Create delivery guidance

**Files:**
- Create: `deliverables/README_FIRST.txt`

**Interfaces:**
- Consumes: the filenames fixed by the approved design.
- Produces: the first document an acceptance reviewer reads before choosing an archive.

- [x] **Step 1: Verify original archive integrity and target absence**

Run:

```bash
cd deliverables
sha256sum -c stageb_cuda_rerun_62227dc.tar.gz.sha256
test ! -e stageB_source_62227dc.zip
test ! -e README_FIRST.txt
```

Expected: checksum reports `OK`, and both absence checks exit zero.

- [x] **Step 1a: Align the original archive filenames with the approved delivery list**

Rename the existing lowercase-`b` archive and checksum files to uppercase-`B`, then update the filename field inside the checksum file:

```bash
mv deliverables/stageb_cuda_rerun_62227dc.tar.gz deliverables/stageB_cuda_rerun_62227dc.tar.gz
mv deliverables/stageb_cuda_rerun_62227dc.tar.gz.sha256 deliverables/stageB_cuda_rerun_62227dc.tar.gz.sha256
```

Expected: the renamed archive still has SHA256 `1321254fe9b941536466f2c69033e59a1a764b1b3672b49f4f43b96cebded372`, and the checksum file references `stageB_cuda_rerun_62227dc.tar.gz`.

- [x] **Step 2: Create the Chinese first-read file**

Create `deliverables/README_FIRST.txt` with:

```text
文件说明：

1. stageB_source_62227dc.zip
   用于 Windows 环境下查看源代码、配置和目录结构。
   该浏览包不用于执行完整 CUDA 仿真。

2. stageB_cuda_rerun_62227dc.tar.gz
   原始 Linux/CUDA 复现包。包含 Linux 符号链接，应在 Linux 环境下解压。

3. stageB_cuda_rerun_62227dc.tar.gz.sha256
   用于校验原始复现包的完整性。

Windows 解压原始复现包时，如遇到符号链接权限提示，
仅查看代码可选择“全部跳过”；完整实验复现请按照 REPRODUCE.md
在指定 Linux/CUDA 环境中进行。

建议：
- Windows 源码浏览：解压 stageB_source_62227dc.zip。
- Linux/CUDA 完整复现：校验并解压 stageB_cuda_rerun_62227dc.tar.gz。
```

- [x] **Step 3: Verify names and instructions**

Run:

```bash
rg -n 'stageB_source_62227dc\.zip|stageB_cuda_rerun_62227dc\.tar\.gz|REPRODUCE\.md|Windows|Linux/CUDA' deliverables/README_FIRST.txt
```

Expected: every delivered filename and the platform guidance appears.

### Task 2: Build the Windows source browsing ZIP

**Files:**
- Create: `deliverables/stageB_source_62227dc.zip`

**Interfaces:**
- Consumes: regular files from `deliverables/stageb_cuda_rerun_62227dc/`, excluding `output/` and `SHA256SUMS`.
- Produces: a ZIP whose archive paths all begin with `stageB_source_62227dc/`.

- [x] **Step 1: Create and validate an isolated staging directory**

Run:

```bash
stage_root=$(mktemp -d /tmp/stageB_source_62227dc.XXXXXX)
mkdir "$stage_root/stageB_source_62227dc"
test -d "$stage_root/stageB_source_62227dc"
```

Expected: a new empty staging root under `/tmp`.

- [x] **Step 2: Copy only allowed regular files**

From `deliverables/stageb_cuda_rerun_62227dc/`, enumerate null-delimited regular files, reject `./output/*` and `./SHA256SUMS`, create each destination parent, and copy with metadata preservation:

```bash
src="$PWD/deliverables/stageb_cuda_rerun_62227dc"
dst="$stage_root/stageB_source_62227dc"
while IFS= read -r -d '' rel; do
  rel=${rel#./}
  mkdir -p "$dst/$(dirname "$rel")"
  cp -p -- "$src/$rel" "$dst/$rel"
done < <(cd "$src" && find . -type f ! -path './output/*' ! -path './SHA256SUMS' -print0)
```

Expected: only regular files are copied, so the `/opt/nvidia/ldpc_decoder_cubin` symbolic link is absent without being dereferenced.

- [x] **Step 3: Validate staging scope before compression**

Run checks for required files and forbidden paths:

```bash
test -f "$dst/CMakeLists.txt"
test -f "$dst/SOURCE_VERSION.txt"
test -f "$dst/REPRODUCE.md"
test -f "$dst/Doc/archive/baselines/stageB_baseline_verification_legacy.md"
test -f "$dst/cuMAC/scripts/run_stageB_baseline_eval_suite.sh"
test -f "$dst/cuMAC/scripts/plot_stageB_baseline_verification_report.py"
test ! -e "$dst/output"
test ! -e "$dst/SHA256SUMS"
test -z "$(find "$dst" -type l -print -quit)"
! rg -l --hidden 'BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY' "$dst"
```

Expected: every command exits zero.

- [x] **Step 4: Create the ZIP**

Run:

```bash
repo_root=$(pwd -P)
(cd "$stage_root" && zip -q -r -9 "$repo_root/deliverables/stageB_source_62227dc.zip" stageB_source_62227dc)
```

Expected: `deliverables/stageB_source_62227dc.zip` is nonempty.

### Task 3: Independently verify the dual-package delivery

**Files:**
- Verify: `deliverables/stageB_source_62227dc.zip`
- Verify: `deliverables/stageB_cuda_rerun_62227dc.tar.gz`
- Verify: `deliverables/stageB_cuda_rerun_62227dc.tar.gz.sha256`
- Verify: `deliverables/README_FIRST.txt`

**Interfaces:**
- Consumes: all four final delivery files.
- Produces: evidence that the original archive is unchanged and the Windows ZIP has the approved scope.

- [x] **Step 1: Test ZIP structure and CRCs**

Run:

```bash
unzip -t deliverables/stageB_source_62227dc.zip
zipinfo -1 deliverables/stageB_source_62227dc.zip > /tmp/stageB_source_62227dc.list
awk -F/ 'NF && $1 != "stageB_source_62227dc" {exit 1}' /tmp/stageB_source_62227dc.list
```

Expected: `No errors detected` and the path check exits zero.

- [x] **Step 2: Independently extract and inspect the ZIP**

Run:

```bash
verify_root=$(mktemp -d /tmp/stageB_source_verify.XXXXXX)
unzip -q deliverables/stageB_source_62227dc.zip -d "$verify_root"
extracted="$verify_root/stageB_source_62227dc"
test ! -e "$extracted/output"
test ! -e "$extracted/SHA256SUMS"
test -z "$(find "$extracted" -type l -print -quit)"
! rg -l --hidden 'BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY' "$extracted"
```

Expected: all scope and secret checks exit zero.

- [x] **Step 3: Compare representative files byte-for-byte**

Run:

```bash
for rel in CMakeLists.txt SOURCE_VERSION.txt REPRODUCE.md \
  Doc/archive/baselines/stageB_baseline_verification_legacy.md \
  cuMAC/scripts/run_stageB_baseline_eval_suite.sh \
  cuMAC/scripts/plot_stageB_baseline_verification_report.py; do
  cmp "deliverables/stageb_cuda_rerun_62227dc/$rel" "$extracted/$rel"
done
```

Expected: all comparisons exit zero.

- [x] **Step 4: Revalidate the untouched Linux/CUDA archive and delivery guide**

Run:

```bash
(cd deliverables && sha256sum -c stageB_cuda_rerun_62227dc.tar.gz.sha256)
test "$(sha256sum deliverables/stageB_cuda_rerun_62227dc.tar.gz | cut -d' ' -f1)" = \
  '1321254fe9b941536466f2c69033e59a1a764b1b3672b49f4f43b96cebded372'
for name in stageB_source_62227dc.zip stageB_cuda_rerun_62227dc.tar.gz \
  stageB_cuda_rerun_62227dc.tar.gz.sha256; do
  test -f "deliverables/$name"
  rg -q -F "$name" deliverables/README_FIRST.txt
done
```

Expected: original archive reports `OK`, its SHA256 matches exactly, and all delivered filenames are documented.

# 2026-11-25 整理归档

收录 11 月 4 日上次提交之后、截至 11 月 23 日的二阶段交付及研究证据。日期表示本次归档时间，不表示当天重跑实验。最新结论见[项目进展](../../stageB_current_progress.md)，历史过程见[11 月 4 日进展快照](../../archive/progress/stageB_progress_20261104.md)。

## 入口

| 目录 | 内容与来源 |
|---|---|
| [minimal_delivery](minimal_delivery/README.md) | 11-23 最小交付包的完整字节副本；报告主场景 120 条、G1–G4 60 条、输入简化 36 条结果，包含复用端点；原 MANIFEST.sha256 保留 |
| [stage2_closure](stage2_closure/完整交付核验.json) | 11-04 完整 220 点实验、90 状态审计、原协议及来源记录；不同于上一归档的三 seed 短窗 |
| [stage2_metrics_revision](stage2_metrics_revision/指标补充核验.json) | 11-05 outcome、队列、逐小区指标扩充；从原轨迹补算，没有重跑 PHY |
| [stage2_generalization](stage2_generalization/protocol.json) | 11-06 混合布点、异构包、小区负载不平衡 G1–G4，seed51–55 |
| [seven_cell](seven_cell/analysis/report.txt) | 11-05 七小区/84 UE 原三臂，三 seed，DA-PLS 排序反转及长尾负结果 |
| [seven_cell_multistart](seven_cell_multistart/analysis/report.txt) | 11-05 三分支：同状态目标改善不保证闭环长期时延 |
| [seven_cell_delay](seven_cell_delay/report.txt) | 11-06 逐包、队列、服务转移和同状态权重诊断 |
| [seven_cell_guarded](seven_cell_guarded/analysis/report.txt) | 11-06 λ=0.1、β=0.98 的有界紧急度与服务保护，P95 仍达 TTL |
| [rsrp](rsrp/report.html) | 11-09 纯标量邻区测量代理，真实原生信道不变 |
| [rsrp_ablation](rsrp_ablation/report.html) | 11-10 空间、频率、单端口、滤波、上报六级有序消融与成本诊断 |
| [rsrp_covariance](rsrp_covariance/report.html) | 11-10 RSRP＋周期性归一化空间协方差；120 新跑＋24 复用端点 |
| report_support/ | `outputs/` 中报告、表10-2和验收演示的源脚本、CSV、文本及核验 JSON；Office/PDF/字体/预览留在本地 |

## 保存范围

`minimal_delivery/` 完整复制，可直接运行其核查器。其余目录保存最终汇总、逐 seed/UE/小区小表、协议、运行身份、审计、源码快照、图表和执行记录。原始逐 TTI/逐包轨迹、大数组、二进制库、压缩交付包和中断运行目录仍留在 `deliverables/`；旧证据不改写。原文件内的绝对路径以及原 manifest 可能指向未收录工件，不代表归档是完整 GPU 发行包。

`manifests/*.json` 逐包记录本次实际复制的相对路径、大小、SHA-256 和来源根目录，同时统计未复制文件。`archive_manifest.json` 覆盖本归档所有文件（自身除外）。`cleanup_manifest.json` 单列本次删除的过期副本及可重建缓存，包含删除前哈希。

二阶段研究代码已纳入主仓库 `training/stage2_*`、`training/stageb_rsrp*`、`training/stageb_sevencell*`。运行库按各包构建说明生成，忽略生成的 `.so`。最小交付包里的冻结源码保留原字节以便核验；其 `code/.worktrees/` 是必需的静态桥接模块，不是 Git 工作树或运行缓存。

## 本次验证

```bash
python3 Doc/evidence/2026-11-25/verify_archive.py
python3 Doc/evidence/2026-11-25/minimal_delivery/verify_results.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python Doc/evidence/2026-11-25/minimal_delivery/replay_sample.py
```

本次核验记录保存在 `validation/`：20 项 RSRP/七小区现有测试；3043 个交付文件哈希、696 项报告数值、504 项汇总值和 40 条零过期记录；从归档源码实际编译的 24 轮/38376 候选离线重放；主工作区原生目标构建；历史干扰配对数据重新统计。它们不构成全量 GPU 实验重跑或研究策略晋级。

当前主工作区还提交了此前遗留的可选动作后干扰插桩和 teacher passthrough 工具。其干扰定义使用实际干扰 UE 预编码和层数；本批冻结报告使用 activity-only PHY，两者不可混为同口径实验。注册默认算法保持不变。

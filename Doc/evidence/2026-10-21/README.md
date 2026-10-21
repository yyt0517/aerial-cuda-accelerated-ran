# 2026-10-21 Stage-B 证据快照

本目录保存当前报告所需的26份小型证据，约714 kB。原始文件按字节复制，不改写其历史结论、时间戳或路径；[manifest.json](manifest.json)记录每份文件的来源、大小及SHA-256。当前解释以[项目进展](../../stageB_current_progress.md)为准。

## 文件职责

| 目录 | 内容与用途 |
|---|---|
| `corrected/` | 三臂10 seed协议、指标定义、逐seed/UE表、均值/区间、短窗变化、统计/物理核验及原生构建与修正patch |
| `historical_comparison/` | 修正current对历史PFQ的整体版本比较、详细指标、配对核验与分窗数据；不是同环境算法比较 |
| `identity/` | seed41 current完整启动身份、冻结加权策略清单及搜索加速一致性记录 |

优先读取：

- [冻结协议](corrected/protocol.json)、[指标字典](corrected/metric_definitions.json)。
- [逐seed数据](corrected/per_seed_metrics.csv)、[均值/标准差](corrected/mean_std.csv)、[配对区间](corrected/paired_mean_ci.csv)。
- [统计核验](corrected/validation.json)、[90状态数值复核](corrected/final_input_validation.json)、[最终复核](corrected/final_review.json)。
- [历史比较逐seed表](historical_comparison/per_seed_comparison.csv)、[历史版本比较核验](historical_comparison/validation.json)。

`final_report.txt`、`report.txt`、`detailed_metrics.txt`是冻结的来源报告，供审计，不作为另一套持续维护的项目进展文档。它们列出的同目录产物指**原始目录**，其中一些重复汇总/脚本没有收入本快照。

## 完整性与复现范围

在仓库根目录可用Python标准库校验全部复制文件：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('Doc/evidence/2026-10-21')
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    data = (root / entry['file']).read_bytes()
    assert len(data) == entry['bytes'], entry['file']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['file']
print(f"PASS: {len(manifest['files'])} evidence files")
PY
```

本次文档整理独立重算了三臂主要指标的均值、样本标准差与配对t区间，并核对历史比较、协议和报告哈希；没有新仿真、训练或全轨迹验证器重跑。JSON中的 `passed` 记录原实验的验证结果，不是本次重新执行所有底层检查。

完整轨迹、checkpoint、专用runner和二进制仍在本地 `.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/`。JSON保留了宿主机 `/home/oai2/...` 与容器 `/opt/nvidia/cuBB/...` 的原始路径；迁移时须按实际挂载映射。`native_correction.patch`只是该实验的一部分修改，不能在main上单独应用后就声称完整复现。

仅克隆仓库可复核小型表格和身份，不能重放完整实验。26份来源快照中不包含原始二进制、checkpoint、大轨迹、pilot重复汇总或PPT/PDF/XLSX导出。

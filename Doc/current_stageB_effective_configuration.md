# Stage-B 当前有效实验配置

更新至 **2026-10-21**。本文描述 `boundary08_corrected_physics_20261020_v1` 的冻结配置；[当前进展](stageB_current_progress.md)维护结果。它不是 `main` 上所有通用脚本的默认配置。

## 1. 实现与版本边界

| 对象 | 身份与状态 |
|---|---|
| 整理前主分支 | `main`，基底 `04a9200`，包含截至 8 月的 independent actor / rescue cohort 实现 |
| 最新实验工作区 | `.worktrees/pfq-independent-reuse-actor`，分支 `research/pfq-independent-reuse-actor` |
| 实验 Git 基底 | `aa1401b542abce8ad2de141d233ff7e0ab0de1ff`；存在未提交修正，不能仅凭 HEAD 复现 |
| 9 月有效性分界 | 实验分支 `7f3b700` 修正容量 bit/byte、PHY slot 索引、GPU MCS 初始化同步 |
| 10 月有效性分界 | 实际服务链路 SVD、显式 3 dB 阴影、同一步动作前输入；修正在实验源码与专用 runner 中 |
| 本次收口 | 文档、历史去重与证据；未把实验 worktree 或现存未提交运行时代码合入主分支 |

原始实验根目录（下称 `FORMAL`）：

```text
.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/
```

仿真器 SHA-256：`621b1ec600e35882362e1198667c46ef393b25f74a682166ca3449123ec0387d`。完整源码、共享库和编译身份见[构建清单](evidence/2026-10-21/corrected/native_build_manifest.json)及 [seed41 current 启动清单](evidence/2026-10-21/identity/seed41_current_launch.json)。这些清单用于定位证据，不是完整可执行交付包。

## 2. 冻结场景与窗口

以[原始协议](evidence/2026-10-21/corrected/protocol.json)为准：

| 项目 | 当前值 |
|---|---|
| 拓扑与用户 | 3 小区、36 UE，全部 boundary 放置，实际 strongest-cell association |
| 几何 | cell radius 500 m，site spacing 1000 m |
| PHY / 资源 | DL、4T4R、Rayleigh、SVD、Type-0；每小区 17 PRG × 16 PRB = 272 PRB |
| 阴影 | 显式 `CUMAC_SHADOW_FADING_STD_DB=3`；lognormal 自然对数标准差为 `3 × ln(10) / 10` |
| 业务 | 每 UE 每 TTI 平均 0.8 包，3000 B/包；配置 offered load 1382.4 Mbps |
| TTI / TTL | 0.5 ms / 200 ms（400 TTI） |
| HARQ / 主动丢包 | HARQ 关闭；last-slot completion guard 关闭，无 proactive drop |
| PFQ demand slack | 0 B，不得与旧 PFQ3000 混用 |
| Seeds | 41–50，三臂各 10 条新轨迹 |
| 共同预热 | TTI 1–4000，三臂都执行 native PFQ |
| 测量 | TTI 4001–6000，2000 TTI = 1 秒 |
| 随访 | TTI 6001–6400，覆盖测量期到达包的完整 TTL |
| 包 cohort | 测量期全部到达包，随访至终态，无截尾 |
| 汇总 | 逐 seed 等权均值、样本标准差；配对差的名义 95% t 区间，df=9 |

三臂同 seed 的流量、拓扑、实际关联、H/W 序列与 4000 TTI 预热一致；第 4001 TTI 动作前完整输入及 CTBS 一致。分叉后队列、OLLA/MCS 与 EMA 可因策略不同而分化，不要求此后完整状态相同。

历史 `rate1.0 / legacyshadow / t4000 / w0`、7cell rate0.85 和 68×4 pilot 属于不同实验。旧 full-KPI / active gate 的 wrapper log 已核验为 **17×16**，不能沿用底稿误写的 68×4。原命令和机制见[历史配置](archive/platform/stageB_legacy_configuration_reference.md)。

## 3. 冻结策略

| arm | 实际执行路径 |
|---|---|
| `pfq0` | 本 TTI fresh preaction 状态下的 native PFQ |
| `full`（DPA） | 冻结 BC actor + 冻结 directed coordinator，输入来自同一步动作前状态 |
| `current` | fresh PFQ → CTBS 裁剪 → 同一冻结 directed coordinator → 服务缺口加权完整局部搜索 |

`current` 不执行 DPA 的 BC actor。服务目标固定为 2400 B/UE/TTI，EMA 为 `0.995 × ema + 0.005 × realized_successful_bytes`，初值 0，计入共同预热。“完整局部搜索”指冻结邻域与停止规则，不代表全局最优。本轮没有重训或调参。

冻结协调器 checkpoint：

```text
FORMAL/boundary08_directed_attention_20261008_v1/training/full.pt
SHA-256: 1e259ecef5b58b3cdf70e1f0e2e73dd6b16082505a547725844034d7bf39534f
```

专用入口是 `FORMAL/boundary08_corrected_physics_20261020_v1/run.py`。PFQ/CTBS 初选来自 `boundary75_pfq_ctbs_replacement_20261016_v1/policy.py`；搜索入口为 `boundary75_search_acceleration_20261020_v1/policy.py`，冻结加权逻辑来自 `boundary75_teacher_weight_20261014_v1/policy.py`。身份见[冻结清单](evidence/2026-10-21/identity/frozen_teacher.json)和[搜索一致性记录](evidence/2026-10-21/identity/search_confirmation.json)。

## 4. 指标口径

| 指标 | 定义与限制 |
|---|---|
| 分段 goodput | 测量窗成功分段 bytes × 8 / 1 秒；可包含最终整包失败的部分字节与预热积压 |
| 完整包 payload 吞吐 | 测量期到达、随访后整包成功的 payload bits / 1 秒 |
| 整包成功率 | 完整成功包数 / 同一成熟到达 cohort 包数 |
| 成功包时延 | 仅完整成功包；`(terminal TTI − arrival TTI + 1) × 0.5 ms` |
| 全包终态时延 | 同一 cohort 中成功、CRC 失败、自然过期包的到达至终态时延 |
| PRG 占用 | 实际分配总数 / `(2000 × 3 × 17)`；未占用不全等于主动 blank |
| TB BLER | 错误 TB / 已发送 TB；不同于包过期率或残留 CRC 包失败率 |
| 置信区间 | seed 为独立单位；未做多重比较校正，不把 TTI 或包当独立重复 |

完整口径见[指标字典](evidence/2026-10-21/corrected/metric_definitions.json)。旧 packet tracker 的 MAC 出队完成时延、UE-macro 包速率与当前成功交付/全包终态指标不同，不能拼表或继承旧阈值。

## 5. 数据核验与本地复现

既有核验包括30条轨迹的动作合法性、逐行/跨 TTI 字节守恒、FIFO 终态、无截尾与配对一致性；物理独立数值核验覆盖90个保存状态，不是全 TTI 物理量重算。

具备原始 worktree、轨迹及 `.venv` 时，可从仓库根目录重算已有统计：

```bash
.venv/bin/python .worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/boundary08_corrected_physics_20261020_v1/summarize_corrected.py
.venv/bin/python deliverables/current_corrected_vs_original_pfq_20261021/compare.py
```

这些命令重新生成本地汇总，不启动仿真。仅克隆主仓库可检查小型证据和 SHA-256，不能直接重放完整 CUDA 实验。通用 `run_stageB_main_experiment.sh` 或旧 StepRsp teacher passthrough 不等价于此专用 runner；新实验必须固定修正源码、编译产物及输入时点。

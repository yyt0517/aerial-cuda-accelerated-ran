# Stage-B 当前有效实验配置

更新至 **2026-11-25**。注册默认算法仍为 2026-10-27 冻结的 **PFQ＋CTBS＋因果服务缺口加权物理搜索，因果出队下界 ρ=0.90**（DPLS）。最新二阶段报告使用原始 PFQ 初选、无 CTBS 的 DPLS/DA-PLS，配置见第 0.2 节；不得通过“最新结果”自动替换注册表。[当前进展](stageB_current_progress.md)维护结果。

## 0. 当前算法与统一入口

算法ID为 `pfq_ctbs_deficit_search_rho090_20261027`，注册在[current_algorithm.json](../training/stageb_hgraph/current_algorithm.json)。执行链为 `fresh PFQ → CTBS裁剪 → 因果服务缺口权重 → 完整物理局部搜索 → 原生执行/更新EMA`。

`D0(t)=sum_u min(Q_u(t),TBS_u(PFQ_CTBS_draft(t)))`，下界为 `ceil(0.90*D0(t))`。每个TTI使用本臂当前状态；不读取参照下界，不加载/推理BC或定向协调器，不构造定向学习特征。PFQ+CTBS同时作为draft和KEEP，因此起点天然可行。比例约束的是草案出队预算，不是实际成功吞吐的损失比例保证。

宿主机统一入口：

```bash
bash cuMAC/scripts/run_stageB_current_algorithm.sh --show-config
bash cuMAC/scripts/run_stageB_current_algorithm.sh --seed 41 --gpu 0
```

入口调用现有 `c_aerial_oai2` CUDA容器及研究worktree中的 `FORMAL/boundary08_rho090_five_seed_20261027_v1/run.py`；`--smoke`运行136TTI，已有同名结果目录拒绝覆盖。当前策略复用 `boundary08_causal_floor_ratio_seed41_20261027_v1/policy.py`，SHA-256为 `f0267e4f00e2f488f488c2047493a45f6116d488f68a010da5587776fc057cdc`，入口将比例固定为90/100。**注册表与此启动脚本目前仅对应 boundary 场景**，不会自动启动 uniform 验证。研究运行时仍依赖原工件，不等同于通用生产调度器集成。

### 0.1 11 月 4 日开发版：M-LWDF-PF＋精确剪枝DPLS

此候选使用 `training/stageb_lns/crossload_native.py` 的 `mlwdf_pf` 权重与冻结 `capacity_bound_q1e6_v1` 内层，未注册成当前默认，也未替换第0节启动入口。Boundary/rate0.8、seed41–43、4000共同PFQ预热＋4000稳定＋4000测量＋400TTL追踪的同窗口PFQ比较见[独立报告](evidence/2026-11-04/mlwdf_pfq_boundary_comparison/report.txt)。这组2秒开发证据与第2节五seed10秒配置分别解释。

| 环节 | 研究候选配置 |
|---|---|
| 权重 | `age=max(1,400−HOLslack)`，`a=−ln(.01)/400`，`w=a*age/max(EMA,1e−9)`后均值归一；空队列服务为0 |
| 成功服务EMA | `.995*EMA + .005*realized_successful_bytes`，含共同预热，未截断PF分母 |
| 初始化与floor | 原始PFQ，无CTBS裁剪；本臂当前PFQ总出队量的90%向上取整 |
| 邻域 | 每PRG换UE、单小区静默、一个小区静默加邻区换UE |
| 物理目标 | 固定本次MCS，协方差/MMSE SINR与原生BLER得到`Σw*min(Q,TBS)*(1−BLER)` |
| 精确规范 | 正权重；逐UEnearest-even量化到1e−6B后int64求和；同一规范键、阈值、候选全集及收敛条件 |
| 加速 | `w*min(Q,TBS)`容量上界、最佳优先精评、键感知停止证书、持久稀疏编辑表与依赖失效；保持规范完整局部搜索等价 |
| AI称谓 | 经典AI启发式搜索；M-LWDF为经典规则，此组合没有训练模型，不代表全局最优或已验证学习收益 |
| 成本边界 | M-LWDF-PF并发墙钟仅描述；相对完整DPLS的公平串行曲线来自旧κ0保存输入，尚无该组合相对PFQ的公平时延比 |

权重形式和原文来源见[跨负载说明](../training/stageb_lns/crossload_README.rst)，精确内层契约见[容量界说明](../training/stageb_lns/bounded_README.rst)。研究源码已纳入主仓库；运行库需按构建脚本生成，原生仿真仍依赖下文worktree/容器与本地大工件。未以新研究结果追改旧冻结报告或原失败门槛。

### 0.2 二阶段长窗交付与后续研究

报告主实验的执行模块为 `training/stage2_closure/`，沿用第 0.1 节的 M-LWDF-PF 权重、原始 PFQ 初始化、90% 当前计划出队下界及精确内层；不加载学习模型。版本身份以[最小交付包](evidence/2026-11-25/minimal_delivery/README.md)内的源码校验、原生入口和原始协议为准，主工作区的可选动作后干扰插桩不参与这些冻结结果。

| 研究 | 场景、seed 与范围 | 协议/证据 |
|---|---|---|
| 二阶段完整矩阵 | Boundary 0.8/1.12/1.28、Uniform 1.0；seed41–50；220 点含 10 点 PFQ 复用 | [运行协议](evidence/2026-11-25/stage2_closure/campaign/protocol.json) |
| 报告主场景 | Boundary0.8、Uniform1.0；六算法×十 seed，共 120 条 | [交付范围](evidence/2026-11-25/minimal_delivery/provenance/package_scope.json) |
| G1–G4 | 36 UE；混合布点/异构包/小区不平衡；seed51–55；三算法共 60 条 | [协议](evidence/2026-11-25/stage2_generalization/protocol.json) |
| 七小区 | 84 UE；seed41–43；原三臂、三分支及 λ0.1/β0.98 保护版本分别冻结 | [研究说明](../training/stageb_sevencell/README.rst) |
| RSRP 系列 | 36 UE；两场景、两权重；seed41–43；标量、有序消融、协方差更新 | [标量](../training/stageb_rsrp/README.rst)、[消融](../training/stageb_rsrp_ablation/README.rst)、[协方差](../training/stageb_rsrp_covariance/README.rst) |

正式轨迹共同使用 4000 预热＋4000 稳定＋20000 测量＋400 随访 TTI（0.5 ms/TTI）；短程 smoke 和保存状态 replay 不算正式质量轨迹。原生模拟器主身份仍为 `a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615`；泛化和七小区用各自独立构建，不能直接继承原二进制身份。G1–G4 缺口公式使用各 UE 配置平均到达字节；七小区掩码扩为 128 位以覆盖 UE64–83。

完整包吞吐、成功包时延、所有 outcome 时延、MAC 字节 goodput 分别统计。原始完整协议可能列出 220/144 点，最小交付包筛选为 120 条主场景、60 条泛化、36 条输入简化结果，含复用端点，不把这些计数相加当作新增实验。

## 1. 实现与版本边界

| 对象 | 身份与状态 |
|---|---|
| 10 月整理前主分支（历史） | `main`，基底 `6ecde90`；此前已于 10-21 整理文档和证据，通用 independent actor / rescue cohort 实现仍保留 |
| 最新实验工作区 | `.worktrees/pfq-independent-reuse-actor`，分支 `research/pfq-independent-reuse-actor` |
| 实验 Git 基底 | `aa1401b542abce8ad2de141d233ff7e0ab0de1ff`；存在未提交修正，不能仅凭 HEAD 复现 |
| 9 月有效性分界 | 实验分支 `7f3b700` 修正容量 bit/byte、PHY slot 索引、GPU MCS 初始化同步 |
| 10 月有效性分界 | 实际服务链路 SVD、显式 3 dB 阴影、同一步动作前输入；修正在实验源码与专用 runner 中 |
| 2026-10-27变更 | 记录因果下界选型、注册并启用ρ0.90研究入口、五seed对照；运行时仍依赖现有worktree与CUDA容器 |
| 2026-10-28/29证据 | 补齐 uniform 同算法五seed、风险直接修复、神经缓存/前缀/过程价值/残差研究；未更换有效算法或默认神经解码器 |

原始实验根目录（下称 `FORMAL`）：

```text
.worktrees/pfq-independent-reuse-actor/output/pfq_independent_reuse/formal/
```

当前仿真器与PFQ五seed参照相同，SHA-256为 `a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615`，采用 `genFastFadingGpu(t)` 连续信道时钟。构建来自 `FORMAL/boundary08_steady_state_20261021_v1/native_build_manifest.json`。旧短窗模拟器 `621b1ec600e35882362e1198667c46ef393b25f74a682166ca3449123ec0387d` 及[旧构建清单](evidence/2026-10-21/corrected/native_build_manifest.json)仅作历史身份，不与当前长窗结果混用。

## 2. 冻结场景与窗口

boundary 以[10-27 五 seed 协议](evidence/2026-10-27/five_seed/protocol.json)为准；uniform 以[10-28 五 seed 协议](evidence/2026-10-28/uniform_dpls_rho090/protocol.json)为准。以下基础设置和窗口相同，布点与业务分别列出：

| 项目 | 当前值 |
|---|---|
| 拓扑与用户 | 3 小区、36 UE，实际 strongest-cell association；布点见下表 |
| 几何 | cell radius 500 m，site spacing 1000 m |
| PHY / 资源 | DL、4T4R、Rayleigh、SVD、Type-0；每小区 17 PRG × 16 PRB = 272 PRB |
| 阴影 | 显式 `CUMAC_SHADOW_FADING_STD_DB=3`；lognormal 自然对数标准差为 `3 × ln(10) / 10` |
| 包大小 | 3000 B/包；到达率和配置输入见下表 |
| TTI / TTL | 0.5 ms / 200 ms（400 TTI） |
| HARQ / 主动丢包 | HARQ 关闭；last-slot completion guard 关闭，无 proactive drop |
| PFQ demand slack | 0 B，不得与旧 PFQ3000 混用 |
| Seeds | 41–45；新增ρ0.90五条轨迹，复用已核验的同协议PFQ五条轨迹 |
| 共同预热 | TTI 1–4000，两臂都执行 native PFQ |
| 各策略稳定化 | TTI4001–8000，不重置队列、OLLA、EMA或信道 |
| 测量 | TTI8001–28000，20000 TTI = 10 秒 |
| 随访 | TTI28001–28400，覆盖测量期到达包的完整 TTL |
| 包 cohort | 测量期全部到达包，随访至终态，无截尾 |
| 汇总 | 逐seed等权均值、样本标准差；ρ0.90−PFQ名义95%配对t区间，n5、df4 |

| 场景 | 布点 | 包/UE/TTI | 配置输入 Mbps | 专用研究 runner |
|---|---|---:|---:|---|
| boundary08 | 全 boundary | 0.8 | 1382.4 | `FORMAL/boundary08_rho090_five_seed_20261027_v1/run.py` |
| uniform10 | uniform | 1.0 | 1728.0 | `FORMAL/uniform10_dpls_rho090_five_seed_20261028_v1/run.py` |

uniform 仍使用 2400 B/UE/TTI 服务目标，未按 3000 B 的业务输入重调；[跨场景身份记录](evidence/2026-10-28/uniform_dpls_rho090/cross_scene_algorithm_identity.json)核对 15 份共享源码一致。表中的配置输入不是实测到达率；两场景负载与几何同时变化，不作只改变几何的消融解释。

两臂同seed的流量、拓扑、实际关联、H/W序列与4000TTI预热一致；第4001TTI动作前完整输入及CTBS一致。分叉后队列、OLLA/MCS与EMA可因策略不同而分化，不要求此后完整状态相同。seed41参与过ρ选型，五seed并非完全未见过的盲测。单seed选型的1秒旧时钟实验与本轮连续时钟长窗口不能按单因素比较或混池。

历史 `rate1.0 / legacyshadow / t4000 / w0`、7cell rate0.85 和 68×4 pilot 属于不同实验。旧 full-KPI / active gate 的 wrapper log 已核验为 **17×16**，不能沿用底稿误写的 68×4。原命令和机制见[历史配置](archive/platform/stageB_legacy_configuration_reference.md)。

## 3. 冻结策略

| arm | 实际执行路径 |
|---|---|
| `pfq0` | 本 TTI fresh preaction 状态下的 native PFQ |
| `full`（历史DPA） | 冻结BC actor＋冻结directed coordinator；仅作旧对照，不是本轮PFQ/ρ0.90五seed主比较臂 |
| `current` / `current_rho090` | fresh PFQ → CTBS裁剪 → 因果服务缺口加权完整物理搜索；当前草案出队下界ρ=0.90 |

`current`不执行BC或定向协调器。服务目标固定为2400B/UE/TTI，EMA为 `0.995 × ema + 0.005 × realized_successful_bytes`，初值0，计入共同预热。权重为 `clip(target/max(EMA,1e-9),1,4)` 后均值归一化。前两轮改善门槛>1B，之后>16B，完整枚举既定局部邻域；256轮仅为安全上限，触及算失败。完整局部搜索不代表全局最优。本轮不重训或重新选择比例。

以下协调器checkpoint仅用于历史版本追溯，当前路径不加载/推理：

```text
FORMAL/boundary08_directed_attention_20261008_v1/training/full.pt
SHA-256: 1e259ecef5b58b3cdf70e1f0e2e73dd6b16082505a547725844034d7bf39534f
```

旧含协调器入口 `FORMAL/boundary08_corrected_physics_20261020_v1/run.py` 保留不变；当前入口见第0节。PFQ/CTBS初选和原加权搜索仍复用相同冻结实现，仅删除学习提案并用本臂当前草案的90%出队量作下界。旧[冻结清单](evidence/2026-10-21/identity/frozen_teacher.json)和[搜索一致性记录](evidence/2026-10-21/identity/search_confirmation.json)用于定位这些基础模块。

## 4. 指标口径

| 指标 | 定义与限制 |
|---|---|
| 分段 goodput | 测量窗成功分段bytes×8/10秒；可包含最终整包失败的部分字节与测量前积压 |
| 完整包 payload 吞吐 | 测量期到达、随访后整包成功的payload bits/10秒 |
| 整包成功率 | 完整成功包数 / 同一成熟到达 cohort 包数 |
| 成功包时延 | 仅完整成功包；`(terminal TTI − arrival TTI + 1) × 0.5 ms` |
| 全包终态时延 | 同一 cohort 中成功、CRC 失败、自然过期包的到达至终态时延 |
| PRG 占用 | 实际分配总数/`(20000 × 3 × 17)`；未占用不全等于主动blank |
| TB BLER | 错误 TB / 已发送 TB；不同于包过期率或残留 CRC 包失败率 |
| 置信区间 | seed 为独立单位；未做多重比较校正，不把 TTI 或包当独立重复 |

完整口径见[当前指标字典](evidence/2026-10-27/five_seed/metric_definitions.json)。旧 packet tracker 的 MAC 出队完成时延、UE-macro 包速率与当前成功交付/全包终态指标不同，不能拼表或继承旧阈值。

## 5. 数据核验与本地复现

旧十seed核验覆盖30条轨迹和90个保存状态，保留作历史。当前五seed额外保留逐seed、逐UE、10个1秒窗/两个5秒半窗、包级cohort生命周期、原生UE×TTI二进制、动作/输入指纹、每次决策/每轮搜索、草案/下界审计。新轨迹逐TTI核对动作/MCS/层数/TBS/出队，逐决策核对独立TBS/队列下界、完整收敛和模型调用为0，并在六个保存状态检查物理输入及首个信道跨界递推。实用稳定性诊断沿用PFQ协议，不等于渐近稳态或容量证明。

uniform 的同协议配对、原生/FIFO、完整 cohort 与稳定性核验见[原始验证](evidence/2026-10-28/uniform_dpls_rho090/validation.json)及[复核记录](evidence/2026-10-28/uniform_dpls_rho090/review_validation.json)。源码和构建身份来自原实验清单；本次整理不重跑原生轨迹或物理验证器。

当前 boundary 使用第0节入口。历史短窗重算命令保留在[10-21 证据索引](evidence/2026-10-21/README.md)及原始报告，不能用它重算本轮长窗。仅克隆主仓库可检查小型证据和 SHA-256，不能直接重放完整 CUDA 实验。通用 `run_stageB_main_experiment.sh` 或旧 StepRsp teacher passthrough 不等价于此专用 runner；新实验必须固定修正源码、编译产物及输入时点。

## 6. 神经研究路径的独立配置边界

`training/stageb_lns` 及相关 `stageb_ppo*`、`stageb_student` 源码已随主仓库保存；`deliverables/tb_*` 保存模型、候选、原生轨迹及版本化脚本。生成的运行库按各模块`build*.py`构建，研究入口仍需具备本地工件和容器挂载；证据目录中的冻结源码/二进制用于审计，不自动成为当前运行库。

原生修复入口默认 `--decoder reference`；`cached`、`verified`、`verified_cover` 和近似物理/残差搜索属于显式实验选项。缓存等价验证保持冻结的87维actor、width16、最多3轮及物理验收；后续候选/排序变化不继承其等价结论。当前有效版本优化缺口加权物理目标 J；风险研究另用因果交付风险代理 R，并保护 J 与原草案 floor。两种物理对照应分别报告，不能用 R 点值代替包结果。

训练、选型、校准、评估 seed 按各轮协议分别固定。过程价值/残差轮为训练41/42/43/46/47/49、选型44、经验校准45、开发评估48/50；这些开发 seed 已有历史暴露。311–315仍留作后续最终测试。各轮原生窗口为1秒或5秒，保存状态回放另计，不能与本文件五seed10秒主验证混池。协议、默认行为及失败门槛见[10-28证据索引](evidence/2026-10-28/README.md)和[10-29证据索引](evidence/2026-10-29/README.md)。

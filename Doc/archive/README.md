# 文档归档索引

本目录保存历史研究过程，不代表当前实现状态或有效性能结论。最新状态见[项目进展](../stageB_current_progress.md)，配置见[当前配置](../current_stageB_effective_configuration.md)。旧文档中的“当前”“主线”“最好”只指原始日期；9月和10月的仿真修正使旧性能数字不能继续充当新版本验收依据。

## 2026-10-21 合并记录

| 原Doc顶层文件 | 处理与保留内容 |
|---|---|
| `stageB_intelligent_algorithm_evolution_training_report.md`、`stageB_boundary_intelligent_algorithm_experiment_report.md`、`stageB_intelligent_algorithm_data_manifest.md` | 合并为[HGraph v9c–v19回顾](hgraph_experiments/stageB_hgraph_v9c_v19_retrospective.md)；保留架构、训练收敛、完整迭代表、负结果、评测与来源；删除重复摘要/导航/写作任务 |
| `stageB_multidimensional_resource_allocation_simulation_verification_report.md`、`stageB_packet_delay_distribution_analysis.md`、`stageB_cochannel_baseline_analysis_reference.md` | 合并为[历史基线报告](baselines/stageB_baseline_verification_legacy.md)；保留验收结构、完整KPI/分布/逐seed表、字段定义、配置未知项和动作后干扰审计；删除重复底稿与结论 |
| `stageB_7cell_tar_gnn_pfq_design.md` | 移至[历史设计与实验日志](hgraph_design/stageB_7cell_tar_gnn_pfq_design.md)；保留独有v9d–v9h、v5/v6、GDPO与7cell机制记录 |
| `current_stageB_effective_configuration.md`的旧内容 | 原配置归入[历史配置参考](platform/stageB_legacy_configuration_reference.md)，顶层重写为10月冻结协议；旧训练README的seed42/v33示例并入其附录 |

合并后的旧v9c跨窗口、跨seed参考差值不再表述为同条件胜出；旧full-KPI/active gate的资源粒度按wrapper日志纠正为17×16，独立68×4 pilot另列。没有删除原始实验数据或图表。

## 其它历史资料

- `platform/`：平台能力、天线/precoding、PRG/RBG、状态字段与旧配置。
- `baselines/`：合并后的基线验收、逐包分布和同频机制证据。
- `hgraph_design/`：HGraph结构设计与历史实验计划。
- `hgraph_experiments/`：v9c–v19回顾、v34–v50流水日志和3cell retrospective。
- `legacy_gnnrl/`：旧GNNRL offline/online/replay/ONNX说明。
- `plans/`：早期总体规划、UE/PRG接管与RL+GNN MVP路线。
- `meeting_notes/`、`code_audit/`：会议记录与历史代码审计。
- [Superpowers历史计划](../../docs/superpowers/README.md)：7—8月任务的判据、命令和设计，保留原目录以便追溯；未勾选复选框不表示代码未实现，已写计划也不表示已验收。

归档中的`output/`与训练运行目录多数是本地生成证据，不随Git分发；链接到本地数据不表示其性能结论仍有效。可随仓库复核的各期小型证据统一从[文档入口](../README.md)进入。

## 2026-10-29 合并与纠错记录

| 对象 | 处理与去向 |
|---|---|
| 主进展中的逐轮实验长摘要、10-20/21历史完整表格 | 主进展集中维护当前结论、关键比较和晋级条件；完整表格和独有边界由10-21、10-27、10-28、10-29各期冻结报告保留，删除重复正文 |
| 两份待纳管临时摘要 `Doc/evidence/2026-10-29/tb_process_value/progress.txt`、`tb_residual/progress.txt` | 内容均被各自相邻的最终`report.txt`覆盖；本次删除快照中的冗余副本，原始`deliverables/`文件与来源清单哈希保留，不删除独有负结果 |
| uniform五seed、104–106直接风险修复、联合修复PHY补表 | 补入主进展和日期索引；补收直接风险修复的八份、PHY补表的四份已有小型证据及复制清单，不新增实验 |
| 主文档旧日期、整理前Git基底、当前入口适用范围 | 统一更新至10-29；基底改为`6ecde90`；明确注册表/启动脚本对应boundary，uniform另有研究runner，旧`current`保留原身份 |
| 缺失的10-28日期索引与复现边界 | 新建日期索引；注明来源清单含未收录工件，证据脚本仍依赖本地研究包，缓存等价不能覆盖后续排序变化 |
| 证据CSV的Git换行转换 | `.gitattributes`为`Doc/evidence/**`关闭文本换行归一化，保留来源字节，使克隆后的SHA-256可核验 |

同名设计/实施计划、事前协议及修订、独有研究笔记和失败核验继续保留；不因过期或文件名相似删除。现有实验源码和演示导出保持本地原状，不混入本次文档提交。

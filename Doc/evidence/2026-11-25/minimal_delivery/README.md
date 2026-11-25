# 二阶段实验最小交付包

对应《3. 基于AI技术的多维协同资源分配研究技术研究报告（二阶段）.docx》，整理日期：2026-11-23。报告身份和SHA-256见 `provenance/report_identity.json`。本包保存实验代码、配置和结果；用户已有的Word报告不重复放入。

## 交付内容与报告对应

| 报告内容 | 源码入口（均在code/下） | 结果文件 |
|---|---|---|
| 第5章：平台单位修正 | `native/`中的冻结cuMAC及报告专用网络源码（位于包根目录） | `results/platform/shadow_statistics.csv`；运行级物理核验JSON |
| 第7章：RR/RRQ/PF/PFQ传统基线 | `training/stage2_closure/campaign.py`、`native.py`、`analyze.py`、`revise_metrics.py` | `results/main/per_seed.csv`、`aggregate.csv` |
| 第8章：DA-PLS权重和精确剪枝 | `training/stageb_lns/crossload_native.py`、`ttl_weights.py`、`bounded_dpls.py`、`bounded_kernel.cpp`；完整枚举为`persistent_dpls.py`、`persistent_kernel.cpp` | `results/search_audit/timing_summary.csv`、`rounds.csv`、逐状态审计JSON |
| 第9章：主场景、弱UE、静默机制 | 同上；`training/stage2_closure/audit.py` | `results/main/`；`results/search_audit/boundary_r0.8_mlwdf_pf_s43_tti018001.json` |
| 第9章：G1–G4泛化实验 | `training/stage2_generalization/` | `results/generalization/`；逐UE业务配置在`protocols/generalization/plans/` |
| 第10章：RSRP与空间协方差输入简化 | `training/stageb_rsrp/model.py`、`training/stageb_rsrp_covariance/`及其依赖 | `results/input_simplification/`；`feedback.csv`为反馈量；`replay/`为单核计时 |
| 附录A：逐小区传输状态 | `training/stage2_closure/revise_metrics.py` | `results/main/cell_per_seed.csv`、`cell_aggregate.csv` |

算法名对应：`pfq`＝PFQ，`deficit`＝DPLS，`mlwdf_pf`＝DA-PLS。输入方案：`exact`＝精确输入，`reported`＝纯RSRP，`prg_t1`＝逐PRG协方差0.5ms更新，`prg_t4`＝2ms更新；`wideband_t1/t4`支持第10章宽带协方差对照结论。

主场景共120条结果（2场景×6算法×10个seed）；泛化场景共60条（4场景×3算法×5个seed）；输入简化共36条（2场景×6输入方案×3个seed，含6条复用主场景及6条复用前次纯RSRP实验）。第10章PFQ的6条参考结果也来自主场景。这些计数不能相加解释为独立新增仿真次数。

## 接收方核查

在解压后的包目录运行：

```bash
python3 verify_results.py
```

只需Python 3标准库。脚本校验文件哈希、实验矩阵、696项报告表格数值、504项汇总数据、40条DA-PLS零过期记录，以及图9-5/表9-4的联合调整案例。报告数值按其显示精度核对，计算使用未舍入的逐seed结果，未把所有seed的包合并计算。

离线调度样例（Linux、Python≥3.10、NumPy、g++、Eigen3）：

```bash
python3 replay_sample.py
```

脚本在临时目录编译包内C++源码，读取`sample/boundary_s43_tti018001.npz`，比较Python完整枚举、C++完整枚举和C++剪枝结果，并逐轮验证上界与选择一致性。样例对应报告图9-5的seed43、TTI18001；不需要GPU、PyTorch或模型权重。运行后自动清理编译文件。Eigen头文件需位于`/usr/include/eigen3`，或调整副本中的构建脚本路径。

交付前已通过上述数值复核及样例实际编译运行：24轮、38376个候选的审计通过，最终分配和物理数值与存档完全一致。记录见`validation/`。样例不代表重新测量报告中的完整性能曲线。

## 完整仿真与版本说明

本包是最小源码与结果交付，不是独立的完整GPU仿真环境。完整重跑需Aerial 25-3-cuBB/CUDA环境、原冻结工作树及其动态库/原生启动和参考文件。原实验脚本保留原路径和二进制身份断言，不能在任意目录直接启动全部实验。原始轨迹中的`metrics.bin`、逐包到达/结局文件、全量信道及状态未打包；从这些原始轨迹重新计算全部统计需另取原始数据。

- 主场景实际模拟器身份：`a68f48668bc623c0bc793c452f1b0cd5d8943929eb26e8eefd25f776d180b615`，配置见`protocols/main/campaign_protocol.json`及每次运行的`launch.json`。
- 原生主入口采用`native/main_experiment/boundary08_steady_state_20261021_v1/instrumented_main.cpp`；网络实现采用`native/main_experiment/boundary08_corrected_physics_20261020_v1/native_network_source.cu`和`.h`。`native/frozen_cuMAC/`提供配套源码，构建时应以这些报告专用文件覆盖相应入口；不要把当前主工作区的后续修改混入。
- G1–G4另外采用`native/generalization/`中的三个场景扩展文件，构建记录见`protocols/generalization/native/build_manifest.json`。
- 运行源文件已与原实验协议的哈希核对；见`provenance/frozen_source_verification.json`、`native_source_verification.json`。本次只补充打包/核查工具，没有更改实验算法。
- `code/.worktrees/`保留冻结桥接和数学辅助模块的原相对层级。部分历史研究模块因Python导入依赖而保留；DA-PLS本身没有神经网络训练，也不依赖训练权重。
- `protocols/`保留原始完整批次协议，因此个别记录会显示220点或144点。实际筛选范围以本说明及`provenance/package_scope.json`为准。原路径仅用于溯源，不表示所有引用文件都在本包。
- 90状态计时审计包含Boundary负载0.8、1.12、1.28，支持第8章报告的耗时范围；只保留这些负载的审计结果，不附无关的高负载业务评测。

完整重跑的原入口：主场景`training.stage2_closure.campaign`→`analyze`→`revise_metrics`/`audit`；泛化`training.stage2_generalization.build`→`campaign --phase setup/smoke/formal`→`analyze`；输入简化`training.stageb_rsrp_covariance.campaign`→`analyze`→`replay`。须先恢复上述平台和路径依赖，使用新的输出目录，具体参数以包内协议和源码为准。

## 文件取舍与校验

保留逐seed/逐UE/逐小区结果、时延分布、实验配置、业务计划、审计证据和原始输入哈希。排除了失败/中断运行、七小区等非报告实验、其他学习训练权重、重复报告与旧压缩包、构建缓存、虚拟环境、Git记录及大体积逐TTI/逐包原始轨迹。

`provenance/source_files.json`记录来源及CSV筛选方式；`MANIFEST.sha256`覆盖最终包内所有其他文件。压缩包旁另附ZIP自身的SHA-256。原项目的`LICENSE`、`ATTRIBUTION.rst`保留在包根目录。

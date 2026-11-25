七小区 Boundary08 三算法对比
==========================

PFQ、DPLS（缺口权重）、DA-PLS（M-LWDF-PF），seeds 41/42/43。
沿用二阶段报告的原始PFQ初选、rho0.90和精确capacity-bound求解器。
七小区/84UE，不裁剪CTBS、不改权重、不重新选参数。
4000共同预热＋4000稳定＋20000测量＋400随访，TTI 0.5ms。

几何扩展
--------

采用已有1+6六边形站点；站间距1000m、半径参数500m。
coop_boundary原来仅支持三角形的两条邻边；现按基站索引顺序在共享
最近邻边上循环分配12个名义UE。中心六邻边各2UE，外围三邻边各4UE。
切向范围±0.28站间距、内缩0–0.02站间距及Voronoi裁剪完全沿用原值。
实际服务关联仍为strongest，名义每站12个不保证关联后恰好12个。
三小区模式仍使用原采样代码。

适配与证据
----------

旧源码、库和已运行实验不改写。build.py复制冻结源码，生成独立仿真器，
差异保存为size_only.diff（包含尺寸、边界邻接和快照维度断言）。
物理计算、连续信道时钟和libcumac均保持原版。
搜索器的UE依赖/资源掩码从64扩为128位，防止UE64–83移位溢出；
小区activity掩码、整数目标、canonical key和搜索规则保持不变。
独立oracle不使用稀疏依赖掩码，逐轮完整枚举并核验所有剪枝上界。

构建与测试（主机求解库使用与旧库相同的g++/Eigen，以保留浮点一致性）::

    .venv/bin/python -m training.stageb_sevencell.build --solver-only
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_sevencell.test_scale
    docker exec -w /opt/nvidia/cuBB c_aerial_oai2 .venv/bin/python -m training.stageb_sevencell.build --native-only --output deliverables/NEW/build

正式运行在容器中，使用已有MPS服务并保留其设置::

    CUDA_DEVICE_MAX_CONNECTIONS=2 CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m training.stageb_sevencell.campaign --output deliverables/NEW/campaign --build deliverables/NEW/build --workers 9

--smoke使用16预热＋32测量＋400随访，仅为接口和统计核验。
所有目录拒绝覆盖；失败试运行保留作为执行记录。
生产轨迹运行期间不要改native/io/campaign/solver/contract或其物理依赖。

后处理（主机）::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m training.stageb_sevencell.analyze --source deliverables/NEW/campaign --output deliverables/NEW/analysis
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m training.stageb_sevencell.audit_saved --source deliverables/NEW/campaign --output deliverables/NEW/analysis/search_audit.json
    .venv/bin/python -m training.stageb_sevencell.report --root deliverables/NEW

每个seed与PFQ逐项匹配拓扑、业务、全程原始信道及共同预热动作/输入。
完整成功包与全部outcome都按到达cohort追踪至终态；P95按seed内求值。
INR使用冻结activity-only PHY模型，在实际调度UE-PRG上池化。
并发墙钟仅为描述统计，不是公平加速比或实时部署时延。

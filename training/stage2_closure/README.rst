二阶段同频报告实验与复现
========================

本目录是 2026-11-04 报告修订所用的冻结实验编排、统计、审计和文档生成工具。
主算法为 M-LWDF-PF 权重与 capacity_bound_q1e6_v1 精确内层。
所有搜索臂从原始 PFQ 开始，保持当前初选计划出队量的 90% 下界。
没有学习模型、主动整包丢弃、额外初选裁剪或负载切换。

运行身份
--------

实际运行配置、源码 SHA-256 和原生仿真器身份见产物目录的 campaign/protocol.json。
每个新实验点的 launch.json 另外记录实际动态库解析、环境参数和依赖哈希。
仿真器和库来自工作区既有冻结工作树；当前主工作区的未提交 C++ 改动不参与仿真。
原报告和修改方案副本及校验值在 source_documents/，原附件没有改写。
完整证据包另含 algorithm_snapshot/，保存本轮冻结执行模块、内层C++源码、
共享库及关键核验模块，并列出校验值。该快照用于代码审阅，不是独立软件发行包；
实际重跑仍需要完整项目、下述冻结仿真器及CUDA环境。原始大体积轨迹留在
campaign/，压缩包包含指标、输入校验值和核验证据。

本机容器名为 c_aerial_oai2，容器工作目录为 /opt/nvidia/cuBB。
主机工作目录为 /data/aerial-cuda-accelerated-ran，只有一张 A800 可用于本次任务。
正式质量轨迹采用 MPS 并发，CUDA_DEVICE_MAX_CONNECTIONS=2，32 个工作进程。
不要对已启动轨迹修改 campaign.py、native.py、冻结物理库或求解器依赖。
只有 done.json 存在并通过统计验证的实验才进入正式指标。

实验矩阵
--------

* Boundary/rate0.8：PFQ、RR、RRQ、PF、等权、缺口、TTL1、TTL2、EXP-PF、M-LWDF-PF。
* uniform/rate1.0：PFQ、RR、RRQ、PF、缺口、M-LWDF-PF。
* Boundary/rate1.12 和 1.28：PFQ、缺口、M-LWDF-PF。
* 每组 seed 41–50；总共 220 点，其中十条已验证的同协议 PFQ 轨迹复用。
* 所有轨迹：4000 共同 PFQ 预热＋4000 稳定＋20000 测量＋400 随访。
* RR/RRQ/PF 是报告第7章明确给出的可复现传统规则；不声称与历史原生基线同一。
* 44–50 单列为本次权重选择之外的稳健性子集，不称完全未见盲测。

运行命令（在容器中；已存在的正式目录不要删除）::

    CUDA_DEVICE_MAX_CONNECTIONS=2 \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/stage2_closure_mps_pipe \
    CUDA_MPS_LOG_DIRECTORY=/tmp/stage2_closure_mps_log \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u -m training.stage2_closure.campaign \
      --output deliverables/stage2_closure_20261104/campaign --workers 32

上述 MPS 路径需要由运行者在容器中预先启动独立 MPS 服务。
没有 MPS 的机器应调整并发数，不能把不同并发条件的墙钟相除作为加速比。
现存中断目录不可直接复用；保留它们作为执行记录，在新输出目录重跑失败点。

统计、审计与文档（主机 Python 环境）::

    .venv/bin/python -m training.stage2_closure.analyze \
      --source deliverables/stage2_closure_20261104/campaign \
      --output deliverables/stage2_closure_20261104/analysis --workers 4
    .venv/bin/python -m training.stage2_closure.audit \
      --source deliverables/stage2_closure_20261104/campaign \
      --output deliverables/stage2_closure_20261104/audit --cpu 128 --repeats 3
    .venv/bin/python -m training.stage2_closure.plot_results \
      --root deliverables/stage2_closure_20261104
    .venv/bin/python -m training.stage2_closure.build_report \
      --root deliverables/stage2_closure_20261104
    .venv/bin/python -m training.stage2_closure.export_results \
      --root deliverables/stage2_closure_20261104

完整统计要求 220 点，完整审计要求三个 Boundary 负载各30个 M-LWDF 保存状态。
进行中可使用 analyze/audit 的 --partial；它不会将结果标记为已完整完成。
先交核心版使用 plot_results/build_report/export_results 的 --core；
核心报告另检查两场景三臂共60点和 Boundary 主负载30个审计状态。
--draft 仅用于不含完整正式数字的文稿排版检查。

Word 导出需要 python-docx，Excel 需要 XlsxWriter，图形需要 matplotlib 与 Noto CJK。
本次文档依赖安装在 /tmp/stage2_report_python，所有科学计算使用项目 .venv。
PDF 和目录更新使用本机 LibreOffice UNO；启动专用 headless 实例后运行::

    /usr/bin/python3 training/stage2_closure/render_doc.py <报告.docx路径>

实例连接名固定为 pipe,name=stage2_doc_render，需与 LibreOffice 的 --accept 一致。
pipeline.py 是本次运行中的连续后处理进程，按完成状态先生成核心版、后生成完整结果。
它不是系统级定时任务，也不会自动重新启动失效或中断的仿真。
prioritize_core.py 按用户的核心优先交付安排，暂挂本批非核心原生子进程；
60个核心点完成后自动恢复。暂停只延长实际等待，不推进仿真时钟或改变队列、
信道、业务到达和搜索参数。暂停与恢复事件保存在 execution_priority.json。
该记录用于解释执行时序；不将并发轨迹的墙钟用于推断公平的单核加速比。

证据边界
--------

完整包按测量期到达 cohort 追踪至终态；禁止以 MAC 出队量冒充完整成功交付。
均时、P95、公平性和最差 UE 均先在 seed 内求值，再对 seed 等权汇总。
paired.csv 是同 seed 方法减 PFQ 的 Student t 区间；未做多重比较校正。
ablation_paired.csv 是 M-LWDF-PF 减各替代权重的同 seed 差值与区间；
weight_paired.csv 仅保留其中的缺口权重对照，便于单独核对核心结果。
所有配对检查拓扑、业务到达和4000步共同预热的动作及输入哈希。
等价性检查采用独立规范完整枚举，逐轮验证上界、缓存、赢家与终态。
保存状态的成本为单核串行相对比较；连续轨迹墙钟含并发争用，只作描述统计。
完整审计在220点全部完成后自动重做保存状态的等价性检查和串行计时，
确保最终成本测量期间本批质量仿真已停止。早期计时保存在
audit/concurrent_timing_archive/，核心交付包中的计时快照保留原样。
精确指规范局部搜索等价；停止门槛下的局部稳定不等于全局最优。
阴影独立采样只验证单位换算，不冒充旧新 PFQ 的单因素反事实吞吐实验。
E2细负载扫描、E6邻域与出队下界扫描、混合布点、动态负载切换不在本轮完成清单内。

2026-11-05 评测扩充修订
----------------------

新增修订目录为 deliverables/stage2_closure_20261105_r2，保留11月4日交付文件原样。
本次从原220条已核验轨迹补算outcome平均/P95、队列与小区指标，没有重跑PHY。
成功包及outcome都采用测量期间到达的成熟包集合，outcome包含所有MAC完成包
（包括CRC损坏包）和自然过期包。报告和新结果表不再报告P99。
小区归集从共同PFQ执行动作恢复实际服务关联，并与360个保存原生状态核对；
不把ue_topology.csv中的名义生成小区标签作为strongest-cell关联结果。

    .venv/bin/python -m training.stage2_closure.revise_metrics \
      --source deliverables/stage2_closure_20261104 \
      --output deliverables/stage2_closure_20261105_r2 --workers 4
    .venv/bin/python -m training.stage2_closure.revise_report \
      --root deliverables/stage2_closure_20261105_r2
    .venv/bin/python -m training.stage2_closure.export_results \
      --root deliverables/stage2_closure_20261105_r2 --tag 评测扩充版

第6章沿用原稿七个场景小节，第7章扩充传统基线，第9章按环境报告算法效果，
第10章集中计算成本与精确性，第11章总结。原始大体积轨迹仍在11月4日目录，
新目录的campaign及cohort引用原始文件；证据包包含补充统计、校验与图表。

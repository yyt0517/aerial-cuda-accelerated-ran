七小区三分支 DA-PLS 闭环实验
==========================

冻结原七小区实验的仿真器、输入、PHY、权重、邻域、阈值和测量定义。
每个当前状态计算三条完整搜索：缺口权重DPLS、PFQ起步的原DA、DPLS终点起步的DA。
所有分支共用原始PFQ的90%计划出队下界，第三分支强制采用DPLS终点。
最终最大化同一DA定点目标；精确同分顺序是原DA、DPLS起点DA、DPLS。
只执行选中动作；三个分支不推进独立历史。在线保证是当时的DA分数不劣于三分支。

原 warm_start 仅在分数优于PFQ时才接纳，无法保证始终从DPLS起步。
build.py复制已经验证的七小区搜索器，唯一内层改动为强制采用合法warm_start。
PFQ下界、候选、剪枝、停止条件和PHY不变；原库与原实验源码不改写。
两个冷启动分支继续调用原库。新库只用于第三分支。

协议：seeds 41/42/43；4000共同PFQ预热＋4000稳定＋20000测量＋400随访。
TTI 0.5ms，每seed正式测量10秒。对照复用原七小区九条轨迹，逐seed验证
拓扑、全程业务和信道、共同预热的完整原生指标以及首个策略输入完全一致。
保存4001、8001、18001、28000时隙的三分支动作/值/权重；独立完整枚举核验各分支。

构建和核验::

    .venv/bin/python -m training.stageb_sevencell_multistart.build
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest training.stageb_sevencell_multistart.test_policy

使用已有MPS配置在容器中运行 campaign 模块，--smoke --seeds 41 可先做短轨迹核验。
分析依次运行 analyze、audit、report 模块；本次完整调用见 run_campaign.sh。
正式运行期间不要修改本包代码或任何冻结依赖；所有源与二进制在启动和结束时核验哈希。
输出目录拒绝覆盖。原始结果保留在 deliverables/seven_cell_boundary08_20261105_v1。

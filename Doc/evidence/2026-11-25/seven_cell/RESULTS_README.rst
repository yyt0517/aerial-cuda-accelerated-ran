七小区 boundary08 实验结果包
============================

先读 analysis/report.txt；图表为 comparison.png/pdf 与 topology.png/pdf。
逐 seed、聚合和算法配对差异在 analysis/ 下；3→7 小区历史对照另列 scaling_*.csv。
campaign/ 中打包了每次运行的逐 UE、逐小区统计、启动参数、汇总和验证证据。
完整原始轨迹、包队列与保存状态保留在工作区同名 campaign/，未装入此轻量结果包。
source_snapshot/ 是运行时源码与求解库快照；reporting_source/report.py 是最终报告生成器，
仅在正式运行结束后完善了报告文字和历史对照，仿真和统计计算代码未改。
复现实验步骤见 source_snapshot/seven_cell/README.rst。

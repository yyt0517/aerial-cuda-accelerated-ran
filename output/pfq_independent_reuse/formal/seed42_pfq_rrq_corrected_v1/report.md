# seed42：修正环境下的 PFQ / RRQ

2026-09-20，已完成。使用同时包含 TBS bits→bytes、PHY slot 索引、GPU MCS 初始化同步修正的 `build.gpu_mcs_sync_v1`。没有修改调度规则、降低负载或启动训练。

## 协议

- topology/traffic seed 均为42，3 cell / 36 UE，coop_boundary、strongest association、omni、Rayleigh、DL Type-0、SVD。
- 每小区17 PRG × 16 PRB，TTI0.5ms；3000-byte packet、Poisson1 packet/UE/TTI、TTL200ms、homogeneous。
- 每组1000 TTI warmup + 4000 TTI measured，统计TTI1001–5000（2秒）；goodput为三个小区合计成功字节吞吐，不是单小区吞吐。
- 两组实际 measured offered 均为 **1729.548 Mbps**；同 offered stream、topology hash、raw channel fingerprint 已断言一致。
- `CUMAC_EXEC_MODE=both`，业务指标来自CPU PHY/队列。slot置换、common-MCS覆盖及GPU固定输入重放均关闭。RR审计随机种子42；PFQ/RRQ仍使用各自正常的queue-aware规则。
- 日志确认分别执行 `queue-aware PF reference` 和 `queue-aware RR baseline`，不是PF/RR。

## 结果

| measured 指标 | PFQ | RRQ |
|---|---:|---:|
| Goodput Mbps | **1388.112544** | **1197.038240** |
| Goodput / offered | 80.2587% | 69.2110% |
| Scheduled dequeue Mbps | 1528.722224 | 1323.872308 |
| TB BLER | 8.9579% | 8.9768% |
| Expired bytes / offered bytes | 10.5367% | 22.7203% |
| Completion delay mean ms | 49.3977 | 55.8018 |
| Completion delay P95 ms | 199.0 | 199.5 |
| 总 backlog mean MB（十进制） | 16.0776 | 22.6772 |
| 总 backlog P95 MB | 17.2311 | 23.3470 |
| 每TTI正出队UE数 | 32.5635 | 32.35575 |
| PRG利用率 | 100% | 100% |

本 seed 下 PFQ 比 RRQ goodput 高 **191.074304 Mbps**。仍有非零过期和backlog，不能据此宣称1729.548 Mbps业务已被稳定承载。completion delay 字段实际统计 MAC 出队完成：packet tracker 的 delivered 标记没有按 TB 成功过滤，因此不是成功端到端交付时延，应与成功 goodput、失败出队、expiry/backlog 一起理解。数值未重算，补充的是统计口径。此结果不能与旧错误环境直接进行算法优劣归因，也不代表多seed结论。

## 验证与归档

两个仿真进程正常结束；各180,000行、共360,000个UE-TTI通过独立TBS重建、bits→bytes上限及流量守恒检查。输入配对及运行前后二进制/共享库hash检查通过。

```text
simulator SHA256 b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644
libcumac SHA256  079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e
```

- 完整指标与输入identity：[`final_report.json`](final_report.json)
- 进度：[`status.json`](status.json)
- 执行入口：[`run_experiment.py`](run_experiment.py)
- 原始数据与配置：`pfq/`、`rrq/` 中的 `command.json`、`reference_manifest.json`、`service_balance_trace.csv`、`packet_outcome_samples.csv`。

训练队列未恢复；本轮到此结束。

# 物理容量单位修复与新训练边界

日期：2026-09-20。旧复现队列停止使用，新训练须等待修正场景可用性确认。

## 原因与修复范围

原 `determineTbsPdsch` 返回 bits，CPU/GPU PDSCH 调用端与容量导出错误地按 bytes 使用；Python CTBS 镜像也保留了该错误。旧 parity 只验证了两端一致，没有验证量纲。

本次修复：

- 保持 C++ TBS 原函数返回 bits；四个 CPU/GPU Type-0/Type-1 容量消费出口及 `scheduledTbsCapacityByActiveUe` 显式除以 8。
- 队列扣减、scheduled dequeue、goodput/failed/expired 计数均为 bytes；内部 `avgRates` 保持 bit/s，因此队列字节转入速率时乘 8，既有 Mbps 特征及 PFQ 速率分子/分母单位一致。
- Python `determine_tbs_bytes` 的小/大 TBS 分支均转换为 bytes，并更改单位语义 hash。
- BC 归档 schema 改为 `independent-bc-ctbs-physical-bytes-v2`；新 reader 拒绝旧 v1，禁止简单改旧字段后重用历史轨迹。
- 首次真实短跑发现另一个出队错误：已有 slot→UE map 的无效 `0xFFFF` 槽会回退为同编号 active UE，产生重复出队。现在已有 map 时无效槽返回无效，仅 map 为空时允许 identity fallback。首轮 PFQ 有 68 行出队超过容量；现场保留在 `failed_attempt1/`，不得与后续修复结果合并。

独立字面量检查包括一 PRG（16 PRB）、11 symbols、MCS0 的 1/2/3/4 层容量分别为 63/123/185/253 bytes；MCS4 四层为 1249 bytes。没有以 Python 镜像输出作为这些期望值。真实逐 UE/TTI 校验还检查累计容量及 `0 <= dequeue <= capacity`。

## 隔离与复现

- 旧二进制保留：`build.reuse_actor/.../multiCellSchedulerUeSelection`，SHA-256 `ac1e809cf5f45aa553545c447915803ac1ea1c00734748477b956367be9beb8f`。
- 修正二进制：`build.capacity_units_v2/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection`。800 TTI 初查使用 SHA-256 `b70c99aba900618d77af4b249ae7e32a6c35364eb9e66c0a5a731b7a5a51ecc0`；增加原始 bits 与信道审计后，当前五组判定实验使用 `4aa34e78114583f41d07dbe5ba3dc2acbbd916de34a2c05657b1df13de25b29b`。两批结果分别归档，不混为同二进制实验。
- 历史冻结 checkpoint 未修改，SHA-256 `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`。
- 短场景目录：[capacity_units_bytes_v2_short](../output/pfq_independent_reuse/formal/capacity_units_bytes_v2_short/)，执行脚本 `run_short.py`，实时状态 `status.json`，完成后写 `report.json`。
- 固定 seed61、3cell/36UE、800 TTI、比较窗口 201–800、TTL 200ms、arrival rate 1.0；PFQ/RRQ 原生执行，旧 DPA 固定权重 argmax、修正 CTBS、guard off。三组采用同二进制、同拓扑及 offered stream。

旧复现队列在用户要求暂停前已于 S1 因缺少 `independent-reward-scales` 参数退出。没有剩余活跃进程，已标记 `paused_capacity_units_invalid`，原错误及所有模型/数据保留。新实验不得直接恢复这个旧目录。若后续进入新训练，还须用新基线产出/确认 S1 所需 reward scales，而非复制旧校准。

## 解释边界与下一步

旧 DPA 在新环境中只是固定权重迁移探针，不是重新训练后的 Full。短跑用于排除物理量纲/出队错误及判断负载严重程度，不证明长期收敛、统计显著性或某组件贡献。

原负载的初步 PFQ/RRQ 结果已显示严重过载：offered 约 1724.64 Mbps，而 goodput 约 291.78/388.46 Mbps，末 200 TTI 过期字节占 offered 比约 77.09%/71.78%。完整三方结果以本目录报告为准。在标准场景可用性得到确认前，不启动大规模 BC/PPO 或消融。

800 TTI 初查已完成。旧 DPA 的 measured goodput 为 387.77 Mbps，expiry 为 43.80%，末 200 TTI expiry 为 69.90%，UE goodput P10 为 0；有 21 个 UE 在 201–800 窗口无出队。这是固定旧权重在修正环境中的迁移表现，不能解释为新 Full 训练结果。

## 五组判定实验（用户确认的下一步）

- 目录：[capacity_units_bytes_v2_decision_v1](../output/pfq_independent_reuse/formal/capacity_units_bytes_v2_decision_v1/)，实时 `status.json`，完成后生成 `report.json` / `report.md`。
- 固定 seed61、3cell/36UE、Rayleigh channel、3000-byte packets、arrival 1.0、TTL 200ms。五组均为 **1000 warmup + 4000 measured**，窗口 1001–5000。
- PF/PFQ/RR/RRQ 分别使用原生 `-b 0/2/1/3`，旧 DPA 固定 checkpoint、argmax、CTBS、guard off。共享 GPU1 顺序执行，避免资源争用；不是混用不同负载的五场实验。
- 验证 offered packet identity、拓扑 SHA-256，以及每 TTI 调度前原始全 active-UE CFR 的 FNV-1a64 指纹序列。信道随机种子来自 topology seed61；不拿策略相关的 post-EQ SINR 当作原始信道相等证据。
- 新增 `tbs_capacity_bits`，直接读取 PHY 原始返回值，不从 byte 导出乘 8 反推。C++ 逐行强制 `dequeued_bytes <= TBS_bits / 8`（避免乘法溢出，等价于用户要求），并验证 byte 容量；Python 再检查独立字面量、CTBS 和所有 UE/TTI（含零分配 UE）。
- 独立 C++ 单位测试同时覆盖正确 63-byte 出队、错误 504-byte 出队、容量和出队共同误用 504 bytes、以及更小的 40-byte 出队；错误实现先失败，修正后通过。C++ 两项相关测试通过，Python 本轮 133 项相关测试通过。
- 五组各 16 TTI 冒烟已完成，输入三类指纹全部相等，单位检查通过。该窗口不足一个 TTL，**不能据其零 expiry 判断场景可用**。

报告口径：

1. offered load、实际分配的 PHY TBS 容量与两者之比；实际分配容量是策略相关、BLER/需求截断前的量，**不是场景最优/可持续可达容量**。另给五组最大实测 goodput，不把它命名为容量上界。
2. goodput、BLER；完成包 completion mean/P95 delay；delivered+expired 的 outcome delay；expiry bytes/offered bytes；弱 UE P10、零服务 UE。
3. 每 TTI 全网 post-queue 总 backlog 的 mean/P95，另给 UE×TTI backlog 分布；正出队 UE/TTI 与 PRG 利用率。
4. PFQ/DPA 将实际 Type-0 bitmap 还原为 `(cell, PRG) -> active UE id | Blank`，记录 token 一致率、完整 TTI 一致率及首个分歧 TTI，含 warmup 的首次分歧与 measured 首次分歧分别报告。分歧后队列/OLLA 状态不同，因此这是**独立轨迹动作一致率，不是同状态因果归因**。

此判定脚本不会自动调负载、采集 BC 或启动训练。须先审阅 5000 TTI 结果，再决定下一步；旧数据保持原状，不缩放复用。

需要先确定是否允许重新校准 offered load；建议只用基线做少量负载探测，固定常规/期限压力两档协议后，重新生成 BC、基线和训练数据。不能根据旧 DPA 是否获胜挑选负载，也不能事后放宽既定验收阈值。旧绝对 goodput、频谱效率和跨单位模型优势不可作为修正物理环境的性能结论。

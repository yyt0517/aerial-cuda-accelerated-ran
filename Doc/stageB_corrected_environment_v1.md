# Stage-B 三项关键修正与实验有效性分界

> 2026-09-20；环境版本：`stageb_corrected_2026_09_20_v1`。
> 这是研究环境的版本标识，不是 Git commit，也不是一个新训练完成的 DPA 模型。

## 1. 决定与影响

容量单位、PHY slot 索引、GPU MCS 初始化同步三个关键错误已修正，并完成各自限定范围的验证。自本记录起，修正前的训练、基线、消融和泛化结果退出“当前有效性能证据”。此前“DPA 稳定超过 PFQ/RRQ”“deadline/attention 已证实有效或无效”等结论均须在修正环境中重新检验。

旧记录没有被伪造或抹除：它们仍描述当时错误环境中的真实观察，可用于追溯实现、学习路径和故障。但同二进制、同输入的 paired 比较不足以证明仿真物理语义正确。不得把旧 goodput、capacity、expiry、选模门槛直接迁移到新环境，也不得把旧容量字段除以 8 后复用训练轨迹。

不是每个旧文件都同时受三个错误影响：前两项污染业务执行；GPU 同步问题独立影响原生 GPU MCS，已完成的 CPU 对照指标未因这一修正变化。不能把旧消融结果全部归因于某一个 bug，也不能保证修复后 DPA 或其组件必然获得优势。

## 2. 三项修正及证据边界

| 问题 | 来源与影响 | 修正和已完成验证 |
|---|---|---|
| TBS bit 被当成 byte | 官方公开 Release 25-3 示例 `network.cu` 的 TBS/队列接缝存在此问题；不等于整个官方 PHY 都错误。容量和出队尺度错误改变了队列、奖励、特征和训练轨迹。 | 在消费边界 bits→bytes，同步出队、统计导出与 Python CTBS；保持内部 rate 单位正确。独立 TBS 重建、流量守恒及 `8 × dequeued_bytes ≤ TBS_bits` 检查。 |
| PHY slot / active UE ID 混用 | 本地 commit `4b05d6b30f024f936976286c6a82d7fee4faa8a9` 引入；slot-packed 信道/precoder 被用 active UE ID 索引，污染 PF/RR 及其他共享该 PHY 的策略。 | CPU/GPU Type-0 信道/precoder 恢复 slot 索引，队列/关联保留 UE ID。固定物理输入只置换内部 slot 后，受控 DL SINR/BLER 不变；PF/RR 同场景对照改善。UL 对称修正未完成运行验证。 |
| GPU MCS 共享计数初始化缺少同步 | 本地 commit `6c31b20aa1ce277a7906b788b5c9b15bae4485f6` 删除初始化屏障；官方 Release 25-3 原有同步。清零与 atomicAdd 竞争，可令有 PRG 的 UE 得到计数 0、MCS −1。 | WB-SINR/CQI Type-0 恢复无条件初始化屏障；PF/RR 首帧固定输入各重放 128 次，计数/MCS 错误由 1/6 条降至 0/0；CPU 长轨迹与全部汇总指标不变。CQI 未运行验证，未证明 GPU-only goodput 增益。 |

完整来源、命令、输入标识与原始结果：

- [容量单位审计](../output/pfq_independent_reuse/formal/mcs_capacity_units_audit_v1/report.md)
- [PHY slot 不变量及索引前后对照](../output/pfq_independent_reuse/formal/phy_slot_invariance_v1/report.md)
- [GPU MCS 同步最小闭环](../output/pfq_independent_reuse/formal/gpu_mcs_sync_v1/report.md)

上述审计目录包含 before/after 或中间失败尝试，不是目录中每个结果都属于修正后基线。仅 bits→bytes 的 `build.capacity_units_v2` 仍有索引问题；`build.phy_slot_v3` 尚未包含 GPU 同步修正；不能作为三项修正完整版本。

## 3. 固定修正代码与构建身份

- 已验证构建目录：`build.gpu_mcs_sync_v1`。
- Git 基底：`f567a839e07ba5f622b54fbacd388ecf9427411e`；存在未提交源码修改，**不能仅凭 HEAD 复现**。
- simulator SHA-256：`b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644`。
- libcumac SHA-256：`079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e`。
- [版本快照](../output/pfq_independent_reuse/versions/stageb_corrected_2026_09_20_v1/README.md)保存二进制、共享库、tracked source patch、变更/新增源码 overlay 和逐文件哈希。不是完整容器快照；容器、依赖及构建配置仍须一并记录。

当前有效短基线为 [seed42 PFQ/RRQ](../output/pfq_independent_reuse/formal/seed42_pfq_rrq_corrected_v1/report.md)：1000 warmup + 4000 measured，offered 1729.548 Mbps，goodput 分别 1388.113 / 1197.038 Mbps。两者仍有过期，不代表负载已被稳定承载。此版本尚无重新训练并验收的 DPA checkpoint。

## 4. 历史材料归档政策

- [文档归档](archive/pre_correction_2026_09_20/README.md)：50 份修改前原文，包括设计、综合报告、消融/复现协议、JSON 参数与两个 README；原文不改写。
- [实验数据归档](../output/pfq_independent_reuse/archive/pre_correction_2026_09_20/README.md)：迁移旧 formal 中 70 个顶层条目，保留全部 7,780 个普通文件、12,374,575,291 bytes；包括原始轨迹、日志、失败尝试、BC、checkpoint、报告及配置。
- 旧 `formal/<campaign>` 路径保留相对符号链接，供历史代码与引用定位；**路径能打开不代表获准复用或恢复队列**。
- 哈希清单记录原路径、归档路径及每个文件 SHA-256；原始 status 保持当时状态，当前有效性以本记录为准。
- 旧冻结 DPA 的 SHA-256 仍为 `94a2e40f0c0a56632e0c0ca63af4fa54d894d35aea38c2dd34afc0cfeadf2d7a`，只能用于历史研究或明确标记的迁移诊断，不能作为新环境已验收模型。
- 旧复现队列不续跑；检查到的旧 equal-budget dispatcher 仍处于暂停状态，未恢复、未终止。其他工作区任务不在本次范围内。

没有删除旧数据，也没有重新计算旧报告来掩盖版本差异。更早的 Doc/archive 材料继续按历史资料处理。归档不是备份到外部存储；仍位于同一工作区。

## 5. 后续新实验的进入条件

本轮只记录、归档、固定版本，**不启动新训练或仿真**。后续须新建 campaign，不能往旧目录追加：

1. 固定修正源码/构建身份；确认所用二进制及运行时 libcumac 都是同一修正版，不依赖旧脚本默认 build 路径。
2. 在共同 topology、traffic、channel、seed、TTL、窗口下重新校准基线，同时看 goodput、BLER、expiry、completion/outcome delay、弱 UE 和 backlog。不要继承旧绝对验收阈值；任何负载或阈值修改均需明确记录并在训练前冻结。
3. 从新环境重新采集 BC；当前 schema 为 `independent-bc-ctbs-physical-bytes-v2`，但 schema 通过不代表同时包含索引/同步修正，仍须校验代码和输入身份。旧 BC、replay、normalizer、optimizer 及旧轨迹不默认复用。
4. 先获得可重复的 Full 训练与独立 argmax 验收，再开始成对组件消融。旧 DPA 可作历史参考，不替代同流程 Full。
5. 保留原研究问题：Full、−Attention、−Features、−Reward；奖励删除范围及平均 log-prob 口径需在新协议中明确，不在本次归档中暗改。训练/验证/测试隔离，选模指标绑定实际 checkpoint。
6. 单位不变量、slot 不变量、GPU 原始计数/MCS 检查通过，只表明已测范围正确；新增模式/拓扑需要相应验证。性能证据不能混合修正前后结果。

当前未证实：修正环境中的 DPA 优势、组件贡献、7cell 泛化、零过期、GPU-only 性能改善及生产级正确性。

## 6. 后续复核：goodput 与 completion 的统计边界

代码确认 goodput 从 `tbErr == 0` 时的队列出队字节累计，再换算 Mbps。历史 seed101 homogeneous DPA 的 TTI1001/UE0，正确 TBS 为 1480 bits（185 bytes），旧记录却出队并计成功 1480 bytes。因此容量错误确实污染 goodput；整场结果因队列反馈改变，不能统一除以 8 修复。

现有 packet tracker 在 MAC 出队完成时标记 `delivered`，不按 TB 成功过滤。`completion_delay` 是 MAC dequeue-completion delay；不能将它称为成功端到端交付时延。修正容量并没有改变这一队列模型，新实验必须分别保留成功字节、失败出队字节、出队完成/过期/未完成包统计，并显式标注无法测得的成功交付时延。

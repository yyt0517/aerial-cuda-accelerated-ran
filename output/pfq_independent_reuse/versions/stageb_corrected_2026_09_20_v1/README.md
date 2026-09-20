# stageb_corrected_2026_09_20_v1

2026-09-20，研究环境快照，不是 Git commit，也不是已训练的 DPA 模型。三项修正：TBS bits→bytes、PHY slot 索引、GPU MCS 初始化同步。

- [manifest.json](manifest.json)：Git 基底、dirty 状态、源码/二进制/共享库 SHA-256。
- `simulator`、`libcumac.so`：从已验证 `build.gpu_mcs_sync_v1` 复制；对应 seed42 PFQ/RRQ 结果。
- [tracked_source.patch](tracked_source.patch)：相对记录的 HEAD，cuMAC/training 范围 tracked 变更。
- `source_overlay/`：该范围的修改/新增源码，包括未跟踪源码；逐文件哈希在 manifest 中。

复原须在隔离目录检出 manifest 的 Git 基底，检查并应用 tracked patch，再叠加 source_overlay，使用对应的 Aerial 容器及依赖重新构建。不能只检出 HEAD；也不能仅运行这里重命名的 simulator 就假定已加载这里的库。运行时必须核对实际 simulator 与 libcumac 的 hash。此快照不是完整容器/工具链或全仓库备份，不能保证重新构建 bit-for-bit 相同。

simulator SHA-256：`b0e56c8b851603a920467d8ca2c2e24e554b4048a207204ee985abedd9a73644`
libcumac SHA-256：`079f4c72ca94358a410584fe1a666914a5da4041acbc4085fada50957363692e`

[修正来源、已测范围及新实验要求](../../../../Doc/stageB_corrected_environment_v1.md)。旧 BC/轨迹不可缩放复用；旧队列不恢复。

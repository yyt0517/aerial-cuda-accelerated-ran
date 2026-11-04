# 整包等待与包龄奖励续训

基于 `stageb_ppo_v2` 上一轮最终 PPO 继续训练。完整等权 DPLS、冻结神经修复与逐候选物理验证保持不变；没有 CTBS 裁剪、需求上限掩码、业务或服务缺口权重。状态、动作候选、网络及 PPO 小批量规则全部复用。

每个时隙执行后的奖励为：

```
(C - E - N × 3000/400 - 2 × 3000/400 × Σ clip((age - 20)/20, 0, 1))/86400
+ γ Φ(next) - Φ(current)
```

C 是真实完整成功包字节，E 是整包过期字节；N 是所有仍未结束的包数，包括仅剩少量字节的包和 CRC 受损包。age 以 0.5 ms 时隙计，TTL 为 400。10 ms 后增加包龄成本，20 ms 后单包瞬时成本封顶，避免极老包的代价无限增长。封顶的是每时隙成本，累计等待代价仍随等待增加。一个在第 d 个时隙完成的包被计费 d−1 次；其固定首时隙不影响策略选择。

这属于所有用户一致的时延目标，不是业务类别加权。奖励从因果 FIFO 读取，独立核验从原生服务轨迹重建包生命周期，显式检查完成、过期以及未结束包的边界。历史适配器 `rewards.csv` 的旧 reward 列仍不用于训练，新奖励见 `long_credit_rewards.csv` 与 rollout 的 `reward_components`。

训练固定 seeds 41–43、4000 PFQ 预热与 512 等权 DPLS 稳定化，然后 24×1024 个决策/seed。恢复旧 actor 与 Adam；因奖励目标改变，critic 与其优化器重新初始化。环境从统一预热重启，不声称原生轨迹无缝续接。每轮最多 192 个 actor 更新；KL 整 epoch 回退机制不变。只评价固定第 24 轮终点，不按开发验证选择检查点。

验证使用既有 seeds 44–45 的 4000+4000+4000+400 协议、相同到达 cohort，并对照上一轮 PPO、同候选池物理贪心、等权 DPLS。同时报告逐 seed 与均值、P95、整包吞吐、过期率、UE P5 和共同成功包时延。奖励调整与训练预算同时改变，旧检查点对照不是严格的同预算奖励消融。44–45 是开发集，不作泛化结论。

```bash
python -m unittest discover -s training/stageb_ppo_tail/tests -v
python -m training.stageb_ppo_tail.campaign --source /absolute/previous/run --out /absolute/new/run
python -m training.stageb_ppo_v2.evaluate --root /absolute/new/run --arms ppo
```

本轮证据输出 `deliverables/ppo_tail_wait_20261030_v1/`。当前依然逐时隙执行完整 DPLS，PPO 只选择已验证方案；不能声称端到端推理提速。计时脚本在相同实时状态上交替执行纯 DPLS 和完整候选池+PPO，单独报告小网络开销与总开销。

## 固定终点结果

固定 24 轮续训完成 73,728 个新在线决策、4,608 次接受的 actor 更新；原生、整包等待/包龄、采样与 GAE 独立核验通过。两开发 seed 的完整包吞吐均值为 1235.202 Mbps，过期率 1.205%，平均/P95 成功包时延 4.341/6.75 ms。上一轮对应 1240.686 Mbps、0.663%、7.012/24.5 ms；等权 DPLS 为 1237.926 Mbps、0.798%、5.887/13 ms。逐 seed 完整保留上一轮吞吐/过期收益且 P95 不高于等权 DPLS：未通过；仅要求相对两个物理基线仍有吞吐/过期收益且 P95 达标：未通过。未更换默认算法，未证明收敛。

用续训起点参数、相同网络和候选结构进行同状态交替计时 256 次：纯 DPLS 平均 136.724 ms，完整候选池+PPO 155.273 ms，开销增加 13.57%；小型 PPO 网络本身约 0.511 ms。当前仍逐时隙运行完整 DPLS，没有推理加速。该数据是研究 Python 流程计时，不是生产部署延迟。

详见 [完整报告](../../Doc/evidence/2026-10-30/ppo_tail_wait/report.txt)、[逐 seed 对照](../../Doc/evidence/2026-10-30/ppo_tail_wait/comparison.csv) 与 [目标门槛](../../Doc/evidence/2026-10-30/ppo_tail_wait/requested_target_gate.json)。

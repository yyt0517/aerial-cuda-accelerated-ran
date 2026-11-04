# 过期保护与包龄代价重新平衡

强包龄奖励版本的两个开发种子都改善了 P95，但吞吐和过期率退步。本轮保留该负结果，恢复其最终 actor 和 Adam，继续固定 12 轮、36,864 个新在线决策；critic 因奖励目标改变重新初始化。环境重新经过同一 PFQ/DPLS 预热，不声称原生轨迹无缝接续。

奖励改为 `(C - 2E - pending_full_bytes/400 - 0.5*age_exposure_bytes/400)/86400` 加原有进度势函数差。包龄 exposure 仍为每个未结束包的 `3000*clip((age-20)/20,0,1)`；时间单位0.5ms。与强包龄版本相比，附加包龄成本降至四分之一，所有用户的过期事件代价统一加倍。整包等待成本保持不变，包括部分传输与受损包，不引入业务类别或服务缺口权重。

状态、候选池、神经修复、精确物理验证、98% 即时成功字节保留、90% PFQ 出队下界及 PPO 优化规则保持不变，CTBS 裁剪关闭。目标仍相对原始服务选择 PPO 的吞吐/过期收益，以及等权 DPLS 的 P95；不把较差的强包龄版本当成放宽目标的新基线。

本轮调整根据已暴露开发 seeds44–45 的上一实验结果作出，不称独立测试。固定最终第12轮，不依据中间验证挑检查点。对照同时保留原PPO、强包龄PPO、同候选池物理贪心和等权DPLS；311–315不使用。

```bash
python -m unittest discover -s training/stageb_ppo_tail_balance/tests -v
python -m training.stageb_ppo_tail_balance.campaign --source /absolute/strong_age_run --reference /absolute/original_service_ppo_run --out /absolute/balanced_run
python -m training.stageb_ppo_v2.evaluate --root /absolute/balanced_run --arms ppo
```

实际学习奖励保存在 `long_credit_rewards.csv` 和各 rollout 的 `reward_components` 中；原生适配器 `rewards.csv` 的兼容 reward 列不用于训练。独立核验用原生包生命周期重新计算整包等待、包龄 exposure 以及双倍过期惩罚。`--reference` 只指定原有收益检查点的对照目录；`--source` 指定实际恢复的强包龄版本，二者用途不同。

## 固定终点结果

固定 12 轮平衡续训完成 36,864 个新在线决策、2,304 次接受的 actor 更新；原生、整包等待/包龄、采样与 GAE 独立核验通过。两开发 seed 的完整包吞吐均值为 1235.232 Mbps，过期率 1.306%，平均/P95 成功包时延 4.951/7.5 ms。原有收益 PPO 对应 1240.686 Mbps、0.663%、7.012/24.5 ms；等权 DPLS 为 1237.926 Mbps、0.798%、5.887/13 ms。逐 seed 完整保留原有 PPO 吞吐/过期收益且 P95 不高于等权 DPLS：未通过；仅要求相对两个物理基线仍有吞吐/过期收益且 P95 达标：未通过。未更换默认算法，未证明收敛。

用两阶段续训前的原服务选择 PPO 参数、相同网络和候选结构进行同状态交替计时 256 次：纯 DPLS 平均 136.724 ms，完整候选池+PPO 155.273 ms，开销增加 13.57%；小型 PPO 网络本身约 0.511 ms。当前仍逐时隙运行完整 DPLS，没有推理加速。该数据是研究 Python 流程计时，不是生产部署延迟。

详见 [完整报告](../../Doc/evidence/2026-10-30/ppo_tail_balance/report.txt)、[逐 seed 对照](../../Doc/evidence/2026-10-30/ppo_tail_balance/comparison.csv) 与 [目标门槛](../../Doc/evidence/2026-10-30/ppo_tail_balance/requested_target_gate.json)。

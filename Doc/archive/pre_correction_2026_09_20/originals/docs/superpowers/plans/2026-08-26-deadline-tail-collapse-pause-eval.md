# Deadline-tail 长训坍塌：暂停与正式评估计划

## 假设

`deadline_tail` reward 在前缀阶段曾改善 goodput，但继续 PPO 更新后会出现 per-UE 服务退化；冻结在训练中观测到的最佳 checkpoint（iter7）并按正式 gate 评估，可以区分“checkpoint 已有可用策略”和“后续更新导致坍塌”。

## 最小执行路径

1. 确认训练进程、输出目录和 `best_iter=7` checkpoint；对当前长训发送可恢复的暂停信号，不删除进程或输出。
2. 使用 `ppo_actor_best.pt`（iter7）做正式 checkpoint evaluation，顺序为 seed42、seed41、seed43。seed42 若 gate 失败，停止后续 seed，避免把未授权的泛化结论混入结果。
3. 记录 goodput、BLER、expiry、UE tail allocation/service、packet delay、rate、backlog/合法性/CTBS parity 等已产生指标，并明确安全前缀 gate 只覆盖前 8 个 iter。
4. 在主报告和导航中记录：坍塌首次出现的大致 iter、last/best 对比、暂停原因、正式评估结论、证据路径以及下一步应验证的 service-history / update-stability 假设。

## 停止条件

- 训练暂停后进程不再推进；`iter7` checkpoint 和 metrics 可读。
- seed42 正式评估失败则不运行 seed41/43，并将其标为未评估。
- 不调 reward 系数、不修改网络/候选/PRG 顺序、不用 actor_last 作为正式候选。

## 复核指标

系统 goodput/BLER、expiry、replay/mismatch、动作合法性、CTBS parity；同时保留 per-UE allocation（特别是 UE2）、UE tail rate、packet delay、backlog 等机制诊断，防止只按平均 goodput 选择 checkpoint。

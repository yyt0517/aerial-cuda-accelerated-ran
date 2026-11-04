# 2026-10-30 研究证据

本轮围绕无 CTBS、等权 PPO 的差距展开诊断，并复用已有神经修复和物理验证训练跨时隙服务选择。首轮结果见 [服务选择报告](ppo_service_choice/report.txt)，强包龄改动见 [首次奖励报告](ppo_tail_wait/report.txt)，奖励续训见 [重新平衡报告](ppo_tail_balance/report.txt)，最新搜索降本实验见 [物理教师与神经候选报告](ppo_search_teacher/report.txt)。

- [冻结训练与对照协议](ppo_service_choice/protocol.json)
- [旧训练覆盖、价值拟合与奖励分解](ppo_service_choice/diagnosis.json)
- [旧 critic 隐层饱和检查](ppo_service_choice/critic_hidden_diagnosis.json)
- [同用户服务不足对照](ppo_service_choice/weak_service_diagnosis.json)
- [价值表示的时间留出诊断](ppo_service_choice/critic_representation_probe.json)
- [最终逐 seed 验证](ppo_service_choice/evaluation.csv)
- [整批与小批量优化对照](ppo_service_choice/optimizer_comparison.csv)
- [同包成功集合的时延对照](ppo_service_choice/matched_packet_comparison.csv)
- [实际优化次数与策略变化](ppo_service_choice/optimizer_activity.png)
- [实现与运行说明](../../../training/stageb_ppo_v2/README.md)

完整轨迹与检查点位于本地 `deliverables/ppo_diagnosis_continue_20261030_v1/`。已有研究默认算法注册表未因此更换；新的学习轨迹关闭 CTBS 裁剪与业务/服务缺口加权。开发 seeds 44–45 已有暴露，不作独立测试集解释。

新增整包等待与包龄奖励实验：

- [冻结协议](ppo_tail_wait/protocol.json)
- [逐 seed 对照](ppo_tail_wait/comparison.csv)
- [完整收益保留与 P95 门槛](ppo_tail_wait/requested_target_gate.json)
- [相同包对照](ppo_tail_wait/matched_packets.csv)
- [同状态推理计时](ppo_tail_wait/inference_timing.json)
- [训练与结果图](ppo_tail_wait/training_and_comparison.png)
- [独立训练审计](ppo_tail_wait/independent_training_audit.json)
- [实现说明](../../../training/stageb_ppo_tail/README.md)

新增过期保护与包龄重新平衡实验：

- [冻结协议](ppo_tail_balance/protocol.json)
- [逐 seed 对照](ppo_tail_balance/comparison.csv)
- [完整收益保留与 P95 门槛](ppo_tail_balance/requested_target_gate.json)
- [相同包对照](ppo_tail_balance/matched_packets.csv)
- [同状态推理计时](ppo_tail_balance/inference_timing.json)
- [训练与结果图](ppo_tail_balance/training_and_comparison.png)
- [独立训练审计](ppo_tail_balance/independent_training_audit.json)
- [实现说明](../../../training/stageb_ppo_tail_balance/README.md)

新增物理教师引导的少候选 PPO：

- [冻结协议](ppo_search_teacher/protocol.json)
- [逐 seed 质量对照](ppo_search_teacher/comparison.csv)
- [独立同状态计时](ppo_search_teacher/inference_timing.json)
- [联合验收](ppo_search_teacher/requested_target_gate.json)
- [独立训练审计](ppo_search_teacher/independent_training_audit.json)
- [结果图](ppo_search_teacher/training_and_comparison.png)

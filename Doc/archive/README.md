# 文档归档说明

本目录保存历史设计、阶段性会议材料和平台能力审计。归档文档只用于追溯背景，不再作为当前实验结论或实现状态的主引用。

## 分类

- [pre_correction_2026_09_20/](pre_correction_2026_09_20/README.md)：三项关键错误修正前的设计、训练/消融/性能文档原文，以及旧实验数据归档索引。旧结论不作为修正环境的性能证据；[有效性分界](../stageB_corrected_environment_v1.md)优先于历史文档中的“当前”状态。
- `platform/`：Aerial 平台能力、天线/precoding、PRG/RBG、候选状态字段审计。
- `hgraph_design/`：HGraph 设计过程、scalable PPO 方案和 v33 训练/评测过程记录。
- `hgraph_experiments/`：HGraph v34-v50 多轮优化、训练、评估的完整流水日志和 3cell retrospective。
- `legacy_gnnrl/`：旧 GNNRL offline/online/replay/ONNX 文档和 readiness checklist。
- `plans/`：早期总体实施计划、UE/PRG 接管路线、历史 RL+GNN MVP。
- `meeting_notes/`：第一次/第二次会后确认材料。
- `code_audit/`：历史代码审计，例如原始 PF call-chain。

当前项目入口请看上层 [`README.md`](../README.md)，当前 Stage-B 详细配置请看 [`current_stageB_effective_configuration.md`](../current_stageB_effective_configuration.md)。

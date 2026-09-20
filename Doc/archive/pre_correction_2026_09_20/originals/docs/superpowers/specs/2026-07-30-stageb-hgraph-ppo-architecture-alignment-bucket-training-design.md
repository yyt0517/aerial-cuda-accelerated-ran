# Stage-B HGraph+PPO 架构对齐与分桶训练设计

## 1. 背景与目标

当前 Stage-B 智能调度器已经具备 HGraph 状态编码、按小区/PRG/UE 的候选评分、PPO 训练、Type-0 动作解码和 Python/C++ online bridge。严格同轨迹的 seed41 试验表明，v9c PyTorch online 路径能够在降低约 10.98% 平均 INR 的同时将 post-1000 goodput 提高约 0.20%，证明“通过 UE/PRG 调度减少有害同频复用”在当前仿真系统中是可实现的。

然而，seed41--50 的历史结果尚未稳定超过 PFQ，困难 seed 中仍存在频谱利用不足、吞吐下降、尾时延和过期恶化。架构审计进一步发现，当前系统并非严格的端到端 HGraph+PPO：候选网络输出经过 UE coverage repair、二次前向、预算贪心选择、需求上限、tail/rescue/completion 偏置和 canonical Type-0 解码后才成为最终动作；训练时的 greedy PRG log-probability 对策略 logits 基本为常数，最终 PRG 动作缺少有效 PPO policy gradient。

本设计的目标是：

1. 先对齐“状态—策略动作—PPO 信用—实际执行动作”因果链路；
2. 在不引入多专家策略的前提下，以 PFQ 为安全基座训练单一 HGraph residual 策略；
3. 在因果链路验证通过后，使用困难 seed 分桶和自适应采样提升 10-seed 均值及最差 seed；
4. 只有在证据表明状态或联合动作表达能力成为主要瓶颈时，才引入动作条件化的跨小区联合干扰头；
5. 最终使 Python、ONNX 和 C++ 使用同一个 canonical 动作语义，并验证实时执行预算。

## 2. 非目标

本轮设计不直接实现以下能力：

- 不引入多专家、按 seed 路由或运行时切换模型；
- 不直接学习功率、MCS、层数、预编码或 HARQ 控制；
- 不在第一轮同时更改 PRG 粒度、图结构、奖励、动作头和部署链路；
- 不将单一 seed 的优势解释为总体超过 PFQ；
- 不以牺牲 packet completion、expiry 或尾时延为代价追求平均 goodput；
- 不把 Python online bridge 的功能可执行性等同于 ONNX/C++ 的实时部署能力。

## 3. 第一性原理判据

令 \(x_{c,p,u,t}\) 表示 TTI \(t\) 时小区 \(c\) 在 PRG \(p\) 上选择 UE \(u\) 或 blank。有效吞吐近似为：

\[
G_t=\sum_{c,p,u}x_{c,p,u,t}
    B\!\left(\mathrm{SINR}_{c,p,u}(s_t,x_t)\right)
    \left(1-\mathrm{BLER}_{c,p,u,t}\right)
\]

队列随服务和到达演化：

\[
Q_{u,t+1}=\left[Q_{u,t}-D_{u,t}\right]^+ + A_{u,t}
\]

要通过强化学习同时改善干扰、频谱效率、吞吐和时延，必须满足四个条件：

1. **可观测性**：状态能够预测候选动作对 SINR、BLER、服务量和队列的边际影响；
2. **动作表达能力**：动作空间能够表达有价值的复用、错频、静默和 UE 服务策略；
3. **因果信用分配**：环境实际执行的动作必须拥有与其一致、可求导的 log-probability，并从回报获得梯度；
4. **执行一致性**：训练、Python 评估、ONNX 导出和 C++ 执行必须对同一观测产生语义一致的动作。

本设计把这四项作为分桶训练前的架构闸门。任一项未通过时，不以增加 PPO 迭代或提高困难 seed 采样比例代替根因修复。

## 4. 当前架构结论

### 4.1 可观测性：部分满足

当前状态已经包含：

- 小区负载、活跃 UE 数、宽带 SINR、历史速率和 TB error；
- UE buffer、平均速率、CQI/RI、HOL delay、TTL slack、近期调度比例和服务缺口；
- PRG top-1/top-2 质量、历史分配、邻区同频质量、冲突率、ICI proxy、post-EQ SINR、期望 goodput 和低 SINR 紧急业务权重；
- UE 到 PRG 的候选边特征，以及相邻小区相同本地 PRG 之间的显式 PRG—PRG 连边。

因此，HGraph 能够识别“某个本地 PRG 风险较高”“某个 UE 在某个 PRG 上更值得服务”“邻区同频 PRG 当前也很紧张”等上下文。

当前不足是：post-EQ SINR 和 ICI proxy 主要反映动作前或保守全复用条件，网络没有显式观测“邻区选择某个具体 UE/blank 后，本候选 UE 的反事实 SINR”。当前结构能够学习相关性驱动的粗粒度干扰规避，但不能准确预测跨小区候选组合的边际干扰。

### 4.2 动作表达：当前场景可用，但存在上限

策略每 0.5 ms 为每个 `(cell, PRG)` 选择一个 UE 或 blank，可以直接控制：

- 邻区相同 PRG 是否同时复用；
- 同一小区的频域 UE 分配；
- 是否通过部分静默换取更高成功率；
- 紧急、弱覆盖或服务不足 UE 的频域机会。

当前策略不能直接控制功率、MCS、空间层、预编码和 HARQ，这些仍由原生链路流程决定。因而策略只能通过 UE/PRG 分配间接影响链路效率。

17 PRG × 16 PRB 具有较低动作维数和较高训练稳定性，但一次 blank 或错误分配会影响较宽频段，无法利用 68 PRG × 4 PRB 的细粒度频率选择性。第一轮架构对齐继续保持 17 PRG，以避免同时改变学习链路和控制粒度；对齐后再通过受控 A/B 试验判断是否切换到 68 PRG。

当前固定场景为每小区 12 个 UE，等于 `max_candidates=12`，不存在候选截断；扩大 UE 数或小区数时，该固定上限和固定 ONNX shape 将成为扩展瓶颈。

### 4.3 因果信用：当前主要瓶颈

当前 greedy budgeted selector 对匹配动作返回常数 log-probability 0，对不匹配动作返回固定 floor，entropy 为 0。由此产生两个问题：

1. 最终 PRG 选择不能通过 PPO ratio 从 advantage 获得有效策略梯度；
2. PPO 的主要梯度落在 UE slot selector，而 canonical candidate decoder 又可以替换或绕过其选择。

历史 v9c seed41 统计中，预算解码器平均改写约 45.85/51 个 PRG 选择，coverage repair 平均约 5.98 次/TTI。这说明最终表现主要来自“候选打分 + 强规则解码”，而不是 PPO 对实际 PRG 动作的端到端优化。

### 4.4 时延信用：状态可见，学习时域不足

0.5 ms TTI、200 ms TTL 对应约 400 步。当前 `gamma=0.99` 的主要折扣时域约 100 步；结合常用 GAE lambda 后，优势估计主要聚焦于十几步。完成包时延又主要在包完成时进入奖励，尚未完成的老包缺少足够即时惩罚。

因此模型能够学习短期 success/goodput，却难以稳定学习“现在为老包保留资源，避免数百步后过期”的策略。后续应优先引入未完成业务的稠密年龄/TTL 势函数，而不是单纯放大完成包 delay 权重。

### 4.5 执行一致性：Python 通过，ONNX 未通过

Python online evaluator 的 canonical 动作可在同一 TTI 由 C++ Type-0 路径执行，seed41 的 INR 和 goodput 改善可以视为真实闭环机制证据。

当前 action-mode ONNX exporter 未复现 historical evaluator 的 coverage repair、二次前向和 canonical budgeted decoder。同一 checkpoint 在 17 PRG ONNX 路径上的 PRG utilization 降至 73.05%，post-1000 goodput 比 PFQ 低 9.63%。ONNX 相对其 PyTorch wrapper 输出一致，只能证明 wrapper 可导出，不能证明其与训练/历史评估动作等价。

## 5. 目标架构

目标链路为：

```text
同一份观测快照
    -> HGraph 编码
    -> PFQ 基础分数 + HGraph residual + blank/reuse 分数
    -> 训练：受约束随机采样 / 评估：同分布 greedy
    -> 最小可行性投影
    -> Type-0 动作
    -> C++ 同 TTI 执行
    -> goodput / BLER / queue / delay / expiry 回报
    -> 与已执行动作完全对应的 log-probability
```

其中“最小可行性投影”只处理原生硬约束：

- UE 存活、关联关系和 cell 合法性；
- 同一 Type-0 slot 的 UE 唯一性；
- PRG mask 合法性；
- 可证明的无需求或需求上限；
- 固定输出 shape 和协议格式。

tail、rescue、completion、usage 等软偏好应由状态、PFQ anchor、模型 residual 或奖励表达，不再由部署后处理大规模改写网络排序。若可行性投影频繁改变动作，应视为策略/动作空间设计错误并触发闸门失败，而不是静默接受。

## 6. 阶段 A：最小因果对齐

### 6.1 直接候选策略

以 `(cell, PRG)` 的候选 UE/blank 类别作为 PPO 的主要离散动作。UE slot 由最终候选选择确定性构造：

- 收集每个小区最终动作实际使用的 UE；
- 分配唯一合法 slot；
- 未使用 slot 置空；
- slot 数不足时在策略采样阶段施加约束，不由事后启发式替换。

原 UE selector 在第一轮可保留用于兼容或辅助预测，但不再把其 log-probability 计入 PPO 主策略损失。只有在证明其动作会因果地约束最终执行结果后，才重新纳入联合策略。

### 6.2 训练采样与 log-probability

训练使用受约束的 sequential sample：

1. 在当前合法候选和剩余需求 headroom 下形成 masked distribution；
2. 从分布采样 UE/blank；
3. 更新该 UE 的已分配 PRG 数和剩余需求；
4. 保存每一步实际使用的 mask、顺序、动作类和 log-probability；
5. PPO 更新时按完全相同的顺序和约束重放 log-probability。

评估和部署使用同一策略分布的 greedy 版本。不得为 greedy 路径定义人为的 0/floor log-probability并用于 PPO。

跨小区同频 PRG 的第一轮动作仍由共享 HGraph 上下文关联，但不新增自回归联合动作依赖。这样可以先隔离并验证 PPO 因果链路；若后续证据表明联合动作表达不足，再进入阶段 D。

### 6.3 因果对齐测试

分桶训练前必须通过：

- 非零 advantage 样本上，candidate action head 参数存在有限、非零梯度；
- 人工扰动被选候选 logit 时，重算 log-probability 和 PPO ratio 按预期变化；
- sample 模式 entropy 大于 0，并能随 entropy coefficient 变化；
- rollout 记录动作、PPO 重算动作和 Python canonical 执行动作逐 token 相同；
- 非硬约束导致的 decoder 改写率低于 5%，目标值为 0；
- C++ 原生拒绝、重复 UE、非法 cell/PRG 动作计数均为 0。

### 6.4 延迟状态与稠密信用

因果对齐完成后、正式分桶前，引入最小的时延增强：

- 每 UE 的临期 backlog，例如 TTL slack 小于 20/60 ms 的 bytes 或 packet count；
- backlog 的年龄加权压力；
- 距上次有效服务的间隔；
- 未完成业务的 tail debt。

奖励采用基于状态的 potential shaping：

\[
r'_t=r_t+\eta\left(\gamma\Phi(s_{t+1})-\Phi(s_t)\right)
\]

其中 \(\Phi\) 由临期 backlog、年龄加权 backlog 和服务债务构成。这样可以为尚未完成的老包提供逐 TTI 信号，同时避免完全依赖数百步后的 expiry。原 goodput、TB error、packet completion 和 expiry 仍保留为真实目标项。

第一轮不同时引入 RNN、极长 GAE 和大幅 reward 权重调整。只有 potential shaping 仍无法解释尾部失败时，再评估 recurrent critic 或更长时间尺度的 value estimator。

## 7. 阶段 B：PFQ-residual 单策略

最终候选分数设计为：

\[
\ell_{c,p,u}
=\alpha\,\widehat{s}^{PFQ}_{c,p,u}
+\Delta_\theta(s_t,c,p,u)
\]

blank 使用独立 logit，并可由 HGraph 学习邻区同频风险、预计失败风险和服务收益的权衡。

设计约束：

- 第一轮固定 \(\alpha=1\)，PFQ 分数做稳定归一化；
- residual head 近零初始化，使初始策略接近 PFQ；
- residual 不使用硬裁剪阻止长期超越 PFQ，但在 warm-up 阶段使用轻量正则限制突变；
- PFQ anchor 是候选排序先验，不直接复制 PFQ 最终动作；
- 训练和评估仍只有一个共享模型，不按 seed 或 bucket 选择专家。

该结构的目的不是把策略限制为 PFQ，而是把学习问题从“重新发现完整调度规律”改为“学习 PFQ 在跨小区干扰、临期业务和过度占用上的系统性误差”。它与当前候选边中已有的 PFQ anchor 能力一致，且更符合“10-seed 均值超过 PFQ、同时约束最差 seed”的目标。

## 8. 阶段 C：分桶训练

### 8.1 初始分桶

依据现有 v9c/v19 相对 PFQ 的 paired seed 表现，初始分桶为：

| 分桶 | seed | 主要问题 | 训练作用 |
|---|---|---|---|
| G：gain anchor | 42、47 | 已出现明显吞吐、expiry 或尾时延增益 | 保留已学会的有效策略，防止修复困难 seed 时回退 |
| U：underfill | 45、48 | PRG 利用下降、吞吐明显受损 | 学习减少无效 blank 和过度保守 |
| T：tail/coverage | 43 | 尾时延、expiry 或覆盖服务恶化 | 加强临期业务和持续未服务 UE 的信用 |
| R：regular | 41、44、46、49、50 | 接近 PFQ 或轻微波动 | 维持总体分布和防止过拟合困难桶 |

分桶表示失败机制，不作为模型输入，不允许策略识别 seed ID。

### 8.2 采样比例和约束

初始训练采样比例：

- G：25%
- U：30%
- T：25%
- R：20%

每 25 个 PPO iteration 为一个 macrocycle。桶权重根据最近一次 paired 验证的 goodput deficit、packet gate 违反程度和不确定性调整，但必须满足：

- G 不低于 20%；
- U 不低于 20%；
- T 不低于 15%；
- R 不低于 15%；
- 任一桶不高于 40%。

每个 macrocycle 结束后执行 seed41--50 的 uniform paired 评估。最终模型选择只使用 uniform 结果，不能用训练采样权重加权指标。

### 8.3 训练流程

1. 从当前 PFQ-residual checkpoint 开始，而不是从困难 seed 单独冷启动；
2. 每次 rollout 记录 seed、bucket、观测、采样动作、执行动作、decoder 改写和关键 KPI；
3. 训练 batch 在 bucket 内保持随机化，不按 seed 连续长时间训练；
4. 每个 macrocycle 同时报告总体和逐桶 PPO ratio、entropy、value error、candidate gradient、override ratio；
5. 若 G 桶连续两个周期明显回退，提高 G floor 或回滚 checkpoint，而不是只继续增加 U/T 比例；
6. 若 U 桶改善来自 PRG utilization 增加但 BLER/goodput未改善，判定为无效填充；
7. 若 T 桶只改善完成包平均时延但 expiry、pending 或 p95 恶化，判定为 survivor bias，不通过；
8. 达到候选门槛后再进行一次固定 seed41--50 的完整 4000 TTI、相同到达轨迹评估。

为防止策略记忆有限 seed，最终候选还应在未参与 bucket 权重更新的额外 seed 上做方向性验证。该验证是泛化检查，不替代 seed41--50 主验收集。

## 9. 验收与停止条件

### 9.1 架构闸门

正式分桶训练前必须满足：

- candidate PRG policy gradient 非零且数值有限；
- 训练采样动作、记录动作和环境执行动作完全一致；
- 非硬约束 decoder 改写率低于 5%，目标为 0；
- Python canonical 与 ONNX/C++ golden trajectory 的最终 UE/PRG bitmap 完全一致；
- C++ 原生动作拒绝和非法动作计数为 0。

### 9.2 主 KPI

最终 seed41--50 paired 评估至少满足：

- full-run 和 post-1000 的 10-seed mean goodput 均高于 PFQ；
- worst-seed goodput 相对 PFQ 不低于 -1.0%；
- pooled expiry rate 不高于 PFQ；
- pooled completed packet ratio 不低于 PFQ；
- 跨 seed 平均 p95 packet delay 不高于 PFQ；
- 吞吐增益不能仅由更高 PRG utilization 或更高 BLER 换取；
- 同时报告均值、标准差、逐 seed delta 和 paired 置信区间，不只报告胜出 seed 数。

在训练初期，checkpoint 可以使用“当前最差 seed deficit 至少缩小 50%”作为过渡门槛；最终交付仍以 -1.0% 的 worst-seed 硬约束为准。

### 9.3 部署门槛

- ONNX checker 和 ONNX Runtime 数值检查通过；
- Python/ONNX/C++ 对固定 golden observations 生成完全一致的离散动作；
- 对完整轨迹的 PRG utilization、goodput、BLER、delay 和 expiry 差异只允许来自数值仿真噪声，不能来自动作算子差异；
- 在目标 GPU/CPU 配置上测量观测构图、推理、解码和写回总时延，p99 不超过 0.5 ms TTI 预算；建议工程目标不超过 0.4 ms；
- 若实时门槛未通过，可以保留为仿真研究策略，但不得标记为可实时部署。

### 9.4 停止和回滚

出现以下任一情况时停止扩大训练并回到最近通过闸门的 checkpoint：

- candidate gradient 消失或 PPO ratio 退化为常数；
- decoder 改写率再次超过 5%；
- 连续两个 macrocycle 的 uniform mean goodput 和 worst-seed 同时恶化；
- expiry、completion 或 p95 任一硬门禁恶化；
- Python 与 ONNX/C++ 动作语义发生漂移。

## 10. 阶段 D：动作条件化联合干扰头

阶段 D 不是分桶训练的前置条件。仅当阶段 A--C 已通过因果和部署闸门，但 U/T 困难桶仍稳定失败，并且失败可归因于“相同本地状态下，不同邻区联合动作产生显著不同结果”时启用。

优先方案是按每个本地 PRG 对小区动作进行自回归选择：

1. HGraph 先生成所有小区/PRG/UE 候选表示；
2. 对同一 PRG 的多个小区按固定或置换随机顺序采样；
3. 后续小区显式接收前序小区已选择的 UE/blank embedding；
4. PPO 保存该联合动作的逐条件 log-probability；
5. 部署 greedy 使用完全相同的条件顺序。

该方案避免枚举 \(13^C\) 个联合组合，但增加推理串行度。若目标部署时延不允许自回归，再考虑 pairwise interference factor 或 learned marginal-INR head。

阶段 D 的进入证据至少包括：

- 已对齐策略在同一观测邻域内对不同联合动作呈现显著 reward 多峰性；
- 当前 pre-action SINR/ICI proxy 无法解释困难 seed 的失败；
- 增加训练量和 bucket 权重不能继续改善；
- 预计串行推理仍能满足 0.5 ms 预算。

## 11. 受控消融顺序

为保证收益可归因，按以下顺序推进：

1. A0：复现当前 v9c/v19 行为和 10-seed 基线；
2. A1：只修正 candidate PRG PPO 因果链路，保留原图、17 PRG 和主要 reward；
3. A2：加入 PFQ-residual 初始化和 blank/reuse 学习；
4. A3：加入临期 backlog 状态与 potential shaping；
5. C1：启用固定初始分桶；
6. C2：启用带 floor/cap 的自适应桶权重；
7. G1：在相同架构上比较 17 PRG 与 68 PRG；
8. D1：仅在满足进入证据时试验动作条件化联合干扰头。

每一步必须与前一步做相同 seed、相同到达轨迹和相同评估窗口的 paired 对比。禁止同时合并多个未经独立验证的改变。

## 12. 数据与可观测性

训练、验证和部署日志至少记录：

- seed、bucket、TTI、checkpoint 和配置 hash；
- actor 原始 logits、采样动作、执行动作和最终 Type-0 bitmap；
- 每步 candidate mask、采样顺序、log-probability、entropy 和 PPO ratio；
- coverage repair、budget selector、canonical decoder 和 C++ legality 各自的改写计数；
- 每 TTI goodput、BLER、PRG utilization、blank/reuse ratio 和动作后 INR；
- completed、expired、pending、HOL/TTL pressure 和 p50/p95 delay；
- 每桶 candidate-head gradient norm、value error 和 auxiliary loss；
- Python、ONNX 和 C++ 的动作一致性摘要及推理时延。

所有最终结论必须能追溯到逐 seed 原始结果和统一聚合脚本。不得从不同生命周期、不同 PRG 粒度或不同动作算子的运行中拼接主表。

## 13. 风险与缓解

### 13.1 受约束采样方差增大

直接对 PRG 候选采样可能降低早期吞吐。采用 PFQ residual 近零初始化、受约束 mask、entropy 退火和短 rollout smoke gate，避免不受控探索。

### 13.2 PFQ anchor 限制策略上限

PFQ 仅作为可学习 residual 的基础分数，不复制其最终动作；warm-up 后减少 residual 正则，使策略可以反转 PFQ 排序或选择 blank。

### 13.3 稠密时延塑形引入代理偏差

使用 potential difference 而不是孤立的启发式 bonus，并始终以真实 completed/expired/p95 KPI 做最终门禁。

### 13.4 过度针对 seed41--50

bucket 不作为模型输入；保持 R/G 采样 floor；最终增加未参与权重更新的额外 seed 方向性评估。

### 13.5 68 PRG 推理和动作维数过大

先在 17 PRG 完成因果闭环，再独立测试 68 PRG 的收益、稳定性和 p99 推理预算。未证明收益前不切换正式训练粒度。

### 13.6 Python 与部署路径再次分叉

canonical 动作算子必须只有一个语义来源；Python、ONNX wrapper 和 C++ 共享同一组 golden trajectory 测试。任何新启发式或合法性规则都必须同时更新一致性测试。

## 14. 设计决策摘要

- 当前 HGraph 能学习有限的粗粒度干扰规避，但缺少联合动作反事实干扰；
- 当前最主要瓶颈不是模型容量或 seed 采样，而是最终 PRG 动作缺少有效 PPO 信用；
- 第一优先级是让 candidate PRG 动作、log-probability 和环境执行动作完全一致；
- 正式分桶策略采用单一 PFQ-residual HGraph，不采用多专家；
- 时延优化使用临期 backlog 状态和 potential shaping 提供稠密信用；
- 初始保持 17 PRG，因果链路稳定后再独立比较 68 PRG；
- Python、ONNX、C++ 动作逐项一致和 p99 实时预算是最终部署硬门槛；
- 只有对齐后的实验证据指向联合动作信息瓶颈时，才进入自回归跨小区干扰头。

## 15. 证据与代码索引

本设计基于以下本地证据和实现入口：

- 多 seed 演进、逐 seed 指标和当前模型配置：`Doc/stageB_intelligent_algorithm_evolution_training_report.md`
- seed41 动作后干扰、Python/ONNX 同轨迹结果及执行差异：`Doc/stageB_cochannel_baseline_analysis_reference.md`
- 状态特征和静态图元数据：`training/stageb_hgraph/feature_snapshot.py`、`training/stageb_hgraph/types.py`
- PRG—PRG、UE—PRG 等图边：`training/stageb_hgraph/edge_generator.py`
- HGraph 消息传递和 candidate action head：`training/stageb_hgraph/model.py`
- budgeted selector、coverage repair、PPO log-probability 和训练入口：`training/stageb_hgraph/online_ppo.py`
- candidate 到 canonical Type-0 动作的 Python 解码：`training/stageb_hgraph/graph_ops.py`
- online bridge 观测和动作时序：`training/stageb_hgraph/bridge_env.py`
- C++ online feature codec 和 Type-0 执行：`cuMAC/examples/onlineTrainBridge/OnlineFeatureCodec.cpp`、`cuMAC/examples/multiCellSchedulerUeSelection/main.cpp`
- C++ ONNX 动作运行时：`cuMAC/examples/customScheduler/GnnRlPolicyRuntime.cpp`
- 动作后链路干扰计算：`cuMAC/src/tools/multiCellSinrCal.cu`

实施计划必须再次核对这些路径的实际函数、配置和测试入口，不能仅依据本文中的逻辑描述推断修改点。

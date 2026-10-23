# SAO：Single-Rollout Asynchronous Optimization for Agentic RL

> 阅读对象：Zhenyu Hou、Yujiang Li、Jie Tang、Yuxiao Dong（Tsinghua University），*Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*，arXiv:2607.07508v1，2026-07-08。
>
> 阅读日期：2026-10-10。本文统一整理[论文正文与附录](https://arxiv.org/html/2607.07508v1)及其 [miles-values 实现（`main-0729`）](https://github.com/yyht/miles-values/tree/main-0729)，按论文方法、实现与扩展、实验和迁移启发组织。实验数字均为作者报告，未在本项目复现。

## 1. 一句话总结

SAO 将异步 agentic RL 的 GRPO group baseline 换成 **每个 prompt 一条 rollout + 学习式 token-level critic**，再通过 **直接使用 rollout logprobs 的双侧 importance masking（DIS）**控制策略滞后，通过 **更快的 critic 更新、冻结 critic attention、skip-observation GAE 和 value pretraining**提高 advantage 的可靠性。

\[
\boxed{
\text{Single rollout}
+\text{DIS}
+\text{Strong token-level critic}
+\text{Skip-observation GAE}
}
\]

关键不是“把 GRPO 的 group size 设成 1”：没有 sibling rewards 后，必须另行解决 baseline、方差与 credit assignment。

**Single rollout 指每个 prompt 采样一条轨迹，不是每条轨迹单独做一次 optimizer step。** 论文仍以 128 条轨迹组成训练 batch。

## 2. 背景：异步为什么不只是一个吞吐优化

### 2.1 Group-wise sampling 引入等待与 staleness

长推理、多轮工具调用、coding agent 的执行时长存在长尾。同一 prompt 的多个 sibling trajectories 要一起计算 group mean，因此：

1. 最早结束的轨迹仍要等待最慢的 sibling；
2. 等待期间 learner 继续更新，数据进入训练时进一步偏离当前 policy；
3. 固定 128 条轨迹的预算，GRPO 的 `16 prompts × 8 samples` 只覆盖 16 个 prompt，而 single rollout 可覆盖 128 个；
4. 在线环境通常只有一次实际交互反馈，不能自然提供同 prompt 的一组独立反馈。

SAO 去掉的是 **prompt 内的 sibling barrier**。异步 batch 聚合、工具执行与 learner 更新仍有各自的调度成本。

### 2.2 一条 agent trajectory 可能跨多个 rollout policy 版本

异步系统可能在一条轨迹的不同生成阶段更新 rollout weights。其行为分布不一定能用一个历史 checkpoint 表示。

因此，拿“最新 old actor”统一重算整条轨迹的概率，并不等于得到真实 behavior probabilities。SAO 使用生成时记录的 **逐 token rollout logprob**，避免维护整个历史 checkpoint 集合。

## 3. DIS：Direct Double-Sided Importance Sampling

来源：论文 §3.1、式 (1)–(3)。

### 3.1 直接的 importance ratio

记 learner policy 为 \(\pi_\theta\)，生成 token 时的 behavior proxy 为 \(\pi_{\mathrm{rollout}}\)，历史为 \(s_t\)，模型生成 action/token 为 \(a_t\)：

\[
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{rollout}}(a_t\mid s_t)}
=\exp\left(\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{rollout}}(a_t\mid s_t)\right).
\]

对比 decoupled PPO 的两段式校正：

\[
\frac{\pi_\theta}{\pi_{\mathrm{old}}}
\quad\text{与}\quad
\frac{\pi_{\mathrm{old}}}{\pi_{\mathrm{rollout}}}.
\]

SAO 不再需要 \(\pi_{\mathrm{old}}\) 的额外 forward；每个 token 直接对比 learner 与该 token 生成时记录的概率。

### 3.2 双侧 gate，而不是普通 PPO clip

定义：

\[
g(\rho_t)
=\mathbf 1\{1-\epsilon_l<\rho_t<1+\epsilon_h\},
\qquad
f(\rho_t)=g(\rho_t)\rho_t.
\]

论文写出的最大化目标为：

\[
L(\theta)
=\widehat{\mathbb E}_t
\left[f(\rho_t)\hat A_t\log\pi_\theta(a_t\mid s_t)\right].
\]

含义是：**超出允许比率区间的 token，无论 advantage 正负，都不参与 policy-gradient 更新；区间内按 importance ratio 加权。**

普通 PPO 的 clipped surrogate：

\[
\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon_l,1+\epsilon_h)\hat A_t\right)
\]

只在特定方向饱和：正 advantage 对上界、负 advantage 对下界。DIS 则对两个方向都进行准入筛选。它也不是把异常 ratio 截到边界后继续训练，而是将其权重置零。

**实现时的梯度语义：** 若以 `weighted_logprob` 实现目标，要核对 ratio 权重与 advantage 的 stop-gradient。希望实现的加权 score gradient 是

\[
g(\rho_t)\rho_t\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\]

若直接让 `rho * A * logprob` 中的 `rho` 也反传，会额外产生导数项。论文展示的式 (1) 没有显式标注 detach；复现需要核对这个细节，不能仅照字面组合张量。

### 3.3 校正的范围

逐 token ratio 校正的是 **已观测历史下的 action 条件概率**；不是整条 trajectory history occupancy 的精确校正。丢弃区间外 token 也引入 bias。

论文明确接受一定 off-policy bias，以换取更低实现成本及更稳定训练。因此 DIS 应理解为实用的异步稳定化机制，而非无偏恢复完全 on-policy 更新的证明。

另外，“strict double-sided”描述的是两侧都筛选，并不意味着数值区间必然很窄。论文实际设置见 §6。

## 4. 强 critic：single rollout 能否工作取决于 baseline 质量

来源：论文 §3.2。

### 4.1 Critic 更新频率高于 actor

每次 actor update 对应 \(K>1\) 次 critic updates，论文设 \(K=2\)。

动机：critic 落后于快速变化的 actor 时，advantage 会产生较大噪声，并进一步破坏 actor。让 critic 更快适应有助于打断这个反馈环。

需要区分：

- critic 学习率更大；
- 每 batch critic 更新次数更多；
- RL 前单独做 value pretraining；
- RL 初期 actor 不动的 critic-only warmup。

这四项作用不同，不能用“多训几步 critic”一概代替。

### 4.2 Frozen-attention critic

作者观察到全参数 critic 的 gradient norm 过大，主要不稳定来自 Full Attention layers，MoE layers 相对稳定。

于是冻结 critic attention modules，优化 MoE projections，以限制 value learning 对已有语义表征的扰动。这是 **critic 的参数冻结策略**，不是冻结 actor attention。

证据基于 Qwen3-30B-A3B MoE backbone，不能直接推出 RWKV 的 time-mixing / recurrent transition 也应冻结；迁移时应先分解各模块 gradient norm、value fit 和训练稳定性。

### 4.3 Value pretraining

随机 value head 的冷启动是瓶颈。作者强调扩大 value pretraining 数据量，为 single-rollout RL 提供更可靠的初始 baseline。

正文未给出足以完整复现的预训练语料规模、混合策略及全部训练细节。不能把社区仓库的 value-pretrain 配方自动当作论文原始设置。

## 5. Multi-turn credit assignment：Skip-observation token-level GAE

### 5.1 不把 observation 当作模型 action

Agent trajectory 交替包含 action 与环境反馈：

\[
[a_0,o_0,a_1,o_1,\ldots].
\]

环境注入的 observation/tool-result tokens 并非 policy 生成。若 GAE 仍按原始 token 序列逐位置递推，即使 policy loss 最后把 observation mask 掉，其 token 数与 value predictions 仍可能影响相邻 actions 的 advantage。

**Skip-observation 不是只在最终 loss 上乘 mask；它改变 GAE 的递推路径。**

### 5.2 在相邻有效 action tokens 间连接 TD residual

令 \(j(t)\) 为当前位置之后的下一个模型生成 token，跳过 observation span：

\[
\delta_t=r_t+\gamma V(s_{j(t)})-V(s_t),
\qquad
\hat A_t=\delta_t+\gamma\lambda\hat A_{j(t)}.
\]

动作末尾连接下一动作首 token，正是论文式 (4)、(5)。压缩掉 observation positions 后执行 GAE，再把结果 scatter 回原位置，是等价实现思路。

Observation **仍保留在模型输入上下文中**，影响下一个 action state；只是它们不作为可优化 action，也不消耗额外 GAE discount steps。此设计不意味着环境反馈从 value estimation 中完全消失。

### 5.3 Length-adaptive GAE

论文采用 VAPO 的 length-adaptive 配方：

\[
\lambda_{\mathrm{policy}}=1-\frac{1}{\alpha l},
\qquad \alpha=1.5.
\]

当 \(\gamma=1\)、\(l\) 较大时：

\[
\lambda^l\approx e^{-1/\alpha}.
\]

于是早期 token 接收到的末端 reward 权重不再因为轨迹长度差异呈同样严重的指数衰减。这个推导描述 **递推中的 credit horizon**，不等于保证不同长度轨迹的总 gradient norm 相同。

Critic 用 \(\lambda_{\mathrm{critic}}=1\)；actor 和 critic targets 可以采用不同的 bias–variance trade-off。`miles-values` 的自适应长度明确使用 `loss_mask=1` token 数，见 §9.5 的实现分析。

### 5.4 Step-level GAE 不一定更好

附录 A.1 试了两种 conversation-turn value：

- 对该 turn 全部 token values 取平均；
- 只取该 turn 最后一个 token，并仅在最后 token 训练 critic。

每 turn 一个 advantage，再广播给该 turn 全部 tokens。相同 400 training steps 的结果：

| Action/value 粒度 | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| Step-level Average | 85.8 | 60.5 |
| Step-level Last-Token | 87.3 | 62.8 |
| Token-level | **89.8** | **66.8** |

作者认为 token-level 信号更能捕捉复杂推理中的逻辑转折。该消融支持此实验设置下的选择，不是所有 agent task 的普遍定律。

## 6. 实验设置与结果

### 6.1 主要配置（论文 §4.1）

| 项目 | Math TIR | Coding agent |
| --- | --- | --- |
| Backbone | Qwen3-30B-A3B-Thinking-2507 | 同左 |
| 初始化 | GPT-OSS-120B 生成的 TIR data，SFT 3 epochs；初始化 actor/critic | 直接以 backbone 开始 RL |
| Batch / group size | 128 / 1 | 几乎相同配置 |
| Max length | 128k tokens | 128k-token context budget |
| Actor LR | `1e-6` | 同左 |
| Critic LR | `5e-6` | 同左 |
| DIS 参数 | `ε_low=0.3, ε_high=5.0` → ratio 区间 **(0.7, 6.0)** | `ε_low=0.8, ε_high=3.0` → **(0.2, 4.0)** |
| Critic updates | 2 / batch | 同左 |
| Critic λ / warmup | 1 / 10 steps | 基本沿用 |
| Actor adaptive λ | `1 − 1/(1.5 × l)` | 基本沿用 |
| Scaffold / turns | Python tools；eval 最多 50 turns | OpenHands；最多 300 turns |

GRPO 对照使用 `16 prompts × 8 rollouts = 128`，与 SAO 的 trajectory batch size 相同，但 **独立 prompt 覆盖数不同**。

评价采用 Pass@1、`temperature=1.0, top_p=1.0`、128k generation budget。AIME2025/HMMT/IMOAnswerBench 平均 16 次 evaluation runs，BeyondAIME 平均 4 次；不是 Pass@16 或 Pass@4。

### 6.2 主结果（Accuracy %，论文 Table 1、2）

Math 部分比较均为 Python-tool setting：

| 方法 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
| --- | ---: | ---: | ---: | ---: |
| TIR SFT | 80.4 | 53.3 | 75.2 | 53.3 |
| GRPO | 84.2 | 54.8 | 76.0 | 55.8 |
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 |
| SAO（论文标作 `w/ DIS only`） | 94.2 | 71.5 | 86.7 | 71.3 |
| Full SAO | **97.3** | **74.8** | **88.3** | **74.0** |

`SAO (w/ DIS only)` 沿用论文行名；Table 1 未逐项列出该变体所有 value-training 配置，不据此猜测完整配方。

| 方法 | SWE-Bench Verified |
| --- | ---: |
| Qwen3-30B-A3B baseline | 23.0 |
| GRPO + DIS | 27.0 |
| SAO | **29.8** |

相对 GRPO + DIS，Full SAO 分别提高 **3.8 / 4.0 / 4.3 / 4.0 / 2.8 个百分点**（四项数学 + SWE）。

训练动态：vanilla GRPO 约 160 steps collapse，vanilla VAPO 约 90 steps collapse；其表格成绩是 collapse 前最后有效结果。DIS 能让 GRPO 稳定，SAO 相对 GRPO + DIS 的差距在约 400 steps 后更明显。论文报告 SAO 可稳定训练约 1,000 steps。

### 6.3 Critic 消融（论文 Table 3、4）

| 方法 | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| SAO | **97.3** | **74.8** |
| Critic 每 batch 仅更新 1 次 | 95.0 | 69.8 |
| Critic 全参数训练、不冻结 attention | 90.6 | 74.5 |
| Vanilla VAPO，无 DIS | 91.3 | 69.0 |
| Running-mean baseline | 79.8 | 55.3 |

数学消融的 running mean 是 **每 prompt 最近 8 个 rewards** 的滑窗，不同于下面在线实验的全局 128-reward window。

Faster critic update 在约 400 steps 后有更高 explained variance；冻结 attention 降低 critic gradient norm。EV 定义为：

\[
\mathrm{EV}=1-\frac{\mathrm{Var}(R-V)}{\mathrm{Var}(R)}.
\]

EV 衡量 residual variance，不衡量常数偏移的校准误差；需要同时看 value loss、return/value 均值以及 downstream reward。

### 6.4 在线模拟

论文 §4.5 使用写作风格偏好变化：Cute → Chuunibyou → Classical。GLM-4.7 judge 分别给 quality/style 二值分数，最终：

\[
r=r_{\mathrm{quality}}\,r_{\mathrm{style}}\in\{0,1\}.
\]

Single-rollout SAO 对偏好切换恢复更快；对照使用最近 **128 个 rewards** 的 running-mean baseline，历史窗口惯性导致适应滞后。

这是受控的非平稳在线模拟；不是开放用户流量上的在线训练实证，也不是证明所有 learned critics 都比所有无参数 baselines 适应更快。

## 7. GLM-5.2 与证据边界

- 论文摘要明确称 SAO 已用于训练开放 GLM-5.2（**750B total / A40B active**）的 agentic RL pipeline。
- 本文详细的 benchmark、消融及稳定性图主要在 **Qwen3-30B-A3B** 上。
- 正文没有给出 GLM-5.2 的同等细致 SAO-only ablation、资源配置与训练配方。因此应将“用于 GLM-5.2 pipeline”和“公开实证来自 Qwen3 backbone”分别记录。
- 论文附录 B 明确限制外推到更小模型、非 agentic RLHF、dense reward 或短 rollout。
- 主要报告的是效果与稳定性，没有可直接引用的 GPU-hour、吞吐倍率、端到端成本对照；额外 critic 成本仍需评估。
- SAO 的 skip-observation、critic freezing、value pretraining 构成组合配方；现有消融不足以将所有收益唯一归因于 single rollout。

## 8. 与 FlashREINFORCE 的关系

结合本目录 [FlashREINFORCE 笔记](FlashREINFORCE.md)：两者都去掉 sibling barrier，但对 variance 与 stale data 的处理不同。

| 维度 | SAO | FlashREINFORCE |
| --- | --- | --- |
| 每 prompt rollout 数 | 1 | 1 |
| Baseline | Learned state/token value | Independent-prompt batch reward mean |
| Advantage | Token-level GAE | 同 trajectory 的 centered reward |
| Credit assignment | Critic + token-level TD/GAE | 无 learned token-level critic |
| Off-policy 更新 | Token IS + 双侧 token gate | Token IS + sequence trust-region gate |
| 数据复用约束 | Critic 多更新；不能据此推成 actor 必须单次 update | One-Batch REINFORCE，每 batch 单次 update 后丢弃 |
| 系统代价 | Critic 参数/optimizer/compute/pretraining | 无 critic，gate 与统计成本 |

选择上，SAO 更值得在 **任务需要细粒度 credit assignment、critic 有可靠预训练且能持续跟上 actor** 时评估；FlashREINFORCE 更适合先验证低成本 single-rollout 异步基线。不同论文的模型、数据和预算不同，不能直接横向比较表格分数。

## 9. SAO 的 miles-values 实现与扩展

实现仓库为 `yyht/miles-values` 的 `main-0729` 分支。本次固定到 commit **`51b84b7f14436e056ce64b665adeb5957c5433ca`**（2026-08-17）；下文源码链接均指向此快照。已阅读 README、共卡设计文档、训练入口、critic/actor 生命周期、advantage/value/policy loss 及相关测试，未执行该仓库训练或测试。

### 9.1. 定位与核心结论

`miles-values` 是 miles 的 fork；miles 又源自 slime。训练依托 **Megatron-LM**，rollout 依托 **SGLang**，分布式编排使用 **Ray**。

它围绕“让有 critic 的 PPO 在长轨迹 agentic RL 中重新可用”组合了四项工作：

| 组件 | 解决什么 | 与 SAO 论文的关系 |
| --- | --- | --- |
| Single-rollout actor–critic | 不等 sibling group，以 state-dependent value 构造 advantage | SAO 的核心方向 |
| Colocate-PPO | Actor/critic 共用 GPU、时分训练 | 仓库系统扩展，不是论文核心贡献 |
| Regression-as-classification value loss | 用 two-hot / HL-Gauss soft labels 训练 critic | 仓库算法扩展；论文使用 MSE |
| Offline value pretrain | 缓解随机 value head 冷启动 | 论文强调重要性，仓库给出更多实践讨论 |

**最值得借鉴的是边界契约：** critic 对外仍输出 scalar values；分类 head 被封装在 value-logit/value-loss 内部；共卡只改变执行时序与传输路径；observation mask 则真正改变 GAE 的递推位置。

“零额外 GPU”只指不为 critic 再分配独立 GPU 组。并非零 critic 参数、零 CPU RAM、零数据搬运或零训练 FLOPs。

### 9.2. 代码导航

| 入口 / 符号 | 职责 |
| --- | --- |
| [`train.py`][sync] | 同步 rollout → critic → actor → weight sync；共卡以 Ray task dependency 排序 |
| [`train_async.py`][async] | Rollout N+1 与 train N 重叠；共卡时显式 await critic，再 await actor |
| [`placement_group.py`][placement] | Critic 复用 actor GPU placement；共卡模式不建立 actor↔critic NCCL group |
| [`MegatronTrainRayActor`][actor] | 初始化、backup/restore、sleep/wake、critic/actor train、save、weight update |
| [`model_provider.py`][provider] | Critic 输出层由 scalar head 改为 `num_bins` logits |
| [`logit_processors.py`][logits] | `predict_values_from_logits`；将分类分布期望转成 scalar value |
| [`advantages.py`][advantages] | 分发 PPO / GRPO / adaptive GAE；actor/critic λ 可不同 |
| [`gae_adaptive.py`][gae] | Mask-aware scan、有效长度 adaptive λ、长序列 chunked scan |
| [`losses.py`][losses] | Policy surrogate、MSE/分类 critic loss、mask-aware EV 充分统计量 |
| [`corrections.py`][corrections] | TIS 与 IcePop 双侧 token ratio gating |
| [`math_utils.py`][math] | PPO clipped surrogate、标准 masked GAE、chunked GAE 等 |
| [`arguments.py`][args] | 组合约束、默认值、async behavior-policy correction 检查 |
| [`scripts/colocate_ppo/README.md`][colocate] | 共卡生命周期、拓扑、配置及作者验证记录 |

### 9.3. Colocate-PPO：时分复用，而不是共享同一模型

#### 9.3.1 GPU placement 与执行顺序

`--colocate-critic` 要求 `ppo` 或 `gae_adaptive` advantage estimator；critic GPU 数量/节点数被设为与 actor 相同，并要求 `offload_train`。

[Placement][placement] 将 critic 放到 actor 相同的 placement group、相同 reordered bundles/GPU IDs。

Actor 与 critic 仍是各自独立的模型和 optimizer，依次使用同一组 GPU：

```text
critic wake_up
  → critic forward values / build returns / train
  → CPU weight backup
  → critic sleep
actor wake_up
  → receive critic values / build advantages / train
  → CPU weight backup
  → actor sleep
weight publication
```

#### 9.3.2 为什么必须 backup/restore

[Actor 源码][actor] 的关键路径：

- Critic 初始化后创建 `weights_backuper`、设置 active tag `critic`，并立即 backup；
- 共卡模式的 critic train 结束再次 backup；actor train 内部 backup tag `actor`；
- `sleep()` 调用 `torch_memory_saver.pause()`，让出 GPU memory；
- `wake_up()` 在 `resume()` 后，**无条件恢复该角色的 CPU weight backup**，再恢复 process groups。

仓库记录：某些 TMS `LD_PRELOAD` hook 场景下，resume 得到的 GPU memory 可能被清零。若没有 restore，value head 可在无 crash 的情况下丢失，表现为 EV 约 0、critic loss 接近零。

这是仓库观察到的生命周期问题，不能把“所有 offload 系统都会清零”当成普遍规则。该设计的可迁移点是 **验证 offload/onload 前后参数、buffers、optimizer state 和通信状态确实保持一致**。

#### 9.3.3 Value transfer：CPU tensors + Ray object store

共卡不能要求 actor 与 critic 同时 wake 后做 NCCL collective，因此：

1. Critic 在 PP last stage 返回 `{"values": tensors_to_cpu(...)}`；
2. Actor 通过 `external_data` 收到对应 worker 的 CPU values；
3. 将 values 搬回 GPU，写入 `rollout_data["values"]`；
4. 跳过 separated-critic 的 `sync_actor_critic_data` NCCL 路径。

同步入口用 `critic_model.submit_train(...)` 的 ObjectRefs 作为 actor task 参数，依靠 Ray dependency 保证 **critic 完整结束并 sleep 后 actor 才开始 wake**。异步入口直接 `await critic.train()` 后再 `await actor.train(external_data=...)`。

该传递依赖 actor/critic 相同的数据并行布局与 worker pairing，不能只看 tensors shape 相同就推定任意异构布局可用。

#### 9.3.4 源码细节：传给 actor 的 values 是本轮 critic 更新前的预测

[`train_critic()`][actor] 的实际顺序（L466–525）：

```text
forward_only(get_values) → rollout_data["values"]
→ compute critic returns → detach targets
→ repeat train for num_critic_epochs
→ return rollout_data["values"] as CPU tensors
```

此路径 **没有在 critic epochs 结束后重新 forward values**。所以共卡 actor 收到的是本轮 critic 更新前的 values；更多 critic updates 会改进后续轮次的 critic，而不是自动让本轮 actor 使用更新后的预测。

这一区别在解释“critic 先训、actor 后训”时很重要：**执行先后不等于本轮 advantage 使用了刚更新的 critic**。

#### 9.3.5 Weight publication 的两种路径

| 模式 | Rollout 布局 | Weight sync |
| --- | --- | --- |
| `train.py --colocate --colocate-critic` | Actor/critic/rollout 全部时分共卡 | `UpdateWeightFromTensor`，使用 CPU backup，经 tensor/IPC 路径发布；actor 可保持 asleep |
| `train_async.py --colocate-critic` | Actor/critic 共卡，rollout 独占另一组 GPU | `UpdateWeightFromDistributed`，需要 live GPU parameters |

异步共卡的后者执行：

```text
sleep 前 disconnect rollout NCCL
→ wake_up / restore
→ reconnect
→ broadcast latest actor weights
→ sleep
```

TMS pause 后 communicator 的 GPU-side buffers 可能无效，不能把“参数已恢复”当作“跨 rollout 的通信状态也自动有效”。

#### 9.3.6 Critic-only warmup 与离线预训练

`--num-critic-only-steps N` 在前 N 个 rollout batches 只更新 critic，actor 不动；它是当前 RL 流程内的 warmup，不等于另一个 offline value-pretrain job。

共卡 async warmup 跳过 actor train/save 和无意义的 weight-version bump。Sync 全共卡且 offload rollout 时，仍走完整 `onload_weights → update_weights → onload_kv`，这是该实现维持 CUDA graph 生命周期的要求。

### 9.4. Regression-as-classification：改变 critic loss，不改变 advantage 接口

#### 9.4.1 Scalar value 来自 categorical expectation

支持点为 \(z_1,\ldots,z_K\)，等距覆盖 \([L,H]\)。Head 输出 logits \(\ell_k\)：

\[
p_k=\operatorname{softmax}(\ell)_k,
\qquad V(s)=\sum_{k=1}^Kp_kz_k.
\]

[`_value_head_output_size()`][provider] 决定 MSE 为 1、classification 为 `num_bins`。[`predict_values_from_logits()`][logits] 将分布转为 scalar；GAE 接口仍只依赖 \(V(s)\)。

MSE 与 classification 的输出层 shape 不同，critic checkpoints 不能直接互换。

#### 9.4.2 Two-hot targets

对标量 return target \(G\)，先 clamp 到 support range，找到邻近支持点 \(z_k\le G\le z_{k+1}\)：

\[
q_k=\frac{z_{k+1}-G}{z_{k+1}-z_k},
\qquad
q_{k+1}=\frac{G-z_k}{z_{k+1}-z_k}.
\]

其余概率为零；期望等于 **clamp 后的** target。Bin 对齐时退化为 one-hot。

#### 9.4.3 HL-Gauss targets

令 bin width 为 \(\Delta z\)，\(\sigma=0.75\Delta z\)。在以支持点为中心的 bin 上积分 Gaussian CDF：

\[
\tilde q_k
=\Phi\left(\frac{z_k+\Delta z/2-G}{\sigma}\right)
-\Phi\left(\frac{z_k-\Delta z/2-G}{\sigma}\right),
\qquad
q_k=\frac{\tilde q_k}{\sum_j\tilde q_j}.
\]

[`_hl_gauss_target_distribution()`][losses] 将 target clamp 到 support 范围，再对边界丢失的 Gaussian mass 归一化。相较 two-hot，它对相邻 bins 提供更平滑的 supervision；支持区间边缘的截断/重归一化使期望不保证精确等于原 target。

Loss 为：

\[
\mathcal L_V=-\sum_kq_k\log p_k.
\]

对 logits 的梯度为 \(p_k-q_k\)，各分量有界；这 **不等于整个网络参数的 gradient norm 有界**。Gaussian soft labels 是对 scalar regression target 的平滑投影，不能仅凭分类 head 就断言它学到了校准良好的真实 return uncertainty。

#### 9.4.4 配置与默认值

[共卡文档][colocate]推荐配置片段：

```text
--value-loss-type classification
--value-num-bins 51
--value-reward-range 0.0 1.0
--value-target-type hl_gauss
--hl-gauss-sigma-ratio 0.75
```

实际 defaults 为 `value_loss_type=mse`、`value_num_bins=2`、`value_target_type=two_hot`，不是 HL-Gauss 自动开启。

MSE 分支使用 PPO-style clipped value loss；classification 分支使用 projected cross-entropy，不套同样的 scalar value clipping。

支持区间应覆盖实际 return targets。若加入 KL reward、dense reward 或改变 reward scale，不能未经检查沿用 `[0,1]`；范围外 targets 会被投影截断。

分类模式另外记录 `value_accuracy / value_entropy / value_confidence`。这些是 bins/distribution 的辅助指标，不是 downstream task success rate。

### 9.5. GAE：Mask 改变递推，而不只是最终 loss

#### 9.5.1 Mask-aware scan

[`gae_adaptive.py`][gae] 的 `_gae_loop()` 从后往前递推：

- `mask=1`：计算 TD residual，更新 next value 和 GAE carry；
- `mask=0`：advantage/return 保持 0，不改变 carry，也不折扣。

`_gae_chunked()` 则压缩有效 positions，以 float64 执行 chunked parallel scan，再 scatter 回原序列。Default chunked threshold 为 **8192 个有效 tokens**，chunk size 为 128。

Outcome reward 加到 **最后一个 `loss_mask=1` token**，而非无条件加到最后 raw response token。跨 CP 时先 gather 全序列 values/KL，再按对应 layout slice 回 rank。

Observation 在模型输入中保留；它们只从可优化 action 序列中跳过。

#### 9.5.2 Adaptive λ 并非选了 estimator 就自动开启

`--advantage-estimator gae_adaptive` 支持：

- `mode=fixed`：使用 CLI `lambd`，仍执行 observation skipping；
- `mode=adaptive`：\(\lambda=1-1/(\alpha l)\)，\(l\) 是有效 token 数。

源码默认是 **fixed mode**。文档中的 adaptive 配置来自 `--custom-config-path` YAML：

```yaml
gae_adaptive:
  mode: adaptive
  alpha: 1.5
  min_length: 50
  chunked_threshold: 8192
  chunk_size: 128
```

低于 `min_length` 的短序列用 λ=1；实现也将可能为负的 λ floor 到 0。

Actor 的 adaptive λ 与 critic target 的 λ 分开。[`advantages.py`][advantages] 中，传入 `critic_lambd` override 会 **强制 fixed mode**，使例如 `critic_lambd=1.0` 真正用于 critic returns。

#### 9.5.3 已有源码测试覆盖什么

[`test_ppo_gae_masks.py`][mask-test] 对标准 PPO masked GAE 验证：

- 跨 observation gap 等价于压缩后的有效 token 参考递推；
- gap 长度不会多消耗 discount；
- terminal reward 放在最后有效 token；
- masked positions 的 advantage/return 为零；
- all-ones mask 与 unmasked GAE 一致；
- truncated rollout 在该实现中也使用 zero terminal bootstrap。

这些是本次阅读到的测试断言，不代表本次运行通过。Truncation-as-terminal 的 zero bootstrap 是实现语义；若迁移到要求 timeout bootstrap 的环境，需要重新确定 target 定义。

### 9.6. DIS 与仓库 policy loss：不能只按名称对齐

#### 9.6.1 三类机制

| 机制 | 源码行为 |
| --- | --- |
| `compute_policy_loss()` | 标准 PPO clipped surrogate，可选 dual-clip PPO |
| `vanilla_tis_function()` | 用旧训练 logprobs / rollout logprobs 的 ratio，经 clamp 后乘到 policy loss |
| `icepop_function()` | 同一 correction ratio，区间外权重 0，区间内保留 ratio |

IcePop 在 [`corrections.py`][corrections] 的形式为：

\[
w_t=
\mathbf 1\{c_{low}\le\exp(\log\pi_{old,train}-\log\pi_{rollout})\le c_{high}\}
\exp(\log\pi_{old,train}-\log\pi_{rollout}).
\]

注意两个细节：

1. [`policy_loss_function()`][losses] 先算 PPO surrogate，再进入 TIS/correction hook；该 hook 的 `train_log_probs` 参数来自存储的 `batch["log_probs"]`，不是本次有梯度 forward 的 current logprobs。
2. IcePop 返回的 `loss_masks` 不变，只将异常 token 的 loss 权重置零；因此该路径默认保留原 valid-token denominator，并非对 surviving tokens 重新取平均。

#### 9.6.2 Direct rollout denominator 是另一开关

`--use-rollout-logprobs` 让 PPO ratio denominator 直接选 `batch["rollout_log_probs"]`，并可跳过 old-policy scoring forward。

但这依旧调用标准 `compute_policy_loss()`。**Direct denominator ≠ SAO 的 strict double-sided current/rollout gate**；普通 PPO ratio clipping 与“区间外无论 advantage 符号都置零”不等价。

另外 [`arguments.py`][args] 明确禁止同时开启 `use_rollout_logprobs` 与 `use_tis`。所以不能将两个 flag 直接拼在一起就声称实现了论文 DIS。仓库具备相关 building blocks；复现 SAO 时需要逐项核对 denominator、gate 使用的 ratio、梯度语义、reduction 与 auxiliary losses。

`dual-clip PPO` 与 `double-sided DIS` 名称相近，但数学机制也不同。

#### 9.6.3 Async correctness 的显式检查

`validate_async_off_policy_correction()` 要求有 critic 的 async training 明确选择至少一种：

```text
use_rollout_logprobs / use_tis / keep_old_actor
```

原因：async 下一 batch 可能在当前 weight publication 前已生成；默认拿当前 actor 重算 denominator 会错误地把该 rollout 当作由当前策略生成。`weight_versions` 的记录只是度量，不会自动执行 off-policy correction。

### 9.7. Async 有两层：入口重叠与持续 rollout worker

[`train_async.py`][async] 默认是 **one-batch-ahead overlap**：

```text
await rollout N
launch rollout N+1
train N
await pending generation before weight publication
publish actor weights
```

源码同步 pending generation 后才更新 weights，避免在默认 generate 调用中途替换权重。因此不能仅因文件名带 async 就推断它复现了论文讨论的“单轨迹跨多个 policy versions”。

仓库另外有 [`examples/fully_async/README.md`][fully-async] 所描述的持续 producer：background worker 不断调用 `generate_and_rm_group`，完成 groups 入队，trainer drain 足够样本。

是否仍有 sibling barrier 取决于 **group 的 sample count**；是否具备持续异步，还取决于 rollout function 和 weight-update 生命周期。`n_samples_per_prompt=1` 才消除了 prompt 内等待整组的结构。

### 9.8. 示例配置与完整 SAO 配方的差异

[`run_qwen35_4b_ppo_colocate_critic_async_8gpu.sh`][script] 是共卡 PPO 拓扑示例，源码配置：

| 参数 | 示例值 |
| --- | --- |
| Model | Qwen3.5-4B-Base |
| GPUs | 4 colocated actor/critic + 4 rollout |
| Parallelism | TP=2、CP=1、PP=1，训练 2 DP |
| Prompt batch × samples | **16 × 8** |
| Global batch | 显式设置 128 |
| Advantage estimator | `ppo` |
| Actor / critic LR | `1e-6 / 5e-6` |
| Critic-only warmup | 10 rollout batches |
| Critic epochs | 默认 **1** |
| λ | 固定 0.95 |
| Value loss | 未另设 classification，走默认 MSE |
| PPO eps | 0.2 / 0.28 |
| Max response | 默认 4096 |

它主要说明 **共卡与 NCCL weight sync 如何组织**，不是“一键完整 SAO”。Single rollout、adaptive mode、HL-Gauss、critic K=2 以及 off-policy 选择需要在具体实验配置中明确指定。README Quick Start 同样是配置片段，不能代替完整可复现实验清单。

也要区分 `rollout batch`、`global batch`、`critic epochs`、`optimizer steps`：只有核对 microbatch/DP accumulation 后，才能判断一次 rollout 实际执行了多少 optimizer updates。

### 9.9. Value pretrain、EV 与 credit assignment

#### 9.9.1 README 的离线预训练经验

[README][readme] 提出：

- Value pretrain 应作为独立训练 job；不能仅靠 `num_critic_only_steps` 代替；
- 从多种 policies 混合的 trajectories 预训练可能更可靠；
- 低预算下，critic pretrain 使用 RL training prompts 的 offline rollouts，可减小分布错位；
- 用固定 prefix 采样多个 suffix、估计 continuation return，构造 `(prefix, return)` supervision；
- 小规模单 policy pretrain 主要改善早期/sample efficiency，未必改变长期最终指标。

这是 README 的研究讨论与实验总结。本次阅读的主入口展示的是 online actor–critic 和 checkpoint 接入（`--critic-load`）；没有据此推定仓库提供了全部 offline 数据构造与预训练 job 的完整复现包。

“Pretrain data 与 RL data 重叠”在这里指 **训练数据分布匹配**，不是建议与 held-out evaluation data 重叠。

#### 9.9.2 Explained variance 的正确聚合

[`value_loss_function()`][losses] 仅对 `loss_mask=1` tokens 输出以下充分统计量：

```text
n
sum(returns), sum(returns²)
sum(returns − values), sum((returns − values)²)
```

聚合后再重建全局 variance / EV，而非平均各 microbatch/rank 的局部 EV。这样避免 tool observations 的零 returns 与非零 predicted values 污染统计，也避免不同 batch 均值造成错误的 variance 聚合。

EV>0 表示 residual variance 小于 return variance；EV<0 表示更差。EV 对常数预测偏移不敏感，也不保证真实 on-policy value calibration；目标来自旧 rollout 时，更不应只凭一个 EV 数字判断 critic 完全可靠。

#### 9.9.3 EVPO 与 PivotRL 是相关路线，不是 SAO 必选项

README 还讨论：

- **EVPO**：当 critic 不可靠时，以 per-group EV 在 critic baseline 与 group mean 之间切换。这种 group fallback 需要多样本组，不能直接套到严格 single-rollout 在线反馈。
- **PivotRL / prefix replay**：选一个有价值的切点，将 prefix 固定为 prompt，只 rollout 和优化 suffix，缩短 credit-assignment 距离。Replay prefix 不接受 policy loss。

这些是仓库 README 引用的相关工作，不应归到 SAO 论文的组成部分，也不能仅凭 README 就声称均已在该分支实现。

### 9.10. 仓库实验报告：怎么解读

以下均来自 README/共卡文档，未由本项目复现。

#### 9.10.1 小模型单轮数学

Qwen3.5-4B-Base，DeepMath-103k；README 配置为 `bs=16, n_rollout=8, actor_lr=1e-6, critic_lr=5e-6, warmup=20, critic_update_steps=1, max_length=16k`。

报告的方向：HL-Gauss 优于 MSE；value pretrain 改善初始 EV；pretrain 与 RL data 匹配更好；PPO + HL-Gauss + pretrain 与 GRPO 大体接近、略有优势。

**这些实验仍有每 prompt 8 条 rollout，不能当作 group=1 SAO 的小模型实证。**

#### 9.10.2 更大模型与 agentic

README 的 Qwen3.5-35A3-Base + DAPO-17k、16k 实验：MSE、HL-Gauss、HL-Gauss-pretrain 训练足够久后都达到接近最终结果（报告 AIME24≈0.91、AIME25≈0.85）；pretrain 组以历史 rollouts 预训练 500 steps，sample efficiency 和 EV 更好。

128k search-agentic 实验使用 single-rollout PPO + 约 **60–70k samples** 的 value pretrain，报告部分 offline benchmarks 提升 **8–10 points**。Agentic benchmark/data 名称未公开，不能从图与概述恢复严格可比较的任务定义、控制变量或成本收益。

#### 9.10.3 三种 GPU 拓扑

[共卡文档][colocate]比较：`2 actor + 2 critic + 4 rollout`、`4 colocated actor/critic + 4 rollout`、`8 全共卡 sync`。都是 8 GPU 总预算；报告 value loss/EV/gradient/TIM 曲线接近，并完成两种权重同步路径验证。

但 eval 结果并非严格相等（例如 AIME25 separated≈0.68、colocated≈0.60–0.63）。而且 actor GPU 数不同，拓扑与并行度一起变化。该记录可支持 **功能正确性与大体相近的训练行为**，不能单凭曲线称为数学等价或无额外 wall-clock 开销；文档没有完整 throughput/GPU-hour 对照。

### 9.11 离线 rollout → value model 的现成训练代码

补充检索日期：2026-10-10。以下已核对源码入口、数据与 loss，未执行训练。

#### 9.11.1 首选参考：ReDiPPO 的 standard critic pretraining

[ReDiPPO 的 critic_pretraining 文档](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/README.md)提供独立离线训练流程，不要求启动 PPO rollout workers。固定快照为 `ecb67b2119b966616a6b8ffbad3db830c4c423fd`，仓库许可证为 Apache-2.0。

主要文件：

- [`scripts/build_standard_critic_data.py`](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/scripts/build_standard_critic_data.py)：将已生成、已评分的推理记录转换为 critic training JSONL。
- [`runner/train.py`](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/runner/train.py)：读取 JSONL，构建 dataset/collator，调用 Hugging Face Trainer 并保存 checkpoint。
- [`runner/pretrainer.py`](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/runner/pretrainer.py)：token-level masked value loss。
- [`runner/response_mask.py`](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/runner/response_mask.py)：通过 fast-tokenizer offsets 构造 assistant response mask。
- [`train.sh`](https://github.com/cii030/ReDiPPO/blob/ecb67b2119b966616a6b8ffbad3db830c4c423fd/critic_pretraining/train.sh)：torchrun 入口，支持可选 DeepSpeed。

原始 rollout record 可以包含：

```json
{"question":"...","model_output":"...","final_correct":true}
```

转换后：

```json
{"prompt":"完整渲染的 question + assistant response","response":"assistant text","label":1.0}
```

`RewardDataCollator` 将 scalar label 展开到每个 token；`PreTrainer.compute_loss()` 使用 sigmoid(value logits) 与 label 的 masked MSE。最关键的对齐为：

```text
predictions[:, :-1]
labels[:, 1:]
response_mask[:, 1:]
```

即用 token t−1 的 hidden-state value 监督 action token t，第一个 response action 使用最后 prompt token 的 value。它不是只在最后 token 训练的 outcome reward model，也不是 pairwise reward-model loss。

训练脚本 defaults：2 epochs、LR `5e-6`、BF16、max length 8192、per-device batch 1、gradient accumulation 4、8 GPUs。该默认 loss 是 sigmoid scalar regression，不是 HL-Gauss。

对于我们的 SAO，只借鉴 **standard critic** 路线即可；无需同时实现论文的 reference-guided critic。现成 loader 假设 Transformer/FlashAttention，并且 mask 主要定位一个 assistant response span；RWKV 原生 BF16/recurrent/padding 与多轮 tool-observation action mask 要接回本项目契约。可选 `filter_balanced` 会改变采样分布，不应未经评估直接用于概率校准。

#### 9.11.2 更接近 SAO 流程：Fireworks cookbook

[`training/recipes/experiment/ppo_value_head_loop.py`](https://github.com/fw-ai/cookbook/blob/a55b26e7c4f3da481a3c9305dd24f42450e5dee2/training/recipes/experiment/ppo_value_head_loop.py)包含真正的 offline value-pretrain 阶段：

- `ValuePretrainData(train, validation)`：保存固定 trajectories，使用 `PromptGroup` 表示。
- `critic_batch_from_groups()`：整理 token data、terminal rewards、action masks。
- `run_value_pretraining()`：仅更新 critic，按 held-out sequence MSE 选择并恢复最佳 checkpoint。
- `critic_value_statistics()`：同时监测 MSE、bias、constant-baseline MSE 和 explained variance。
- `sao_config()`：衔接 DIS、adaptive GAE、critic K=2，以及更新后重新预测 values 的在线流程。

源码明确将该 recipe 标为 experimental。它依赖 Fireworks Training API/Tinker-style SDK 的独立 projection head，适合参考 pretrain → validation → online SAO 的组织方式，不是可直接在我们的本地 FSDP worker 上运行的 standalone trainer。分类宽度的 projection 在此路径仍以连续 value regression loss 优化，不能据此当成 HL-Gauss 实现。

#### 9.11.3 分类式 value loss 的现成组件

[`lucidrains/hl-gauss-pytorch`](https://github.com/lucidrains/hl-gauss-pytorch)提供 `HLGaussLoss` 与 `HLGaussLayer`，能完成 scalar target → Gaussian histogram labels → categorical logits loss → scalar expectation。

它是可复用的 loss/head 组件，不包含 LLM rollout dataset、action-mask alignment 或分布式 critic training。对于本项目，优先参考已有 `miles-values` 的 target projection，实现时继续保持 scalar value 接口。

**实施原型建议：ReDiPPO 的离线数据/collator/训练 loss → SkyRL 原生 FSDP critic；SAO 在线调度参考上述 cookbook；HL-Gauss 作为后续对照。** 不必搬入整个外部 RL 框架，也不需要双 critic。

### 9.12 已移植到本项目的离线 value 预训练

2026-10-10，按 ReDiPPO standard critic 的 data/collator/action-before-token loss 原型移植，保留 Apache-2.0 来源说明，训练运行时改为本项目原生 FSDP2：

- `skyrl/train/dataset/value_dataset.py`：读取精确 rollout token IDs、完整 action mask、terminal outcome / 显式 returns，支持多轮工具轨迹；过长轨迹整条剔除，截断轨迹要求提供含 bootstrap 的 returns。
- `skyrl/train/main_value.py`：独立 critic pretraining，复用 `FSDPStrategy`、optimizer 和 checkpoint/export；全局 action-token normalization，每 50 步及末步进行 held-out 验证/保存，以 sequence MSE 选取 `best/`，记录 W&B 曲线。
- `skyrl/backends/skyrl_train/workers/model_wrapper.py`：RWKV critic 原生 BF16、禁用 enclosing autocast、单次右 padding recurrent forward，再还原 token 位置；其他模型继续走原有路径。
- `examples/train/rwkv/README.md`：输入契约、启动命令及 online critic checkpoint 接入说明。

与 ReDiPPO 的明确差异：使用 SkyRL **raw scalar MSE**，不应用 binary-outcome sigmoid；不引入双 critic、VeRL/DeepSpeed 运行时或文字重分词 mask。该入口初始化独立 critic，不实现 SAO online DIS / adaptive GAE，也不改变现有 rollout 采样与在线 RL 算法。

验证：RWKV 相关测试分组运行 **69 passed**（数据/预训练 40、model wrapper 29），含采集→builder→训练数据契约、CLI、resume 去重和真实 RWKV HF checkpoint 的 BF16 / value-head 往返。GPU 训练验收尚未完成：本机最小 CUDA 内存查询就触发 OOM；没有据此宣称真实模型训练、收敛或多 GPU FSDP 已通过。数据构建链路已补齐：`examples/train/rwkv/rollout.py --sampling rollout --save-rollouts` 保存全部已评分的精确 token 轨迹；`examples/train/rwkv/build_value_data.py` 流式生成 `train.jsonl`、`validation.jsonl` 和 provenance manifest，按 question UUID 固定分组，不重分词、不做正负样本重平衡。完整 token IDs 按 question 暂存到磁盘，评分后先原子发布 shard，再记录 completed question；resume 不重复追加同一 question。默认剔除空响应、超长样本和没有显式 bootstrap returns 的截断样本，并记录计数，因此 MC 数据目标是满足完成/长度条件的轨迹分布。

### 9.13 本项目在线 GAE 与 DIS + Score-Centering

2026-10-15，在线部分补齐到原生接口：

- `advantage_estimator=gae_adaptive`：通过原生 registry 分发，按 `response_mask & loss_mask` 跳过 observation/padding，只在有效 action 之间连接 TD residual 和 trace。Actor 默认使用 `max(0, 1 - 1/(1.5*l))`，critic 独立使用 `gae_adaptive.critic_lambd=1`；`mode=fixed` 可单独使用原有 `algorithm.lambd` 作为 actor lambda。内部 FP64 累积，输出 FP32 targets；不强制 whitening，由原生 Trainer 的 normalization/reduction 配置控制。
- Response-level terminal reward 在该 estimator 下放到最后有效 action；已有 token-level rewards 保持生成器提供的 action-boundary 位置。完整轨迹在原生同步/fully async Trainer 中共用该 estimator。
- `policy_loss_type=rollout_is` 复用已有 DIS；可选 `rollout_is.score_centering=true` 复用现有 FSDP/RWKV top-k sampler-support 采集、worker forward/backward 和 SC helper。对完整 `ratio * double-sided gate` 加权 score 做中心化，包含相同加权函数下的 head/tail correction，不再次按 sampled-action gate 屏蔽 correction。
- SC 组合使用 temperature 1、top-p 1、top-k -1、无 penalty 的 rollout 分布，默认捕获 128 个 raw candidates；capture width 不改变 sampling top-k。该实验选项不叠加其他 off-policy correction 或 support replay。
- 现有 `gae` 和其他算法的默认路径保留。配置和启动示例见 `examples/train/rwkv/README.md` 的 Online SAO and Score-Centering。

验证：相关回归 **428 passed**；当前环境未安装 FlashAttention 的 3 个既有 packed-layout 测试未运行。覆盖 compressed-action GAE 对照、actor/critic lambda 分离、8k/BF16 trace 精度、Ray registry 重建、SC 加权梯度/常数 advantage 漂移抵消/top-k tail、FSDP worker train/eval 通道与原有 RWKV critic。Ruff、Black、gitleaks 通过。本轮未进行真实 RWKV GPU 端到端训练，尚未验证 SAO + SC 的收敛和效果收益。

### 9.14 RWKV critic 配方审计与 Channel-mixing 修正

2026-10-21，重新对照原论文 §2、§3.2、§4.1、§4.3，以及固定版本的 `miles-values` 和 Fireworks experimental cookbook。两份代码是参考实现，不能视为论文作者公开的完整原始训练配方。

已确认的偏离与证据：

- **可训练模块。** 论文冻结 attention、训练 MoE projections；此前 RWKV critic 的 NoRA 仅选择 `r_proj/k_proj/v_proj/o_proj`，均位于 Time-mixing。现有 Transformers 的 `RwkvFeedForward` 已向 FlashRWKV2 传递 Channel-mixing `mlp.key/mlp.value` 的 A/B 参数及 `use_nora`，并非缺少接口。离线 value 入口和在线示例现改为 Channel-mixing adapters + 全量 FP32 value head；冻结 Time-mixing 是 RWKV 迁移假设，不是论文已证明的 RWKV 结论。OPD 训练 policy，默认包含 Time-mixing、Channel-mixing 与 `lm_head`，不套用 critic 的冻结策略。
- **原训练数据。** 远端 `rwkv-g1l-1b-temp-6331-sao/value-data/manifest.json` 记录每域正误 1:1、训练 5742 条中 2871 条正例，并剔除 14815 条截断轨迹；采样采用 eval 的 top-p/top-k/penalty，另混入数学 benchmark，部分预算为 24576。在线数据采用 temperature 1、top-p 1、top-k -1、无 penalty、8192 预算。未经采样权重修正，平衡后的回归不估计自然策略分布下的 value。多策略预训练本身可以作为 warm start，但不能据此宣称已对当前 actor 校准。后续 `critic-calibration/value-data` 已改用固定 actor 的自然 rollout，保留预算截断为失败。
- **两次更新不等于两遍训练。** 原运行 `train_batch_size=128`、`policy_mini_batch_size=128`、`critic_mini_batch_size=64`，共用 `update_epochs_per_batch=1`。Critic 做两次半批更新，每条数据仅使用一次。论文 §4.3 将 K=2 对照明确写为 `critic-train-epoch=2`；`miles-values` 的 `train_critic()` 也用独立 `num_critic_epochs` 重建同批 iterator。最小协议修正应为 critic 独立两遍完整 batch，actor 一遍，并相应计算 critic scheduler 的步数。
- **更新后 value 的使用。** SkyRL 当前先计算 GAE，再并行更新 critic/actor。Fireworks 的 `sao_config()` 使用两次完整 critic 更新，再刷新 value 计算 actor advantages；但该版本 `miles-values` 传给 actor 的仍是更新前 value。两份参考实现确有差异，不能把刷新策略说成已定位的统一实现 bug。若采用论文描述与 Fireworks 的顺序，应显式选择 post-update value，保留最初的 PPO value anchors/critic targets，且不与 actor 更新并行。
- **目标与 reduction。** 论文使用平方 value error；`miles-values` 的 MSE head 输出 raw scalar，在线用 clipped MSE，默认 reduction 为逐序列有效 token 平均。ReDiPPO 的 sigmoid + MSE、后续 sigmoid/BCE/L2 读取器、全骨干冻结和校准扫描均是额外选择。固定 `8192` 分母改变了训练权重，应视为独立算法选择，不能称为复现参考实现必需的修复。

原论文没有公开完整的离线 value 数据比例、平衡/截断规则及全部优化细节，因此这些部分只能区分已证实的数据契约问题和迁移假设，不能补写成作者配置。原崩溃前的数据重加权、Time-mixing-only 适配和更新协议偏离，与后续修复过程中加入的 sigmoid/读取器试验也应分别归因。模块选择已修正；更新顺序、独立 critic epochs 和自然分布预训练构成后续最小方案，在确认方案前不启动采样或训练实验。

参考定位：[原论文](https://arxiv.org/html/2607.07508v1#S3.SS2)、[critic 更新循环](https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/megatron_utils/actor.py#L466-L523)、[raw value 转换](https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/logit_processors.py#L17-L25)、[序列 reduction](https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/cp_utils.py#L92-L155)、[Fireworks SAO 更新顺序](https://github.com/fw-ai/cookbook/blob/a55b26e7c4f3da481a3c9305dd24f42450e5dee2/training/recipes/experiment/ppo_value_head_loop.py#L946-L1002)。

## 10. 对 SkyRL / RWKV 的启发（迁移判断，不是已实现结论）

1. **先确保数据契约。** rollout 持久保存生成时的 token logprobs；trainer 使用同一 tokens、完整工具上下文及 loss masks。跨版本轨迹不能用某个统一 old checkpoint 冒充逐 token behavior policy。
2. **将 policy loss mask 与 GAE scan mask 分开审查。** Tool observations 不能接受 policy gradient，也不应在 skip-observation GAE 中消耗递推步数；terminal reward 放在最后有效 action token。
3. **先检验 critic 再缩小 rollout count。** 记录 mask-aware EV、value loss、各模块 gradient norm、positive/negative advantages，避免冷启动 critic 破坏 actor。
4. **Frozen attention 不能机械替换为冻结 RWKV recurrent layer。** RWKV 的状态转移结构不是论文的 Full Attention/MoE 分工，需要独立消融。
5. **不要一次改完所有机制。** 在现有基线下依次比较直接 behavior ratio / gate、single rollout、critic update frequency、value pretrain；以相同 rollout budget、wall-clock 和 eval protocol 报告结果。
6. **GPU 共卡是另外一项系统优化。** `miles-values` 展示了 actor/critic 时分复用，但其 Megatron + SGLang/TMS 生命周期不等于本项目 FSDP + vLLM 的现成实现；需要评估参数/optimizer/offload/weight-sync 的完整生命周期与总成本。
7. **保持 critic 接口简单。** 将 classification head → scalar value 转换留在 critic 内部，不把 categorical details 扩散到 Trainer/Generator/Environment；独立评估 MSE / HL-Gauss / pretrain。
8. **区分论文目标与实现开关。** 不把 PPO dual clip、IcePop correction、direct rollout denominator 当作可互换开关，逐项核对 gate ratio、梯度语义及 reduction。
9. **公平比较预算。** 除 trajectory 数，还报告独立 prompt 数、generated/action tokens、critic compute、训练时间、behavior lag 和 gate rejection。

## 11. 来源

- [arXiv 摘要、作者与 v1 日期](https://arxiv.org/abs/2607.07508v1)
- [论文全文](https://arxiv.org/html/2607.07508v1)：§3 方法，§4 设置/结果/消融/在线模拟，Appendix A 粒度对照，Appendix B 限制。
- [PDF](https://arxiv.org/pdf/2607.07508v1)
- [miles-values 实现分支](https://github.com/yyht/miles-values/tree/main-0729)。以下源码来源固定到本次阅读的 commit。
- [分支对应的 commit](https://github.com/yyht/miles-values/commit/51b84b7f14436e056ce64b665adeb5957c5433ca)
- [README][readme]：定位、研究扩展、实验概述与相关论文。
- [Colocate-Critic PPO 文档][colocate]：共卡设计、配置、验证。
- README 引用的三篇作者博客（本文未用它们替代源码/论文证据）：[SAO 与小规模实验](https://zhuanlan.zhihu.com/p/2064452938018322060)、[credit assignment](https://zhuanlan.zhihu.com/p/2064646116927344943)、[value pretrain](https://zhuanlan.zhihu.com/p/2066503866091410176)。

[readme]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/README.md
[sync]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/train.py#L110-L127
[async]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/train_async.py#L69-L128
[placement]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/ray/placement_group.py
[actor]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/megatron_utils/actor.py
[provider]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/megatron_utils/model_provider.py#L27-L31
[logits]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/logit_processors.py#L12-L27
[advantages]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/advantages.py#L107-L129
[gae]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/gae_adaptive.py
[losses]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/losses.py
[corrections]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/corrections.py#L34-L66
[math]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/backends/training_utils/loss_hub/math_utils.py#L254-L278
[args]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/miles/utils/arguments.py
[colocate]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/scripts/colocate_ppo/README.md
[script]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/scripts/colocate_ppo/run_qwen35_4b_ppo_colocate_critic_async_8gpu.sh
[mask-test]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/tests/fast/backends/training_utils/test_ppo_gae_masks.py
[fully-async]: https://github.com/yyht/miles-values/blob/51b84b7f14436e056ce64b665adeb5957c5433ca/examples/fully_async/README.md

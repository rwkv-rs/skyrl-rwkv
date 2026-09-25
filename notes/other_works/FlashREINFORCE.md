# FlashREINFORCE：Critic-Free Single-Rollout Asynchronous RL

> 阅读对象：NVIDIA，*FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models*。本文按论文与官方实现整理核心动机、数学形式、训练流程和实验设计。

## 1. 一句话总结

FlashREINFORCE 把“每个 prompt 采样多个 sibling rollouts、等待整组完成”的 GRPO 式流程，改成**每个 prompt 只采样一条轨迹**，并用一个刚完成的独立轨迹 batch 完成一次 critic-free policy-gradient 更新：

\[
\boxed{\text{One-Batch REINFORCE}
\;+
\text{Sequence Trust Region}
\;+
\text{Sample-Mean Optimization}}
\]

三部分分别解决：

1. **没有 sibling rollout / critic 时如何获得有正有负的学习信号**：对独立 prompt 的 reward 做 batch mean centering；
2. **异步 rollout 的 policy staleness 和 history-distribution mismatch**：用行为策略的 token log-prob 做 importance sampling，并以整条轨迹为单位做 divergence gate；
3. **长失败轨迹的负反馈被 token 数放大**：先在轨迹内取平均，再在轨迹之间取平均。

它的关键工程性质是：**一条轨迹完成即可进入异步队列；一个新 batch 只做一次 optimizer update，随后立即丢弃。** 因而不需要 sibling-rollout barrier、critic、PPO ratio clipping 或 reference-model forward。

## 2. 问题背景与设计动机

### 2.1 GRPO/组相对方法在 agentic RL 中的代价

对于长 CoT、多轮工具调用和环境交互，轨迹长度与完成时间差异很大。GRPO 通常为同一个 prompt 生成多个回答，然后用组内 reward 的相对值构造 advantage。这会带来：

- 固定 rollout 预算下，多个样本被消耗在同一个 prompt，覆盖的 prompt 数量变少；
- 必须等待同组 sibling trajectories，异步执行中形成同步屏障；
- 在线交互或只能得到一条 trajectory 的环境不容易使用 group baseline；
- 当 rollout 与 learner 解耦时，不同轨迹来自不同旧 policy，staleness 使训练更容易不稳定。

去掉 group 后又有两个直接问题：

- 单条轨迹没有同 prompt 的相对 baseline；若 reward 是二值的，未经处理的 REINFORCE 通常只会强化成功样本，失败样本没有学习信号；
- 若失败轨迹很长，给整条轨迹一个负 advantage 会让“失败轨迹中的有用 token”也被惩罚，而且 token-level reduction 会让长失败得到更大的权重。

FlashREINFORCE 的原则不是重新设计一个复杂的 critic，而是重新组织普通 REINFORCE 的数据归一化、off-policy 校正和轨迹准入。

### 2.2 异步数据的定义

第 \(i\) 条轨迹包含：

- 行为策略 \(\mu_i\) 生成的历史 \(h_{i,t}\)；
- 在该历史下实际采样的 token/action \(a_{i,t}\)；
- 轨迹长度 \(T_i\)；
- terminal scalar reward \(R_i\)。

rollout worker 必须在生成时保存实际行为概率：

\[
\mu_i(a_{i,t}\mid h_{i,t}),
\]

不能依赖 learner 之后重算的“old probability”，因为训练与推理栈、模型版本、router 或数值实现可能已经不同。

这里的 **fresh** 只表示这条 trajectory 尚未被某次 update 使用，不表示它一定来自最新 policy。staleness 由行为策略相对当前 learner 的漂移决定，交给 sequence trust region 处理。

## 3. 数学形式

### 3.1 Token-level importance sampling

当前 learner 为 \(\pi_\theta\)，对每个已存储的 action 计算：

\[
\rho_{i,t}(\theta)
=\frac{\pi_\theta(a_{i,t}\mid h_{i,t})}
{\mu_i(a_{i,t}\mid h_{i,t})}
=\exp\left(
\log\pi_\theta(a_{i,t}\mid h_{i,t})
-
\log\mu_i(a_{i,t}\mid h_{i,t})
\right).
\]

它校正的是**在固定、已观测历史上的条件 action distribution**。它并没有完全校正历史本身仍来自 \(\mu_i\) 这一事实；长轨迹上 history occupancy 的差异会累积，因此还需要轨迹级 gate。

### 3.2 One-Batch REINFORCE：用 batch mean 产生 signed feedback

一次 learner update 取接下来完成的 \(B\) 条独立 prompt 轨迹，计算 batch reward 均值：

\[
\bar R=\frac{1}{B}\sum_{j=1}^{B}R_j,
\qquad
A_i=R_i-\bar R.
\]

- \(A_i>0\)：该轨迹比当前 batch 的平均水平好，增加其 action 的概率；
- \(A_i<0\)：该轨迹低于平均水平，降低其 action 的概率；
- 对二值 reward，只要 batch 同时包含成功和失败样本，失败样本就能提供负反馈。

\(\bar R\) 是一个无参数的、critic-free control variate；它不需要学习 value model，也不需要同一 prompt 的 sibling samples。每个 batch 只使用一次，然后丢弃。

### 3.3 Local off-policy objective 与历史分布误差

令 \(J(\pi)\) 表示当前 policy 的 expected return，令 \(d_{\mu,t}\) 表示行为策略在第 \(t\) 步产生 history 的分布，\(A_\mu(h,a)\) 是行为策略下的 continuation advantage。论文定义一个保持 behavioral histories 和 behavioral continuation values 不变的 local surrogate：

\[
\mathcal L_\mu(\pi)
=J(\mu)+
\sum_{t=1}^{H}
\mathbb E_{h\sim d_{\mu,t}}
\mathbb E_{a\sim\pi(\cdot\mid h)}
\left[A_\mu(h,a)\right].
\]

在 common support 条件下，其梯度可以用行为数据写成：

\[
\nabla_\theta \mathcal L_\mu(\pi_\theta)
=
\mathbb E_{\tau\sim\mu}
\left[
\sum_{t=1}^{H}
\rho_t(\theta) A_\mu(h_t,a_t)
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\right].
\]

在 \(\pi=\mu\) 处，该式退化为普通 REINFORCE 梯度。重要性比率修正了 action 条件分布，但 history 仍由 \(\mu\) 产生，因此 \(J(\pi)-\mathcal L_\mu(\pi)\) 会随着每一步 policy movement 累积。论文的 coupling 分析给出如下量级关系：

\[
\left|J(\pi)-\mathcal L_\mu(\pi)\right|
\lesssim
A_{\max} H^2\,\bar\kappa,
\]

其中 \(H\) 是固定 token horizon，\(A_{\max}\) 是 advantage 上界，\(\bar\kappa\) 是沿 horizon 平均的行为策略与 learner 的 per-history KL 漂移。该结果的直接启发是：不要只在 token 级别独立截断，而应对整条 trajectory 做准入控制。

### 3.4 Sequence Trust Region：按整条轨迹筛选 stale data

对每个 token，将完整 vocabulary distribution 投影成“采样 action vs 其余 action”的 Bernoulli 分布。令

\[
p_{i,t}=\mu_i(a_{i,t}\mid h_{i,t}),
\qquad
q_{i,t}=\pi_\theta(a_{i,t}\mid h_{i,t}),
\]

定义 sampled-action Bernoulli KL proxy：

\[
d_{i,t}
=p_{i,t}\log\frac{p_{i,t}}{q_{i,t}}
+(1-p_{i,t})
\log\frac{1-p_{i,t}}{1-q_{i,t}}.
\]

对不同长度的轨迹进行公平比较，使用轨迹内平均值：

\[
\bar D_i=\frac{1}{T_i}\sum_{t=1}^{T_i}d_{i,t}.
\]

设阈值为 \(\delta\)，整条轨迹的 stop-gradient 准入 mask 为：

\[
m_i=\mathbf 1[\bar D_i\le\delta].
\]

若一条轨迹不通过 gate，则删除它的**完整 stored-history contribution**，而不是只删除漂移较大的 token。

这个 Bernoulli KL 是完整 categorical KL 的 data-processing lower bound：\(\bar D_i\) 超过阈值时，可以确认完整分布漂移至少很大；反过来通过 gate 并不能保证 full-vocabulary KL 足够小。例如，采样 token 的概率不变、但其余 token 之间发生大幅质量转移时，proxy 可能仍然很小。因此它是低成本的实用筛选器，不是严格的 full-distribution trust-region certificate。

### 3.5 Sample-Mean Optimization：移除显式长度权重

将每条轨迹内的 policy-gradient 项先平均，再在 batch 内平均：

\[
\widehat{\mathcal J}_{\mathrm{FR}}(\theta)
=
\frac{1}{B}\sum_{i=1}^{B}
\frac{m_i A_i}{T_i}
\sum_{t=1}^{T_i}
\rho_{i,t}(\theta).
\]

实际实现使用等价的 detached-ratio score-function loss：

\[
\mathcal L_{\mathrm{FR}}(\theta)
= -\frac{1}{B}\sum_{i=1}^{B}\frac{1}{T_i}
\sum_{t=1}^{T_i}
\operatorname{stopgrad}\left(m_i A_i\rho_{i,t}\right)
\log\pi_\theta(a_{i,t}\mid h_{i,t}).
\]

其梯度方向为：

\[
-\nabla_\theta\mathcal L_{\mathrm{FR}}
=
\frac{1}{B}\sum_{i=1}^{B}\frac{m_i A_i}{T_i}
\sum_{t=1}^{T_i}
\rho_{i,t}
\nabla_\theta\log\pi_\theta(a_{i,t}\mid h_{i,t}).
\]

对比两种 reduction：

- **token mean / token sum**：batch 中所有 token 直接汇总，长轨迹权重与 \(T_i\) 相关；
- **sample mean**：每条 trajectory 先除以自身长度，每个被准入样本拥有相同的外层权重。

因此 sample mean 保留了失败轨迹的负反馈，但避免“失败越长，负更新越大”的显式长度放大。论文明确指出，这是一种有意的 trajectory-normalized practical approximation，不是对未归一化 policy-gradient surrogate 的精确 Monte Carlo 等价变换。

### 3.6 完整的一次更新

给定一个新完成的 trajectory batch \(\mathcal B\)：

1. 计算 \(\bar R\) 与 \(A_i\)；
2. 用当前 learner 重算每个 sampled token 的 \(\log\pi_\theta\)；
3. 用 rollout 时保存的 \(\log\mu_i\) 得到 \(\rho_{i,t}\)；
4. 计算每条轨迹的 \(\bar D_i\)，得到整条轨迹 mask \(m_i\)；
5. 用 sample-mean loss 做**一次** optimizer step；
6. 丢弃 \(\mathcal B\)，继续消费后续完成的轨迹。

该流程不包含：learned critic、PPO-style ratio clipping、reference model forward、同 prompt sibling wait 或同一 collected batch 的 replay。

## 4. 训练架构与实现要点

### 4.1 异步流水线

\[
\text{rollout workers}
\rightarrow
\text{completed-trajectory queue}
\rightarrow
\text{learner batch}
\rightarrow
\text{one update}
\rightarrow
\text{discard}
\]

不同 worker 的轨迹可以来自不同 policy snapshot。与同步 GRPO 的“一个 prompt 生成 \(G\) 个回答并等待整组”相比，FlashREINFORCE 只需每个 prompt 一条轨迹，因此相同 rollout budget 可以覆盖更多 prompt，并可直接适配完成时间不规则的多轮 agent。

### 4.2 对 SkyRL/RL 系统的接口启发

若在现有 Trainer/Generator/InferenceEngine 中实现，至少需要保证：

- rollout 端保存**实际生成 token 的 behavior log-prob**，而不是训练端事后重算的 old log-prob；
- trajectory 中保留 token、history 对齐关系、terminal reward、有效长度 \(T_i\) 和 tool/environment turn 边界；
- trainer 端对 rollout policy 与当前 policy 使用完全一致的 tokenization、response mask 和 action 定义；
- sequence gate 必须在一次 update 前冻结，mask 不能在反向传播中变化；
- batch mean 的 reward 必须跨独立 prompts 计算，而不是错误地在同 prompt 的 sibling 维度上计算；
- sample mean 的归一化要在 trajectory 内完成，不能先把所有 token flatten 后再做全局平均；
- 由于只做一次更新，不能把同一 batch 拆成多个 sequential minibatches 后继续使用，否则后三个 minibatch 面临额外 policy drift。

## 5. 实验设计

### 5.1 总体设置

论文覆盖三类场景：长 CoT 数学推理、多轮 Python tool use、交互式决策。默认使用 FlashREINFORCE 的 one-pass update；报告的 benchmark accuracy 使用 avg@N，即每个问题多次采样后的平均准确率，benchmark mean 对任务等权。

### 5.2 长 CoT 异步稳定性

- **模型**：DeepSeek-R1-Distill-Qwen-1.5B；
- **训练数据**：DPPO 的 1,460-problem MATH subset；
- **每次 update**：128 条 trajectory，每个 prompt 一条；
- **异步 policy lag**：约 4；
- **训练跨度**：6,000 cumulative updates；
- **指标**：AIME24/25 avg@32、response length、token entropy。

结果：训练保持稳定，AIME24/25 mean avg@32 从 21.7 提高到最后 checkpoint 的 33.7；训练过程中 response length 下降、entropy 趋于稳定。该实验主要验证 sequence trust 在长轨迹和异步 staleness 下的稳定作用，而不只是展示单点精度。

### 5.3 数学推理：与 GRPO 的 rollout budget 对比

- **模型**：Qwen2.5-Math-1.5B；
- **训练集**：去重后的 7.5k DAPO-Math prompts；
- **评测**：MATH-500、AMC23、Minerva、AIME25、OlympiadBench；
- **评测方式**：avg@16；
- **FlashREINFORCE**：峰值 checkpoint step 2,000，256k rollouts；
- **报告的 GRPO baseline**：step 1,000，512k rollouts；
- **其他对比**：C-RF + NTF。

| 方法 | Step | Rollouts | MATH-500 | AMC23 | Minerva | AIME25 | OlympiadBench | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base model | 0 | 0 | 35.1 | 30.9 | 6.7 | 5.4 | 19.4 | 19.5 |
| C-RF + NTF | 1,000 | 512k | 69.6 | 55.5 | 18.4 | 9.4 | 23.9 | 35.3 |
| GRPO | 1,000 | 512k | **71.0** | **57.1** | 18.2 | 10.0 | 24.9 | 36.3 |
| FlashREINFORCE | 2,000 | **256k** | 69.2 | 52.5 | **22.8** | **10.8** | **34.5** | **38.0** |

FlashREINFORCE 的五任务平均分为 38.0，比报告的 GRPO baseline 高 1.7 个百分点，同时使用一半的 training rollouts。需要注意：这是 rollout 数量和报告 checkpoint 的比较，不等价于已经证明 GPU-hours、wall-clock 或总成本也恰好减半。

### 5.4 Python-tool 多轮训练

每次 update 使用 128 条 trajectory：FlashREINFORCE 为 128 个 prompt 各采样一次，GRPO 为 32 个 prompt 各采样四次。评测为 AMC23、Minerva、AIME25 的 avg@N，并统计平均 tool calls。

#### Qwen2.5-7B-Instruct

| 方法 | Step | AMC23 | Minerva | AIME25 | Mean | Tool calls |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 600 | 51.2 | 32.3 | 7.5 | 30.3 | 0.00 |
| FlashREINFORCE | 600 | 60.6 | 28.8 | 21.7 | **37.0** | **3.25** |

GRPO 在约 200 steps 后基本停止调用 Python tool，而 FlashREINFORCE 保持了工具使用行为，平均 3.25 次调用，并在三任务 mean 上领先 6.7 个百分点。

#### Qwen3-30B-A3B（MoE）

| 方法 | Step | AMC23 | Minerva | AIME25 | Mean | Tool calls |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 800 | 93.8 | 40.4 | 46.7 | 60.3 | 1.19 |
| FlashREINFORCE | 800 | 95.6 | 44.9 | 60.8 | **67.1** | **1.94** |

两者使用相同的 102.4k rollout budget；FlashREINFORCE 在约 8 的 policy lag 下运行，GRPO 约为 1，仍领先 6.8 个百分点。论文还报告 FlashREINFORCE 在该 30B MoE 设置下至少运行到约 1,450 updates，说明方法可以扩展到更大模型和更高异步 lag。

### 5.5 ALFWorld 交互式决策

- **模型**：Qwen2.5-7B-Instruct；
- **训练数据**：1,024 个 ALFWorld games；
- **每次 update**：64 games × one rollout；
- **评测**：140 个 seen games + 134 个 unseen games；
- **训练奖励**：terminal environment success；
- **评测 checkpoint**：step 200，12.8k training trajectories；
- **约束**：最多 50 environment actions、总 token budget 16,384、generation budget 8,192。

| 方法 | Rollouts per prompt | Seen | Unseen |
|---|---:|---:|---:|
| GRPO（published） | 8 | 78.6 | 76.8 |
| C-RF + NTF（published） | 1 | 90.5 | 86.3 |
| FlashREINFORCE | 1 | **98.3** | **96.5** |

相较最强的已发表 baseline，FlashREINFORCE 在 seen/unseen 上分别提升 7.8/10.2 个百分点。由于 C-RF + NTF 也使用一条 rollout per prompt，这一结果不能简单归因于 prompt diversity。

## 6. 消融实验与论文结论

### 6.1 Fresh one-batch update vs sequential minibatches

比较两种使用同等数据和 optimization budget 的方式：

- 每次收集 128 条，立刻做一次 update；
- 先收集 512 条，再拆成四个连续的 128 minibatches 更新。

后者的后三个 minibatch 在 learner 已经更新 1–3 次后才使用，即使有 token IS 和 trust gate，也会产生额外 drift。论文结果支持“每个 fresh batch 只更新一次，然后丢弃”。

### 6.2 Signed feedback vs positive-only feedback

去掉负 advantage 项后，reward-zero 轨迹不再提供抑制错误行为的信号，方法近似变成 positive-only update。论文的对比支持保留 batch-centering 产生的 signed feedback，尤其是在工具行为和失败恢复相关任务上。

### 6.3 Sample mean vs token mean

token mean 会让长失败轨迹按 token 数获得更大负权重；sample mean 能显著缓解 response length 上升、truncation 增加和 reward 下降的联动。该设计不是为了改变单个 token 的信号，而是为了改变 trajectory-level weighting。

### 6.4 Sequence-level gate vs token-level gate

逐 token gate 只移除局部高漂移 token，但不能阻止其它 token 继续使用同一条已偏离的 stored history；sequence-level gate 直接拒绝整条 trajectory，更符合 history-distribution mismatch 随时间累积的分析。论文报告数学实验中 token-local 方案可能 collapse，而 sequence-level admission 保持稳定。

### 6.5 阈值敏感性

Qwen2.5-Math-1.5B 上比较了 \(\delta=10^{-3}\) 与 \(3\times10^{-3}\) 的 sequence threshold；两者都能训练，说明阈值是重要的稳定性旋钮，但不应被误解为理论上自动最优的常数。实际应结合 policy lag、学习率、响应长度和 gate rejection rate 监控。

## 7. 方法边界与实践评价

1. **批均值 baseline 依赖 batch 的 reward diversity**：若一个 batch 几乎全成功或全失败，centered advantage 的有效区分度会降低。
2. **token IS 不是完整 off-policy correction**：它只校正 action probability，不能完全修复旧 policy 产生的 history occupancy；sequence gate 也只是 proxy。
3. **通过 gate 不代表严格安全**：Bernoulli projection 可能漏掉未采样 token 之间的分布迁移，因此需要监控 rejection rate、KL proxy 分布、ratio 分布和实际训练稳定性。
4. **Sample mean 是有意的工程近似**：它牺牲了未归一化 policy-gradient 的严格形式，换取对长失败轨迹更合理的贡献控制。
5. **结果比较需注意协议差异**：数学实验使用论文中报告的 GRPO baseline；不同模型的 policy lag、checkpoint step、rollout budget 和评测采样数并不完全相同，不能据此宣称 FlashREINFORCE 在所有 RL 任务上普遍优于 GRPO。
6. **实现最容易出错的地方是 behavior log-prob 对齐**：必须保存推理引擎真正使用的 token probability，并确保 tokenizer、mask、tool-call token 和 trainer policy 完全对齐。

## 8. 最终理解

FlashREINFORCE 并没有提出一个新的 critic 或复杂的 advantage estimator，而是把普通 REINFORCE 重新整理成适合异步 agentic RL 的形式：

\[
\boxed{
\text{独立 prompt 的 reward centering}
\rightarrow
\text{token IS + trajectory trust}
\rightarrow
\text{trajectory-normalized policy gradient}
}
\]

其主要收益来自三个互补的控制面：batch mean 让单 rollout 也能产生 signed feedback；sequence gate 限制 stale trajectory 对 history distribution 的破坏；sample mean 防止长失败轨迹主导更新。对于长 CoT、工具调用和交互环境，这种“少等待、少重复 rollout、一次性消费数据”的训练范式比单纯把 GRPO 的 group size 调小更系统。

## 9. 论文与代码链接

- **论文 PDF**：<https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf>
- **项目主页**：<https://yifanzhang-pro.github.io/FlashREINFORCE/>
- **论文/参考实现仓库**：<https://github.com/yifanzhang-pro/FlashREINFORCE>
- **NVIDIA Molt 集成仓库**：<https://github.com/NVIDIA-NeMo/labs-molt>

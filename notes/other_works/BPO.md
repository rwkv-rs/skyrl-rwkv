# Bellman Policy Optimization（BPO）

> 论文：Zhuoqing Song, Haotian Xu, Xikun Zhang, Lidong Bing，*Bellman Policy Optimization*，arXiv:2609.15987v1（2026-09-14）。

## 1. 一句话概括

BPO 是一种用于 RLVR 的 **critic-free policy optimization** 方法。它从 Policy Mirror Descent（PMD）出发，在“自回归生成 + 只有序列末端奖励”的设定下，利用 Bellman 方程把每个 token 的 advantage 写成相邻状态价值之差；沿整条 response 求和后，中间价值项发生 telescoping，只剩下终端 verifier reward 与 prompt-level 初始价值。因此，训练不需要 value model，也不需要对每个中间 prefix 做额外 rollout。

## 2. 问题设定与动机

给定 prompt `x`，模型自回归生成 response `y = (y_1, ..., y_T)`，状态为

\[
 s_t=(x,y_{<t}), \qquad y_t\sim \pi(\cdot\mid s_t).
\]

RLVR 的奖励通常是终端奖励 `R(x,y)`（例如数学答案是否正确），中间 token 没有直接奖励。记 rollout/reference policy 为 `\mu`，当前训练 policy 为 `\pi`。

标准 PMD 的 token-level 更新需要

\[
 A^\mu(s_t,y_t)=Q^\mu(s_t,y_t)-V^\mu(s_t),
\]

也就是需要估计每一个中间 prefix 的 value。对长链路数学推理而言，训练一个可靠 critic 成本高且容易产生误差；BPO 的关键观察是，这些中间 advantage 在 terminal-reward 场景中可以通过 Bellman 方程整体消掉。

## 3. 核心数学推导

### 3.1 PMD 的局部最优条件

PMD 可以理解为在保持靠近 rollout policy `\mu` 的同时，按 advantage 调整动作分布。其典型正则化形式为

\[
\pi^+(\cdot\mid s)
=\arg\max_p\left\{
\mathbb E_{a\sim p}[A^\mu(s,a)]
-\eta D_{\mathrm{KL}}(p\Vert \mu)
\right\},
\]

其中 `\eta` 是 KL 温度。对应的最优性条件可写成“当前 policy 的 log-ratio 与 advantage（以及状态相关归一化项）匹配”。直接实现时，需要在每个 visited state 估计 `A^\mu`。

### 3.2 Bellman telescoping

对于确定性的 token transition，选择 token `y_t` 后到达 `s_{t+1}`，有

\[
 Q^\mu(s_t,y_t)=V^\mu(s_{t+1}),
\]

从而

\[
 A^\mu(s_t,y_t)
=V^\mu(s_{t+1})-V^\mu(s_t).
\]

对一条完整 response 求和：

\[
\sum_{t=1}^{T}A^\mu(s_t,y_t)
=V^\mu(s_{T+1})-V^\mu(s_1)
=R(x,y)-V^\mu(x),
\]

其中终止状态的 value 就是 verifier 给出的终端奖励，且

\[
 V^\mu(x)=\mathbb E_{y\sim\mathbb P_\mu(\cdot\mid x)}[R(x,y)].
\]

因此，原本需要一串中间 value 的 PMD 条件，可以改写为只依赖：

- 完整 response 的终端奖励 `R(x,y)`；
- 同一 prompt 在 rollout policy 下的期望奖励 `V^\mu(x)`；
- 当前 policy 与 rollout policy 的 token-level log probability / KL 项。

论文将这种改写后的 trajectory-level residual 记为 `\delta(x,y;\pi,\mu)`，并给出如下形式的 critic-free 目标（符号正负取决于 log-ratio 的定义约定）：

\[
 \min_{\pi\in\Pi_\mu}
 \mathcal L(\pi)
 =\mathbb E_{x\sim\mathcal D,\,y\sim\mathbb P_\mu(\cdot\mid x)}
 \left[
 \phi(x)\frac{\delta(x,y;\pi,\mu)^2}{2\eta}
 \right],
\]

其中 `\phi(x)>0` 是任意正的 prompt weighting。论文 Theorem 1 证明：在 rollout policy 可达的状态上，这个目标与原 PMD 目标具有相同的唯一最优 completion distribution。直观上，Bellman difference 的累加形成 telescoping；因此并非“用一个粗糙 baseline 代替 critic”，而是对 terminal-reward 问题的等价重参数化。

### 3.3 Prompt value 的估计

对同一 prompt 采样一组 response `\{y_i\}_{i=1}^G`，用 reward 均值估计初始 value：

\[
 \widehat V^\mu(x)=\frac1G\sum_{i=1}^{G}R(x,y_i).
\]

相应的中心化回报为 `R_i-\widehat V^\mu(x)`。这也是 BPO 与 GRPO 的共同点之一：使用 group rollout 做 prompt-level reward normalization；区别在于 BPO 的 policy-gradient 权重不是普通 importance ratio。

## 4. 实际 BPO loss

理论目标不能直接原样用于常规 token-level SGD，论文依次做了以下近似：

1. **group rollout 估计初始 value**：用同 prompt 的 reward 均值估计 `V^\mu(x)`，并构造中心化 terminal signal；
2. **线性化 squared residual / 取其梯度**：将 trajectory-level residual 转成可实现的 token-level policy loss；
3. **full reverse KL → binary KL**：对每个 sampled token，把词表划成“该 token”和“其他 token”两个事件，避免遍历整个词表；
4. **additive smoothing 与 clipping**：限制概率接近 1 时的数值爆炸，并截断极端权重。

Binary KL 的关键恒等式是：

\[
\nabla_\theta\left[
\log \pi_\theta(y_t\mid s_t)
+D_{\mathrm{KL}}^{\mathrm{bin}}
 (\mu(\cdot\mid s_t)\Vert\pi_\theta(\cdot\mid s_t);y_t)
\right]
=
\frac{1-\mu(y_t\mid s_t)}{1-\pi_\theta(y_t\mid s_t)}
\nabla_\theta\log\pi_\theta(y_t\mid s_t).
\]

所以 BPO 的 mismatch-correction weight 使用的是**互补概率比**，而不是 GRPO 中常见的直接 importance ratio `\pi/\mu`：

\[
 \omega_t
 =\frac{1-\mu(y_t\mid s_t)}{1-\pi(y_t\mid s_t)}.
\]

工程实现中使用平滑后的版本，例如

\[
 \widetilde\omega_t
 =\frac{1+\epsilon-\mu(y_t\mid s_t)}
 {1+\epsilon-\pi(y_t\mid s_t)},
 \qquad
 \bar\omega_t=\min(\widetilde\omega_t,C),
\]

其中 `\epsilon` 防止分母过小，`C` 是上限。于是实际 token-level loss 可概括为

\[
 \mathcal L_{\mathrm{BPO}}
 =-\sum_{i,t}
 \widehat A_i\,M_{i,t}\,
 \bar\omega_{i,t}
 \log\pi_\theta(y_{i,t}\mid s_{i,t}),
\]

其中 `\widehat A_i` 是由 terminal reward 与 group prompt value 构成的 response-level signal，`M_{i,t}` 是 response/token mask。论文的完整实现还包含相应的 normalization、masking 和 clipping 细节。

### 直觉比较

- **GRPO**：主要依赖 group reward advantage，通常对 rollout/current policy 使用直接 ratio 或 clipping；
- **BPO**：仍然是 critic-free、group-rollout 方法，但其 policy mismatch 修正来自 Bellman/PMD 推导，使用 ` (1-\mu)/(1-\pi) ` 的互补概率比；
- **核心收益**：不训练中间状态 value，同时理论上保留 PMD 的最优解；
- **主要近似来源**：group value 估计、binary KL、平滑与 clipping。因此“理论等价”与“工程 loss 完全等价”需要区分。

## 5. 实验设计

### 5.1 训练设置

- **任务**：RLVR 数学推理；
- **训练数据**：DAPO-Math-17k；
- **主模型**：Qwen3-30B-A3B-Base；
- **对比方法**：GRPO-ClipHigher、GSPO、CISPO、DPPO；
- **控制变量**：尽量保持模型、rollout、奖励/verifier、采样与训练超参数一致，只替换 policy loss，以比较优化目标本身；
- **评测**：AIME 2024、AIME 2025、AIME 2026；使用 `Avg@32`，即每题采样 32 个回答后计算平均正确率，再在题目上取平均；报告各方法训练过程中的 peak checkpoint。

### 5.2 主结果

论文报告 BPO 在三个 AIME 集合上的平均 peak accuracy 为 **50.5%**，且三个 benchmark 上均取得最高结果。相对基线的提升约为：

| 方法 | 三个 AIME 的平均 peak accuracy（论文报告） |
|---|---:|
| GRPO-ClipHigher | 39.5% |
| GSPO | 43.5% |
| DPPO | 46.4% |
| CISPO | 47.4% |
| **BPO** | **50.5%** |

对应提升约为 **11.0、7.0、4.1、3.1 个百分点**。训练曲线显示 BPO 的优势并非只来自某个孤立 checkpoint；论文还报告在最终训练步骤上 BPO 仍优于强基线。

### 5.3 消融实验

论文在 **Qwen3-4B-Base** 上研究 BPO 实现中的稳定性超参数，主要包括：

- additive smoothing 的大小；
- mismatch-correction weight 的 truncation / clipping 上限；
- 不同的平滑与截断组合；
- 实际 binary-KL 近似及其 token-level loss 组件。

结论是：在一段合理的 smoothing 与 truncation 范围内，性能变化相对平稳，说明结果不是依赖一个极窄的数值超参数点；同时 clipping 仍然是实际训练中控制极端概率比的重要稳定化机制。

## 6. 方法评价与对 SkyRL/RWKV 的启示

### 优点

1. **不需要 critic**：减少 value model 的显存、训练和同步成本；
2. **理论动机清晰**：不是简单经验 baseline，而是由 PMD + Bellman equation 得到的 trajectory-level reformulation；
3. **适合 terminal verifier reward**：数学、代码等 reward 可在完整 response 结束后计算的任务尤其适用；
4. **token-level 可实现**：binary KL 将全词表 KL 的梯度化为一个 sampled-token scalar weight；
5. **与现有 group rollout 框架兼容**：可以复用 response 采样、verifier、group reward center 和 policy logprob 计算。

### 局限与实现风险

1. 理论等价性依赖 terminal reward、正确的 Bellman transition 和 rollout-policy 可达状态；对多轮环境、dense reward、随机 transition 需要重新推导；
2. 实际算法包含 binary-KL、smoothing、clipping 等近似，理论目标与工程 loss 并不完全相同；
3. 互补概率权重在 `\pi(y_t\mid s_t)\to1` 时敏感，必须严格实现 smoothing、clipping 和数值稳定的 logprob；
4. group reward 均值是 prompt value 的 Monte Carlo 估计，group size、reward sparsity 和 prompt difficulty 会影响方差；
5. 论文的主实验是数学 RLVR，尚未证明对一般 agentic、多轮、工具调用环境同样有效。

对 SkyRL/RWKV 集成而言，最重要的接口要求是：rollout 与 trainer 必须对同一批 token 严格对齐，并保存 `R(x,y)`、group-level `\widehat V^\mu(x)`、rollout logprob `\mu(y_t|s_t)`、trainer logprob `\pi(y_t|s_t)` 和 response mask；随后在 loss 中计算平滑互补概率权重。权重同步错误或 tokenizer/template 不一致会直接破坏 BPO 的 mismatch correction。

## 7. 链接

- 论文（Hugging Face Papers）：<https://huggingface.co/papers/2609.15987v1>
- 论文（arXiv abstract）：<https://arxiv.org/abs/2609.15987>
- 论文（arXiv HTML）：<https://arxiv.org/html/2609.15987v1>
- 代码仓库：截至本笔记整理时，论文页面和 arXiv 页面未给出官方代码仓库链接；未发现作者公开的 BPO 官方实现。

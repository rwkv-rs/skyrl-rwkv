# Score Centering：稳定训练-推理不一致下的 Off-policy RL

> 阅读对象：Martin Marek、Max Ryabinin，*Score Centering Stabilizes Off-policy Reinforcement Learning*，arXiv:2609.20807v1（2026-09-17）。本文按核心思路、数学表示和实验设计整理。

## 1. 问题与核心结论

LLM RL 通常由 sampler/inference engine 生成 rollout，再由 trainer 计算梯度。理想情况下二者表示同一个策略，但实际会因不同 kernel、浮点运算顺序、量化、KV cache 精度、不同代码实现及异步更新造成 training-inference mismatch（TIM）：

\[
q_\theta(y)\neq p_\theta(y),
\]

其中 \(q\) 是 sampler 分布，\(p\) 是 trainer 分布。论文的核心判断是：TIM 导致的不稳定主要不是瞬时误差，而是每个训练步骤积累的 **drift（漂移）**。trainer 用 sampler 产生的 token 计算 score 时，score 的 sampler 期望不再为零；于是即使 reward 没有任何学习信号，梯度仍会把 trainer 蒸馏向 sampler。随后 trainer 权重同步回 sampler，偏差被反馈放大，最终导致 reward/accuracy collapse。

论文提出 **score centering（SC）**：对每个 prefix 的 trainer score 减去 sampler 下的期望 score。它是一个不需要重要性比率、截断或 mask 的加性修正；在 sampler 分布下精确消除 drift。SC 可独立使用，也可与 TIS/MIS 等 importance sampling 方法组合。

## 2. 数学表示

### 2.1 On-policy policy gradient

对 rollout \(y\) 及 reward \(R\)，标准 policy gradient 为

\[
\nabla_\theta \mathbb E_{p_\theta}[R]
=\mathbb E_{p_\theta}\left[R\nabla_\theta\log p_\theta(y)\right].
\]

令单个 prefix 下采样 token 为 \(y_t\)，词表 token 为 \(v\)：

\[
p_v=p_\theta(v\mid y_{<t}),\quad
q_v=q_\theta(v\mid y_{<t}),\quad
s_v=\nabla_\theta\log p_v.
\]

在 on-policy 情形，
\[
\mathbb E_p[s_{y_t}]=\sum_v p_v\nabla_\theta\log p_v
=\nabla_\theta\sum_vp_v=0.
\]

### 2.2 TIM 下的 drift 分解

实际更新使用 \(q\) 采样、\(p\) 计算 score。定义 sampler 下的期望 score：

\[
\bar s=\mathbb E_q[s_{y_t}]=\sum_vq_vs_v.
\]

对单个 prefix 使用协方差恒等式：

\[
\mathbb E_q[R s_{y_t}]
=\underbrace{\mathbb E_q[R]\,\bar s}_{\text{drift}}
+\underbrace{\operatorname{Cov}_q(R,s_{y_t})}_{\text{signal}}.
\tag{1}
\]

当 \(p=q\) 时 \(\bar s=0\)，drift 消失；当存在 TIM 时，一般 \(\bar s\neq0\)。drift 只依赖 reward 的均值，不依赖哪一个 token 真正带来成功，因此是伪学习信号；其方向等价于用 sampler 作为 teacher 对 trainer 做交叉熵蒸馏。

### 2.3 Score centering

定义 centered score：

\[
\tilde s_{y_t}=s_{y_t}-\bar s.
\tag{2}
\]

则

\[
\begin{aligned}
\mathbb E_q[R\tilde s_{y_t}]
&=\mathbb E_q[R]\mathbb E_q[\tilde s_{y_t}]
+\operatorname{Cov}_q(R,\tilde s_{y_t})\\
&=\operatorname{Cov}_q(R,\tilde s_{y_t}).
\end{aligned}
\tag{3}
\]

因为 \(\mathbb E_q[\tilde s]=\bar s-\bar s=0\)，drift 在每个 prefix 被精确取消。SC 与理想 on-policy 更新的区别只剩下 covariance 是在 \(q\) 而不是 \(p\) 下估计；因此在严重 staleness 时，SC 与 IS 组合通常更好。

### 2.4 与 importance sampling 的关系

完整序列的精确 IS 为

\[
\nabla_\theta\mathbb E_p[R]
=\mathbb E_q\left[\frac{p(y)}{q(y)}R\nabla_\theta\log p(y)\right].
\]

它是乘性校正，能恢复训练分布，但 ratio 可能有重尾，因此实践中要 clip/mask，重新引入 bias 和 drift。SC 是加性校正，不依赖 sampled token 的随机 ratio；两者作用正交，可先以 IS 重加权 score，再减去其 sampler 期望：

\[
\tilde s^{(w)}_{y_t}=w_{y_t}s_{y_t}
-\mathbb E_q[w_vs_v].
\]

### 2.5 Top-k 实现

保存完整 sampler vocab 分布代价过高。论文只记录 top-k（默认 \(k=128\)）logprobs，对 tail 使用 trainer 分布重构：令 \(H\) 为 top-k 集合，

\[
\hat q_v=
\begin{cases}
q_v,&v\in H,\\
\rho p_v,&v\notin H,
\end{cases}
\qquad
\rho=\frac{1-\sum_{v\in H}q_v}{1-\sum_{v\in H}p_v}.
\]

利用 \(\sum_vp_vs_v=0\)，期望 score 只需计算 head：

\[
\mathbb E_{\hat q}[s]
=\sum_{v\in H}(q_v-\rho p_v)s_v.
\]

对应 scalar loss（\(\operatorname{sg}\) 表示 stop-gradient）：

\[
\mathcal L
=-R\left(\log p_{y_t}-
\sum_{v\in H}\operatorname{sg}[q_v-\rho p_v]\log p_v\right).
\tag{4}
\]

实验显示 \(k=128\)，甚至 \(k=32\)，都与 full SC 基本一致；额外开销约可忽略（报告中约 1% wall-clock）。

## 3. 实验设计

### 3.1 统一比较框架

- **模型/任务**：Qwen3-0.6B-Instruct + Countdown；Qwen3-30B-A3B-Base + INTELLECT-2 数学子集。
- **共同目标**：全部使用 REINFORCE + group-centered rewards，只替换 TIM correction，避免把 DAPO 等复合算法的其他组件混入比较。
- **控制变量**：相同 sampler、trainer、optimizer、advantages，每 batch 一次 SGD；IS 方法均使用 sampler 记录的概率。
- **比较方法**：vanilla PG、naive IS、TIS、MIS、PPO、DAPO、GSPO、TOPR、DPPO，以及 SC、TIS+SC、MIS+SC。
- **评估指标**：训练过程 accuracy/reward 曲线、collapse 时间、不同 mismatch 强度下的稳定性和最终训练准确率。

### 3.2 受控 synthetic weight noise

首先固定一个 sampler-trainer 权重偏移：

\[
\theta_{sampler}=\theta_{trainer}+\Delta\theta,
\qquad
\Delta\theta\sim\mathcal N(0,\sigma^2I).
\]

使用三档噪声，逐渐增加 TIM。结果：小噪声下强方法差异不明显；噪声增大后 drift 累积并提前 collapse。最大噪声下只有 SC 及 SC+TIS/MIS 持续稳定；TIS、MIS 在较大噪声下也会崩溃。

### 3.3 量化 mismatch

只量化 sampler，trainer 保持 bf16，分别测试 sampler 的权重、activation、KV cache 量化组合，包括 FP8/INT8 权重和 activation、FP8/FP4 或 INT4 KV cache。结果：SC 单独或与 TIS/MIS 组合通常最稳；PPO/DAPO 的 ratio clip 对 policy staleness 有效，但不能处理纯数值量化误差。

在 Qwen3-30B-A3B-Base 上，FP8 sampler 尚可由 vanilla PG 稳定训练；加入更激进的 FP4 KV 后 PG 很快 collapse，而 SC 约达到 52%、TIS 约 51%；INT8 sampler + INT4 KV 时 SC 约 30%，TIS 约 12%，其他方法低于 5%。MoE 实验未 replay router indices，因此还包含 trainer/sampler expert routing 差异。

### 3.4 Staleness mismatch

sampler 每 64 steps 才同步一次，构造严重异步 rollout。结果：SC 能稳定训练，但 SC+TIS 或 SC+MIS 最佳。这验证 SC 消除漂移、IS 修正 sampler/trainer 分布差异的互补性；当 trainer 在同步间隔内移动很远时，单独 SC 的 covariance 仍在 sampler 分布下估计，组合方法更有优势。

### 3.5 消融与规模验证

附录比较 full score centering 与 top-k 近似（\(k=32,128\)），并验证组合 IS 后应对加权 score 做 centering。总体结论是：top-k 近似几乎不损失效果；轻微 TIM 下 SC 与 IS 接近，严重 quantization 下 SC 明显更稳，严重 staleness 下 TIS/MIS + SC 最好。

## 4. 局限与对 SkyRL 的启示

1. SC 消除了 drift，但不完全恢复训练分布；严重 staleness 时需要与 IS 组合。
2. 精确 SC 需要 sampler 的 next-token 分布，工程实现依赖 rollout engine 暴露 top-k logprobs。
3. 论文 headline 结果使用了刻意放大的 mismatch，代表“短实验中可观察的严重 TIM”，不能直接等同于所有生产配置。
4. 对 SkyRL，最直接的接口要求是 rollout 记录 sampled token 的 top-k logprobs、top-k ids 和 trainer logprobs；训练侧按 prefix 计算 \(\rho\) 与 correction loss，并确保 token、advantage/loss mask 严格对齐。
5. RWKV/其他非标准推理后端若存在 sampler-trainer 数值差异，应优先记录 top-k 分布并验证 SC；异步权重同步时建议 SC + 合适的 IS，而不是只依赖 PPO-style clipping。

## 链接

- 论文 Hugging Face 页面：[https://huggingface.co/papers/2609.20807v1](https://huggingface.co/papers/2609.20807v1)
- 论文 arXiv：[https://arxiv.org/abs/2609.20807](https://arxiv.org/abs/2609.20807)
- 论文 PDF：[https://arxiv.org/pdf/2609.20807](https://arxiv.org/pdf/2609.20807)
- 官方代码仓库：[https://github.com/martin-marek/score-centering](https://github.com/martin-marek/score-centering)

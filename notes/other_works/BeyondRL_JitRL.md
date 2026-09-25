# Beyond RL 与 JitRL：把优化/适应改写为采样与推理时策略更新

> 阅读对象：Shengyu Feng，*A Gallery of Methods Beyond RL — Part I: Sampling Methods*；Yibo Li 等，*Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates*。前者是一篇方法综述/视角文章，后者提出了具体的无梯度 test-time learning 算法 JitRL。

## 1. 一句话总结

两者共享一个方向：**冻结基础模型，把训练时的参数更新转移成推理时的分布重加权、候选搜索或记忆驱动的策略更新**。

- Feng 的文章将“寻找高质量输出”写成从 reward/objective tilted distribution 中采样；重要性采样、Twisted SMC 和 MCMC 分别对应一次性重加权、逐步粒子筛选和局部链式搜索。
- JitRL 则针对交互式 agent 的 continual learning：保存历史轨迹，从相似状态检索经验，估计 action advantage，并将 advantage 直接加到冻结 LLM 的 action logits 上，不做反向传播或权重更新。

因此，前者主要回答“如何在固定模型上更好地找样本”，后者主要回答“如何用在线经验在固定模型上即时改变 action 分布”。

## 2. Feng：Beyond RL 的采样视角

### 2.1 从优化目标到能量分布

设候选解为 `x`，基础分布为 `p_0(x)`，需要最小化的 objective 为 `g(x)`，温度/尺度为 `β>0`。定义目标分布：

\[
p_\beta(x)=\frac{1}{Z_\beta}p_0(x)\exp\left(-\frac{g(x)}{\beta}\right),
\qquad
Z_\beta=\int p_0(x)\exp\left(-\frac{g(x)}{\beta}\right)dx.
\]

若用 reward `r=-g` 表示，则：

\[
p_\beta(x)\propto p_0(x)\exp\left(\frac{r(x)}{\beta}\right).
\]

高 reward 解的概率更大，但仍保留基础模型的先验；`β` 越小，分布越尖锐，越接近只寻找最优解。核心转变是：**不一定要学习一个新参数化策略，只要能从这个目标分布采样，就能在推理时获得更好的解。**

### 2.2 Importance Sampling

从易采样的 proposal `q(x)` 中取得 `x^i`，再赋予：

\[
w^i\propto\frac{p_0(x^i)\exp(-g(x^i)/\beta)}{q(x^i)}.
\]

归一化后重采样；`Z_β` 不需要知道。如果 `q=p_0`，则：

\[
w(x)\propto \exp(-g(x)/\beta)=\exp(r(x)/\beta).
\]

优点是实现简单、无需更新模型；缺点是只在完整 trajectory 结束后修正。如果 proposal 与目标分布差距大，尤其是长 reasoning sequence 中，权重会退化为“少数样本占据几乎全部质量”，有效样本数低、方差高。

### 2.3 Twisted Sequential Monte Carlo

将完整序列拆成前缀逐步生成。维护粒子集合 `x_{1:t}^{(i)}`，每一步：

1. 用基础模型扩展每个粒子；
2. 用 twist/value function 估计当前前缀未来可能获得的质量；
3. 按增量权重重加权；
4. 重采样，及时淘汰低潜力前缀；
5. 继续生成直到结束。

目标仍是 reward-tilted 分布，但 credit/selection 被分散到各个时间步，而不是最后一次性处理。可用：

\[
\tilde w_t^{(i)}
\approx \frac{\text{target prefix density}\times\text{future-quality estimate}}
{\text{proposal prefix density}}.
\]

直觉上，twist function 近似“从当前 prefix 出发的未来 reward”，所以 SMC 能在长轨迹中改善 importance sampling 的权重退化。代价是需要额外的 value/twist 估计、粒子复制和 resampling。

### 2.4 MCMC / Metropolis-Hastings

构造一个以 `p_β(x)` 为 stationary distribution 的 Markov chain。对当前序列 `x` 提议 `x'`，接受概率为：

\[
\alpha(x,x')=\min\left(1,
\frac{p_\beta(x')q(x\mid x')}{p_\beta(x)q(x'\mid x)}\right).
\]

在 LLM reasoning 中，一个自然 proposal 是随机保留前缀、用基础模型重新生成 suffix，再按上式接受/拒绝。这样模型参数不变，但 sequence 会经过多轮局部重写，逐渐接近 sharpened distribution（例如 `p_0(x)^α` 或 reward-tilted distribution）。

若 proposal 能利用梯度，可采用 Langevin 风格更新：

\[
x'=x+\eta\nabla_x\log p_\beta(x)+\zeta,
\qquad \zeta\sim\mathcal N(0,\sigma^2I),
\]

但离散 token 空间通常需要在连续表示或 logits 空间设计近似 proposal。

### 2.5 与 RL 的关系

RL 通过梯度更新 `π_θ`，使高 reward trajectory 在未来更容易被采样；采样方法直接在当前 `π_0` 上花更多 inference compute，从 target distribution 中找到高 reward trajectory。两者可以组合：采样结果可用于蒸馏/训练，RL policy 也可作为更好的 proposal。

## 3. JitRL 的问题设定与核心流程

### 3.1 问题

LLM agent 在部署中持续遇到相似任务，但传统 continual RL 需要收集数据、反向传播、更新 checkpoint，成本和延迟都高，而且在线更新存在灾难性遗忘风险。JitRL 将 adaptation 放到每次 action selection：

- 冻结 base LLM `π_θ`；
- 维护跨 episode 的非参数 memory；
- 根据当前 state 检索相似历史经验；
- 用历史 return 估计 `Q`、`V` 和 advantage；
- 把 advantage 加到当前 action logits；
- episode 完成后评分并将新经验写回 memory。

伪代码：

```text
初始化冻结 LLM πθ 和跨 episode memory M
for episode:
    for t=0,...,T-1:
        s_t = observe()
        D_t = retrieve(M, s_t)
        A_hat(s_t, ·) = estimate_advantage(D_t)
        z'_t(·) = z_θ(s_t, ·) + β A_hat(s_t, ·)
        a_t ~ softmax(z'_t)
        execute(a_t)
    score episode / steps
    计算 discounted returns，写入 M
```

## 4. JitRL 数学表示

### 4.1 记忆与 return

对轨迹中的 state-action 对保存 `(s_i,a_i,G_i)`，其中：

\[
G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u.
\]

用 n-gram Jaccard 等相似度检索当前状态附近的历史样本。记检索到的 state 集合为 `N(s)`，检索到相同 action 的集合为 `N(s,a)`，则：

\[
\widehat V(s)=\frac{1}{|\mathcal N(s)|}\sum_{i\in\mathcal N(s)}G_i,
\qquad
\widehat Q(s,a)=\frac{1}{|\mathcal N(s,a)|}\sum_{j\in\mathcal N(s,a)}G_j.
\]

优势估计：

\[
\widehat A(s,a)=\widehat Q(s,a)-\widehat V(s).
\]

实现中需要处理没有匹配 action、样本数过少和 advantage 尺度不稳定等情况；通常需要平滑、归一化或截断。

### 4.2 KL 正则策略改进的闭式解

给定冻结策略 `π_θ` 和检索得到的 advantage，考虑：

\[
\pi^*=\arg\max_{\pi'}\left\{
\mathbb E_{a\sim\pi'(\cdot\mid s)}[\widehat A(s,a)]
-\frac{1}{\beta}D_{\mathrm{KL}}(\pi'(\cdot\mid s)\|\pi_\theta(\cdot\mid s))
\right\}.
\]

对 `π'` 做变分优化，得到：

\[
\boxed{
\pi^*(a\mid s)=
\frac{\pi_\theta(a\mid s)\exp(\beta\widehat A(s,a))}
{\sum_{a'}\pi_\theta(a'\mid s)\exp(\beta\widehat A(s,a'))}
}
\]

若 base LLM logits 为 `z_θ(s,a)`，由于 softmax 会吸收状态相关归一化项：

\[
\boxed{z'(s,a)=z_\theta(s,a)+\beta\widehat A(s,a)}.
\]

因此 JitRL 不是 policy-gradient：它是一个**检索驱动的 exponentiated-advantage / logit reweighting**。`β` 是 test-time adaptation 强度：过大可能过度相信噪声经验，过小则适应不明显。

如果模型只提供黑盒 API，可将候选 action 的置信度/评分转换成近似 logits；若能取得 token/action logprobs，则可直接在候选 action 空间进行更新。实现还可以把 memory 中的历史动作并入当前候选集，避免 base model 概率极低但经验上有效的 action 被完全漏掉。

## 5. 两篇材料的统一理解

| 方面 | Beyond RL 采样方法 | JitRL |
|---|---|---|
| 被优化对象 | reward-tilted sequence distribution | 当前 state 下的 action distribution |
| 反馈来源 | 完整 reward/objective，或 twist/value | 历史 trajectory 的 return/advantage |
| 更新方式 | importance weighting、resampling、MCMC moves | 直接修改 logits |
| 是否更新参数 | 否 | 否 |
| 计算预算 | 多候选、粒子扩展、链式重写 | retrieval、action scoring、每步 logit adjustment |
| 主要风险 | weight degeneracy、mixing 慢、计算量大 | stale/偏置 memory、检索错误、advantage 噪声 |

可以将 JitRL 看成一种局部、在线、记忆增强的 inference-time policy improvement；而 TSMC/MCMC 更像全局的 trajectory-level inference-time search。一个可能的组合是：用 JitRL 的 advantage 作为 proposal/twist 信号，再用粒子搜索或局部 MCMC 扩展 action/trajectory。

## 6. 实验设计

### 6.1 JitRL 评测环境

论文在两类长期交互任务上评测：

1. **WebArena / WebArena-Lite**：网页导航与操作任务，使用自动 evaluator 判断任务成功；报告跨 episode 的平均和最终 success rate。
2. **Jericho**：文本交互式游戏，至少包含 Library、Zork1、Zork3；报告平均/最终游戏得分。

比较对象包括：

- Static：冻结模型、无历史经验；
- Memory：加入记忆但不做完整 JitRL advantage/logit 更新；
- Reflexion：用反思文本改进后续行为；
- AWM、EvoTest 等 training-free/test-time 方法；
- GRPO、SFT、WebRL 等训练型方法（在可比设置下作为参考）。

控制变量应包括相同 base model、相同任务顺序、相同 episode 数、相同 action budget 和相同评测器。关键指标不应只看一次任务成功率，还应看：

- 随 episode/时间的平均 reward 与最终 reward；
- success/score 的跨 episode 曲线（continual adaptation 速度）；
- 相对 Static 的增益；
- memory 检索和 action scoring 的额外延迟与 API 成本；
- 是否出现错误经验累积、过度 exploitation 或跨任务负迁移。

### 6.2 论文报告的主结果

WebArena 主表报告的 success rate（Avg / Final）为：

| 方法 | Avg | Final |
|---|---:|---:|
| Static | 35.63 | 36.30 |
| Memory | 41.36 | 43.00 |
| Reflexion | 41.08 | 42.12 |
| AWM | 39.37 | 40.32 |
| EvoTest | 39.24 | 42.49 |
| **JitRL** | **46.98** | **51.35** |

WebArena-Lite 的最终成功率报告为 JitRL 60.00%、WebRL 46.06%、SFT 23.00%。Jericho 上，JitRL 在 Library、Zork1、Zork3 的 Avg / Final 分别为：

| 游戏 | JitRL Avg / Final |
|---|---:|
| Library | 25.9 / 30 |
| Zork1 | 53.0 / 69 |
| Zork3 | 3.1 / 5 |

论文还报告相对于其他 training-free 方法以及部分 fine-tuning 方法的优势，并声称由于不训练模型，货币成本可降低 30 倍以上。具体复现实验时应以论文/代码仓库当前版本的配置和表格为准。

### 6.3 建议的消融实验

为了验证机制而非只验证总效果，至少应做：

- `β` 扫描：适应强度与稳定性的 trade-off；
- 无 `Q-V` baseline：只用 `Q` 或 raw return，验证 advantage centering 的价值；
- 不同检索器：n-gram Jaccard、embedding、精确 state/action 匹配；
- memory size / top-k 检索数；
- 是否将 memory action 并入候选集；
- token-level、action-level、trajectory-level 的 logit 调整；
- 无跨 episode memory、只保留当前 episode；
- reward 噪声、错误 evaluator 和分布变化下的鲁棒性；
- 与 best-of-N、importance sampling、TSMC/MCMC 的 inference compute-matched 对比。

## 7. 对 SkyRL/RWKV 的启发

- JitRL 的 memory、retrieval 和 advantage estimator 可以作为 Generator/Environment 外侧的 test-time adaptation 层，不要求修改 RWKV 权重。
- 若 RWKV/vLLM 接口能返回 token/action logprob，可把 `β A` 注入候选 action logits；若只能返回序列概率，需要在 action 边界处做聚合。
- 采样视角提示：rollout 不一定只能“采一次然后训练”，也可以把算力用在粒子扩展、prefix resampling 或 trajectory rewrite；这适合研究 rollout-time scaling 与 RL 的 trade-off。
- 必须记录 token、action、return、logprob 的严格对齐关系；memory 经验若来自不同 policy，需要保存 policy/version、时间戳和 evaluator 版本，避免 stale experience 造成错误 advantage。

## 8. 链接

### 论文与代码

- [Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates（arXiv:2601.18510）](https://arxiv.org/abs/2601.18510)
- [JitRL 代码仓库](https://github.com/liushiliushi/JitRL)
- [JitRL OpenReview 版本](https://openreview.net/pdf?id=us2YPNouOm)

### Beyond RL 与相关实现

- [A Gallery of Methods Beyond RL — Part I: Sampling Methods](https://shengyu-feng.github.io/blog/2026/08/27/beyond-RL/)
- [TSMC4MATH：Twisted Sequential Monte Carlo for math reasoning](https://github.com/Shengyu-Feng/TSMC4MATH)
- [Twisty：Twisted Sequential Monte Carlo for Language Models](https://github.com/smahsramo/twisty)
- [RLD4CO：Regularized Langevin Dynamics for Combinatorial Optimization](https://github.com/Shengyu-Feng/RLD4CO)

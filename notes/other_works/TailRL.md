# TailRL：Tail-Likelihood Reinforcement Learning

> 阅读对象：Shrinivas Ramasubramanian 等，*Tail-Likelihood Reinforcement Learning*，arXiv:2609.02987v1（2026-09-02）。方法名为 **TailRL**。

## 1. 一句话总结

TailRL 不再只最大化一次采样的平均回报，而是对所有回报阈值的**上尾概率**同时做最大似然：

\[
\boxed{
\text{continuous reward}
\;\longrightarrow\;
\text{tail events at every threshold}
\;\longrightarrow\;
\text{log tail-likelihood}
\;\longrightarrow\;
\text{harmonic mixture of Best-of-}k
}
\]

它的直接实现也很简单：对同一个输入采样一组 rollout，按 reward 排序；每一段 reward gap 由仍然超过该阈值的 rollout 平分，得到新的 rollout-level advantage，再把该 advantage 接入原有的 REINFORCE/GRPO/RLOO 等 policy-gradient pipeline。

核心收益是：**越稀有但越高的 reward，得到的梯度权重越大**。因此策略不会只收敛到容易获得的中等质量解，而会保留并放大通向稀有高质量解的概率质量；这与部署时的 Best-of-\(k\)、Pass@\(k\) 和 test-time sampling scaling 更一致。

## 2. 问题背景与核心直觉

### 2.1 平均回报与 Best-of-many 部署目标不一致

设输入为 \(x\)，policy 为 \(\pi_\theta(z\mid x)\)，一次 rollout 为 \(z\)，标量 reward 为 \(r(x,z)\)。普通 RL 优化：

\[
J_{\mathrm{RL}}(\theta;x)
=\mathbb E_{z\sim\pi_\theta(\cdot\mid x)}[r(x,z)].
\]

这只关心 reward 分布的均值。但生成式策略在训练和推理时经常会采样多次，并从候选中选出最好的一条：

\[
\mathrm{Best\text{-}of\text{-}k}(\theta;x)
=\mathbb E\left[\max_{i\le k}r(x,z_i)\right],
\qquad z_i\overset{\mathrm{i.i.d.}}{\sim}\pi_\theta(\cdot\mid x).
\]

两个策略可能有相同的 mean reward，却有完全不同的高 reward 尾部：

- 策略 A：大多数 rollout 都是稳定的中等质量解；
- 策略 B：平均值相近，但小概率产生非常好的解。

当推理时增加采样数时，策略 B 的 Best-of-\(k\) 增长会更快。只优化 mean reward 可能让策略收缩到“容易得到的安全解”，从而丢失本来可以由更多 inference samples 找到的稀有好解。

### 2.2 把连续 reward 看作一族二值成功事件

对每个阈值 \(\tau\)，定义事件

\[
E_\tau=\{r(x,z)>\tau\}.
\]

一个连续 reward 因而对应一族二值任务：策略是否能超过 \(\tau\)？TailRL 对每个 \(\tau\) 都优化该成功事件的 log-likelihood，而不是只选一个固定阈值或一个固定的 Best-of-\(k\) 预算。

直觉上，普通 expected-reward RL 是对所有阈值的 tail probability 做算术平均；TailRL 则做几何平均。几何平均对很小的概率更敏感，所以高 reward、低覆盖率的区域不会被中等 reward 区域淹没。

### 2.3 与已有目标的关系

- **Expected-reward RL：** 只对应 \(k=1\) 的 Best-of-\(k\) 目标。
- **MaxRL：** 当 reward 是二值时，TailRL 精确退化为最大化成功概率的 log-likelihood。
- **PKPO：** 选择一个预先指定的 \(k_{\mathrm{opt}}\)，直接优化 Best-of-\(k_{\mathrm{opt}}\)；TailRL 不需要选择单一预算，而是同时覆盖所有预算。
- **GRPO/RLOO：** 它们主要改变 baseline/归一化方式；在同一个输入内部，通常仍按原始 reward 做相对加权。TailRL 改变的是 reward distribution 对 policy gradient 的目标权重。
- **不是额外的 entropy/diversity regularizer：** TailRL 不额外添加“保持多样性”的正则项，而是直接改变优化目标，使稀有高 reward 的 tail event 具有更大的边际价值。

## 3. 数学表示

### 3.1 Tail probability 与 expected reward

以下先假定 \(r(x,z)\in[0,1]\)。定义阈值 \(\tau\in[0,1]\) 处的上尾概率：

\[
p_\theta(x,\tau)
:=\Pr_{z\sim\pi_\theta(\cdot\mid x)}
\bigl(r(x,z)>\tau\bigr).
\]

利用 layer-cake 表示，expected reward 等于 tail-probability 曲线下的面积：

\[
\boxed{
J_{\mathrm{RL}}(\theta;x)
=\mathbb E[r(x,z)]
=\int_0^1p_\theta(x,\tau)\,d\tau.
}
\]

因此 expected reward 对每个阈值的 tail probability 使用相同的线性权重。

### 3.2 TailRL 的 population objective

TailRL 在每一个阈值上应用 MaxRL 的 log-likelihood 原则，并对均匀采样的阈值取期望：

\[
\boxed{
J_{\mathrm{TailRL}}(\theta;x)
:=\int_0^1\log p_\theta(x,\tau)\,d\tau
=\mathbb E_{\tau\sim\mathrm{Unif}[0,1]}
\left[\log p_\theta(x,\tau)\right].
}
\]

也可以写成：

\[
\exp\bigl(J_{\mathrm{TailRL}}(\theta;x)\bigr)
=\exp\left(\int_0^1\log p_\theta(x,\tau)d\tau\right),
\]

即所有 tail probabilities 的几何平均（严格来说是连续阈值上的几何平均）。当某个高 reward 阈值非常难达到时，\(p_\theta(x,\tau)\) 很小，\(\log p_\theta(x,\tau)\) 及其梯度会显著关注该区域。

### 3.3 “随机阈值审计”的解释

可以把 TailRL 看成对策略进行连续的 threshold audit。取 \(L\) 个覆盖 reward 区间的阈值 \(\tau_1,\ldots,\tau_L\)，对每个阈值独立采样一个 rollout，并要求所有事件同时成功：

\[
E_L=\bigcap_{\ell=1}^L\{r(x,z_\ell)>\tau_\ell\}.
\]

由于各 rollout 独立：

\[
\Pr_\theta(E_L\mid x)
=\prod_{\ell=1}^L p_\theta(x,\tau_\ell).
\]

使用每个审计阈值的平均 log-likelihood：

\[
\frac1L\log\Pr_\theta(E_L\mid x)
=\frac1L\sum_{\ell=1}^L\log p_\theta(x,\tau_\ell).
\]

当阈值分辨率趋于无穷时，该 Riemann sum 变为：

\[
J_{\mathrm{TailRL}}(\theta;x)
=\lim_{L\to\infty}\frac1L\log\Pr_\theta(E_L\mid x)
=\int_0^1\log p_\theta(x,\tau)d\tau.
\]

这说明 TailRL 不是任意加权 reward，而是在所有质量等级上要求策略维持覆盖。

### 3.4 TailRL policy gradient：逆 tail-probability 加权

对 population objective 求梯度：

\[
\nabla_\theta J_{\mathrm{TailRL}}(\theta;x)
=\int_0^1
\frac{1}{p_\theta(x,\tau)}
\nabla_\theta p_\theta(x,\tau)\,d\tau.
\]

定义 score function：

\[
S_\theta(x,z):=\nabla_\theta\log\pi_\theta(z\mid x).
\]

将 \(\nabla p_\theta\) 写成 score-function 形式，可得：

\[
\boxed{
\nabla_\theta J_{\mathrm{TailRL}}(\theta;x)
=\mathbb E_{z\sim\pi_\theta}
\left[
\left(
\int_0^{r(x,z)}
\frac{d\tau}{p_\theta(x,\tau)}
\right)
S_\theta(x,z)
\right].
}
\]

因此一次 rollout 的理论权重是：

\[
w_\theta(x,z)
:=\int_0^{r(x,z)}\frac{d\tau}{p_\theta(x,\tau)}.
\]

它有两个重要性质：

1. reward 越高，积分覆盖的阈值范围越大；
2. 被策略很少达到的阈值具有更大的 \(1/p_\theta\) 权重。

所以 TailRL 会特别强化“高且稀有”的 rollout，而不是简单地把 reward 本身作为 advantage。

### 3.5 与 Best-of-\(k\) 的调和分解

对 \(k\) 个独立 rollout，Best-of-\(k\) reward 可以写成：

\[
\begin{aligned}
B_k(\theta;x)
&:=\mathrm{Best\text{-}of\text{-}k}(\theta;x)\\
&=\int_0^1\left[1-(1-p_\theta(x,\tau))^k\right]d\tau.
\end{aligned}
\]

利用

\[
-\log p=\sum_{k=1}^{\infty}\frac{(1-p)^k}{k},
\]

可以得到论文的核心定理：

\[
\boxed{
J_{\mathrm{TailRL}}(\theta;x)
=\sum_{k=1}^{\infty}
\frac{B_k(\theta;x)-1}{k}
}
\]

以及在相应正则条件下：

\[
\boxed{
\nabla_\theta J_{\mathrm{TailRL}}(\theta;x)
=\sum_{k=1}^{\infty}
\frac1k
\nabla_\theta B_k(\theta;x).
}
\]

因此 TailRL 的梯度是所有 Best-of-\(k\) 梯度的 harmonic mixture：

- \(k=1\) 的 expected reward 梯度占据最直接的一项；
- 更大的 \(k\) 对应更强的 inference-time selection；
- 权重 \(1/k\) 由 log 的级数展开自动产生，不是额外调出来的超参数。

这给出了训练目标与部署目标之间的直接联系：TailRL 同时为多种 inference budget 优化，而不是训练时先决定一个目标 \(k\)。

### 3.6 二值 reward 时精确退化为 MaxRL

若 \(r(x,z)\in\{0,1\}\)，令成功概率为：

\[
q_\theta(x):=
\Pr_{z\sim\pi_\theta(\cdot\mid x)}(r(x,z)=1).
\]

所有非平凡阈值 \(\tau\in[0,1)\) 都对应同一个成功事件，因此：

\[
p_\theta(x,\tau)=q_\theta(x),
\]

从而：

\[
\boxed{
J_{\mathrm{TailRL}}(\theta;x)
=\int_0^1\log q_\theta(x)d\tau
=\log q_\theta(x)
=J_{\mathrm{MaxRL}}(\theta;x).
}
\]

此时 \(B_k\) 就是 Pass@\(k\)，上面的 Best-of-\(k\) 分解也变为 MaxRL 的 harmonic Pass@\(k\) 分解。

## 4. 有限 rollout 下的可执行估计器

population objective 需要知道所有阈值上的真实 \(p_\theta(x,\tau)\)，实际训练只能拿到有限个 rollout。论文的关键处理是：**\(N\) 个 rollout 不只是降低方差，还自然定义了 TailRL 的 order-\(N\) 截断目标。**

### 4.1 Order-\(T\) 截断目标

定义：

\[
J_{\mathrm{TailRL}}^{(T)}(\theta;x)
:=\sum_{k=1}^{T}
\frac{B_k(\theta;x)-1}{k}.
\]

其梯度为：

\[
\nabla_\theta J_{\mathrm{TailRL}}^{(T)}
=\sum_{k=1}^{T}\frac1k\nabla_\theta B_k.
\]

等价的 tail-probability 形式为：

\[
\boxed{
\nabla_\theta J_{\mathrm{TailRL}}^{(T)}
=\int_0^1
\frac{1-(1-p_\theta(x,\tau))^T}
{p_\theta(x,\tau)}
\nabla_\theta p_\theta(x,\tau)d\tau.
}
\]

这个权重随 \(T\) 变化：

\[
\frac{1-(1-p)^T}{p}
\begin{cases}
=1,&T=1,\\
\to 1/p,&T\to\infty.
\end{cases}
\]

所以：

- \(T=1\)：恰好是 expected-reward RL；
- \(T\) 越大：越强调难达到的高 reward 阈值；
- \(T\to\infty\)：趋近 population TailRL。

### 4.2 基于 reward 排序的 rollout 权重

对同一输入采样 \(N\) 个 rollout：

\[
z_1,\ldots,z_N\overset{\mathrm{i.i.d.}}{\sim}\pi_\theta(\cdot\mid x),
\qquad r_i=r(x,z_i).
\]

对第 \(i\) 个 rollout 定义：

\[
\boxed{
\omega(r_i)
:=\int_0^{r_i}
\frac{d\tau}
{\sum_{j=1}^N\mathbf 1\{r_j>\tau\}}.
}
\]

含义是：在每一个阈值 \(\tau\) 上，把一单位 credit 平分给所有超过该阈值的 rollout；一个阈值上幸存的 rollout 越少，每个幸存 rollout 分到的 credit 越多。

令 reward 排序为：

\[
r_{(1)}\le r_{(2)}\le\cdots\le r_{(N)},
\qquad r_{(0)}:=0.
\]

则可通过一次排序和累加精确计算：

\[
\boxed{
\omega(r_{(i)})
=\omega(r_{(i-1)})
+\frac{r_{(i)}-r_{(i-1)}}{N-i+1},
\qquad \omega(r_{(0)})=0.
}
\]

最终的 critic-free score-function estimator 为：

\[
\boxed{
 g_{\mathrm{TailRL}}^{(N)}(x)
=\sum_{i=1}^N
\omega(r_i)\,S_\theta(x,z_i).
}
\]

论文证明：

\[
\mathbb E\left[g_{\mathrm{TailRL}}^{(N)}(x)\right]
=\nabla_\theta J_{\mathrm{TailRL}}^{(N)}(\theta;x).
\]

也就是说，有限 rollout estimator 对 order-\(N\) 目标是无偏的；其计算复杂度主要是 reward 排序的 \(O(N\log N)\)。

### 4.3 Centered advantage

实际实现中对组内权重做中心化：

\[
\bar\omega=\frac1N\sum_{j=1}^N\omega(r_j),
\qquad
\boxed{A_i=\omega(r_i)-\bar\omega.}
\]

然后把 \(A_i\) 广播到该 rollout 的 response tokens 上，作为 policy-gradient advantage。

中心化的作用：

- 降低组内梯度方差；
- 可以直接替换 GRPO/RLOO pipeline 中的 advantage calculation；
- 不需要 learned critic；
- 不改变 rollout、模型结构和 policy loss 的主体形式。

因为 baseline 是从同一组样本估出来的，中心化 estimator 的严格无偏目标是 order-\((N-1)\) 的梯度，而不是未中心化 estimator 对应的 order-\(N\) 梯度。实践中这通常是可接受的 variance-reduction trade-off。

### 4.4 与普通 group advantage 的差别

普通 reward-centered 方法通常形如：

\[
A_i^{\mathrm{standard}}
= r_i-\frac1N\sum_j r_j
\quad\text{或其标准化版本}.
\]

TailRL 则是：

\[
A_i^{\mathrm{TailRL}}
=\left[
\sum_{j=1}^{\operatorname{rank}(i)}
\frac{r_{(j)}-r_{(j-1)}}{N-j+1}
\right]-\bar\omega.
\]

差异不在于是否有 baseline，而在于 reward 到 advantage 的映射：

- 普通 RL：所有 reward level 的 tail gradient 线性同权；
- TailRL：高 reward 区间按“该区间有多少 rollout 能达到”反比分摊 credit；
- 高 reward 且只有少数 rollout 达到时，会产生明显更大的相对 advantage。

### 4.5 一个统一的 tail-objective 视角

可定义：

\[
J_\phi(\theta;x)
=\int_0^1\phi\bigl(p_\theta(x,\tau)\bigr)d\tau.
\]

则：

\[
\nabla_\theta J_\phi
=\int_0^1\phi'\bigl(p_\theta(x,\tau)\bigr)
\nabla_\theta p_\theta(x,\tau)d\tau.
\]

不同目标只是在 tail gradient 上使用不同的边际权重：

| 目标 | \(\phi(p)\) | tail-gradient 权重 \(\phi'(p)\) |
|---|---:|---:|
| Expected reward | \(p\) | \(1\) |
| TailRL population | \(\log p\) | \(1/p\) |
| TailRL order-\(T\) | \(-\sum_{\ell=1}^{T}(1-p)^\ell/\ell\) | \([1-(1-p)^T]/p\) |
| PKPO | \(1-(1-p)^{k_{\mathrm{opt}}}\) | \(k_{\mathrm{opt}}(1-p)^{k_{\mathrm{opt}}-1}\) |

这张表也说明了 TailRL 和 PKPO 的区别：PKPO 的目标在某个选定的 \(k_{\mathrm{opt}}\) 附近集中，而 TailRL 的权重由所有阈值和所有 sampling budgets 共同决定。

## 5. 训练流程与伪代码

### 5.1 Drop-in 训练流程

```text
for each input x:
    sample N independent rollouts z_1, ..., z_N from π_θ
    evaluate scalar rewards r_1, ..., r_N

    sort rewards: r_(1) <= ... <= r_(N)
    ω_(0) = 0
    for i = 1, ..., N:
        ω_(i) = ω_(i-1) + (r_(i) - r_(i-1)) / (N - i + 1)

    map ω_(i) back to the original rollout order
    A_i = ω_i - mean_j(ω_j)
    broadcast A_i to valid response tokens

    run the existing policy-gradient loss with A_i
    perform the optimizer update
```

在 LLM 训练中，若 rollout \(z_i=(y_{i,1},\ldots,y_{i,T_i})\)，一个简化的 sequence-level policy-gradient loss 可以写为：

\[
\mathcal L_{\mathrm{TailRL}}
=-\frac1N\sum_{i=1}^N A_i
\sum_{t=1}^{T_i}
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}),
\]

实际实现还需要沿用原训练框架的 response mask、token reduction、ratio/clipping 或 KL 约束；TailRL 主要替换的是 rollout-level advantage 的计算。

### 5.2 重要实现性质

- 不需要 value critic；
- 不需要额外的 reward model 或 reference model；
- 不需要预先选择 threshold；
- 不需要预先选择唯一的 inference budget \(k\)；
- 不修改模型架构；
- 可以作为现有 group-based policy optimization 的 advantage estimator；
- rollout group 必须对应同一个输入，否则“同一 reward distribution 的尾部”没有意义；
- reward 最好有连续的 ordinal 信息；如果 reward 几乎只有一个二值层级，方法退化到 MaxRL/其有限版本。

## 6. 实验设计

论文用四个互补设置检验四个问题：

1. **目标是否正确：** population TailRL 是否能利用标量 reward 达到甚至超过使用 ground-truth label 的 supervised objective？
2. **有限估计是否正确：** 随着 rollout group 变大，order-\(N\) estimator 是否接近 population objective？
3. **稀有成功是否能被 bootstrapping：** 当初始策略几乎找不到高 reward 轨迹时，TailRL 能否从少量 rare successes 学习？
4. **部署时 scaling 是否更好：** 训练后的策略是否能从更多 inference samples 中持续获益，而不是早早饱和或坍缩到安全次优解？

总体对比方法主要包括：

- **GRPO**：组内相对 reward 的 critic-free policy optimization；
- **RLOO**：leave-one-out baseline；
- **PKPO**：选择固定 \(k_{\mathrm{opt}}\) 优化 Best-of-\(k_{\mathrm{opt}}\)；
- **TailRL**：population 形式或有限 rollout estimator。

除 advantage/目标估计方式外，各任务尽量保持模型、数据、优化和 rollout 条件一致，以区分“目标函数收益”和额外模型/训练成本。

### 6.1 ImageNet Object Localization

**任务与策略。**

- 输入：ImageNet 图像；
- 输出：目标物体的 bounding box，不要求输出类别；
- backbone：ResNet-50；
- policy head：四个 categorical heads，分别采样 box center \((\hat x_c,\hat y_c)\)、width \(\hat w\)、height \(\hat h\)；
- rollout：一个采样出的 bounding box；
- reward：预测框与 ground-truth box 的 IoU，属于 \([0,1]\) 的连续标量。

**实验问题。**

1. 只给 scalar IoU 的 RL，能否接近直接看 ground-truth coordinates 的 supervised learning？
2. 随着训练 rollout 数 \(N\) 增加，有限 TailRL 是否接近可以精确计算的 population TailRL？
3. 在相同或更小的训练 rollout budget 下，TailRL 是否优于 expected-reward baselines？

**对比与预算。**

- supervised：L1、GIoU、L1+GIoU，其中 L1+GIoU 是最强的 supervised reference；
- TailRL：训练 group size \(N\in\{16,64,256,1024\}\)；
- GRPO/RLOO：主要使用 \(N=1024\)；
- PKPO：用固定 \(k_{\mathrm{opt}}\) 的 Best-of-\(k\) 目标；
- 因为 box action space 可枚举/可计算，population TailRL 可以作为 exact objective 直接训练和比较。

**评测指标。**

- \(\mathrm{CorLoc}@0.5\)：greedy box 的 IoU 超过 0.5 的图像比例；
- \(\mathrm{CorLoc}@0.75\)；
- mean IoU；
- inference 时 \(\mathrm{Best\text{-}of\text{-}k\) IoU，例如 Best-of-1024；
- gradient-level convergence：有限 estimator 与 exact population gradient 的接近程度。

**主要观察。**

- population TailRL 在 CorLoc@0.5 和 CorLoc@0.75 上高于 L1+GIoU，同时 mean IoU 可比；只用 scalar IoU 也可以达到甚至超过使用 ground-truth box 的监督目标；
- 增大训练 group size 后，有限 TailRL 曲线按预期逐步靠近 population objective；\(N=1024\) 时最接近；
- 在 matched \(N=1024\) 下，TailRL 在 CorLoc、mean IoU 和 Best-of-1024 等指标上优于 GRPO/RLOO；
- 即使 TailRL 只用 \(N=16\)，也能在 CorLoc@0.5 和 mean IoU 上超过用 \(N=1024\) 训练的 expected-reward baselines，说明收益不只是降低估计方差；
- 将连续 IoU 粗暴二值化会丢失 ordinal 信息：论文的 binary-reward 对照显示，TailRL 能得到更强的高 IoU 覆盖，而固定阈值的 MaxRL 只会关注一个成功层级。

### 6.2 Text-Maze：从低成功率初始化开始学习

**任务与初始化控制。**

- 环境：用文本表示的 \(17\times17\) maze；
- rollout：模型生成描述路径的 token sequence；
- reward：结合到达目标的接近程度、路径长度与最短路径关系；
- 只有沿最短路径抵达目标时才得到 reward 1，称为 shortest-path success；未成功的轨迹也有“离目标更近”的 partial credit；
- 通过改变 supervised pretraining 程度，构造一系列覆盖率不同的初始 policy，shortest-path success 大约从 1%（代码仓库中给出的 sweep 上限约 0.83%）降到 0.01% 左右；
- held-out validation set：1024 个 maze；
- 默认每个输入训练使用 \(N=16\) 个 rollout。

**比较。**

从相同初始化分别进行 TailRL、GRPO、RLOO 和 PKPO post-training，主要报告 post-training 的 Pass@1，并观察不同 initialization 和 rollout budget 下的学习行为。

**主要观察。**

- 当初始策略经常成功时，TailRL、GRPO、RLOO 的差距变小；
- 当 shortest-path success 降至约 0.01% 的低覆盖区域时，GRPO/RLOO 很难可靠提升，TailRL 仍能利用偶尔出现的 rare high-reward paths 进行 bootstrapping；
- PKPO 通常能改善，但整体低于 TailRL，尤其在 poor initialization 区域；
- 增大训练 rollout budget 时，TailRL 从 \(N=4\) 到 \(N=16\) 的提升明显，说明需要足够概率在 group 中看到 rare excellent rollout；
- 该实验隔离了 TailRL 最重要的使用场景：高质量解可达，但在初始 policy 下非常稀有。

### 6.3 GUI Grounding：Vision-Language Model 的 inference scaling

**任务与模型。**

- 模型：Qwen2.5-VL-3B、Qwen2.5-VL-7B；
- 训练数据：GTA1 grounding corpus；
- 输入：专业软件截图和自然语言 click instruction；
- 输出：要点击的像素坐标；
- 评测：ScreenSpot-Pro；
- reward：结合 click 与 target 的距离、点击落入 target 的 bonus 以及格式 bonus，具有连续和二值成分。

**训练与评测。**

TailRL、GRPO、RLOO 使用相同训练配置，仅替换 advantage estimator。评测时增加 inference rollout 数 \(k\)，报告：

- Pass@\(k\)：\(k\) 次点击中至少一次命中 target 的概率；
- Best-of-\(k\) reward：\(k\) 个候选中最高的连续 reward；
- smoothed training-batch accuracy。

**主要观察。**

- 在 Pass@1 上，TailRL 与 RLOO 大致相当，GRPO 较弱；
- 随着 inference samples 增加，TailRL 的 Pass@\(k\) 持续增长，而 RLOO 较早 plateau，说明 TailRL 保留了更多“非零但稀有的成功候选”；
- 3B：TailRL 用 8 次 inference rollouts 达到 RLOO 的 Pass@1024，等价于约 \(128\times\) 的采样节省；
- 7B：TailRL 用 4 次 rollouts 达到 RLOO 的 Pass@1024，等价于约 \(256\times\) 的采样节省；
- 单次 TailRL rollout 在两个模型规模上都超过 GRPO 的平均 Pass@1024；
- 该实验验证了 TailRL 与 Best-of-many deployment 的 alignment，而不只是训练 batch 上的平均 reward 提升。

### 6.4 Code Runtime Optimization：逃离安全但次优的 shortcut

**任务与 reward。**

- 模型：Qwen3-1.7B；
- 数据：PIE corpus 中的慢 C++ competitive-programming 程序；
- rollout：模型改写输入程序，目标是保持正确并加速；
- 执行流程：编译、运行测试用例；错误输出得到 0 reward；通过全部测试的程序按相对输入程序的 speedup 得分；
- 用 gem5 对执行时间进行仿真，避免普通 timing noise 制造虚假的 speedup；
- 每个程序使用 \(N=16\) 个 training rollouts；
- 对比 TailRL、GRPO、RLOO，使用相同 one-epoch compute budget。

该任务专门构造一个容易被 expected-reward RL 选择的安全解：直接复制输入程序总能通过测试，获得约 \(1\times\) 的 reward，但没有任何优化。训练初始分布中大约 74.4% rollout 不正确，23.5% 正确但无加速，只有约 2.1% 同时正确且更快。

**指标与结果。**

- training mean reward；
- correctness / pass-all-tests 比例；
- policy entropy；
- held-out test set 的 per-problem Best-of-1024 speedup。

GRPO 和 RLOO 很快坍缩到复制输入的 shortcut：正确率很高、mean reward 接近 \(1.0\)，但 entropy 下降一到两个数量级，Best-of-1024 约为 0.98× 和 0.96×。TailRL 保持更高 entropy，继续探索低概率的非平凡 rewrite；single-rollout correctness 可能较低，但在候选池中保留了更多真正加速的程序，held-out Best-of-1024 平均 speedup 达到约 **7.7×**。论文还报告 TailRL 在一个 held-out problem 上找到约 27× 的 verified speedup，而 pretrained model 约为 5.05×，GRPO/RLOO 主要复现原程序。

这里的结果是在相同 one-epoch 预算下得到的 matched-compute 对比；论文指出 TailRL 在约 step 300 时仍在提升，因此数字不应被理解为所有方法的最终收敛上限。

## 7. 结果的统一解释

四个实验分别对应 TailRL 的四个预测：

| 现象 | expected-reward RL 的倾向 | TailRL 的处理 |
|---|---|---|
| 连续质量层级 | 只优化平均 IoU/平均 reward | 对每个质量阈值维护 coverage，连续 reward 信息不被二值化丢掉 |
| rare success | 组内很少见时，信号容易被平均掉 | 通过幸存 rollout 数反比分配高阈值 credit |
| 增加 inference samples | 策略可能已经坍缩，新增样本没有新候选 | 保留高 reward 但低概率的候选，Best-of-\(k\) 持续增长 |
| 安全次优 shortcut | 复制/中等解频率高，平均 reward 看起来足够好 | 继续提升稀有高阈值 tail，抵抗过早 mode collapse |

从目标函数角度看，TailRL 同时做两件事：

1. 提升 mean/低阈值 tail（因为它包含 \(k=1\) 的 expected-reward 分量）；
2. 额外提升高阈值、低覆盖率 tail（因为其梯度权重接近 \(1/p_\theta\)）。

因此它不是简单追求更大的 entropy，而是让 entropy/探索保留在“可能产生高 reward 的方向”上。Code optimization 中 entropy 只是结果性诊断，真正的驱动力仍是 tail-likelihood objective。

## 8. 局限、实现风险与对 SkyRL 的启发

### 8.1 局限与适用条件

1. **需要有序的 scalar reward。** 如果 reward 只提供极少的二值层级，TailRL 的连续阈值优势变弱，并退化到 MaxRL 类目标。
2. **必须已经有少量高质量信号。** TailRL 会放大 rare high-reward rollout；如果策略的 support 中完全没有可行的高质量解，目标本身不能凭空创造该解。
3. **极端稀有 reward 的估计方差仍然存在。** 有限 group 中必须至少观察到高 reward 候选，增大 \(N\) 可以改善覆盖，但会增加 rollout 成本。
4. **对 reward hacking 仍敏感。** TailRL 强调高 reward；如果 verifier/reward function 有漏洞，错误的高 reward 也可能被放大。
5. **population objective 的 log 需要正 support。** 某阈值若 \(p_\theta(x,\tau)=0\)，其 log-likelihood 为 \(-\infty\)；理论和实践都依赖合理的 support/regularity 条件。
6. **reward scale 需要处理。** 论文主体假定 reward 在 \([0,1]\)，一般有界区间可以先做仿射归一化；若 reward 未界且重尾，需要谨慎设计截断或归一化，否则高 reward gap 可能导致不稳定的 advantage。

### 8.2 对 SkyRL/RWKV 接入的接口启发

如果将 TailRL 接入 SkyRL native Trainer/Generator/InferenceEngine，最小修改应位于 rollout advantage 计算，而不是 Trainer 的模型结构：

- Generator 为每个 prompt 采样 group，并确保 group 内 token、response mask、sequence reward 严格对齐；
- reward 计算后按 prompt 分组排序，使用 reward gaps/survivor counts 计算 \(\omega_i\)；
- 将中心化后的 \(A_i\) 广播到有效 response tokens；
- 保持 trainer policy、rollout policy 的 token/logprob 对齐，以及 optimizer step 后的 weight sync；
- 使用与论文一致的 sequence-level reward，不能把不同 prompt 的 reward 混到同一个 tail distribution 中；
- 训练日志同时记录 mean reward、tail quantiles、group 中最高 reward 的频率、Best-of-\(k\)/Pass@\(k\) 和 advantage 分布，否则很难判断方法是否真的改善高尾覆盖；
- 对 RWKV 这类 recurrent/stateful model，需要额外确认每条 rollout 的 hidden-state/reset、token mask 和 sequence score 计算一致，避免把模型状态误差误判为 TailRL 的 reward weighting 效果。

## 9. 总结

TailRL 的核心不是“给高 reward 加一个手工权重”，而是将连续 reward 重新解释为所有阈值上的 binary success events，并最大化这些 event 的平均 log-probability：

\[
\boxed{
\text{TailRL}
=\int\log\Pr(r>\tau)d\tau
\quad\Longleftrightarrow\quad
\text{harmonic mixture of all Best-of-}k\text{ objectives}
}
\]

它由此得到三个相互一致的结论：

1. 梯度按逆 tail-probability 加权，稀有高 reward rollout 更重要；
2. \(N\) 个 rollout 可用排序后的 reward gaps 构造无 critic 的有限 estimator，只需替换 advantage；
3. 训练目标与 inference-time Best-of-many/Pass@\(k\) 直接对齐，能够在 rare success、采样扩展和安全次优 shortcut 场景中优于只优化平均 reward 的方法。

## 参考链接

- 论文（用户指定页面）：<https://huggingface.co/papers/2609.02987v1>
- 论文 arXiv：<https://arxiv.org/abs/2609.02987>
- 论文 HTML：<https://arxiv.org/html/2609.02987v1>
- 官方代码仓库：<https://github.com/Zanette-Labs/TailRL>
- 官方实现中的 TailRL advantage estimator：<https://github.com/Zanette-Labs/TailRL/blob/main/experiments/text_maze/verl/trainer/ppo/core_algos.py>
- 项目主页：<https://zanette-labs.github.io/TailRL-website/>

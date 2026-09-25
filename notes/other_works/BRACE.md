# BRACE：Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL

> 阅读对象：Guanqun Zhao、Zijun Xie、Binbin Zheng、Jiafeng Lu、Enlei Gong、Zeyu Chen，*BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL*，arXiv:2609.09783v1（2026-09-09）。本文整理论文的核心问题、方法、数学表示与实验设计。

## 1. 一句话总结

BRACE 针对异步 LLM-RL 中 **critic 用旧行为策略产生的轨迹训练，因而拟合了 stale policy value 而非当前 policy value** 的问题，提出一种带锚定 Monte-Carlo 尾部的 Bellman residual correction：

\[
\boxed{\text{前 }k\text{ 个 policy token 做重要性校正}\;+
\;\text{后续尾部固定权重并 telescoping 到 terminal reward}}
\]

它将两个本来冲突的目标拆开：

1. **policy correction**：只在短前缀内修正行为策略 \(\mu\) 与目标策略 \(\pi\) 的差异，避免长轨迹重要性比率乘积爆炸/衰减；
2. **reward propagation**：无论轨迹多长，都让 terminal reward 通过一个恒定权重的 Monte-Carlo tail 传回早期状态。

BRACE 只修改 critic target 和由此得到的 advantage，actor 仍使用标准 PPO clipped objective。

## 2. 问题背景：异步训练中的 critic-side bias

异步 RL 中，生成与训练并行，轨迹由行为策略 \(\mu\) 生成，消费轨迹时 actor/critic 已更新到目标策略 \(\pi\)。因此：

\[
V_\gamma^\mu(s)=\mathbb E_\mu\left[\sum_{k\ge 0}\gamma^k r_{t+k}\mid s_t=s\right],
\qquad
V_\gamma^\pi(s)=\mathbb E_\pi\left[\sum_{k\ge 0}\gamma^k r_{t+k}\mid s_t=s\right].
\]

普通 GAE/\(\lambda\)-return 在 \(\mu\) 采样的轨迹上回归，精确回归时收敛到 \(V^\mu\)，但 actor 的 advantage 需要接近 \(V^\pi\)。于是：

\[
\mathbb E_{y\sim\pi}[R-V^\mu(s_t)]
=V^\pi(s_t)-V^\mu(s_t)
=:b(s_t)\ne 0.
\]

这个偏差通常在序列早期最大，因为早期决策后面还包含更长的、由 \(\mu\) 产生的 suffix。batch-level advantage whitening 只能去掉整体均值，不能去掉跨 prompt、跨 position 的 value gap。

### 2.1 为什么直接套 V-trace 不合适

对 terminal-only reward，若 V-trace correction window 不包含最终 reward，则 target 对 terminal return 没有直接依赖：

\[
\frac{\partial V_{\mathrm{targ}}(s_s)}{\partial R}=0.
\]

要让 reward 进入早期状态，window 必须覆盖到终点；但此时 target 中出现长距离的重要性比率乘积：

\[
\gamma^{T-s}\prod_{i=s}^{T-1}c_i,
\qquad
c_i=\lambda\min\left(\bar c,\frac{\pi(a_i\mid s_i)}{\mu(a_i\mid s_i)}\right).
\]

其对数的期望随 horizon 线性变化，因而该系数可能随序列长度指数衰减；同时方差/样本间 dispersion 增大。于是一个 window 无法同时满足：

- 足够长，以传播 sparse terminal reward；
- 足够短，以控制 off-policy ratio product。

## 3. BRACE 的核心方法

### 3.1 基本重要性权重

对 policy token 定义：

\[
w_t=\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)},
\qquad
\rho_t=\min(\bar\rho,w_t),
\qquad
c_t=\lambda\min(\bar c,w_t),
\]

以及截断 Bellman residual：

\[
\delta_t^V=\rho_t\left(r_t+\gamma V(s_{t+1})-V(s_t)\right).
\]

\(\rho_t\) 控制当前 residual 的 correction strength，\(c_t\) 控制 residual 向前传播时的 trace coefficient。

> 在 agentic trajectory 中，\(k\) 只计算 policy tokens；retrieved observations、tool outputs 不消耗 correction horizon。落在非-policy token 上的 reward 折叠到前一个 policy token。

### 3.2 capped correction horizon

对状态 \(s_s\)，令 correction horizon 为 \(k\)，则 capped target 为：

\[
V_{\mathrm{targ}}^{(k)}(s_s)
=V(s_s)+\sum_{t=s}^{T}
\gamma^{t-s}
\left(\prod_{i=s}^{\min(t,s+k)-1}c_i\right)\delta_t^V.
\]

关键点：

- residual 的求和仍然走到 \(T\)，所以 terminal reward 一定能进入 target；
- product 最多只包含前 \(k\) 个 ratio，因此 correction variance 不随完整轨迹长度积累；
- \(k\) 越大，越接近对目标策略的 off-policy correction，但方差更高；\(k\) 越小，越接近行为策略 value，但更稳定。

在理想条件下，该 target 的 fixed point 接近截断目标策略 value：

\[
V_{\gamma}^{\pi_{\bar\rho}}(a\mid s)
\propto \min\left(\bar\rho\,\mu(a\mid s),\pi(a\mid s)\right),
\]

并且随着 \(\bar\rho\) 增大而接近 \(V_\gamma^\pi\)。

### 3.3 为什么还需要 Monte-Carlo tail

在 capped target 中，product 在 \(j=\min(s+k,T)\) 后冻结为：

\[
\Pi_k(s)=\prod_{i=s}^{j-1}c_i.
\]

如果尾部仍逐 token 使用 \(\rho_t\)，则 residual coefficient 在 tail 中变化：

\[
\gamma b_{t-1}-b_t
=\gamma^{t-s}\Pi_k(s)(\rho_{t-1}-\rho_t),\qquad t>j.
\]

这些项的净和可以是 \(O(1)\)，但 total variation 随 tail 长度增长，给 critic label 引入约为
\(O(\sqrt{T-s-k})\) 的随机噪声，并破坏 tail 的 telescoping。

BRACE 因此将 correction horizon 之后的 weight 固定为 1：

\[
\tilde\rho_t^{(s)}=
\begin{cases}
\rho_t,&t<j,\\
1,&t\ge j.
\end{cases}
\]

于是 tail 完全 telescoping：

\[
\begin{aligned}
\Pi_k(s)\sum_{t=j}^{T}\gamma^{t-s}
\left(r_t+\gamma V(s_{t+1})-V(s_t)\right)
=\Pi_k(s)\left[\gamma^{T-s}R-\gamma^{j-s}V(s_j)\right].
\end{aligned}
\]

这就是“anchored Monte-Carlo tail”：它只用前缀的固定 correction weight \(\Pi_k(s)\)，但将完整 terminal reward 传递回来，不在长尾继续乘新的 importance ratios。

### 3.4 最终 BRACE target

原始形式：

\[
\boxed{
V_{\mathrm{targ}}^{\mathrm{BRACE}}(s_s)=V(s_s)+
\sum_{t=s}^{T}\gamma^{t-s}
\left(\prod_{i=s}^{\min(t,s+k)-1}c_i\right)
\tilde\rho_t^{(s)}
\left(r_t+\gamma V(s_{t+1})-V(s_t)\right)
}
\]

等价的“corrected prefix + anchored return”形式：

\[
\boxed{
V_{\mathrm{targ}}^{\mathrm{BRACE}}(s_s)=V(s_s)
+\sum_{t=s}^{j-1}\gamma^{t-s}
\left(\prod_{i=s}^{t-1}c_i\right)\delta_t^V
+\Pi_k(s)\left[\gamma^{T-s}R-\gamma^{j-s}V(s_j)\right]
}
\]

该表达式直接体现了 BRACE 的设计：前缀负责校正 policy mismatch，尾部负责稳定地传播 realized return。

### 3.5 对 actor 与 critic 的使用

论文使用由 BRACE target 形成的 advantage：

\[
\hat A_s=r_s+\gamma V_{\mathrm{targ}}^{\mathrm{BRACE}}(s_{s+1})-V_\phi(s_s).
\]

critic 使用 PPO 风格 clipped value loss：

\[
\mathcal L_V=\frac{1}{|\mathcal D|}\sum_{t\in\mathcal D}\frac12
\max\left[
(V_\phi(s_t)-V_{\mathrm{targ}}(s_t))^2,
(V_\phi^{\mathrm{clip}}(s_t)-V_{\mathrm{targ}}(s_t))^2
\right],
\]

其中

\[
V_\phi^{\mathrm{clip}}(s_t)=
\operatorname{clip}(V_\phi(s_t),V_{\mathrm{old}}(s_t)-\epsilon_v,V_{\mathrm{old}}(s_t)+\epsilon_v).
\]

actor 仍采用标准 PPO clipped surrogate；BRACE 与 PPO 的主要区别是 value target 和对应 advantage，而不是 actor trust-region 目标。

### 3.6 一次 BRACE iteration

1. 异步 rollout 得到轨迹、\(\log\mu(a_t\mid s_t)\)、terminal reward \(R\)。
2. 用当前 actor/critic 计算 \(\log\pi(a_t\mid s_t)\) 与 \(V_\phi(s_t)\)。
3. 计算 \(w_t,\rho_t,c_t,\delta_t^V\)。
4. 对每个位置 \(s\)，令 \(j=\min(s+k,T)\)，计算 \(\Pi_k(s)\) 和 BRACE target。
5. 用 BRACE target 更新 critic。
6. 构造 \(\hat A_s\)，用 PPO clipped objective 更新 actor。
7. 异步同步 actor 权重到 rollout pool；下一批轨迹继续带有相应 staleness。

## 4. 理论/算法直觉

### 4.1 \(k\) 是 bias-variance knob

BRACE 不再试图用一个完整 horizon 的 product 同时完成 correction 和 reward propagation，而是让 \(k\) 单独控制 correction depth：

- \(k=0\)：退化为 Monte-Carlo-style target，稳定但不做 policy correction；
- 较小 \(k\)：主要修正早期、最受 stale critic bias 影响的 token；
- 较大 \(k\)：更接近目标策略 value，但 ratio product 更长、variance 更大；
- \(k\to\infty\)：退化为未截断的 V-trace-like correction。

### 4.2 作用对象与已有异步方法的区别

论文中的 PPO-EWMA、AReaL、KPop、IcePop 等主要作用于 actor：改变 trust region、reweight gradient 或 mask stale tokens，但 critic 仍然回归 stale trajectories 的普通 target。BRACE 直接修正 critic regression target，因此 actor 获得的 advantage 本身更加接近当前策略的 value baseline。

BRACE 也可以与 actor-side correction 组合，而不是替代这些方法。

## 5. 实验设计

### 5.1 任务、模型与异步设置

| 任务 | 模型 | 任务类型 | 训练/rollout 配置摘要 | 评测 |
|---|---|---|---|---|
| Search-R1 | Qwen3-8B | 多轮搜索增强问答 | 2 nodes；train/rollout 1/1；每 prompt 4 samples；prompt/response 4096/8192；staleness \(S=9\) | NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Musique、Bamboogle，greedy EM |
| BrowseComp-Plus | Qwen3-30B-A3B | 多轮深度研究 agent | 4 nodes；train/rollout 3/1；每 prompt 1 sample；prompt/response 4096/32768；\(S=6\) | mean@1，LLM judge |
| DAPO-Math-17k | Qwen2.5-7B-Instruct | 数学推理 | 2 nodes；每 prompt 8 samples；prompt/response 2048/16384；\(S=5\) | AIME 2024/2025/2026，mean@32、pass@32 |
| GSM8K-tools | Qwen2.5-7B | 多轮工具增强数学 | 2 nodes；每 prompt 8 samples；prompt/response 2048/8192；\(S=13\) | mean@4、pass@4 |

所有方法使用 fully asynchronous VeRL pipeline、disaggregated rollout/training pools 和 partial rollout；实验运行在 H800 上。训练侧使用 SGLang rollout 与 Megatron trainer，权重通过 NCCL checkpoint engine 同步。

### 5.2 Baselines

- **PPO**：只使用 PPO clipped surrogate；critic 使用普通 GAE target；
- **PPO-EWMA**：使用 proximal/EWMA reference 的 actor-side correction；
- **AReaL**：解耦 trust region 的异步策略优化；
- **KPop**：按 Bernoulli-KL 对 token 做 adaptive masking；
- **IcePop**：按 \(\pi/\mu\) ratio 区间过滤 token；
- **BRACE**：actor 与 PPO 相同，区别在 critic target/advantage。

比较重点是：在相同 model、data order、step budget、异步 pipeline 下，critic-side correction 是否能带来独立收益。

### 5.3 默认超参数

- \(\gamma=1,\lambda=1\)；
- correction horizon \(k=100\)；
- \(\bar\rho=1.2,\bar c=1.1\)；
- actor learning rate \(10^{-6}\)，critic learning rate \(10^{-5}\)；
- critic warm-up 5 steps；
- PPO clip low/high = 0.20/0.28；value clip = 0.5；
- KL penalty 关闭，entropy coefficient 为 0；
- 使用 advantage whitening；
- 评测：检索/问答任务 greedy；数学任务 temperature 0.6、top-p 0.95，报告 mean@N 与 pass@N。

## 6. 主要结果

### 6.1 主结果表

论文报告 BRACE 在 16 个主要指标中的 13 个取得最好结果：

| 方法 | Search-R1 NQ | Search-R1 HotpotQA | BrowseComp-Plus mean@1 | AIME24 mean@32 | AIME24 pass@32 | AIME25 pass@32 | AIME26 mean@32 | GSM8K-tool mean@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 0.264 | 0.251 | 0.206 | 0.110 | 0.332 | 0.349 | 0.082 | 0.842 |
| PPO-EWMA | 0.302 | 0.255 | 0.199 | 0.121 | 0.375 | 0.403 | 0.089 | 0.854 |
| AReaL | 0.341 | 0.274 | 0.245 | 0.142 | 0.338 | 0.391 | 0.095 | 0.926 |
| KPop | 0.324 | 0.270 | 0.183 | 0.131 | 0.380 | 0.379 | 0.093 | 0.889 |
| IcePop | 0.280 | 0.271 | 0.231 | 0.147 | 0.391 | 0.393 | 0.102 | 0.923 |
| **BRACE** | **0.383** | **0.285** | **0.269** | **0.144** | **0.409** | **0.414** | **0.103** | **0.937** |

代表性结论：

- BrowseComp-Plus：BRACE mean@1 = 0.269，相比最强 baseline AReaL 的 0.245 提升 2.4 个百分点；
- Search-R1：BRACE 在 7 个 QA set 上总体最强，NQ 从 PPO 的 0.264 提升到 0.383；
- 数学：BRACE 取得 AIME24/AIME25 的最佳 pass@32，并取得 AIME25/AIME26 的最佳 mean@32；
- GSM8K-tools：BRACE mean@4 = 0.937，优于 AReaL 的 0.926；
- BRACE 的优势在长 credit-assignment 距离的任务上更明显。

### 6.2 critic bias 与训练动态

论文使用两个诊断量：

\[
\hat b=\widehat{\mathbb E}_\pi[R]-\widehat{\mathbb E}_\mu[R],
\qquad
\hat g_\pi=\left|\bar v-\widehat{\mathbb E}_\pi[R]\right|.
\]

在 Search-R1 中，BRACE 后期的 value bias gap 约为 0.07，而 baseline 大约为 0.08–0.10；在 DAPO-Math 中，PPO 的 policy-value separation 后期约 0.06–0.08，BRACE 维持在约 0.02。这支持收益主要来自 critic bias reduction，而非只来自 actor-side reweighting。

BrowseComp-Plus 上，BRACE 同时表现为：

- 更高 entropy（约 0.44，对比 PPO 约 0.31）；
- 更短 response（约 26k 降至 15k tokens）；
- 更少 retrieval turns（约 23 降至 13）；
- 更高准确率，因此不是靠策略坍缩到窄行为模式获得收益。

### 6.3 消融实验

1. **去掉 \(k\)-cap（\(k\to\infty\)）**：恢复 full-horizon importance correction。性能从早期开始落后，BrowseComp-Plus 约停在 0.33，而 BRACE 约达到 0.39（论文图中训练曲线指标）。
2. **去掉 Monte-Carlo tail**：tail telescoping 被破坏，terminal reward 被衰减，长 tail 的 alternating value terms 引入随 horizon 增长的噪声；曲线后半程更不稳定，最终约低 0.01。
3. **staleness sweep**：在 BrowseComp-Plus 上，\(S=5\) 约 0.39，\(S=10\) 约 0.35，\(S=15\) 至 \(S=50\) 约 0.29–0.31。BRACE 在 staleness 增大时质量下降，但没有失稳；从 \(S=15\) 到 \(S=50\) 的额外下降小于 0.02。
4. **cap size sweep**：\(k=10\) 明显落后；\(k=20\) 与 \(k=100\) 相差小于 0.01，说明只要覆盖 stale bias 集中的近端 horizon，继续增大 \(k\) 收益有限。
5. **\(\bar\rho\) sweep**：\(\bar\rho\le 2\) 较稳定；继续放宽会因 weighted residual variance 增大而下降。
6. **\(\bar c\) sweep**：对性能更敏感；增大 \(\bar c\) 会携带更长 product，重新引入 ratio accumulation，因此默认 \(\bar c=1.1\)。

### 6.4 训练效率

在 BrowseComp-Plus 上：

| 方法 | step time | throughput | 相对同步 PPO |
|---|---:|---:|---:|
| Synchronous PPO | 546.43 s | 58.64 tokens/s/GPU | 1.00× |
| Asynchronous PPO | 219.20 s | 134.79 tokens/s/GPU | 2.49× |
| BRACE | 222.47 s | 122.15 tokens/s/GPU | 2.46× |

BRACE 相比 asynchronous PPO 只增加约 1.5% step time，仍比 synchronous PPO 快 2.46×。在 GSM8K-tools 上，BRACE 因 anchored tail 需要完整 rollout 的计算，step time 约 348s、throughput 约 660 tokens/s；论文强调该开销是常数级，而非随训练 staleness 增长。

## 7. 对 SkyRL/RWKV 的启发

1. **异步 rollout 不只需要 actor correction**：如果 trainer 使用 critic/GAE，必须显式记录 rollout policy 的 token logprob，并在 trainer 侧计算当前 policy logprob，才能构造 \(w_t\)。
2. **correction horizon 应按 policy token 计数**：工具返回、检索文档等 observation 不应消耗 \(k\)，否则 agentic trajectory 的有效 correction depth 会被错误缩短。
3. **terminal reward 与 ratio correction 应解耦**：对长上下文模型，不能简单把整段轨迹的 ratio product 传到终点；BRACE 的 anchored tail 可作为 value-target 设计原型。
4. **需要监控 critic-side diagnostics**：除 actor KL/ratio clipping 外，应记录 \(\hat b\)、\(\hat g_\pi\)、terminal reward coefficient、effective sample size 和 max advantage。
5. **RWKV 实现时需严格区分 policy/value token mask**：BRACE target 和 advantage 只应在 policy-generated tokens 上计算；tool/observation token 需要 mask 或按论文规则折叠 reward。
6. **建议先做小规模验证**：在同步或低 staleness 环境中验证 BRACE target 退化行为，再逐步增加异步 version gap；重点检查 \(k=0\)、\(k\to\infty\)、on-policy \(w_t=1\) 三个边界。

## 8. 局限与需要谨慎的地方

- BRACE 的 correction 仍是截断的：其 fixed point 更准确地说与 \(V^{\pi_{\bar\rho}}\) 和 \(V^\mu\) 之间插值，而非无条件等于 \(V^\pi\)。
- \(k\) 需要覆盖主要 stale bias 深度；过小会保留 bias，过大则重新增加 variance。
- 论文实验主要采用 terminal/sparse reward；dense reward、极长上下文及更复杂 reward decomposition 仍需单独验证。
- 论文未提供独立的 BRACE 专属代码仓库链接；实现依赖 VeRL、Megatron 与 SGLang，复现时需要注意论文版本与实验 pipeline 的一致性。

## 9. 链接

### 论文

- arXiv 摘要页：[https://arxiv.org/abs/2609.09783](https://arxiv.org/abs/2609.09783)
- arXiv HTML 全文：[https://arxiv.org/html/2609.09783v1](https://arxiv.org/html/2609.09783v1)
- arXiv PDF：[https://arxiv.org/pdf/2609.09783v1](https://arxiv.org/pdf/2609.09783v1)

### 代码/基础框架

- **BRACE 专属官方代码仓库**：截至本文整理时，论文页面未提供独立 GitHub 仓库链接。
- VeRL（论文实验使用的 RL 训练框架）：[https://github.com/volcengine/verl](https://github.com/volcengine/verl)
- Megatron-LM（训练后端）：[https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- SGLang（rollout/inference 后端）：[https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)

# RISE：Recursive Improvement via Self-Extrapolating Policy Distillation

> 阅读对象：arXiv: [2609.05295](https://arxiv.org/abs/2609.05295)，论文标题为 *RISE: Recursive Improvement via Self-Extrapolating Policy Distillation*，Yang Li、Semih Yavuz、Shafiq Joty，v1（页面标注提交于 2026-09-04）。本文重点整理其方法、数学形式、实验设计，以及与 OPSD 的区别。

## 1. 一句话总结

RISE 不使用外部 teacher，也不把正确答案作为 privileged context 输入给 teacher；它先用 RLVR 得到一个“有验证奖励支撑的更新方向”，再沿该方向外推出一个临时的 future teacher，最后用 OPD 将这个 future teacher 的逐 token 分布蒸馏回当前策略：

\[
\boxed{\text{RLVR 定方向}\;\rightarrow\;\text{trajectory extrapolation 造 teacher}\;\rightarrow\;\text{OPD 做细粒度投影}}
\]

核心直觉是：当前模型自己的训练轨迹，包含了它朝更优策略移动的方向；如果这条轨迹在局部近似低维、近似线性，那么“沿最近一次有效更新再走一点”可以作为当前模型的近似 future self。

## 2. 为什么需要 RISE

- **RLVR 的问题**：结果奖励通常是 sequence-level 的。同一条回答中的所有 token 共享一个 outcome advantage，无法区分哪些推理步骤有帮助、哪些步骤导致失败。
- **OPD 的优点与瓶颈**：teacher 可以在 student 自己访问的 prefix 上给出完整的 next-token 分布，因此监督稠密；但效果受 teacher 质量限制。
- **外部 teacher 的问题**：teacher 可能没有见过 student 生成的 prefix，存在 prefix/distribution mismatch。
- **OPSD 的问题**：teacher 虽与 student 共用模型，但依赖 privileged information（如正确解答/推理轨迹）；模型未必能通过 in-context learning 正确利用这些信息，且 teacher 看到的条件与 student 推理时的条件不一致。

RISE 的选择是：不再问“如何更好地使用一个给定 teacher”，而是直接从 RLVR 训练轨迹中构造一个动态 teacher。

## 3. 核心方法与数学表示

### 3.1 记号

第 \(n\) 轮中：

- 当前策略：\(\pi_{\theta_n}\)；
- 从当前策略采样 \(y\sim\pi_{\theta_n}(\cdot\mid x)\)，由 verifier 得到结果奖励 \(R(x,y)\)；
- 经过 GRPO 等 RLVR 更新得到中间 checkpoint \(\pi_{\theta'_{n+1}}\)；
- \(\pi_{\theta_a}\) 是 anchor，默认可取上一 checkpoint，也可取 EMA anchor；
- \(\beta>1\) 是外推系数。

### 3.2 统一的 self-extrapolation

令 \(\varphi\) 把策略映射到一个可以做线性运算的表示空间：

\[
\varphi(\pi_{\mathrm{future}})
=\varphi(\pi_{\theta_a})
+\beta\left[\varphi(\pi_{\theta'_{n+1}})-\varphi(\pi_{\theta_a})\right],
\qquad \beta>1.
\]

当 \(\beta=1\) 时，teacher 就是 post-RLVR checkpoint；当 \(\beta>1\) 时，teacher 被推到当前 RLVR 更新方向的前方。\(\beta\) 不是“知道真实最优策略”，而是对最近一次已经由奖励验证过的局部方向做保守外推。

### 3.3 Weight-space 与 logit-space 两种实现

#### A. Weight-space RISE

令 \(\varphi(\pi)=\theta\)：

\[
\theta_{\mathrm{future}}
=\theta_a+\beta(\theta'_{n+1}-\theta_a).
\]

再用 \(\theta_{\mathrm{future}}\) 得到普通模型作为 teacher。它产生一个跨位置一致的完整模型，但需要 materialize 一份 future 权重并做 teacher forward。

#### B. Logit-space RISE

对 student rollout 中的上下文 \(s_t=(x,y_{<t})\)，令 \(\varphi(\pi)=\log\pi(\cdot\mid s_t)\)：

\[
\log \pi_{\mathrm{future}}(\cdot\mid s_t)
= \log\pi_{\theta_a}(\cdot\mid s_t)
+\beta\left[
\log\pi_{\theta'_{n+1}}(\cdot\mid s_t)
-\log\pi_{\theta_a}(\cdot\mid s_t)
\right]+C,
\]

其中 \(C\) 用于归一化。等价地：

\[
\pi_{\mathrm{future}}(v\mid s_t)
\propto
\pi_{\theta_a}(v\mid s_t)^{1-\beta}
\pi_{\theta'_{n+1}}(v\mid s_t)^{\beta}.
\]

因此，post-RLVR 后概率上升的 token 会被进一步放大，概率下降的 token 会被进一步压低。这是一个 geometric mixture，而不是简单的 arithmetic mixture。

若记 \(\pi_{\theta_a}\) 和 \(\pi_{\theta'_{n+1}}\) 为 stop-gradient 的分布，则 reverse-KL 有如下分解：

\[
\begin{aligned}
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{future}})
={}&-(\beta-1)D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_a})\\
&+\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta'_{n+1}})+\log Z.
\end{aligned}
\]

第一项系数为负，推动当前策略继续远离 anchor、延续 RLVR 方向；第二项把策略约束在 post-RLVR checkpoint 附近，避免直接跳到过远的 future point。这解释了为什么 RISE 不是简单的“把模型权重往前推一步”。

weight-space 计算的是 \(f(\theta_a+\beta\Delta\theta)\)，logit-space 计算的是 \(f(\theta_a)+\beta\Delta f\)，后者是前者在 anchor 附近的一阶近似；神经网络中二者不必相同。

### 3.4 RISE 的 OPD loss

在当前策略的 rollout prefix 上，将 future policy 作为 stop-gradient teacher：

\[
\mathcal L_{\mathrm{RISE}}
=\mathbb E_{y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s_t)
\middle\|
\operatorname{sg}[\pi_{\mathrm{future}}(\cdot\mid s_t)]
\right)
\right].
\]

论文实际训练中使用 JSD 代替 reverse KL，以利用其有界性并提高稳定性；分布计算采用 top-\(K\)+tail bucket，而非完整 vocabulary。默认 \(K=100\)，代码任务因分布更尖锐使用 \(K=20\)。teacher 在整个 OPD phase 中固定。

### 3.5 每一轮训练流程

1. **RLVR phase**：从 \(\pi_{\theta_n}\) 采样 rollout，计算 verifier reward，用 GRPO 等得到 \(\theta'_{n+1}\)。
2. **Teacher construction**：从 \(\theta_a\) 与 \(\theta'_{n+1}\) 在 weight-space 或 logit-space 外推得到 \(\pi_{\mathrm{future}}\)。
3. **OPD phase**：从 \(\theta'_{n+1}\) 出发，最小化 \(\mathcal L_{\mathrm{RISE}}\)，得到最终的 \(\theta_{n+1}\)。
4. **更新 anchor**：
   \[
   \theta_a\leftarrow(1-\eta)\theta_a+\eta\theta_{n+1}.
   \]
   \(\eta=1\) 是上一 checkpoint；\(\eta<1\) 是 EMA anchor。

RLVR rollout 被直接复用于 OPD，因此没有额外的 sampling cost；但因为 rollout 是 RLVR 更新前由旧策略生成的，OPD 相对于更新后的 student 有轻微 off-policy 性质。论文实验认为该影响很小。

### 3.6 为什么必须“外推后再蒸馏”

直接把 \(\theta_{\mathrm{future}}\) 作为下一策略，会把超出 trust region 的噪声和错误一并放大。OPD 在这里相当于一个 token-level trust-region projection：让策略朝 future teacher 移动，但不完全采用它。

论文的消融显示，直接采用 future weights 的 Math Avg 几乎不变：

- Qwen3-8B：GRPO 60.0，w/o OPD 60.3，完整 RISE 约 62.5–62.7；
- Qwen3-1.7B：GRPO 45.4，w/o OPD 45.6，RISE(logit) 50.2。

因此收益主要来自“future distribution 的逐 token 投影”，而不是简单地进行更长的 weight-space step。

## 4. 理论直觉与假设

### 4.1 teacher gap + distillation error

论文给出 OPD 的次优性分解。对最优策略 \(\pi^*\)、student \(\pi_\theta\)、teacher \(\pi_T\)：

\[
J(\pi^*)-J(\pi_\theta)
\le
\underbrace{J(\pi^*)-J(\pi_T)}_{\text{teacher gap}}
+
T\sqrt{\overline{\mathcal L}_{\mathrm{OPD}}/2}.
\]

其中 \(\overline{\mathcal L}_{\mathrm{OPD}}\) 是 student 自己访问的 prefix 上的平均 token-level reverse KL。普通 OPD 优化第二项，但 teacher gap 一旦 teacher 固定就成为能力上限；RISE 的目标是用 training trajectory 构造一个更接近未来最优策略的动态 teacher。

### 4.2 为什么要衰减 \(\beta\)

在理想的线性轨迹

\[
\varphi(\pi_{\theta_n})
=\varphi(\pi_{\theta_0})+\alpha_n d
\]

下，若最优点对应 \(\alpha^*\)，外推 teacher 比当前 student 更接近最优点的条件是：

\[
1<\beta<2\frac{\alpha^*}{\alpha_n}-1.
\]

随着 \(\alpha_n\to\alpha^*\)，安全的 \(\beta\) 范围变窄。因此 RISE 使用从 \(\beta_0=1.2\) 线性衰减到 1 的 schedule：早期可以更激进，后期避免 overshoot。

这不是一个无条件的收敛定理：它依赖局部近似线性，且实际 OPD 只做有限步更新。论文在 Qwen3-1.7B/8B 的 checkpoint 轨迹上测得第一主成分解释约 68.2% 方差，前三个主成分解释约 86.6%/88.5%，说明轨迹低维但并非严格 rank-1。

## 5. 实验设计

### 5.1 任务、模型与数据

| 任务族 | 模型与训练数据 | 主要评测 |
|---|---|---|
| 数学推理 | Qwen3-8B、Qwen3-1.7B、Qwen3-1.7B-Base + DAPOMath；OLMo3-7B-Instruct-SFT + OpenR1-Math-46K | MATH-500、AIME’24/’25、AMC’23、Minerva、OlympiadBench；GPQA-Diamond、IFEval、MMLU-Pro 作 OOD |
| 多领域 | Qwen3-4B-Base；混合数学与 STEM 数据 | 数学套件 + GPQA、SuperGPQA、MMLU、TheoremQA |
| 代码 | Qwen3-8B-Base + Skywork-OR1-Code | HumanEval+、MBPP+、LiveCodeBench |
| Agent | Qwen2.5-3B-Instruct + GIGPO 设置 | ALFWorld、WebShop |

### 5.2 对比方法

- Base/SFT：起点；
- GRPO：RLVR-only baseline；
- GRPO+SDPO：在 GRPO loss 上添加 privileged-teacher KL；
- SDAR：使用 teacher–student gap 对 GRPO advantage 做 token-level gating；
- RLSD：用 teacher 的逐 token 信号重加权 GRPO advantage；
- RISE(logit)、RISE(weight)：本文两种 teacher 构造方式。

后三个 OPSD 类 baseline 使用同一类 privileged teacher：一个 prompt 的 rollout group 中若存在正确答案，就把该正确 sibling solution 作为 teacher 的 privileged context；若 group 中没有正确 rollout，则该 prompt 通常只有 RLVR loss。RISE 不需要先找到一条正确 sibling solution。

### 5.3 训练与评测控制

- VeRL，单节点 8 GPU，每个训练集 1 epoch；
- AdamW，学习率 \(10^{-6}\)，constant，无 warmup；
- prompt 最长 2,048，response 最长 8,192；
- 每个 prompt 采样 8 条 rollout，温度 1.0；
- 各方法共享 rollout、batch、学习率等训练设置；
- RISE：\(\beta_0=1.2\)，线性衰减至 1；Qwen 使用 \(\eta=0.1\) EMA anchor，OLMo 使用 \(\eta=1\)；
- 评测使用 temperature 1、top-p 1、无 top-k 限制；同时报告平均正确率 avg@N 和至少一个样本正确的 pass@N，以区分平均质量与解空间覆盖。

### 5.4 主要结果

数学 Math Avg（单个代表性 seed）：

| 配置 | GRPO | RISE(logit) | RISE(weight) |
|---|---:|---:|---:|
| Qwen3-8B | 60.0 | 62.5 | **62.7** |
| Qwen3-1.7B | 45.4 | **50.2** | 49.2 |
| OLMo3-7B | 47.6 | **56.4** | 53.7 |

代表性观察：

- OLMo3-7B 上 AIME’24 从 GRPO 的 30.2 提高到 RISE(logit) 的 46.9，Math Avg 提高 8.8 个百分点；
- Qwen3-1.7B 上 logit RISE 提高 4.8 个百分点；
- Qwen3-8B 上 weight RISE 提高 2.7 个百分点；
- 不存在始终优于另一种的 extrapolation space，稳定的是“使用 extrapolated teacher”这一原则本身；
- OOD 能力没有明显损失，例如 Qwen3-8B OOD Avg 从 GRPO 的 70.6 提到 RISE(weight) 的 72.0。

其他任务：

- **多领域**：Qwen3-4B-Base 上 Math Avg 为 GRPO 40.2、RISE(logit) 43.3、RISE(weight) 44.8；STEM Avg 为 45.5、47.4、47.5。
- **代码**：RISE 在 HumanEval+、MBPP+、LiveCodeBench 上早期收敛更快；最终结果与 GRPO 接近，可能因为这些 benchmark 很快饱和。
- **Agent**：Qwen2.5-3B-Instruct 上，ALFWorld 为 GRPO 75.0、RISE(weight) 84.4；WebShop Score 为 79.8 对 86.3，Acc 为 63.3 对 74.2。
- **sample efficiency**：训练早期 RISE 已经达到更高准确率；在 Qwen3-1.7B 的 AIME’24 上，RISE(logit) 的 pass@16 比 GRPO 提高 9.6 个百分点，而 avg@16 提高 6.3 个百分点，说明收益不仅是把已有答案变得更尖锐，也扩大了可解题覆盖。

### 5.5 关键消融与成本

- **去掉 RLVR**：只沿 self-distillation 产生的方向递归外推，约 60 steps 内 MATH-500 降至 2.4%，生成长度从约 2K 爆到 8K 上限，reward 归零。说明外推本身没有判断方向好坏的能力，必须由 outcome reward grounding。
- **去掉 OPD**：直接采用 future policy，收益消失，说明不是“多走一步”带来的收益。
- **\(\beta\)**：\(\beta_0\in[1.2,1.5]\) 较稳定；\(\beta_0=2.0\) 会发散。固定 \(\beta\) 比衰减 schedule 差，支持后期缩小安全外推范围的分析。
- **anchor**：Qwen3-1.7B 上 logit RISE 的 Math Avg 从 \(\eta=1\) 的 46.2 提到 \(\eta=0.1\) 的 50.2；但 OLMo 上 EMA 反而有害，说明 anchor 需按训练轨迹噪声调节。
- **重采样**：OPD 阶段从 post-RLVR policy 重新采样，与复用原 RLVR rollout 的结果基本相同；复用 rollout 可以省掉约一半 sampling cost。
- **计算匹配**：GRPO 增加一个相同 rollout 上的 gradient pass 后，Qwen3-8B Math Avg 60.5，仍低于 RISE(weight) 62.7；Qwen3-1.7B 为 47.3，低于 RISE(logit) 50.2。因此增益不只是更多 optimizer steps。
- **墙钟时间**：相对 GRPO 约为 1.3–1.6 倍；sampling 数量不增加，额外成本主要来自 OPD gradient phase、anchor/future teacher forward 和缓存 logits。

## 6. RISE 与 OPSD 的区别

这里的 OPSD 指 Zhao et al. 的 *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*（[arXiv:2601.18734](https://arxiv.org/abs/2601.18734)，[代码](https://github.com/siyan-zhao/OPSD)）。RISE 论文中 GRPO+SDPO、SDAR、RLSD 也被作为 OPSD/privileged self-distillation 类 baseline，但它们不完全等同于原始 OPSD 的单一 loss。

### 6.1 OPSD 的标准形式

给定带参考解的训练集
\(\mathcal S=\{(x,y^*)\}\)：

- student 只看到问题：\(p_S(\cdot\mid x)\)；
- teacher 是同一模型，但额外看到正确答案/推理轨迹：\(p_T(\cdot\mid x,y^*)\)；
- rollout 只由 student 产生：\(\hat y\sim p_S(\cdot\mid x)\)；
- teacher 和 student 在相同的 student prefix 上计算 next-token 分布，并最小化逐 token divergence：

\[
\mathcal L_{\mathrm{OPSD}}
=\mathbb E_{(x,y^*)\sim\mathcal S}
\mathbb E_{\hat y\sim p_S(\cdot\mid x)}
\left[
\sum_{t=1}^{|\hat y|}
D\left(
 p_T(\cdot\mid x,y^*,\hat y_{<t})
 \middle\|
 p_S(\cdot\mid x,\hat y_{<t})
\right)
\right].
\]

\(D\) 可以是 forward KL、reverse KL 或 JSD；实现中还常用 pointwise clipping。teacher 梯度停止，student 学习把“有答案条件下的自己”内化为“无答案条件下的自己”。

### 6.2 对照表

| 维度 | OPSD | RISE |
|---|---|---|
| teacher 来源 | 同一模型 + privileged context（正确答案、参考 CoT 等） | 当前模型的 RLVR 更新轨迹外推出的 future policy |
| 是否需要正确参考解 | 需要；原始 OPSD 直接使用 \(y^*\)，RISE 论文的 OPSD baseline 通常需要 group 中至少一条正确 rollout | 不需要参考解/privileged context；但需要 verifier reward 来产生可靠 RLVR 方向 |
| student rollout | student 自己采样，保持 on-policy | RLVR 阶段由当前策略采样；通常复用这些 rollout 做 OPD |
| teacher 与 student 的差异 | 主要来自输入条件不同 | 主要来自 checkpoint 位移被放大，且 teacher 位于当前策略“前方” |
| teacher 是否动态 | 常见实现是 frozen teacher 或慢速 EMA teacher；核心定义并不要求沿训练方向外推 | 每一轮由最新 RLVR update 重新构造，天然 non-stationary、递归更新 |
| 能否超过 teacher 能力上限 | 直接蒸馏倾向于逼近 privileged teacher，存在 teacher ceiling | teacher 由 RLVR 方向外推，原则上可超过 post-RLVR student；但受 reward、局部线性和 \(\beta\) 安全范围限制 |
| 监督信息性质 | 参考解携带了 student 当前没有的任务特定信息 | 不增加新任务信息，只把已有 RLVR 的稀疏 outcome update 转换为稠密 token target |
| RL 的位置 | 可以独立使用 OPSD，也可与 GRPO/SDPO 等联合 | RLVR 是必需的 grounding phase；OPD 是后续 refinement/projection phase |
| 失败模式 | teacher 可能不会有效利用 privileged context；privileged prefix 与 student 条件不匹配，可能造成错误蒸馏或模式坍缩 | 外推方向可能错误或 reward-hacking；\(\beta\) 过大、轨迹非线性时会 overshoot；没有 RLVR grounding 时会退化/发散 |
| 采样与成本 | 原始 OPSD 强调以较少 rollout 获得稠密监督；是否需要正确参考取决于数据设置 | 相对 GRPO 不增加 sampling，但多一个 OPD phase，墙钟约 1.3–1.6 倍 |

### 6.3 最重要的概念差异

1. **“更有信息的 teacher” vs. “更靠前的 teacher”**

   OPSD 通过给同一个模型看正确答案，使 teacher 在输入信息上占优；RISE 不给 teacher 额外输入，而是把模型刚刚通过 RLVR 获得的参数/logit 位移向前延伸。前者是 **privileged-information asymmetry**，后者是 **training-dynamics asymmetry**。

2. **静态/条件 teacher vs. 动态/轨迹 teacher**

   OPSD 的 teacher 质量取决于模型能否在 privileged context 下正确 rationalize；RISE 的 teacher 质量取决于最近 RLVR 位移是否是“朝正确方向的局部切线”。RISE 不是简单地把 OPSD 的 privileged context 删除，而是替换了 teacher 的生成机制。

3. **是否引入新的任务信息**

   OPSD 可以使用 student 自己没有生成出来的正确解，因此在 student 完全不会做但训练集有答案时仍可能提供信号。RISE 没有这种信息注入能力：它不创造新的正确解，只放大 RLVR 已验证的方向；若 verifier reward 没有区分度，RISE 也没有理由产生可靠改进。

4. **是否只是模仿**

   OPSD 的基本动作是把 student 拉向 privileged teacher。RISE 的 \(\beta>1\) 先把 post-RLVR update 外推到“未来”，再用 OPD 做受控投影，因此其目标不是复现当前 teacher，而是把 RLVR 的 outcome-level 方向转成 token-level 的继续改进信号。

5. **二者的适用场景不同**

   - 有高质量参考解、但 verifier/RL 信号弱：OPSD 可能更直接；
   - 有可靠可验证奖励、但没有额外参考解或强模型：RISE 更合适；
   - 任务奖励容易被 hack：RISE 会放大错误方向，需要 reward uncertainty、保守 \(\beta\) 或多目标约束；
   - 模型 ICL/rationalization 较弱：OPSD 的 privileged teacher 未必可靠；
   - 训练轨迹不低维或更新高度噪声：RISE 的线性外推假设会失效。

### 6.4 一个容易混淆的地方

RISE 与 OPSD 都在 student 自己访问的 prefix 上提供 dense token-level supervision，因此从“loss 形式”看很相似；真正的区别不在于是否用了 KL/JSD，而在于 **teacher target 是如何得到的**：

\[
\begin{aligned}
\text{OPSD:}&\quad
\text{same model}+\text{privileged context}\ \longrightarrow\ \pi_T;\\
\text{RISE:}&\quad
(\pi_{\mathrm{anchor}},\pi_{\mathrm{post\text{-}RLVR}})
\xrightarrow{\text{extrapolation}}
\pi_{\mathrm{future}}.
\end{aligned}
\]

因此可以把 RISE 看成一种 **self-extrapolated OPD**，但不能把它简单等同于“没有 privileged context 的 OPSD”。它同时改变了 teacher 的来源、时间动态和与 RLVR 的组合顺序。

## 7. 总体评价与局限

RISE 最有价值的观点是把 RLVR 与 OPD 的关系从“两个 loss 加权”改写成：

> RLVR 负责判断最近的移动方向是否由 outcome 支撑；OPD 负责把这个方向转化为逐 token 的密集更新；外推负责让 teacher 不停留在当前 checkpoint。

实验支持其在数学、STEM、代码和多轮 Agent 任务上的收益，且收益不是简单增加 rollout 或 gradient pass。但方法的可靠性依赖三个条件：

1. verifier/reward 的方向确实可信；
2. 训练轨迹在局部足够低维、近似线性；
3. \(\beta\)、anchor 和 top-\(K\) 近似足够保守。

论文尚未给出判断“何时不应继续外推”的一般准则；reward hacking 也可能被 RISE 放大。这是它相对 OPSD 的主要交换：RISE 去除了 privileged teacher ceiling，却把风险转移到了 trajectory extrapolation 和 reward grounding 上。

## 参考链接

- RISE（用户指定页面）：<https://huggingface.co/papers/2609.05295>
- RISE arXiv HTML：<https://arxiv.org/html/2609.05295v1>
- RISE arXiv：<https://arxiv.org/abs/2609.05295>
- RISE arXiv source（论文 LaTeX/图表）：<https://arxiv.org/e-print/2609.05295>
- RISE 官方代码：论文、arXiv HTML/source 与 Hugging Face 页面未提供独立的公开 RISE 代码仓库；因此不将其他项目误标为 RISE 实现。
- VeRL（论文实验使用的 RL 基础框架）：<https://github.com/volcengine/verl>
- OPSD（相关 baseline 的代码）：<https://github.com/siyan-zhao/OPSD>
- OPSD 论文：<https://arxiv.org/abs/2601.18734>

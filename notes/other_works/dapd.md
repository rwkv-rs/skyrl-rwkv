# DAPD：Dual-Anchored Policy Distillation

> 论文方法整理。DAPD 面向带 privileged information 的 on-policy self-distillation（OPSD），核心问题是 **privilege illusion**：训练时 teacher 能看到参考解，但推理时 student 看不到；student 因而可能学会“仿佛已经知道答案”的不可靠行为。

## 1. 问题背景与直觉

给定问题 `x`、参考完成 `y*`，当前策略为 `p_θ`。训练时从无参考信息的 student prompt 采样 rollout：

\[
y \sim p_\theta(\cdot\mid x).
\]

传统 OPSD 在 rollout 前缀 `y_{<t}` 上构造两个同模型分布：

- **None**：`p_None = p_θ(· | x, y_<t)`，推理时真正可用的信息；
- **Cross**：`p_Cross = p_θ(· | x, y_<t, y*)`，额外看到参考完成，是 privileged teacher。

OPSD 的目标是

\[
\mathcal L_{\mathrm{OPSD}}
=\mathbb E_{(x,y^*)\sim\mathcal D,\,y\sim p_\theta}
\left[\frac1{|y|}\sum_t
D(\operatorname{sg}[p_{\mathrm{Cross}}]\|p_{\mathrm{None}})\right].
\]

问题在于 `Cross` 与 `None` 的信息条件不一致：teacher 的部分行为依赖 `y*`，但 student 在部署时无法复现。这会把“可复现的正确指导”和“依赖特权信息的行为”纠缠到同一更新中，造成错误自信、unsupported answer claims 等现象。

DAPD 的两个设计要求：

1. 用信息条件匹配的中间目标，把 reference 与 rollout 行为对齐；
2. 同时使用可靠但 off-policy 的 reference，以及 student-reachable 但可能错误的 rollout 作为指导源。

## 2. 三种条件分布与 Self bridge

对任意 completion `s∈{y,y*}`，记另一个 completion 为 `\bar{s}`：

- **None**：
  \[
  p_{\mathrm{None}}^s=p_\theta(\cdot\mid x,s_{<t})
  \]
  不接收 privileged completion，对应推理条件。
- **Cross**：
  \[
  p_{\mathrm{Cross}}^s=p_\theta(\cdot\mid x,s_{<t},\bar{s})
  \]
  接收另一个 completion，提供特权指导。
- **Self**：
  \[
  p_{\mathrm{Self}}^s=p_\theta(\cdot\mid x,s_{<t},s)
  \]
  接收自身正在预测的完整 completion。

`Self` 是关键桥接分布：它与 `Cross` 都看到完整 completion，信息条件匹配；同时它和 `None` 使用同一策略参数，因此可以把 matched-information 的监督传回实际推理策略。

## 3. 三个基础损失

对每个 completion 源 `s`，DAPD 定义三个有方向的 distillation objective（第一项 stop-gradient，第二项为可训练 student）：

### 3.1 Entangled Distillation

保留原始 OPSD 的 privileged supervision：

\[
\mathcal L_{\mathrm{ent}}^s
=\mathbb E\left[D(\operatorname{sg}[p_{\mathrm{Cross}}^s]\|p_{\mathrm{None}}^s)\right].
\]

### 3.2 Inference Anchor

让 Self 逼近无特权的 None：

\[
\mathcal L_{\mathrm{infer}}^s
=\mathbb E\left[D(\operatorname{sg}[p_{\mathrm{None}}^s]\|p_{\mathrm{Self}}^s)\right].
\]

该项将 Self 连接到推理时可用的行为。

### 3.3 Privileged Anchor

在双方都拥有 completion 的条件下对齐 Self 与 Cross：

\[
\mathcal L_{\mathrm{priv}}^s
=\mathbb E\left[D(\operatorname{sg}[p_{\mathrm{Cross}}^s]\|p_{\mathrm{Self}}^s)\right].
\]

该项保留 privileged teacher 中有价值的知识，但避免直接让 privileged `Cross` 监督 inference-time `None`。

## 4. Dual-Path Anchoring（DPA）

对有序方向 `s→\bar{s}`，DPA 组合两条信息匹配路径。

### 4.1 Unconditioned path

目标是对齐两个 inference-relevant 的 None 分布：

\[
\mathcal L_{\mathrm{uncond}}^{s\to\bar{s}}
=\mathcal L_{\mathrm{infer}}^{\bar{s}}
+\mathcal L_{\mathrm{ent}}^s.
\]

例如 rollout-to-reference 方向 `y→y*`：

\[
\mathcal L_{\mathrm{uncond}}^{y\to y^*}
=\mathcal L_{\mathrm{infer}}^{y^*}+\mathcal L_{\mathrm{ent}}^y.
\]

直觉是：`L_ent^y` 将 rollout-side None 向 privileged Cross 推进；`L_infer^{y*}` 将 reference-side Self 向 reference-side None 推进。由于 reference-side Self 与 rollout-side Cross 都由同一模型、相近的 `(x,y*)` 条件产生，二者构成 proxy bridge，联合更新会间接缩小两个 None 分布之间的差异。

### 4.2 Privileged path

当两侧都带有 completion 信息时，直接使用 Privileged Anchor：

\[
\mathcal L_{\mathrm{priv-path}}^{s\to\bar{s}}
=\mathcal L_{\mathrm{priv}}^s.
\]

### 4.3 DPA objective

\[
\mathcal L_{\mathrm{DPA}}^{s\to\bar{s}}
=\mathcal L_{\mathrm{uncond}}^{s\to\bar{s}}
+\mathcal L_{\mathrm{priv}}^s.
\]

DPA 的重点不是删除 privileged supervision，而是分别在无特权和有特权的 matched-information 条件下组织监督。

## 5. Dual-Source Anchoring（DSA）

单向 reference guidance 只利用了参考解。DAPD 认为两种源互补：

- `y*`：正确性可靠，但可能远离当前策略分布；
- `y`：on-policy、student-reachable，但可能错误。

因此 DSA 对两个方向都应用 DPA：

\[
\mathcal L_{\mathrm{DAPD}}
=\lambda\mathcal L_{\mathrm{DPA}}^{y\to y^*}
+(1-\lambda)\mathcal L_{\mathrm{DPA}}^{y^*\to y},
\qquad \lambda\in[0,1].
\]

其中 `y→y*` 是 reference-guided 方向，`y*→y` 是 rollout-guided 方向。`λ` 越大，越依赖参考指导。

展开后，参考引导方向使用

\[
\mathcal L_{\mathrm{infer}}^{y^*}+\mathcal L_{\mathrm{ent}}^y+\mathcal L_{\mathrm{priv}}^y,
\]

rollout 引导方向使用

\[
\mathcal L_{\mathrm{infer}}^y+\mathcal L_{\mathrm{ent}}^{y^*}+\mathcal L_{\mathrm{priv}}^{y^*}.
\]

## 6. 实现层面的目标与权重

论文实验中，`D(sg[q] || p)` 使用 full-vocabulary、component-clipped forward KL。teacher/student logits 先除以温度 `T=1.1`，每个词项的贡献裁剪到 `c=0.05`：

\[
\ell_c(q,p)=\sum_{v\in\mathcal V}
\min\{q_v(\log q_v-\log p_v),c\}.
\]

不使用 importance-sampling ratio；每个分布直接在对应 sampled prefix 上计算。

官方实现中六项损失的一个主配置为：

| loss | 权重 |
|---|---:|
| `entangled_rollout` | `2/15` |
| `inference_reference` | `2/15` |
| `privileged_rollout` | `2/15` |
| `entangled_reference` | `2/5` |
| `inference_rollout` | `2/5` |
| `privileged_reference` | `4/5` |

权重和为 2；实验还根据模型规模调节 source balance 与 Privileged Anchor 权重。训练中：

1. 当前 LoRA-on policy 采样 rollout，并产生所有 trainable distributions；
2. 两个 Entangled Distillation teacher 使用周期性复制的 LoRA snapshot；
3. Inference Anchor、Privileged Anchor 的 teacher 使用 LoRA-off base policy；
4. 所有 teacher distribution detach，仅 student 侧反传；
5. 每隔一定步数更新 snapshot。

## 7. 数学上的解释

DPA 的 unconditioned path 可用三角不等式解释。对 reference prefix 与 rollout prefix 的配对，令

\[
n^*=p_{\mathrm{None}}^{y^*},\quad
s^*=p_{\mathrm{Self}}^{y^*},\quad
c^y=p_{\mathrm{Cross}}^y,\quad
n^y=p_{\mathrm{None}}^y.
\]

则

\[
\mathrm{TV}(n^*,n^y)
\leq \mathrm{TV}(n^*,s^*)
+\mathrm{TV}(s^*,c^y)
+\mathrm{TV}(c^y,n^y).
\]

其中第一项由 Inference Anchor 压低，第三项由 Entangled Distillation 压低，中间项是 Self–Cross bridge consistency。使用 Pinsker 不等式，可得到期望意义下的上界：

\[
\mathbb E[\mathrm{TV}(n^*,n^y)]
\leq \sqrt{\frac{\mathcal E_{\mathrm{infer}}}{2}}
+\epsilon_{\mathrm{bridge}}
+\sqrt{\frac{\mathcal E_{\mathrm{ent}}}{2}}.
\]

因此，在 bridge 足够一致时，同时优化两项可以显式收紧 inference-time None 行为之间的差异。另一个序列级结论是：若 on-policy 前缀上的 `KL(Cross || None)` 累积损失为 `E_ent`，则长度为 `H` 的 None/Cross 序列分布满足类似

\[
\mathrm{TV}(N,P)\leq \sqrt{\frac{H}{2}\mathcal E_{\mathrm{ent}}}.
\]

这说明 token-level 的 Entangled Distillation 能够通过 autoregressive hybrid bound 传递到序列级行为，但它本身仍需要 anchor 来解决 teacher/student 的信息不对称。

## 8. 实验设计

### 8.1 模型、数据与训练

- **模型规模**：Qwen3-1.7B、4B、8B、14B、32B。
- **训练数据**：OpenThoughts；Reasoning 使用 math domain，Coding 使用 code domain，Instruct 使用 math/code/science 混合域；评测样本从训练集中排除。
- **训练方式**：LoRA，rank 64、scale 128；学习率 `5e-6`；线性 500-step schedule、无 warmup；gradient clipping 0.1；bfloat16、gradient checkpointing；effective batch size 32；seed 42。
- **硬件/软件**：8×A100-80GB；PyTorch 2.8、Transformers 4.57.1、DeepSpeed 0.18.2、vLLM 0.11.0。
- **训练 rollout**：temperature 1.1、top-p 0.95、top-k 20，最多生成 1,024 tokens，训练上下文最多 20,000 tokens。
- **privileged prompt**：将完整参考解插入 teacher user message 的显式 delimiters 中；student rollout prompt 只含问题。Self 则插入正在预测的 completion。参考解只用于训练分布构造，不出现在推理 prompt 中。

### 8.2 评测任务与指标

1. **Reasoning**：AIME24、AIME25、HMMT25；每题生成 12 个 thinking-enabled samples，temperature 1.0、top-p 0.95，最多 38,912 new tokens，验证最终答案；使用 Avg@12。
2. **Coding**：LiveCodeBench v5，官方 Pass@1 evaluator。
3. **Tool/instruction following**：BFCL v3 官方多轮 function-call accuracy、IFBench 官方 verifier score。
4. 结果同时报告六任务平均分、不同模型规模的 reasoning Avg@12、OOD 表现，以及 wrong claims（unsupported answer assertion）来直接衡量 privilege illusion。

### 8.3 Baselines

- Base：原始 Qwen3。
- OPSD：当前 student rollout 上的 `Cross → None` distillation。
- SDFT：on-policy sample + 带 expert demonstration 的 EMA self-teacher。
- SDPO：基于成功 peer rollout 或环境反馈的 EMA teacher。
- Purified OPSD：用 pointwise mutual information correction 去除可由 reference 单独预测的 privileged 成分。
- DOPD：按 advantage gap 与相对概率动态路由 teacher/student supervision。

所有 baseline 使用相同 backbone、数据、on-policy budget，并按各自方法实现优化。

### 8.4 主结果

Qwen3-4B、六任务平均：

| 方法 | AIME24 | AIME25 | HMMT25 | LCB v5 | BFCL v3 | IFBench | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 75.56 | 65.56 | 42.50 | 52.11 | 61.38 | 29.67 | 54.46 |
| OPSD | 76.67 | 67.78 | 43.33 | 52.26 | 61.32 | 30.67 | 55.34 |
| **DAPD** | **77.22** | **72.22** | **46.39** | **53.31** | **61.91** | **33.00** | **57.34** |

DAPD 相比 OPSD 平均提升 **+2.00**。跨规模 reasoning Avg@12 的 DAPD / OPSD / gain 为：

| 规模 | Base | OPSD | DAPD | DAPD−OPSD |
|---|---:|---:|---:|---:|
| 1.7B | 36.85 | 42.04 | 43.98 | +1.94 |
| 4B | 61.20 | 62.59 | 65.28 | +2.69 |
| 8B | 65.00 | 65.00 | 67.41 | +2.41 |
| 14B | 68.80 | 68.89 | 70.93 | +2.04 |
| 32B | 70.00 | 70.28 | 73.06 | +2.78 |

OOD 评测中，DAPD 平均分为 49.64，相比 OPSD 提升 +1.37；其中 LCB v5 提升 +4.82。定性案例显示，OPSD 在推导失败后会“回忆”一个未被支持的答案，DAPD 则继续基于题目完成推导。训练后期（step 250–300）DAPD 的 wrong claims 相比 OPSD 降低约 73%。

### 8.5 消融实验

**(a) Dual-Path**：对每个 guidance source 依次加入 Entangled Distillation、Inference Anchor、Privileged Anchor。只用 Entangled Distillation 最弱；加入 Inference Anchor 后提升；再加入 Privileged Anchor，reference guidance 达到 63.89、rollout guidance 达到 65.09，说明无特权路径和有特权路径均必要。

**(b) Dual-Source**：reference-only 与 rollout-only 均低于同时使用两者；完整 DAPD 达到 65.28，验证可靠性与可达性的互补性。

**(c) 权重/规模敏感性**：最优 reference guidance 权重 `λ` 从 1.7B 的 0.5 降至 4B/8B/14B 的 0.2。较小模型 rollout 噪声更大，需要更强 reference 纠偏；较大模型 rollout 更可靠，可以增加 on-policy guidance。

**(d) Privileged Anchor 权重**：reference-side Privileged Anchor 权重随规模大致从 1.7B 的 0.5 增至 4B 的 1、8B/14B 的 2；不同 source 的最优权重不完全相同，说明固定统一权重并非普适最优。

**(e) Reference-free**：用两个独立 rollout `u,v` 替代 reference；dual-rollout 在 1.7B/4B/8B 相比 OPSD 分别提升 +0.46/+2.41/+2.41。进一步从四个候选中用 verifier 选出 reference-side rollout，最佳分数可达到 43.15、66.11、67.59，说明 DSA 可以扩展到无 curated reference 的场景；但 verified variant 依赖可用的 correctness signal。

## 9. 优点、代价与适用条件

### 优点

- 从监督结构而非简单过滤 teacher logits 的角度解决 privilege illusion；
- 保留 privileged teacher 的正确性指导，同时让训练目标包含 inference-matched anchor；
- 同时利用 reference 的可靠性与 rollout 的 student reachability；
- 不增加推理时开销，且可扩展到 dual-rollout / verified-rollout。

### 代价与限制

- 每个 batch 要构造并计算多个 None/Cross/Self 分布，训练计算和显存开销明显增加；
- `λ`、Inference/Privileged Anchor 权重可能需要按模型规模和架构重新调节；
- verified-rollout 需要自动正确性信号，开放式任务未必具备；
- 方法依赖 Self–Cross bridge 与共享参数产生有效传递，理论分析建立在 bridge consistency、局部梯度兼容等假设上。

## 10. 论文与代码

- 论文（arXiv）：[DAPD: Dual-Anchored Policy Distillation](https://arxiv.org/abs/2608.01735)
- 论文 HTML：[arXiv HTML](https://arxiv.org/html/2608.01735)
- 官方代码仓库：[uanu2002/DAPD](https://github.com/uanu2002/DAPD)
- 代码中的核心目标实现：[dapd/objective.py](https://github.com/uanu2002/DAPD/blob/main/dapd/objective.py)

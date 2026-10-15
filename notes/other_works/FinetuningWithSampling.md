# Finetuning with Sampling：SFT Learns Better Than You Think

> 论文：Aayush Karan、Sitan Chen、Yilun Du（Harvard University），*Finetuning with Sampling: SFT Learns Better Than You Think*，arXiv:2610.02140v1，2026-10-01。
>
> 阅读日期：2026-10-10。阅读范围为论文正文、附录 A–C、PDF 表格，以及官方仓库 [`aakaran/finetuning-with-sampling`](https://github.com/aakaran/finetuning-with-sampling)；代码固定到 **`aa7080885f114fdc5d141f19c09f53008a1c061e`**（2026-10-05）。实验数字均为作者报告，未在本项目复现。

> 项目状态：已按可靠性标准撤回 SamplingSFT 工程实现和生成、筛选数据，保留普通 Nemotron CoT SFT。官方仓库 [issue #1](https://github.com/aakaran/finetuning-with-sampling/issues/1) 报告最终数学轨迹正确率 61.2%，未复现论文附录 C.1 的 95.33%；检索时该 issue 无作者回复。这是尚未解决的复现问题，不能据此认定作者造假。下文实验数字仍仅代表论文声称。

## 1. 一句话总结

**不修改 SFT loss，而是先把专家答案改写成更适合目标模型学习的答案。** 用目标模型在「题目 + 专家解答 + 部分回答」上下文中反复续写，再用同一个模型在「只有题目」的上下文中打分，逐步获得信息尽量不变、模型自身 likelihood 更高的训练数据，最后做普通 SFT。

\[
\boxed{
\text{Expert traces}
\xrightarrow{\text{model-specific projection sampling}}
\text{Boosted traces}
\xrightarrow{\text{ordinary SFT}}
\text{Finetuned model}
}
\]

最重要的阅读区分：

- **理论**：保持正确性/语义等价的 Metropolis–Hastings（MH）链，目标是 base distribution 在正确轨迹集合上的 information projection。
- **实验与公开代码**：附录 C.2 明确改用「likelihood 更高就接受」的贪心近似；源码具体比较的是 **平均 token log-likelihood**，不计算 MH 接受概率和反向 proposal ratio。
- 因而它是有理论动机的数据重写与搜索方法，**公开实现不是具有精确 MH 收敛保证的 sampler**。

## 2. 动机：off-policy 信息与 on-policy 可学习性

普通 SFT 能直接利用专家轨迹，尤其适合模型靠自己反复采样也做不对的难题；但专家写法可能远离模型已有分布，更新会更大，容易损害泛化和原有能力。

RL / on-policy self-distillation 从模型自身的生成分布出发，通常更有利于保留能力；但在模型极难采到正确答案时，纯 RLVR 尤其是 group-relative 方法会缺少有效训练信号。

本文选择调整 **数据分布**，而非增加新的 policy-gradient、importance weighting 或 KL-regularized SFT loss：

1. 专家轨迹提供模型当前难以自行获得的信息；
2. 模型自身将这些信息重新表达；
3. 用无专家提示时的 likelihood 判断这种表达是否更接近 learner；
4. 将改写结果作为固定数据集做 SFT。

这里的「more on-policy」指 **更靠近固定初始模型的分布**，不是每次训练更新后重新 rollout 的严格 online on-policy learning。

## 3. 理论：Information projection + MH

### 3.1 目标分布

固定题目 \(q\)，记初始模型为 \(p(x\mid q)\)，专家轨迹为 \(\tau\)。令 \(\mathcal C\) 为与专家轨迹信息等价的回答集合：

- 数学/化学：借助专家信息得到正确最终答案；
- 事实/医学：保留关键事实或医学信息，以自动 grader 判断。

令 \(\mathcal P_{\mathcal C}\) 为只在 \(\mathcal C\) 上有概率质量的分布。论文 Definition 2、Proposition 1 给出：

\[
p_{\mathcal C}
=\arg\min_{\pi\in\mathcal P_{\mathcal C}}
D_{\mathrm{KL}}(\pi\Vert p)
=\frac{p(x\mid q)\mathbf 1[x\in\mathcal C]}{Z(q)},
\qquad
Z(q)=\sum_{x\in\mathcal C}p(x\mid q).
\]

即 **模型在「回答正确/信息等价」条件下的分布**，不是直接挑出单个 likelihood 最大的答案，也不是 power sampling 的 \(p^\alpha\)。要求 \(Z(q)>0\)，才存在这一条件分布。

证明的关键恒等式：对支撑集在 \(\mathcal C\) 内的 \(\pi\)，

\[
D_{\mathrm{KL}}(\pi\Vert p)
=D_{\mathrm{KL}}(\pi\Vert p_{\mathcal C})-\log Z(q).
\]

若 \(Z\) 很小，直接从模型采样再筛选正确答案的 rejection sampling 很昂贵。作者希望用专家轨迹引导 proposal，避免完全依赖模型盲目探索。

### 3.2 精确 MH 接受概率

当前回答为 \(x\)，候选为 \(x'\)，proposal kernel 为 \(\kappa\)：

\[
A(x',x)=\min\left\{1,
\frac{p_{\mathcal C}(x')\kappa(x\mid x')}
{p_{\mathcal C}(x)\kappa(x'\mid x)}
\right\}.
\]

若当前和候选都在 \(\mathcal C\) 内，归一化常数消去：

\[
A(x',x)=\min\left\{1,
\frac{p(x')}{p(x)}
\frac{\kappa_{\mathcal C}(x\mid x',\tau)}
{\kappa_{\mathcal C}(x'\mid x,\tau)}
\right\}.
\]

随后采样 \(u\sim U(0,1)\)，在 \(u\le A\) 时接受。正确的 proposal correction 不能一般性地省略；MH 也可能接受 base likelihood 较低的候选。

### 3.3 KL 单调下降的保证针对分布

Proposition 3：若转移核 \(\Psi\) 以 \(p_{\mathcal C}\) 为 stationary distribution，令 \(\pi_{k+1}=\Psi(\pi_k)\)，则由 KL 的 data processing inequality：

\[
D_{\mathrm{KL}}(\pi_{k+1}\Vert p_{\mathcal C})
\le D_{\mathrm{KL}}(\pi_k\Vert p_{\mathcal C}),
\]

再由上一恒等式得到：

\[
D_{\mathrm{KL}}(\pi_{k+1}\Vert p)
\le D_{\mathrm{KL}}(\pi_k\Vert p).
\]

这是 **采样分布到 base distribution 的 KL** 的保证，不是每条回答 likelihood 必然增加，也不是 SFT 后模型的 KL 必然减少。长期收敛还需要不可约、非周期等条件；论文没有给出实际 LLM 链的 mixing-time bound。

## 4. Projection sampling：逐块扩展、随机切点重写 suffix

论文 Algorithm 1 不直接反复重生成整条长回答，而是按块逐步扩展：

1. 从已有 prefix 出发，在专家提示下生成下一个 block；
2. 在当前 **整段已生成回答** 中均匀采样一个切点；
3. 保留切点之前的 prefix，从那里重采样 suffix；
4. 按接受规则保留当前回答或替换为候选；
5. 每个 block 重复 \(N_{\mathrm{MCMC}}\) 次，然后继续扩展。

**切点可以落在此前 block 中，不是只编辑最新 block。** 因而 prefix 在下一阶段仍可能被重写，不能把它理解成所有旧 blocks 永久冻结。

Proposal 的基本形式为：

\[
\kappa_{\mathcal C}(\cdot\mid x_{<m},\tau,q)
=p_{\mathrm{base}}\bigl(\cdot\mid
\text{instruction}(q,\tau,x_{<m})\bigr).
\]

提示要求先研究专家解答，再接着部分回答用自己的话推理，最终答案与专家解答一致。专家信息只用于 **proposal**；base scoring 和最终 SFT prompt 不包含专家解答。

### 实际接受规则

附录 C.2 为减少反向 proposal scoring 成本，改成「生成了更高 likelihood 的候选就交换」。[`boost_math.py`][math-sampler] 与 [`boost_sci.py`][sci-sampler] 的实际分数为：

\[
s(x)=\frac1{|x|}\sum_{t=1}^{|x|}
\log p_{\mathrm{base}}(x_t\mid q,x_{<t}),
\qquad
\text{accept iff }s(x')>s(x).
\]

这更准确地描述为 **expert-conditioned suffix search + base-likelihood hill climbing**。它没有精确 MH 的 detailed balance；用平均 logprob 而非 sequence logprob，在不同长度候选间也会改变目标。

此外，实现靠提示词维持正确性，不在每次 proposal 接受前运行 verifier；因此「information-preserving」是经验近似，不是硬约束，见 §6.3。

## 5. 实验设置

| 任务 | 训练 / 测试 | 模型 | 专家信息与评测 |
| --- | --- | --- | --- |
| Chemistry | SciKnowEval Chemistry L-3：1,800 / 600 | Qwen2.5-7B-Instruct；Olmo-3-7B-Instruct | GPT-5 解答；选择题与化学方程式配平 |
| Math | MATH levels 3/4/5：8,230 / 1,024 | Qwen2.5-3B | 原数据专家解答；额外 AMC、MATH500、GSM8K 泛化评测 |
| Medical | HuatuoGPT-o1 SFT：19,704；RL 集随机取 1,000 条测试 | Qwen2.5-7B-Instruct | 原数据专家回答；GPT-5-mini judge |

- 主表是 **single-shot accuracy**；另用 Olmo 化学的 Pass@k 做分布分析。
- 化学/医学的 prior capabilities：MMLU、GPQA、AMC、MATH500、GSM8K。
- 数学的 prior capabilities：Chemistry、MMLU、GPQA。
- 对照：原始专家 SFT、只做一次专家提示重写的 Rewrite SFT（0-step）、OPSD；数学另有 GRPO、使用专家信息的 UFT，以及 Sampling SFT 后接 GRPO。
- SFT 搜索：epochs `{1,2}`、LR `{5e-5,1e-5,5e-6}`、batch size `{16,32,64}`；医学 epochs 扩展到 `{1,2,4,6}`。AdamW、cosine scheduler、gradient norm clipping=1；H100/H200。
- 论文 §5.1 写 maximum length=1,856、MCMC steps=10、**block number=32**。Algorithm 1 的 `B` 则是 block size，两处术语不能直接互换，代码设置见 §7.3。

## 6. 主要结果与证据边界

以下 accuracy 统一转换为 **百分数**；差值用「百分点」。`Avg.` 保留原表给出的数值。

### 6.1 Qwen 主结果（Table 1）

#### Chemistry：Qwen2.5-7B-Instruct

| 方法 | Chemistry | MMLU | GPQA | AMC | MATH500 | GSM8K | Prior Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 34.3 | 68.7 | 25.3 | 42.2 | 73.4 | 91.8 | 59.7 |
| SFT | 61.8 | 58.6 | 24.2 | 27.7 | 62.2 | 87.1 | 52.0 |
| Rewrite SFT | 61.3 | 64.0 | 27.3 | 31.3 | 67.6 | 88.7 | 55.8 |
| OPSD | 61.8 | 65.1 | 22.2 | 34.9 | 71.6 | 90.1 | 56.8 |
| **Sampling SFT** | **66.0** | **69.2** | **30.3** | 37.4 | 67.0 | 89.1 | **58.6** |

化学相比 base 提高 **31.7 pp**、相比 OPSD 提高 **4.2 pp**；prior 平均也优于其他 finetuning 方法。但并非每个旧任务都最好：MATH500/GSM8K 仍低于 OPSD，AMC 仍低于 base。

#### Math：Qwen2.5-3B

| 方法 | MATH(3/4/5) | AMC | MATH500 | GSM8K | New Avg. | Chemistry | MMLU | GPQA | Prior Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 31.5 | 13.3 | 24.5 | 57.9 | 31.8 | 28.3 | 65.1 | 33.3 | 42.2 |
| SFT | 24.3 | 10.0 | 16.8 | 45.5 | 24.2 | 22.2 | 64.8 | 29.8 | 38.9 |
| OPSD | 26.7 | 8.4 | 33.2 | 52.4 | 30.2 | 24.2 | 65.2 | 31.3 | 40.4 |
| GRPO | 45.7 | 24.9 | 31.3 | 80.8 | 45.7 | 27.8 | 65.2 | 31.3 | 41.4 |
| UFT | 47.0 | **29.3** | 29.7 | 74.6 | 45.2 | 28.3 | **65.3** | 32.8 | 42.1 |
| Sampling SFT | 49.5 | 27.7 | 58.2 | 78.2 | 53.4 | 26.6 | 65.1 | 34.3 | 42.0 |
| **Sampling SFT + RL** | **54.5** | 24.1 | **65.2** | **83.0** | **56.7** | **28.5** | 65.2 | **35.4** | **43.0** |

- 原始专家 SFT 对四项数学评测全部退化；Sampling SFT 全部改善。
- Sampling SFT 的 MATH(3/4/5) 比 UFT 高 **2.5 pp**；MATH500 比 GRPO 高 **26.9 pp**，但不是所有 benchmark 都超过 RL。
- SFT 后接 GRPO 的平均表现继续提升，说明这也可以是 **RL 初始化方案**，而不必将 SFT 和 RL 当作互斥路线。

#### Medical：Qwen2.5-7B-Instruct

| 方法 | Medical | MMLU | GPQA | AMC | MATH500 | GSM8K | Prior Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 35.3 | 68.7 | 25.3 | 42.2 | 73.4 | 91.8 | 59.7 |
| SFT | 44.8 | 61.4 | 21.2 | 19.3 | 38.4 | 36.4 | 35.3 |
| Rewrite SFT | 45.7 | 57.1 | 27.3 | 30.1 | 59.2 | 82.1 | 51.2 |
| OPSD | **46.6** | 55.3 | 24.7 | 33.7 | 62.5 | 74.2 | 50.1 |
| Sampling SFT | 45.8 | 52.3 | **29.8** | 31.3 | 61.2 | 83.3 | **51.6** |

医学 Sampling SFT 的新任务 accuracy **低于 OPSD 0.8 pp**；prior 平均比原始 SFT 高 **16.3 pp**（按表中 rounded Avg.），但仍明显低于 base，且 MMLU 是所有方法中最低。应总结为改善综合 trade-off，不能概括成「不遗忘」或「所有指标最好」。

**原表数值注意：** 两处 Qwen2.5-7B-Instruct base 的 `Prior Avg.=59.7` 与五个显示分数的算术均值 **60.28** 不一致；本笔记保留原表，不据此反推精确 retention 比例。

### 6.2 Olmo 与模型专属数据（Appendix B、Table 2）

| Olmo-3-7B-Instruct 方法 | Chemistry | Prior Avg. |
| --- | ---: | ---: |
| Base | 32.8 | 60.1 |
| SFT | 56.7 | 59.0 |
| Rewrite SFT | 56.3 | 60.4 |
| OPSD | **59.7** | 60.0 |
| Sampling SFT | 58.3 | **61.7** |

Olmo 上也不是新任务 accuracy 最优，但原有能力平均更好。

两项额外化学实验尤其值得关注：

- 用 **Olmo-specific boosted data** 训练 Qwen：Chemistry **57.33%**，低于 Qwen 原始专家 SFT 的 61.8%。
- Qwen 用 **50% boosted + 50% 原始专家数据**：**62.14%**，介于原始 SFT 和全 boosted SFT 之间。

这支持数据变换应匹配 **具体 learner checkpoint**；它不是可无条件跨模型复用的通用「高质量数据清洗」。

### 6.3 信息保留不是 100%（Appendix C.1）

作者报告 boosted training dataset 的正确率：

| 数据与模型 | 正确率 |
| --- | ---: |
| Chemistry / Qwen | 94.33% |
| Chemistry / Olmo | 93.94% |
| Math / Qwen | 95.33% |
| Medical / Qwen | 95.86% |

这些是 **改写后训练数据的正确率**，不是 finetuned model 的 test accuracy。说明提示约束会失败，严格的 \(\operatorname{supp}(\pi)\subseteq\mathcal C\) 假设在实践中不完全满足。

更细的风险：Appendix A 的 molar-weight 示例中，boosted 回答把 oxane 说成「five-membered ring」，原答案说的是 six-atom cyclic ether；最终选项仍正确。**最终答案对齐不保证中间推理事实完全保真。**

### 6.4 采样规模与「超越 sharpening」

- Figure 3：专家答案 → boosted 答案，长度归一化的 base log-likelihood 分布明显上移。
- Figure 5：chemistry/Qwen 用 `{0,2,4,6,8,10}` steps，作者报告 KL gap 随采样步数下降，SFT accuracy 总体提高。0-step 是只重写，不是原始专家数据。
- Figure 4：Olmo 化学的 Pass@k 曲线直到 **k=64** 仍高于 base 和 OPSD；部分难题从 base 多次未成功变成约 **67.2% / 53.1%** 成功率。

这支持「不只是提升既有高概率答案的 top-1」的经验判断；但有限次采样未成功并不证明 base 对正确回答的真实概率为零，也不能据此证明支持集出现了数学意义上的新元素。

**KL 估计也需区分：** 经多轮提案与筛选得到的最终 sampler 分布 \(\pi_k\)，不等于单次 proposal \(\kappa\)。公开代码保存的 proposal token logprobs 不能直接作为 \(\log\pi_k(x)\)，因此 Figure 5 的 KL 结果应作为作者报告，而不能由这些日志直接宣称精确重建。

## 7. 官方实现：数据生成与 SFT 的边界

### 7.1 代码导航

| 文件 / 函数 | 实际职责 |
| --- | --- |
| [`boost_math.py`][math] | MATH 数据读取、分片、sampling、最终 grading、JSONL；支持按 `idx` resume |
| [`boost_sci.py`][sci] | Chemistry MCQ / equation balancing；Qwen 和 Olmo sampling |
| `proposal_vLLM()` | 专家条件下生成 suffix，返回新 tokens 和 proposal logprobs |
| `get_logprobs_vLLM()` | 只用题目上下文，通过 `prompt_logprobs` 计算回答的平均 base logprob |
| `proj_samp_vLLM()` | 逐块扩展 + 每块多次随机切点 suffix 重写 + 贪心接受 |
| [`constants.py`][prompts] | Chemistry / Math / Medical proposal prompts |
| [`scripts/boost_math.sh`][math-script]、[`scripts/boost_sci.sh`][sci-script] | Slurm array 0–14，每 shard 一张 H100 |
| [`scripts/train_sft.sh`][train] | 外部 DFT/VeRL 的 FSDP SFT 示例；本仓库没有独立 SFT trainer |
| [`eval_math.py`][eval-math]、[`eval_sci.py`][eval-sci]、[`eval_ood_math.py`][eval-ood] | 单次回答、解析与 grading；OOD loader 名称为 AMC / GSM8K / MATH-TTT |

公开快照提供化学/数学数据与脚本；医学只有 proposal 常量，没有对应的 `boost_med.py` / evaluation pipeline。OPSD、GRPO/UFT、Pass@k 与 KL scaling 的完整训练/分析复现脚本也不在该快照中。

### 7.2 Proposal 和 target scoring 不是同一个条件分布

源码的 proposal 输入为：

```text
chat_template(instruction(question, expert_solution, decoded_prefix))
+ exact prefix token IDs
→ generated suffix
```

同一 prefix 既作为提示中的文本，又作为 assistant generation 的 token prefix；不能简化成只在题目后附专家答案。

Target scoring 则为：

```text
chat_template(question) + candidate response token IDs
→ prompt_logprobs for response only
→ average logprob
```

它不会给题目 token 算分，也不会把「已知专家答案」时的 proposal likelihood 当作 learner 自身的 likelihood。

`get_logprobs_vLLM()` 通过 `generate(max_tokens=1, prompt_logprobs=0)` 完成 scoring，额外生成的一个 token 不参与当前回答分数。实现仍需要独立的 base-scoring 调用，附录的近似主要省去的是反向 proposal 概率计算，并非省掉所有 scoring。

### 7.3 默认参数与论文记号不完全一致

| 参数 | Math 脚本 | Chemistry 脚本 |
| --- | --- | --- |
| Model ID | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-7B-Instruct` / `allenai/Olmo-3-7B-Instruct-SFT` |
| Max new tokens | 1856 | 1856 |
| CLI `block_num` | **32** | **58** |
| 实际 `jump_size=1856/block_num` | **58 tokens** | **32 tokens** |
| Steps / block | 10 | 10 |
| Proposal temperature | 0.6 | 0.6 |
| Proposal repetition penalty | 1.1 | 1.05 |
| Model prompt mode | chat | chat |

注意：

- Algorithm 1 的 `B` 是 **tokens/block**；源码 `block_num` 是 **block count**，且要求整除 1856。
- §5.1 声称各任务 block number=32，chemistry 的公开默认值却是58，不能把默认脚本当成正文参数的完全复现。
- 论文称 Olmo-3-7B-Instruct，代码具体加载 `Olmo-3-7B-Instruct-SFT`；数学使用非 Instruct 的 Qwen2.5-3B，却硬编码 chat mode。复现应固定实际 checkpoint 和模板。
- Sampling 脚本虽有 `--seed`，实际切点 RNG 固定 `random.seed(0)`，生成 `SamplingParams` 未传该 CLI seed；仅保存参数值不能确保完全可复现。

### 7.4 输出语义

`boosted_<shard>.jsonl` 主要包含：

```text
idx, prompt, solution, gen,
target_log_prob_cur, token_logprobs,
runtime_sec, answer, is_correct
```

- `gen`：最终 boosted response；`solution`：原专家答案。
- `target_log_prob_cur`：无专家条件下的 **平均 base logprob**，不是 joint probability。
- `token_logprobs`：各片段在其生成时 **专家 + 当时 prefix** 条件下的 proposal logprobs；多轮替换后是拼接的历史记录，不能当成最终 response 的 base logprobs 或完整 sampler density。
- `is_correct`：整条生成完毕后才打分，sampling 内部不使用；错误样本也照常写出。README 没有明确说明最终 SFT 是否过滤这些样本。
- `trace_boosted_<shard>.jsonl`：记录每块初始生成与被接受候选的 prefix/continuation，以及 EOS 信息；用于重建生成上下文，不保存完整 teacher vocabulary logits。普通 SFT 不依赖它。

普通 SFT 的数据边界是 `prompt=question`、`response=gen`，**不要将专家解答或重写指令放进最终训练 prompt**。README 建议合并 JSONL 后转换为 parquet，但快照没有完整转换脚本。

### 7.5 训练、评测的复现完整度

- README 引用 [`yongliang-wu/DFT`](https://github.com/yongliang-wu/DFT) 的训练设置；示例实际调用 `verl.trainer.fsdp_sft_trainer`。示例超参为 LR `5e-5`、2 epochs、global batch16、BF16、max length2048、4 GPUs；不是提出新 DFT loss。
- 示例将 train file 同时设为 validation file，并关闭周期测试，不提供独立验证集选择 checkpoint 的完整流程。
- 评测脚本默认 temperature0.6、max tokens1856；MMLU/GPQA 转交 lm-evaluation-harness。
- 当前评测代码有未接齐的引用：`eval_math.py` 调用了未定义的 `format_prompt`；另外两个 eval 脚本也未定义它，且存在与目录不一致的 grader imports。这里是源码静态检查，不是已运行的失败报告。

因此该仓库更适合作为 **sampling 算法和数据格式的参考实现**，不能将其视作已验证的一键全实验复现包。本次没有安装仓库环境或运行其生成、训练、评测。

## 8. 成本与方法评价

### 8.1 离线一次性，不等于便宜

论文式 (9)，用 \(b\) 明确表示 tokens/block，避免与 block count 混淆：

\[
\mathbb E[\text{resampled tokens}]
\approx |\mathcal D|\frac{N_{\mathrm{MCMC}}T^2}{4b}.
\]

这是随机切点 suffix regeneration 的估算，尚不包含初始扩展、专家提示 prefill、base scoring、grader、SFT 及调度成本。

以 math defaults `T=1856, b=58, N=10` 代入，近似为 **每条满长回答 148,480 个重采样 tokens**，约80倍回答长度；EOS 提前终止会减少实际成本。Chemistry defaults 的 block 更小，满长预算更高。

优势是这笔采样开销发生在训练前，部署不需要 MCMC；但数据与特定模型匹配，换模型后未必能摊销复用。论文没有给出完整、等算力的 GPU-hour / wall-clock 成本对照。

### 8.2 最有价值的贡献

1. **把 off-policy 学习问题部分转为数据适配问题**：保留普通 SFT，训练接入成本低。
2. **明确分离信息来源与 learner likelihood**：专家用于提案，自身分布用于打分。
3. **比一次性「用自己的话重写」更强**：Qwen 化学和数学收益明显，采样步数成为离线 compute axis。
4. **可与 RL 组合**：数学 Sampling SFT + GRPO 的结果优于单独使用两者。
5. **展示 model-specific data 的必要性**：跨模型 boosted data 迁移会退化。

### 8.3 不应超出证据的结论

- 精确 MH 理论不能直接覆盖贪心、length-normalized、无逐次 verifier 的公开实现。
- 提高样本 likelihood 本身不保证整个数据分布的 KL 下降：KL 还依赖数据分布自身的 entropy。
- 信息保留约94–96%，且最终答案正确不保证 reasoning factuality。
- 新任务和旧任务存在 trade-off：医学与 Olmo 化学的新任务 accuracy 没有超过 OPSD；医学 MMLU 遗忘仍严重。
- 主要实验为3B/7B模型、短于约2k的回答；尚未验证长 agent trajectory、工具交互、RWKV 或大规模持续在线学习。
- 多次改写的算力、数据长度/风格、超参搜索及外部专家成本都可能影响比较；现有结果不是严格等总预算的「SFT普遍优于RL」证明。

## 9. 对 SkyRL / RWKV 的启发（迁移判断，非已实现功能）

**最自然的落点是离线 SFT 数据构造，不是改现有 RL loss。**

建议的最小对照为：同一组 training prompts、同一 RWKV checkpoint、同一专家轨迹，比较原始专家 SFT → 0-step rewrite SFT → 多轮搜索 SFT → 搜索 SFT 后接原生 RL。固定 SFT token budget，并分别记录采样预算与训练预算。

关键契约：

1. **Learner-specific scoring**：proposal/scorer 使用明确版本的目标 RWKV checkpoint；不能直接拿 Qwen/Olmo 的 boosted dataset 就视为 RWKV on-policy 数据。
2. **上下文与 recurrent state 分离**：专家条件 proposal 与无专家 base scoring 必须各自从正确的 prompt/state 开始；随机切点重写后，suffix 依赖的新 recurrent state 必须来自保留的准确 prefix。
3. **Token/template 对齐**：固定 RWKV prompt template、tokenizer、token IDs、EOS、response loss mask；不能将历史 proposal logprobs 当作 learner/rollout probabilities。
4. **分开记录 decoding 设置**：论文 proposal 的 temperature0.6、repetition penalty 是其离线搜索配方，不是本项目 RL rollout 的配置；不能改变原有 rollout/eval contract 来迁就该论文。
5. **验证信息保真**：至少统计最终正确率、错误类型与专家/生成长度；复杂推理可抽样检查中间事实。若加入 verifier filtering，明确标注为扩展并报告保留率，不把它当成论文已有硬约束。
6. **成本与旧能力一起评测**：除新任务 reward，报告旧任务 retention、acceptance rate、mean base logprob、generated/scored tokens、runtime；不能只看改写答案的 likelihood。
7. **不改模型原生路径**：SFT 继续复用 SkyRL 原生训练接口，后续 RL 保持 Trainer/Generator/InferenceEngine/Environment、权重同步与既有 reward pipeline。

它与 [FlashREINFORCE](FlashREINFORCE.md) / [SAO](SAO.md) 解决不同阶段的问题：本文让专家数据更适合 **训练初始化**，后两者处理 **online RL rollout、baseline、advantage 与 staleness**，可以组合但不能互相替代。

## 10. 本项目状态

SamplingSFT 的专用采样实现、命令入口和测试已撤回。普通 SFT 使用
examples/train/rwkv/prepare_sft.py 和 examples/train/rwkv/run_rwkv_sft.sh，
直接读取 Nemotron 原始 reasoning_content 与 content；按题划分数据的逻辑
位于普通 SFT 脚本中，不再依赖采样目录。

## 11. 来源

- [arXiv 摘要与 v1 元数据](https://arxiv.org/abs/2610.02140v1)
- [论文全文](https://arxiv.org/html/2610.02140v1)：§4 理论/算法，§5 实验，Appendix A 示例、B 额外实验、C 信息保留与实际采样近似。
- [PDF](https://arxiv.org/pdf/2610.02140v1)：Table 1（p.9）、Table 2（p.18）；表格数据以 PDF 为准。
- [项目页](https://aakaran.github.io/finetuning_with_sampling/)
- [官方仓库固定快照](https://github.com/aakaran/finetuning-with-sampling/tree/aa7080885f114fdc5d141f19c09f53008a1c061e)
- [README][readme]：sampling 输出、外部 SFT 训练、数据转换与评测说明。
- 以下源码链接均固定到上述 commit，不随默认分支更新。

[readme]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/README.md
[math]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/boost_math.py
[sci]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/boost_sci.py
[math-sampler]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/boost_math.py#L161-L251
[sci-sampler]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/boost_sci.py#L159-L244
[prompts]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/constants.py
[math-script]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/scripts/boost_math.sh
[sci-script]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/scripts/boost_sci.sh
[train]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/scripts/train_sft.sh
[eval-math]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/eval_math.py
[eval-sci]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/eval_sci.py
[eval-ood]: https://github.com/aakaran/finetuning-with-sampling/blob/aa7080885f114fdc5d141f19c09f53008a1c061e/eval_ood_math.py

# OPD²：On-Policy Delta Distillation

> 阅读对象：Byeongho Heo、Jaehui Hwang、Sangdoo Yun、Dongyoon Han，*On-Policy Delta Distillation*，arXiv:2607.15161v1（2026-07-16）。本文整理其核心思路、数学表示与实验设计。

## 1. 一句话总结

OPD² 不再直接把 reasoning teacher 的输出分布当作蒸馏目标，而是比较 **reasoning-tuned teacher 与其 base model 的差异**，用这个差异提取“reasoning tuning 新学到的能力”。具体地，它以 delta signal 为主要 token-level advantage，并通过中心化（centering）和与普通 OPD 信号的方向一致性约束，避免训练发散及偏离 teacher。

---

## 2. 背景与问题

设学生策略为 \(\pi_\theta\)，teacher 为 \(\pi^*\)，输入为 \(x\)，学生 on-policy 采样序列为 \(y\sim\pi_\theta(\cdot\mid x)\)。普通 OPD 在学生实际访问的状态上，让 teacher 对学生采样的 token 打分，因此相比 teacher-generated SFT 能减轻 exposure bias，并提供密集的 token-level 监督。

但普通 OPD 的目标是“模仿 teacher 当前输出”，其中混合了：

- teacher 在预训练阶段已经具备的语言、风格与一般知识；
- teacher 在 reasoning SFT/RL 阶段新增的推理能力。

对于 reasoning post-training，真正希望迁移的是第二部分。论文的基本假设是：teacher 与其 instruction/reasoning tuning 之前的 base model 之间的变化，可以作为这部分能力的 learning trace。

---

## 3. 普通 OPD 的数学形式

知识蒸馏在 on-policy 数据上的目标可写为：

\[
\mathcal L_{\mathrm{OPD}}(\theta)
=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[D_{\mathrm{KL}}\left(\pi_\theta(y\mid x)\,\middle\|\,\pi^*(y\mid x)\right)\right].
\]

工程实现采用 RL-like 的 sampled-token gradient。对上下文 \((x,y_{<t})\) 和学生采样 token \(y_t\)，普通 OPD reward 为：

\[
R_t^{\mathrm{OPD}}
=\log\pi^*(y_t\mid x,y_{<t})
-\log\pi_\theta(y_t\mid x,y_{<t}).
\]

训练梯度为：

\[
\nabla_\theta J_{\mathrm{OPD}}
=\mathbb E\left[
\sum_{t=1}^{T}R_t^{\mathrm{OPD}}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right].
\]

它只对 sampled token 施加增强/抑制信号，不直接对未采样 token 求蒸馏梯度。

---

## 4. OPD² 的核心方法

### 4.1 Delta signal：蒸馏 teacher 的“变化”

令 \(\pi^*_{\mathrm{base}}\) 表示 teacher 在 reasoning/instruction tuning 之前的 base model。OPD² 定义：

\[
\boxed{
R_t^\Delta
=\log\pi^*(y_t\mid x,y_{<t})
-\log\pi^*_{\mathrm{base}}(y_t\mid x,y_{<t})
}
\]

直觉：

- \(R_t^\Delta>0\)：reasoning-tuned teacher 相比 base 更偏好该 sampled token，应提高其概率；
- \(R_t^\Delta<0\)：该 token 更像 base model 的偏好，或不是 reasoning tuning 带来的能力，应抑制；
- teacher 与 base 从相同预训练权重出发，差分通常比 teacher-student 差分更能反映 reasoning 相关的相对变化。

论文的 token/词语分析显示，delta signal 往往增强 `hence`、`thus`、`however`、`regardless` 等逻辑连接词，抑制 `perhaps`、`tackle`、`consider`、`analyze` 等较泛化的解题叙述或不确定表达。

### 4.2 Centering：将 reward 变成 action-relative advantage

仅使用 \(R_t^\Delta\) 存在一个问题：它不含学生策略项，因此在理想化情况下可能把概率推向 delta reward 最大的 token（接近 one-hot），而不是在 teacher 对齐点停止。

对学生当前分布下的期望 reward 做中心化。普通 OPD advantage 为：

\[
A_t^{\mathrm{OPD}}
=R_t^{\mathrm{OPD}}
-\mathbb E_{\tilde y_t\sim\pi_\theta(\cdot\mid x,y_{<t})}
\left[\log\pi^*(\tilde y_t\mid x,y_{<t})
-\log\pi_\theta(\tilde y_t\mid x,y_{<t})\right].
\]

Delta advantage 为：

\[
A_t^\Delta
=R_t^\Delta
-\mathbb E_{\tilde y_t\sim\pi_\theta(\cdot\mid x,y_{<t})}
\left[\log\pi^*(\tilde y_t\mid x,y_{<t})
-\log\pi^*_{\mathrm{base}}(\tilde y_t\mid x,y_{<t})\right].
\]

实际实现只在学生分布的 top-\(k\) token 上近似期望，论文使用 \(k=1024\)，以降低 full-vocabulary 计算和显存开销。

### 4.3 Joint conditioning：只保留共同下降方向

为兼顾 reasoning trace 与 teacher alignment，OPD² 只在两个 advantage 方向一致时更新：

\[
A_t^{D^2}=
\begin{cases}
A_t^\Delta,&A_t^\Delta A_t^{\mathrm{OPD}}>0,\\
0,&\text{otherwise}.
\end{cases}
\]

含义是：

1. 用 \(A_t^{\mathrm{OPD}}\) 判断该 token 的更新方向是否仍与 teacher-student 对齐；
2. 方向一致时，用 \(A_t^\Delta\) 提供更新幅度；
3. 方向冲突时不更新，避免只追逐 teacher/base 差异而远离 teacher 的输出分布。

因此，当学生已经接近 teacher，普通 OPD advantage 接近零，OPD² 的更新也会自然关闭。

最终梯度：

\[
\boxed{
\nabla_\theta J_{\mathrm{OPD^2}}
=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}A_t^{D^2}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right]
}
\]

### 4.4 训练流程

1. 从混合问题集取一个问题，学生按当前策略 rollout 一条 completion。
2. 对同一上下文和 sampled tokens，分别计算 student、reasoning teacher、teacher-base 的 log-probability。
3. 计算 \(R_t^{\mathrm{OPD}}\)、\(R_t^\Delta\)，再计算 top-1024 近似期望并中心化。
4. 应用符号一致性门控，得到 \(A_t^{D^2}\)。
5. 以 sampled-token advantage 进入 GRPO/PPO-style clipped loss；论文实现基于 TRL 的 `GRPOTrainer`，但启用单 completion、token-level signal，并禁用 group normalization。

官方代码的实现等价地对学生、teacher、teacher-base 做 forward；用学生 top-k 概率作为期望权重，并对符号冲突的 token 将 signal 置零。

---

## 5. Delta signal 的实证分析设计

论文先不训练模型，而是分析 OPD signal 与 delta signal 的差异：

### 5.1 Word cloud

- 学生：Qwen3-1.7B；teacher：Qwen3-4B-Thinking-2507；teacher-base：Qwen3-4B-Base。
- 从 OpenMathReasoning 采样 10k 数学问题。
- 用学生生成 response，分别计算 OPD、base 与 delta 信号，并按信号强度生成词云。
- 为处理 reward 整体偏负，先进行中心化。

### 5.2 Token-level 可视化

- 使用数学、科学、代码各一个简单问题。
- 人工构造/筛选包含错误推理的输入，而不是使用真实 rollout。
- 对每个 token 比较 OPD 与 delta 的正负方向，观察 delta 是否更能抑制错误推理片段。

### 5.3 大规模统计

- 学生：Qwen3-8B；teacher：Qwen3-30B-A3B-Thinking-2507；base：Qwen3-30B-A3B-Base。
- Math、Code、Science 各随机采样 10k 问题，使用学生生成 reasoning trace。
- 统计将 OPD 替换为 delta 后，某 token 的 signal 增强/抑制幅度至少为 1 的比例。

---

## 6. 主实验设计

### 6.1 模型与 teacher 配置

学生覆盖 Qwen3-1.7B、Qwen3-4B、Qwen3-8B，以及 Gemma4-E4B-it；teacher 使用同系列更大的 reasoning/instruction 模型，teacher-base 使用相应 `-Base` 模型：

| 学生 | Teacher（示例） | Teacher-base | 模式 |
|---|---|---|---|
| Qwen3-1.7B | Qwen3-4B-Instruct-2507 | Qwen3-4B-Base | no-think |
| Qwen3-4B/8B | Qwen3-30B-A3B-Instruct-2507 | Qwen3-30B-A3B-Base | no-think |
| Qwen3-1.7B/4B/8B | 对应 Thinking-2507 | 对应 Base | thinking |
| Gemma4-E4B-it | Gemma-4-31B-it | Gemma-4-31B | thinking |

Qwen3 同时评估 no-think 与 thinking；Gemma4 只评估 thinking。

### 6.2 训练数据

构造三领域 1:1:1 的混合数据：

- Math：OpenMathReasoning；
- Science：OpenScienceReasoning-2；
- Code：OpenCodeReasoning。

制作 100k questions 的 balanced subset；训练实际少于 30k samples，每个问题最多使用一次，不使用数据集原有 reasoning trace/answer 作为监督，只使用问题并让学生在线生成 completion。

### 6.3 训练设置

- 方法：OPD、ExOPD、OPD²；ExOPD 使用论文所述 \(\lambda=1.25\)。
- 每个问题只生成 1 条 completion，并分别送入 student、teacher、teacher-base。
- 基于 TRL `GRPOTrainer` 改造：single completion、token-level signals、关闭 group normalization。
- forward 后 softmax temperature：0.7；rollout temperature：0.7。
- 最大 completion length：8k；训练步数：100。
- AdamW，learning rate \(5\times10^{-6}\)，cosine decay，gradient clipping 1.0。
- 关闭 reference-model KL regularization；所有 reward 统一乘 0.1，减少过于频繁的 gradient clipping。
- centering 的期望使用 top-k=1024 近似。
- vLLM rollout：Qwen3-1.7B 使用 colocate，其余模型使用独立 server 节点。
- 报告最终 step 结果，不挑选训练中 peak checkpoint。

### 6.4 评测任务与指标

全部使用 pass@1，并对重复采样取平均：

| 领域 | Benchmark |
|---|---|
| Math（7） | AIME24、AIME25、AMC23、HMMT25、MATH500、OlympiadBench、ReasoningGym Math |
| Code（4） | CodeContests、CodeForces、LiveCodeBench v5、ReasoningGym Algorithm |
| Science（3） | GPQA、SuperGPQA、SciBench |

对比对象：原始 student、普通 OPD、ExOPD、OPD²。实验同时覆盖不同模型规模、thinking/no-think 模式和 Qwen3/Gemma4 两个模型系列。

---

## 7. 主要结果与诊断实验

### 7.1 主结果

OPD² 在 Qwen3 的 no-think 模式中，Math/Code/Science 三个领域、1.7B/4B/8B 三个规模上都取得最高平均分。例如 Math 平均分：

| Student | 原始 | OPD | ExOPD | OPD² |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 34.8 | 51.0 | 51.4 | **54.6** |
| Qwen3-4B | 45.8 | 64.0 | 66.4 | **70.3** |
| Qwen3-8B | 46.9 | 65.9 | 67.8 | **71.6** |

在 thinking 模式中，普通 OPD/ExOPD 可能损伤原有能力；OPD² 仍在三个规模上取得最高 Math、Code、Science 平均分。以 Math 为例，Qwen3-1.7B/4B/8B 的平均分分别从 59.2/73.3/73.7 提升至 62.7/74.8/75.9。

Gemma4-E4B-it 也呈现类似趋势：Math 平均分从 60.6 提升至 OPD² 的 67.8；Code 上 OPD² 虽未超过原始模型，但比 OPD/ExOPD 更能保留原有能力；Science 平均分从 47.0 提升到 48.8。

### 7.2 Training dynamics

在 Qwen3-4B no-think 的训练曲线中，所有方法早期都快速提升；OPD/ExOPD 较早达到峰值，随后 plateau 或下降。OPD² 早期提升更大，并在后续 step 中保持更高性能。因此作者认为优势不是偶然的单个 checkpoint 峰值，而是整个短训练轨迹中较稳定的收益。

### 7.3 Ablation

消融三项：

1. **No delta signal**：将 \(A_t^\Delta\) 替换成 \(A_t^{\mathrm{OPD}}\)；
2. **No condition**：移除 \(A_t^\Delta A_t^{\mathrm{OPD}}>0\) 门控；
3. **No centering**：将中心化的 \(A_t^\Delta\) 替换为原始 \(R_t^\Delta\)。

最重要的组件是 delta signal。以 no-think 的 Math/Code/Science 为例，OPD² 的 54.6/29.4/38.8 在去掉 delta 后降至 50.5/22.5/35.9；thinking 模式也有明显下降。去掉 condition 或 centering 的影响较小且不完全一致，说明它们主要起稳定化和约束作用。

### 7.4 计算成本

相对于 OPD，OPD² 需要额外加载并 forward teacher-base：

| 方法 | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Gemma4-E4B |
|---|---:|---:|---:|---:|
| OPD | 4.4h | 7.3h | 7.6h | 12.7h |
| OPD² | 5.5h（+24%） | 9.3h（+28%） | 9.6h（+27%） | 13.8h（+8%） |

OPD² 的额外开销与 ExOPD 接近；作者指出 reward computation 尚未充分优化，仍有降低空间。由于 OPD 类方法通常只需较少训练步数，额外成本在短 post-training 场景下相对可接受。

---

## 8. 对 SkyRL/RWKV 集成的启示

1. **需要三套 forward/logprob**：student、reasoning teacher、teacher-base 必须对同一 sampled token 和同一上下文计算 log-prob；因此 rollout token、chat template、loss mask 必须严格对齐。
2. **delta 不等于 teacher logprob**：不能把 teacher logprob 直接作为 reward；核心是 \(\log p_{teacher}-\log p_{base}\)。
3. **期望项使用 student 分布**：centered delta 的 baseline 是在学生当前 token 分布下的期望，工程上可采用 top-k 近似。
4. **只更新符号一致 token**：需要保留普通 OPD advantage 以实现 gate；冲突位置 advantage 置零。
5. **短训练和曲线评估很重要**：论文只训练 100 steps，且 OPD/ExOPD 后期可能退化；实现时应保存中间 checkpoint、记录 reward/advantage 分布并运行定期评测。
6. **teacher-base 版本必须匹配**：teacher 与 teacher-base 应来自同一模型家族/相同预训练起点，只相差 reasoning/instruction tuning，否则 delta 的语义会混入架构或预训练差异。

## 9. 论文与代码

- 论文（arXiv）：<https://arxiv.org/abs/2607.15161>
- 论文 HTML：<https://arxiv.org/html/2607.15161v1>
- 官方代码仓库：<https://github.com/naver-ai/opd2>
- 官方代码中实现入口：<https://github.com/naver-ai/opd2/blob/main/src/open_r1/trainers/opd2_trainer.py>
- 相关训练框架：<https://github.com/huggingface/trl>

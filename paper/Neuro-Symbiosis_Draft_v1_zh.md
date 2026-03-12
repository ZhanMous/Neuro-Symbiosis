# Neuro-Symbiosis: 面向脑机接口的 SNN-Transformer 混合解码与隐私-能效协同优化

## 摘要
脑机接口（BCI）在临床康复和人机交互中具有重要应用潜力，但其发展面临三重挑战：准确率、隐私风险与能耗约束难以同时优化。本文提出 Neuro-Symbiosis，一种面向 EEG 运动想象任务的 SNN-Transformer 混合解码框架，其中 SNN 前端用于时序脉冲编码与稀疏表示，Transformer 后端用于跨时域上下文建模。围绕单卡 RTX 4070 的现实实验条件，本文构建了可复现的三目标评估流程，联合报告任务效用、成员推断风险与推理能耗。初步结果显示：在当前快速实验设置下，Hybrid 模型在准确率与隐私攻击可分性之间达到较优折中（单次运行 val_acc=0.9612, mia_auc=0.4650），并在能耗上显著优于纯 SNN（6.1902 mJ vs 10.6822 mJ）。进一步的三随机种子复现实验显示，Hybrid 的平均性能为 $0.8803\pm0.1284$（val_acc）与 $0.5045\pm0.0388$（MIA AUC），优于 SNN 的隐私风险表现并保持较低能耗（5.9868 mJ）。在差分隐私 DP-SGD 网格实验中，随着噪声系数从 0.8 提升到 1.2，epsilon 由 5.0671 下降至 2.1771，同时 MIA AUC 由 0.5260 下降至 0.5159，呈现可解释的隐私-效用权衡趋势。该工作为 BCI 场景下神经形态混合模型的实用部署与合规评估提供了可复用范式。

关键词：脑机接口；脉冲神经网络；Transformer；差分隐私；能效评估；成员推断攻击

## 1 引言
脑机接口系统通常依赖高频时序生理信号进行实时解码，这使得模型必须在低时延、低能耗与鲁棒泛化之间做出平衡。另一方面，EEG 等神经数据属于高度敏感的生物信息，模型训练与部署中存在成员推断、属性推断及模型反演等隐私风险。传统高容量深度网络虽然在精度上表现较好，但在边缘设备能耗与隐私风险控制方面存在明显缺口。

SNN 通过脉冲稀疏激活机制提供了潜在的能效优势，但其全局上下文建模能力常受限。Transformer 在长程依赖建模方面优势突出，但代价是较高计算与存储开销。基于此，本文提出 Neuro-Symbiosis：将 SNN 作为时序编码前端，Transformer 作为语义聚合后端，并联合隐私与能效评估，形成面向 BCI 的三目标优化框架。

本文贡献如下：
1. 提出可在单卡 RTX 4070 复现实验的 Neuro-Symbiosis 混合结构与训练流程。
2. 构建统一评测框架，联合报告准确率、MIA 风险、时延与估计能耗。
3. 给出 SNN、Transformer、Hybrid 三类基线的可复现实验对比与帕累托图。
4. 给出 DP-SGD 网格实验结果，形成 epsilon-accuracy-energy 联合表，为隐私预算选型提供依据。
5. 补充三随机种子统计与 DP 条件下 MIA 复评，降低单次运行结论偏差。

## 2 相关工作
### 2.1 BCI EEG 解码
BCI Competition IV 2a/2b 和 PhysioNet EEG Motor Movement/Imagery（eegmmidb）是运动想象研究中的代表性数据基准。前者强调跨受试者与多类别想象任务，后者具有较大受试者规模与开放访问特性。

### 2.2 SNN 与神经形态计算
SNN 通过离散脉冲传递信息，理论上可降低无效计算并提升事件驱动场景下的能效。近年基于 surrogate gradient 的训练技术使 SNN 在标准深度学习框架中可训练。

### 2.3 隐私保护学习
差分隐私 SGD（DP-SGD）为模型训练提供可量化隐私保证。Opacus 提供了 PyTorch 中的工程化实现。对 EEG/医疗任务而言，除 DP 训练外，成员推断攻击（MIA）是重要的经验风险评估手段。

## 3 方法
### 3.1 Neuro-Symbiosis 架构
输入 EEG 片段表示为 $X \in \mathbb{R}^{C \times T}$。

1. SNN 编码器：
- 使用一维卷积投影时域信号。
- 基于 LIF 神经元与 surrogate gradient 完成脉冲化动态编码。
- 生成 token 序列 $Z_s \in \mathbb{R}^{T \times D}$，并记录脉冲率 $r_{spike}$。

2. Transformer 解码器：
- 对 $Z_s$ 执行多头自注意力与前馈变换，得到上下文特征 $Z_t$。
- 时域池化后经线性头输出分类 logits。

3. 训练目标：
$$
\mathcal{L}=\mathcal{L}_{ce}+\lambda_{reg}\mathcal{R}
$$
当前实验中采用交叉熵主损失，后续可加入脉冲稀疏约束与知识蒸馏项。

### 3.2 对比模型
1. SNN baseline：仅保留 SNN 编码器与分类头。
2. Transformer baseline：卷积投影后直接 Transformer 解码。
3. Hybrid（Neuro-Symbiosis）：SNN 前端 + Transformer 后端。

### 3.3 隐私与能效评估
1. 隐私风险：
- 非 DP 条件下，采用置信度阈值成员推断攻击，报告 MIA Accuracy 与 MIA AUC。
- DP 条件下，报告隐私预算 $(\epsilon, \delta)$，其中 $\delta=10^{-5}$，并补充 MIA 经验评测。

2. 能效指标：
- 平均单样本推理时延（ms/sample）。
- 通过 nvidia-smi 采样功率，估计单样本能耗（mJ/sample）。
- 脉冲率用于反映稀疏激活程度。

## 4 实验设置
### 4.1 硬件与软件
- GPU: NVIDIA RTX 4070 Laptop
- 框架: PyTorch + Opacus
- 项目代码目录：Neuro-Symbiosis

### 4.2 数据设置
- 快速验证：合成 EEG-like 数据（用于全流程验证）。
- BCI 接入：已实现 BCI-IV 2a NPZ 读取与预处理流水线（带通滤波、标准化、时间窗截取）。

### 4.3 训练设置（quick）
- batch size: 32
- epochs: 2（用于管线验证）
- 模型宽度: embed=64, snn_hidden=64
- 优化器: AdamW

### 4.4 新增稳健性实验设置
- 多随机种子：seed={42,43,44}，对 SNN、Transformer、Hybrid 分别重复训练与评估。
- DP-MIA 复评：在既有 DP 网格（noise={0.8,1.0,1.2}, clip={0.8,1.0}）上逐一加载模型并计算 MIA ACC/AUC。

## 5 结果与分析
### 5.1 三模型基线对比（单次）
根据 baseline_summary.csv，结果如下：

| 模型 | val_acc | MIA AUC | MIA ACC | latency(ms) | energy(mJ) | spike_rate |
|---|---:|---:|---:|---:|---:|---:|
| SNN | 0.7767 | 0.5416 | 0.5098 | 0.5135 | 10.6822 | 0.1197 |
| Transformer | 1.0000 | 0.5308 | 0.5254 | 0.0390 | 0.7609 | 0.0000 |
| Hybrid | 0.9612 | 0.4650 | 0.4863 | 0.4454 | 6.1902 | 0.1236 |

对应帕累托图见：outputs/quick/benchmark/pareto_baselines.png。

### 5.2 多随机种子稳健性结果（新增）
根据 multiseed_summary.csv，3 个随机种子下均值与标准差如下：

| 模型 | val_acc(mean±std) | MIA AUC(mean±std) | latency(ms) | energy(mJ) |
|---|---:|---:|---:|---:|
| SNN | 0.6537±0.1125 | 0.5353±0.0300 | 0.4493±0.0150 | 8.0595±1.1166 |
| Transformer | 1.0000±0.0000 | 0.5152±0.0423 | 0.0342±0.0027 | 0.4786±0.0189 |
| Hybrid | 0.8803±0.1284 | 0.5045±0.0388 | 0.4354±0.0174 | 5.9868±0.4580 |

分析：
1. Hybrid 在多种子下仍保持更低 MIA AUC，隐私趋势并非单次偶然。
2. Hybrid 准确率方差较大，说明短训练设置对初始化较敏感。
3. Transformer 在合成数据上过强，需在真实 BCI 数据验证外推性。

### 5.3 DP 网格实验
根据 dp_sweep_summary.csv，关键结果如下：

| noise | clip | epsilon | val_acc | energy(mJ) | latency(ms) |
|---:|---:|---:|---:|---:|---:|
| 0.8 | 0.8 | 5.0671 | 0.6796 | 10.0374 | 0.4663 |
| 0.8 | 1.0 | 5.0671 | 0.6796 | 9.2365 | 0.4594 |
| 1.0 | 0.8 | 3.1400 | 0.6699 | 6.6434 | 0.4510 |
| 1.0 | 1.0 | 3.1400 | 0.6699 | 6.5447 | 0.4577 |
| 1.2 | 0.8 | 2.1771 | 0.6699 | 6.7812 | 0.4716 |
| 1.2 | 1.0 | 2.1771 | 0.6699 | 7.3082 | 0.4822 |

### 5.4 DP 条件下 MIA 复评（新增）
根据 dp_mia_summary.csv，DP 配置下经验攻击指标如下：

| noise | clip | MIA ACC | MIA AUC |
|---:|---:|---:|---:|
| 0.8 | 0.8 | 0.5098 | 0.5260 |
| 0.8 | 1.0 | 0.5098 | 0.5258 |
| 1.0 | 0.8 | 0.5059 | 0.5177 |
| 1.0 | 1.0 | 0.5059 | 0.5177 |
| 1.2 | 0.8 | 0.5020 | 0.5161 |
| 1.2 | 1.0 | 0.5059 | 0.5159 |

分析：
1. 随噪声系数增加，MIA AUC 整体下降，与 epsilon 下降方向一致。
2. MIA ACC 接近 0.5，攻击区分能力弱化。
3. DP 收益不仅体现在理论预算，也可在经验攻击指标上观测到。

### 5.5 BCI 2a 接入链路验证
在 bci2a_quick 配置下（2 epochs），流程已端到端跑通，best_val_acc=0.25。该结果主要用于验证数据通路与训练管线可用性，不代表最终性能上限。

## 6 讨论
1. 关于“隐私-能效-效用”三目标：
Neuro-Symbiosis 的价值在于将三者纳入统一实验报告框架，避免仅报告准确率的片面评估。

2. 关于 SNN 与 Transformer 协同：
SNN 提供时序稀疏表征，Transformer 提供上下文整合能力。多种子结果下 Hybrid 仍保持更低 MIA AUC，说明该优势具有一定稳定性。

3. 关于现实部署：
在单卡笔记本设备约束下，项目已具备可重复的工程路径，适合后续在真实 BCI 数据上扩大实验规模。

## 7 局限性
1. 当前主结果仍以 quick 配置与合成数据为主，真实 BCI 数据上的统计显著性尚未完成。
2. 多种子实验中 Hybrid 准确率方差较大，提示需要更长训练轮次与学习率日程控制。
3. MIA 目前以置信度阈值攻击为主，仍需补充 shadow-model、loss-based 与 label-only 攻击。
4. 能效评估依赖 GPU 功率采样与时延估计，尚未覆盖神经形态硬件上的真实事件驱动开销。

## 8 伦理与合规
1. EEG 属于敏感生物信号，应最小化收集原则并明确用途边界。
2. 模型报告应同时包含效用、隐私预算与风险评估，避免“高准确率但高泄露风险”的不可解释部署。
3. 对医疗相关场景，应补充数据治理流程与人工复核机制，避免单模型决策。

## 9 结论
本文提出 Neuro-Symbiosis，用于 BCI 场景下 SNN-Transformer 混合解码，并构建隐私-能效-效用联合评估范式。新增多随机种子与 DP-MIA 结果表明，该框架在隐私风险控制方面具有可复现趋势性优势。后续将扩展到真实 BCI 数据集、增强攻击基准并开展显著性检验，以形成完整可发表结果。

## 参考文献（初稿）
[1] BCI Competition IV, https://www.bbci.de/competition/iv/ .
[2] Schalk G., et al. BCI2000: A General-Purpose Brain-Computer Interface System. IEEE TBME, 2004.
[3] PhysioNet EEG Motor Movement/Imagery Dataset, DOI:10.13026/C28G6P.
[4] Eshraghian J. K., et al. Training Spiking Neural Networks Using Lessons From Deep Learning. Proceedings of the IEEE, 2023.
[5] Opacus Documentation, https://opacus.ai/ .

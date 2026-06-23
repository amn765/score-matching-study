# 阶段 6 · 复现、源码与前沿（约 4 天）

> 目标：把前五阶段的理论与 toy 代码，收束成一次"真·图像生成"的复现，再读官方源码补齐工程细节，最后把整条脉络写成一篇自己的小结。学完你具备独立做 score-based 科研实验的能力。
> （梯度提示：不引入新理论，全部是"把已学的拼起来 + 落到真实数据 + 输出自己的理解"。）

主线资料：[song 官方源码 score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) + 你前几阶段的全部代码。

## 任务清单

### ☐ 6-1 ⭐ 综合复现：在 MNIST 上训练 score 模型并生成样本
- 把阶段 4/5 的代码升级到图像：score 网络换成小 U-Net，数据用 MNIST（或 CIFAR-10 子集）。
- 选一套体系（推荐 VP-SDE 或 NCSN），训练到能生成可辨认的数字。
- **完成标志**：生成的样本人眼能认出是手写数字（不要求 SOTA 质量）。

### ☐ 6-2 读官方源码，对照自己的实现找差距
- 通读 `score_sde_pytorch` 的训练循环、SDE 定义、采样器，对照你的实现逐一比对。
- 重点看：噪声尺度/时间调度、损失加权 λ(t)、网络条件 t 的注入方式（如 Fourier 特征 + 时间 embedding）。
- **完成标志**：列出至少 3 处"官方做了而我没做/做错"的细节。

### ☐ 6-3 理解工程要点：EMA、调度、PC 采样器
- 逐一搞懂这些"论文不强调但实战必备"的细节：

| 工程技巧 | 作用 | 不做会怎样 |
|------|------|------|
| EMA（参数滑动平均） | 稳定生成质量 | 样本噪点多、不稳定 |
| 损失加权 λ(t) | 平衡各时间尺度 | 大/小噪声学不均衡 |
| Predictor-Corrector | 逆向 SDE 步 + Langevin 校正 | 纯 EM 采样质量偏低 |
| 时间/噪声 embedding | 网络感知 t | 单网络无法服务多尺度 |

- **完成标志**：能解释每个技巧解决什么问题。

### ☐ 6-4 拓展阅读：score matching 的下游与亲戚
- 横向了解（读 intro/abstract 级别即可），把 score matching 放进更大的图景：
  - **条件生成**：classifier guidance / classifier-free guidance 如何修改 score；
  - **加速采样**：DDIM、DPM-Solver、consistency models；
  - **统一视角**：flow matching / rectified flow 与 score-based 的关系。
- **完成标志**：能说出至少两个"基于 score 的后续方向"及其核心想法。

### ☐ 6-5 ⭐ 费曼自测：写一篇 score matching 脉络小结
- 本阶段压轴。写一篇 800~1500 字的小结（博客/笔记皆可），把这条线讲清楚：
  Hyvärinen 显式 SM → DSM/SSM 可扩展化 → NCSN 多尺度 → SDE 统一 → 工程落地。
- 要求：用自己的话，给出每一步"解决了上一步的什么问题"。讲不清楚的地方就是没懂，回去补。
- **完成标志**：小结能让一个"只懂 DDPM"的同学读完理解 score matching 全貌。

### ☐ 6-6 自测：合上资料画出知识地图
合上所有资料，在白纸上画一张图，包含并连接：
1. score 定义、EBM、Langevin；
2. explicit SM、DSM、SSM 的关系与取舍；
3. NCSN、DDPM、VE/VP-SDE、probability flow ODE 的统一关系。

## 完成标志
- 完成一次 MNIST/CIFAR 级别的 score 模型复现，能生成可辨认样本。
- 读懂官方源码并补齐 EMA、PC 采样等工程细节。
- 产出一篇自己的 score matching 脉络小结，并能默画完整知识地图——至此具备独立科研复现能力。

<!--
约定：本文件 ### ☐ 数量 = 6。
-->

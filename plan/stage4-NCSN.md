# 阶段 4 · NCSN：噪声条件 score 网络与退火 Langevin（约 4 天）

> 目标：把"单一噪声尺度不够用"这个痛点，升级成多噪声尺度的 NCSN，并用退火 Langevin 采样真正生成像样的样本。这是 score-based 生成模型从"玩具"走向"能打"的关键一跃（Song & Ermon, NeurIPS 2019）。
> （梯度提示：方法上只是把上一阶段的 DSM"按多个 σ 叠起来"，但加入了流形假设与退火采样两个新直觉。）

主线资料：[Song & Ermon 2019《Generative Modeling by Estimating Gradients of the Data Distribution》](https://arxiv.org/abs/1907.05600)。

## 任务清单

### ☐ 4-1 通读 NCSN 论文，抓住三个核心问题
- 带着问题读：(1) 单噪声尺度有什么问题？(2) 为什么用多个噪声尺度？(3) 怎么采样？
- **完成标志**：能用三句话概括论文要解决的痛点和办法。

### ☐ 4-2 ⭐ 理解多噪声尺度的两大动机
- 本阶段核心直觉，搞懂这两点其余都顺：
  - **流形假设**：真实数据集中在低维流形上，流形外 score 没有定义/估不准；加噪把数据"撑开"填满空间，让 score 处处可学。
  - **低密度区问题**：远离数据的区域样本极少，score 估计方差大；大噪声尺度先提供"粗糙但全局"的指引，小噪声再精修。
- 思考题：回看阶段 2-3 你记下的"边界假设/流形"疑问，NCSN 是怎么正面回应的？
- **完成标志**：能讲清"为什么一个 σ 不够，需要一串 σ"。

### ☐ 4-3 理解 NCSN 网络设计与噪声尺度选取
- 网络是**噪声条件**的：`s_θ(x, σ)`，一个网络服务所有噪声尺度（σ 作为条件输入）。
- 噪声尺度取几何级数 `σ_1 > σ_2 > … > σ_L`，相邻尺度的分布要有足够重叠。训练目标是各尺度 DSM 损失的加权和（权重常取 σ²）。
- **完成标志**：能写出多尺度加权 DSM 训练目标，并解释 σ 为何取几何级数。

### ☐ 4-4 ⭐ 代码：在 2D toy 上训练一个小 NCSN
- 用一个接收 `(x, σ)` 的 MLP，在八高斯/双月上按多尺度 DSM 训练：
  ```python
  sigmas = torch.logspace(0, -2, 10)        # 几何级数：1 → 0.01
  def ncsn_loss(net, x):
      i = torch.randint(0, len(sigmas), (x.shape[0],))
      sig = sigmas[i].unsqueeze(1)
      x_t = x + sig * torch.randn_like(x)
      target = (x - x_t) / sig**2
      out = net(x_t, sig)                    # 网络以 σ 为条件
      return (((out - target)**2).sum(1) * sig.squeeze()**2).mean()
  ```
- **完成标志**：多尺度损失稳定下降，网络在不同 σ 下都能给出合理 score 场。

### ☐ 4-5 ⭐ 代码：实现退火 Langevin dynamics 采样
- 从最大 σ 到最小 σ 逐级采样，每级跑若干步 Langevin，步长按 σ 缩放：
  ```python
  def annealed_langevin(net, x, sigmas, n_steps=100, eps=2e-5):
      for sig in sigmas:                     # 从大到小
          step = eps * (sig / sigmas[-1])**2
          for _ in range(n_steps):
              s = net(x, sig.expand(x.shape[0],1))
              x = x + 0.5*step*s + step**0.5*torch.randn_like(x)
      return x
  ```
- 可视化整个退火过程：粒子如何从弥散噪声逐级收拢到数据分布。
- **完成标志**：生成样本清晰还原 toy 分布的多个模式。

### ☐ 4-6 思考题：噪声尺度的工程选择
- 思考题：σ_max 该多大？（提示：要能连接相距最远的两个数据点）σ_min 该多小？（提示：接近数据本身的噪声水平）级数取多密？
- 读论文/后续 technique 报告里关于 σ 选取的经验法则，对照你的实验。
- **完成标志**：能给出选 σ_max / σ_min / L 的定性原则。

### ☐ 4-7 自测：口述 NCSN 的价值
合上资料，能流畅回答：
1. 单噪声尺度 DSM 有哪两个具体问题？NCSN 如何分别解决？
2. 为什么一个网络要"以 σ 为条件"，而不是训 L 个独立网络？
3. 退火 Langevin 与普通 Langevin 的区别是什么？为什么从大噪声开始？

## 完成标志
- 能讲清多噪声尺度的流形假设与低密度区动机。
- 实现了噪声条件 score 网络与退火 Langevin 采样，在 2D toy 上生成像样样本。
- 能说出噪声尺度选取的工程原则。

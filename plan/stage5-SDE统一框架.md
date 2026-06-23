# 阶段 5 · SDE 统一框架（Song et al. 2021）（约 4 天）

> 目标：站上整个领域的制高点——用随机微分方程（SDE）把 NCSN（VE）和 DDPM（VP）统一成一个连续时间框架，理解逆向 SDE 与 probability flow ODE。学完你将拥有看待所有 score-based / diffusion 模型的统一视角。
> （梯度提示：把前面学的"多个离散噪声尺度"取连续极限就是前向 SDE，是自然的推广，新工具只有一条 Anderson 逆向 SDE 公式。）

主线资料：[Song et al. 2021《Score-Based Generative Modeling through Stochastic Differential Equations》(ICLR Outstanding Paper)](https://arxiv.org/abs/2011.13456) + [配套 Colab](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3)。

## 任务清单

### ☐ 5-1 通读 SDE 论文，建立"连续化"大图景
- 带着一个问题读：前面那串离散噪声尺度 σ_1…σ_L，如果让 L→∞ 会变成什么？（答案：一个连续时间的加噪过程，即前向 SDE）
- **完成标志**：能说清"离散噪声尺度 → 连续 SDE"的直觉。

### ☐ 5-2 ⭐ 理解前向 SDE 与三种实例（VE / VP / sub-VP）
- 前向 SDE 一般形式：`dx = f(x,t) dt + g(t) dw`，把数据逐渐扰动成噪声。
- 三个实例对应你已学的体系：
  - **VE-SDE**（方差爆炸）= NCSN 的连续极限；
  - **VP-SDE**（方差保持）= DDPM 的连续极限；
  - **sub-VP-SDE** = 改进版，likelihood 更好。
- **完成标志**：能写出 VE/VP 的 f、g，并说出各对应哪个离散模型。

### ☐ 5-3 ⭐ 理解逆向 SDE（Anderson 定理）—— score 在哪里出现
- 核心公式：前向 SDE 存在一个逆向 SDE，把噪声变回数据：
  ```
  dx = [ f(x,t) − g(t)²·∇x log p_t(x) ] dt + g(t) dw̄
  ```
- 关键：逆向过程里出现了 `∇x log p_t(x)`，也就是 **每个时间 t 的 score**。我们用网络 `s_θ(x,t)` 估计它，就能反向采样。
- **完成标志**：能指出逆向 SDE 公式里哪一项是 score，并解释"为什么生成 = 解逆向 SDE"。

### ☐ 5-4 理解 probability flow ODE
- 存在一个确定性 ODE，与逆向 SDE 有**完全相同的边际分布** `p_t`：
  ```
  dx = [ f(x,t) − ½·g(t)²·∇x log p_t(x) ] dt
  ```
- 它带来两个好处：确定性采样（可用高级 ODE 求解器加速），以及可精确计算 likelihood（用瞬时变量替换公式）。
- 思考题：为什么去掉随机项后，确定性 ODE 还能保持和 SDE 一样的边际分布？
- **完成标志**：能说清 probability flow ODE 的两个用途。

### ☐ 5-5 把 DDPM = VP、NCSN = VE 对齐，完成大一统
- 回到阶段 0 那张对照表，现在把它补全：DDPM 是 VP-SDE 的离散化，NCSN 是 VE-SDE 的离散化，二者只是同一框架下 f、g 的不同选择。
- **完成标志**：能在一张图里画出"离散 ↔ 连续、VE ↔ VP"的四格对应关系。

### ☐ 5-6 ⭐ 代码：实现 VP/VE 训练目标 + Euler-Maruyama 采样
- 训练：时间 t 连续采样，目标是时间加权的 DSM `E_t[ λ(t)·‖s_θ(x_t,t) − ∇log p_t(x_t|x_0)‖² ]`。
- 采样：用 Euler-Maruyama 数值解逆向 SDE：
  ```python
  def em_sampler(score, sde, shape, N=1000):
      x = sde.prior_sample(shape)
      ts = torch.linspace(1, 1e-3, N)
      dt = -1.0 / N
      for t in ts:
          f, g = sde.reverse(x, t, score)      # 逆向漂移与扩散
          x = x + f*dt + g*(abs(dt)**0.5)*torch.randn_like(x)
      return x
  ```
- **完成标志**：在 2D toy（或 MNIST，留到阶段 6）上用 SDE 采样器生成样本。

### ☐ 5-7 思考题：probability flow ODE 与精确 likelihood
- 思考题：为什么用 ODE 形式能算出精确的 log-likelihood，而原始 EBM 因为配分函数算不了？（提示：连续归一化流 / 瞬时变量替换）
- **完成标志**：能讲清 score-based 模型如何"曲线救国"地得到 likelihood。

### ☐ 5-8 自测：口述 SDE 框架的完整逻辑
合上资料，能从头讲一遍：
1. 前向 SDE 是怎么从离散噪声尺度推广来的？VE 和 VP 各对应谁？
2. 逆向 SDE 的形式是什么？score 出现在哪一项？
3. probability flow ODE 和逆向 SDE 什么关系？各有什么用？

## 完成标志
- 能写出前向/逆向 SDE 与 probability flow ODE，并指出 score 的位置。
- 能把 DDPM、NCSN 统一进 VP/VE-SDE 框架。
- 实现了基于 SDE 的训练目标与 Euler-Maruyama 采样器。

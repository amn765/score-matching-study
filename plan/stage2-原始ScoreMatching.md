# 阶段 2 · 原始 Score Matching（Hyvärinen 2005）（约 4 天）

> 目标：吃透 score matching 的开山之作。理解"我们想最小化和真实 score 的差距，但真实 score 不可知"这个矛盾，以及 Hyvärinen 如何用分部积分把它变成可计算的目标。学完你能亲手训出第一个 score 网络。
> （梯度提示：本阶段第一次出现完整的数学推导，但只用到分部积分这一个工具，难度只比上一阶段高一点。）

主线资料：[Hyvärinen 2005《Estimation of Non-Normalized Statistical Models by Score Matching》](https://www.jmlr.org/papers/v6/hyvarinen05a.html) + [Yang Song 博客 Score Matching 一节](https://yang-song.net/blog/2021/score/)。

## 任务清单

### ☐ 2-1 写出显式 score matching 目标，并看清它为何不可直接算
- 目标：让网络 `s_θ(x)` 逼近真实 score，最小化 `J = (1/2)·E_p[ ‖s_θ(x) − ∇x log p_data(x)‖² ]`。
- 矛盾点：`∇x log p_data(x)` 是真实数据分布的 score，我们**根本不知道** p_data，所以这个目标看似无法优化。
- **完成标志**：能写出这个目标，并说清"卡在哪里"。

### ☐ 2-2 ⭐ 亲手推导 Hyvärinen 恒等式（分部积分）
- 本阶段最核心的一步。把 `J` 展开为三项：`‖s_θ‖²`、`‖∇log p‖²`、交叉项 `−s_θ·∇log p`。
- 第二项不含 θ，优化时是常数可丢。对交叉项用分部积分（关键：`∇log p · p = ∇p`），把它转成只含 `∇·s_θ`（散度/Jacobian 的迹）的形式。
- 推出最终可算目标：
  ```
  J(θ) = E_p[ trace(∇x s_θ(x)) + (1/2)·‖s_θ(x)‖² ] + const
  ```
  其中 `∇x s_θ` 是 score 网络对输入 x 的 Jacobian。
- **完成标志**：能在白纸上从 `J` 一路推到上式，每一步说出用了什么。

### ☐ 2-3 厘清推导依赖的边界假设
- 分部积分丢掉了边界项，这要求 `p_data(x)·s_θ(x) → 0`（x 趋于无穷时密度足够快地趋 0）。
- 思考题：对图像这种"分布支撑在低维流形上"的数据，这个假设会不会出问题？（先记下疑问，阶段 4 的 NCSN 会回应它）
- **完成标志**：能说清这个目标在什么前提下成立。

### ☐ 2-4 代码：实现 explicit score matching 训练一个小 MLP
- 用一个 2~3 层 MLP 当 `s_θ(x)`，在 1D 或 2D toy 上按 2-2 的目标训练。
- Jacobian 的迹用自动微分逐维求（小维度可行）：
  ```python
  def sm_loss(score_net, x):
      x = x.requires_grad_(True)
      s = score_net(x)                      # (B, D)
      norm = 0.5 * (s**2).sum(1)
      tr = 0
      for d in range(x.shape[1]):           # trace(∂s/∂x)，逐维求导
          grad = torch.autograd.grad(s[:,d].sum(), x, create_graph=True)[0]
          tr = tr + grad[:, d]
      return (tr + norm).mean()
  ```
- **完成标志**：loss 能稳定下降。

### ☐ 2-5 ⭐ 验证：用学到的 score 采样并对比真实分布
- 把训练好的 `s_θ` 喂给阶段 1 的 Langevin 采样器，生成样本，叠在真实分布上看吻合度。
- 若不像，调步长/迭代步数/网络容量。这是你第一次"端到端"完成"学 score → 采样生成"。
- **完成标志**：生成样本大致还原 toy 分布的形状。

### ☐ 2-6 思考题：为什么这个方法到高维就崩了
- trace(Jacobian) 需要 D 次反向传播（D = 数据维度），对图像（D 上万）完全不可行。
- 思考题：有没有办法**不算完整 Jacobian** 还能估计它的迹？（这正是下一阶段 SSM 的随机投影思路）
- **完成标志**：能说清原始 score matching 的"可扩展性瓶颈"具体卡在哪。

### ☐ 2-7 自测：口述完整推导链条
合上资料，能从头讲一遍：
1. 显式 SM 目标是什么？为什么不能直接优化？
2. 分部积分怎么把它变成可算的 `trace(∇s) + ½‖s‖²`？用了什么边界假设？
3. 这个方法的致命缺点是什么？

## 完成标志
- 能独立推导 Hyvärinen 恒等式，并说清每步与边界假设。
- 实现了 explicit score matching，并用 Langevin 验证采样。
- 能清楚指出 trace(Jacobian) 的高维瓶颈，理解后续方法的动机。

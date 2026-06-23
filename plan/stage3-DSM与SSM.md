# 阶段 3 · Denoising & Sliced Score Matching（约 4 天）

> 目标：掌握让 score matching 真正可扩展的两条路线——去噪 score matching（DSM）和切片 score matching（SSM）。最关键的是：你将亲手证明 **DDPM 的 ε-prediction 损失 ≡ denoising score matching**，让你已有的 DDPM 知识在这里彻底闭环。
> （梯度提示：DSM 用一个巧妙的等价彻底绕开 Jacobian，是本阶段最该花时间的一步。）

主线资料：[Vincent 2011《A Connection Between Score Matching and Denoising Autoencoders》](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) + [Song 2019《Sliced Score Matching》](https://arxiv.org/abs/1905.07088)。

## 任务清单

### ☐ 3-1 ⭐ 推导 Denoising Score Matching（Vincent 2011）
- 对数据加高斯噪声：`x̃ = x + σ·ε`，得到扰动分布 `q_σ(x̃)`。
- 核心结论：对加噪分布做 score matching，等价于让网络预测"去噪方向"。已知条件分布 score 解析可得：
  ```
  ∇x̃ log q_σ(x̃ | x) = (x − x̃) / σ²
  ```
- DSM 目标：`E[ ‖ s_θ(x̃) − (x − x̃)/σ² ‖² ]`。注意：**完全没有 Jacobian，只是一个回归损失**。
- **完成标志**：能推出条件 score 的解析式，并写出 DSM 目标。

### ☐ 3-2 理解 DSM 学到的是"加噪分布"的 score
- 关键认识：DSM 学到的不是干净数据的 score，而是 `q_σ` 的 score；σ 越小越接近真实 score，但小 σ 在低密度区估计方差大。
- 思考题：这是不是意味着我们需要"多个噪声尺度"来兼顾精度与覆盖？（为阶段 4 NCSN 埋下伏笔）
- **完成标志**：能说清 σ 的"精度 vs 覆盖"权衡。

### ☐ 3-3 ⭐ 把 DDPM 接上：证明 ε-prediction ≡ DSM
- 这是本阶段对你最有价值的一步。把你熟悉的 DDPM 前向 `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε` 套进 DSM。
- 推导：DDPM 的 score `∇ log q(x_t|x_0) = −ε / √(1−ᾱ_t)`，所以预测噪声 `ε_θ` 和预测 score `s_θ` 只差一个 `−1/√(1−ᾱ_t)` 的系数。
- 结论：**DDPM 的简化损失 `E‖ε − ε_θ‖²` 就是带权重的 denoising score matching**。你训过的 DDPM 一直在做 score matching。
- **完成标志**：能写出 ε_θ 与 s_θ 的换算关系，并说清两套损失为何等价。

### ☐ 3-4 推导 Sliced Score Matching（Song 2019）
- 另一条路线：保留原始 score matching 目标，但用随机投影避开完整 Jacobian。
- 核心技巧 Hutchinson estimator：`trace(A) = E_v[ vᵀ A v ]`，v 是随机向量（如 Rademacher）。于是 `trace(∇s)` 用 `vᵀ(∇s)v` 估计，只需 **一次** 向量-Jacobian 积。
- SSM 目标：`E_x E_v[ vᵀ∇x s_θ(x) v + ½(vᵀs_θ(x))² ]`。
- **完成标志**：能解释 Hutchinson estimator 怎么把 D 次反传压成 1 次。

### ☐ 3-5 代码：实现 DSM 并在 toy 上训练采样
- 实现 DSM 损失（注意它只是一个加权回归，非常简洁）：
  ```python
  def dsm_loss(score_net, x, sigma=0.1):
      x_tilde = x + sigma * torch.randn_like(x)
      target = (x - x_tilde) / sigma**2
      return ((score_net(x_tilde) - target)**2).sum(1).mean()
  ```
- 训完用 Langevin 采样，和阶段 2 的 explicit SM 结果对比。
- **完成标志**：DSM 训练比 explicit SM 更快更稳，能采出 toy 分布。

### ☐ 3-6 代码：实现 SSM 并三方对比成本
- 实现 SSM 损失（用随机投影），与 explicit SM、DSM 在同一 toy 上对比训练速度与样本质量。
- 记录：每步耗时、最终样本质量、实现复杂度，填一张对比表：

| 方法 | 是否需要 Jacobian | 单步成本 | 适用规模 |
|------|------|------|------|
| Explicit SM | 需要完整 trace | 高（D 次反传） | 仅低维 |
| Sliced SM | 一次向量-Jacobian 积 | 中 | 中高维可行 |
| Denoising SM | 完全不需要 | 低（一次回归） | 高维主流 |

- **完成标志**：能用数据说出三种方法的成本差异。

### ☐ 3-7 自测：口述 DSM / SSM / DDPM 的关系
合上资料，能流畅回答：
1. DSM 为什么能完全避开 Jacobian？它学到的是谁的 score？
2. SSM 的 Hutchinson 技巧是什么？解决了原始 SM 的什么问题？
3. 为什么说"你训 DDPM 时其实一直在做 score matching"？

## 完成标志
- 能推导 DSM 目标，并证明它与 DDPM ε-prediction 等价。
- 能解释 SSM 的随机投影技巧。
- 实现了 DSM 与 SSM，并对三种方法做了成本对比。

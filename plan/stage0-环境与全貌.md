# 阶段 0 · 环境与全貌：从 DDPM 接到 score 视角（约 2 天）

> 目标：搭好实验环境，并把你已经掌握的 DDPM 知识"翻译"到 score 语言。学完你能说清 score function 是什么、为什么生成模型想要它。
> （梯度提示：本阶段几乎不引入新数学，只做"已知知识的视角切换"，是整条路线最缓的一级台阶。）

主线资料：[Yang Song 博客《Generative Modeling by Estimating Gradients of the Data Distribution》](https://yang-song.net/blog/2021/score/) + 你已读过的 DDPM 论文。

## 任务清单

### ☐ 0-1 搭好实验环境并跑通一个 2D toy 数据集
- 建一个干净的 Python 环境，装 `torch`、`numpy`、`matplotlib`。
- 亲手敲一段最小代码，采样一个二维分布（如"八高斯"或"双月"）并画散点图：
  ```python
  import torch, matplotlib.pyplot as plt
  # 八高斯：8 个均匀分布在圆上的高斯
  k = torch.randint(0, 8, (2000,))
  ang = k.float() / 8 * 2 * torch.pi
  centers = torch.stack([ang.cos(), ang.sin()], 1) * 4
  x = centers + 0.2 * torch.randn(2000, 2)
  plt.scatter(x[:,0], x[:,1], s=4); plt.axis('equal'); plt.show()
  ```
- **完成标志**：能稳定生成并画出这个 2D 分布，后续每个阶段都会复用它。

### ☐ 0-2 ⭐ 把"已知的 DDPM"和"score 世界"对齐成一张表
- 这是本阶段核心：不是学新东西，而是给已有知识换坐标系。
- 自己填这张对照表（先凭直觉填，本阶段末再回来校正）：

| DDPM 里的概念 | Score 世界里的对应 | 一句话联系 |
|------|------|------|
| 前向加噪 q(x_t \| x_0) | 一族被噪声扰动的分布 | 每个 t 对应一个 noise level |
| 预测噪声 ε_θ | 预测 score s_θ | 二者只差一个系数 |
| 逐步去噪采样 | Langevin / 逆向 SDE 采样 | 都靠 score 指路 |
| 离散 T 步 | 连续时间 SDE | DDPM 是它的离散化 |

- 思考题：你训练 DDPM 时网络输出的 `ε_θ(x_t, t)`，凭直觉它和"指向高密度区的方向"有什么关系？

### ☐ 0-3 理解并可视化 score function ∇x log p(x)
- 定义：score 就是对数密度关于"数据 x"的梯度 `∇x log p(x)`（注意不是对参数求导）。
- 对一个已知的 2D 高斯 `N(0, σ²I)`，手算它的 score = `-x / σ²`，然后画出 score 向量场（quiver 图），观察箭头都指向哪里：
  ```python
  xs = torch.linspace(-3, 3, 20)
  gx, gy = torch.meshgrid(xs, xs, indexing='xy')
  s = -torch.stack([gx, gy], -1)          # σ=1 时 score = -x
  plt.quiver(gx, gy, s[...,0], s[...,1]); plt.axis('equal'); plt.show()
  ```
- **完成标志**：你能指着 quiver 图说出"箭头指向概率密度更高的地方"，并解释为什么这正是采样时想要的方向。

### ☐ 0-4 自测：口述 score 的直觉
合上资料，能流畅回答：
1. score function 的数学定义是什么？它是对谁求导？
2. 为什么有了 score 就能采样，而不需要知道归一化常数 Z？
3. 用一句话把"DDPM 预测噪声"和"估计 score"联系起来。

## 完成标志
- 环境能跑、能画 2D toy 分布与 score 向量场。
- 能对着对照表，把 DDPM 的每个组件映射到 score 视角。
- 能口述 score 的定义与直觉，为后面所有阶段打好"共同语言"。

<!--
约定：本文件 ### ☐ 数量 = 4，必须等于 index.html 该阶段 count 与 tracker tasks.length。
-->

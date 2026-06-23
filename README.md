# 🌊 Score Matching 学习计划

> 目标：从「有生成模型基础、懂 DDPM 推导」到「科研 + 数学 + 代码全打通：能推导各类 score matching 目标、亲手实现、读懂并复现论文」。
> 总时长：约 3~4 周（每天 2~3 小时，宁慢勿断）。

## 设计思路

你已经懂 DDPM 推导，所以这套计划**不从零讲扩散**，而是用一条主线把你的已有知识接进 score-based 世界：

**先换视角 →（阶段 0-1）先会用 score 采样 →（阶段 2-3）再亲手造每一种 score matching 目标 →（阶段 4-5）逐级推广到 NCSN 与 SDE 统一框架 →（阶段 6）落到真实数据复现并读源码。**

梯度刻意铺得很平：阶段 0 几乎不引入新数学，只把 DDPM「翻译」成 score 语言；阶段 1 把"采样"和"学 score"解耦，先用真实 score 跑通 Langevin；阶段 2 才出现第一个完整推导，且只用分部积分一个工具。每个阶段难度只比上一个高一点点，每个任务都是可勾选、有"完成标志"的小目标——这就是「看得见进步」和「梯度不陡峭」的来源。

特别地，**阶段 3 会让你亲手证明「DDPM 的 ε-prediction 损失 ≡ denoising score matching」**，让你已有的 DDPM 知识在这里彻底闭环。

## 学习路线总览

| 阶段 | 主题 | 时长 | 里程碑 |
|------|------|------|--------|
| 0 | [环境与全貌：从 DDPM 接到 score 视角](plan/stage0-环境与全貌.md) | 约 2 天 | 能口述 score 定义，把 DDPM 映射到 score 视角 |
| 1 | [Score、能量模型与 Langevin 采样](plan/stage1-能量模型与Langevin.md) | 约 3 天 | 能用真实 score 在 toy 分布上跑通 Langevin |
| 2 | [原始 Score Matching（Hyvärinen 2005）](plan/stage2-原始ScoreMatching.md) | 约 4 天 | 能独立推导 Hyvärinen 恒等式并训出 score 网络 |
| 3 | [Denoising & Sliced Score Matching](plan/stage3-DSM与SSM.md) | 约 4 天 | 能证明 DDPM ≡ DSM，掌握 DSM/SSM 实现 |
| 4 | [NCSN：噪声条件 score 网络与退火 Langevin](plan/stage4-NCSN.md) | 约 4 天 | 能训练 NCSN 并用退火 Langevin 生成样本 |
| 5 | [SDE 统一框架（Song et al. 2021）](plan/stage5-SDE统一框架.md) | 约 4 天 | 能把 DDPM/NCSN 统一进 VP/VE-SDE，实现 SDE 采样 |
| 6 | [复现、源码与前沿](plan/stage6-复现与前沿.md) | 约 4 天 | 复现 MNIST 级 score 模型，产出脉络小结与知识地图 |

全程约 **45 个可勾选任务**，完成即从「懂 DDPM」走到「能独立做 score-based 科研复现」。

## 在线访问（手机也能学）

| 入口 | 网址 |
|------|------|
| 📖 学习站（阅读文档 + 勾选任务） | https://amn765.github.io/score-matching-study/ |
| 📊 打卡面板（进度 / 热力图 / 连续天数） | https://amn765.github.io/score-matching-study/tracker/ |
| 💾 GitHub 仓库 | https://github.com/amn765/score-matching-study |

> 已部署到 GitHub Pages，手机浏览器打开后选「添加到主屏幕」即可当 App 用。
> 两个页面共享同一份进度（同一浏览器内），换设备用面板里的「导出 / 导入进度」同步。

## 怎么使用这套计划

1. **打开进度面板**：双击 `tracker/index.html`。每完成一个任务就勾选，进度条、热力图、连续天数会实时更新。进度存在浏览器 localStorage 里，换浏览器/电脑前记得用「导出进度」备份。
2. **每天的节奏**：打开当前阶段的 markdown（在 `plan/` 文件夹里直接看，或部署后在学习站读）→ 挑 1~2 个任务 → 动手推导/写代码验证 → 回到面板打勾。即使某天没时间，也点一下「今日打卡」保住连续天数。
3. **黄金法则**：
   - 所有公式**亲手推一遍**、所有代码**亲手敲一遍**，不要复制粘贴。
   - 每个任务做完后，合上资料**口述一遍学到了什么**——讲不出来就是没懂。
   - 卡住超过 1 小时就先跳过、做标记，往往学到后面会自然解开（比如阶段 2 的"流形疑问"会在阶段 4 解开）。

## 推荐资料（全程通用）

- [Yang Song 博客《Generative Modeling by Estimating Gradients of the Data Distribution》](https://yang-song.net/blog/2021/score/)（最佳总览，反复读）
- [Hyvärinen 2005《Estimation of Non-Normalized Statistical Models by Score Matching》](https://www.jmlr.org/papers/v6/hyvarinen05a.html)
- [Vincent 2011《A Connection Between Score Matching and Denoising Autoencoders》](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)
- [Song & Ermon 2019《Generative Modeling by Estimating Gradients of the Data Distribution》(NCSN)](https://arxiv.org/abs/1907.05600)
- [Song et al. 2021《Score-Based Generative Modeling through SDEs》](https://arxiv.org/abs/2011.13456) + [官方源码](https://github.com/yang-song/score_sde_pytorch)

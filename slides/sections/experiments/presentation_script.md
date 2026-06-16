# EliteAlpha 答辩 — 实验章节讲稿（约 3 分钟）

> 对应 `sections/experiments/experiments.tex` 的 8 张 slides。
> 全长约 700 字，正常语速 3 分钟内。
> 反复强调三个数字：**+75% / 15.7% / 1.46**。

---

## Slide 1 — Experimental Setup（约 15 秒）

实验在中证 500 上做，**5 年测试期，2021 年 6 月到 2026 年 5 月**。回测用 TopkDropoutStrategy，每天选 50 只股票，对标 SH000905 指数。指标我们既看信号质量（IC、Rank IC），也看实际收益（AR、IR、MDD）。Baseline 包括 LightGBM、LSTM、Transformer 三个传统模型方法，以及 AlphaAgent —— 我们框架的直接对照组。

---

## Slide 2 — Reproducibility Surprise（约 40 秒）

在复现 AlphaAgent 的过程中我们发现一个**意料之外的问题**：它的 LightGBM 配置用了非常强的 L1 正则——λ=205.7。这套参数本来是为 Alpha158 的 158 维特征调的，但 AlphaAgent 实际只用 4 个 base features 加 1 个 LLM 因子。我们直接看 LightGBM 训完的 feature importance：**LLM 因子的 importance 是 0**，被 L1 完全剪掉了。

这意味着：**LLM 收到的 IC 反馈跟它生成的因子无关**，永远是 0.0166 左右。梯度信号被压平，LLM 就在原核心公式上换装饰——这就是我们说的"attractor collapse"。我们 EliteAlpha 通过**直接 per-factor IC 评分**修复了这个问题。

> **附：可能的提问 — 为什么会跟 LightGBM 有关系？**
>
> LightGBM 不是检测脚本，是 AlphaAgent 框架自身的设计：
> `LLM 生成因子 → 4 base + 因子 → LightGBM 训练 → IC → 反馈给 LLM`
> 它自带的打分器就是 LightGBM。bug 就出在这个内置打分器 L1=205 太狠，把因子贡献剪光。EliteAlpha 的修复是绕过 LightGBM 直接算 per-factor IC。

---

## Slide 3 — Table 2（约 40 秒）

Table 2 是主对比。注意三列：

**第一**，IC 这列 EliteAlpha 是 0.0084，是最低的。但这其实是因为我们**直接用因子值当 signal，没过模型平滑**，跟模型 pred 比 IC 天然偏低。

**第二**，更公平的对比是 **Rank IC** —— 排序质量。EliteAlpha 是 0.0322，**强于 LightGBM 和 AlphaAgent**，仅次于 LSTM。

**第三**，最重要的：**Portfolio 表现 EliteAlpha 全场第一**——年化超额 15.7%，信息比率 1.46，都是最高。**比最强的模型基线 LSTM 还高 5 个百分点**。也就是说，我们用一个手写的因子表达式，做到了比训练完整深度学习模型更强的收益。

---

## Slide 4 — Figure 3 累计超额收益（约 25 秒）

这张图直观看 5 年累计走势。**黑色粗线就是 EliteAlpha**，5 年累计超额 +75%，最上面那条。第二是 LSTM 54%，AlphaAgent 43%，LightGBM 34%，Transformer 33%。EliteAlpha 不仅终点最高，而且**全程稳定上升**，没有大的回撤。

---

## Slide 5 — Figure 4 因子来源 yearly IC（约 25 秒）

按因子来源比较。每条线代表一个"因子家族"：RSI 是经典手工因子，Alpha158 是工程化特征库，AlphaAgent 是 LLM 挖的因子，**EliteAlpha 是我们 MAP-Elites 找出来的**。

看下面 Rank IC 那个 panel：**EliteAlpha 黑粗线在 2021-2024 都是最高的**，2025-2026 跟 Alpha158 持平但仍然领先于 RSI 和 AlphaAgent。说明我们的因子在**排序能力上长期稳定**，没出现明显的 alpha decay。

---

## Slide 6 — Figure 5 mining trajectory（约 25 秒）

这张图最能体现 EliteAlpha 的核心 contribution。

左上角红色那 5 个点是 AlphaAgent，**完全水平**——5 轮迭代下来 IC 没动过，因为反馈机制坏了。

黑色这条是 EliteAlpha，46 轮 mining 里 archive 最佳质量**从 0.002 涨到 0.006，整整 3 倍**。下面那条灰色虚线是 archive 平均质量，也在持续上升。

→ **MAP-Elites 真的在探索**，AlphaAgent 卡在一个 attractor 里没出来。

---

## Slide 7 — Archive Coverage（约 15 秒）

我们的 archive 是 5 类因子 × 5 个 depth 桶 = 25 个 cell。46 轮跑完填了 **13 个 cell，覆盖了 5 个类别中的 4 个**。AlphaAgent 5 轮只覆盖 1 个 cell。**多样性差距 13 倍**。

---

## Slide 8 — Takeaways（约 15 秒）

总结四点：

1. 我们**发现并修复**了 AlphaAgent 评测机制的 bug；
2. attractor collapse 是 LLM mining 普遍隐患，我们用 archive 解决；
3. **多样性方面 13:1 碾压**；
4. **性能方面 AR 15.7%、IR 1.46，超过最强基线**——而且**不用训练任何模型**。

谢谢，准备回答问题。

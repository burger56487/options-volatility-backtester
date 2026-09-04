# 2. 定价引擎

## 2.1 模型清单

统一接口（`PricingRequest` / `PricingResult`）之下实现：

- Black--Scholes--Merton 解析解（含连续股息、五种 Greeks）；
- CRR 二叉树（欧式/美式，提前行权处理）；
- 有限差分（显式/隐式/Crank--Nicolson，稳定性检查）；
- 蒙特卡洛（对偶变量 + 控制变量 + 标准误/置信区间）；
- Heston 随机波动率（特征函数 + 全截断 Euler MC + Fourier 半解析定价）；
- Merton 跳跃扩散（Poisson 加权解析级数 + compound-Poisson MC）；
- Dupire 局部波动率。

## 2.2 数值验证（实测 2026-09-04）

固定参数：S=100, K=100, T=0.5, r=4%, q=1%, σ=25%。
BS 参考价格：7.7215522303。

| 方法 | 参数 | 绝对误差 |
|---|---|---|
| CRR | 100 步 | 1.74e-2 |
| CRR | 200 步 | 8.72e-3 |
| CRR | 400 步 | 4.36e-3 |
| CRR | 800 步 | 2.18e-3 |
| Crank--Nicolson | 400 时间步/800 空间步 | 3.50e-2 |
| Merton MC（跳跃项） | 20 万路径 vs 解析级数 | 9.6e-4 |

CRR 步数翻倍误差约减半（O(1/N)），收敛阶写入回归测试
（`tests/pricing/test_pricing_batch2.py`）。

## 2.3 退化验证

- Heston σ_v→0（v0=θ）半解析定价精确回到 BS，误差 0
  （`tests/pricing/test_heston_semianalytic.py`）；
- Merton 跳跃强度=0 时级数解与 BS 误差 <1e-9；
- Heston/Merton MC 与对应基准在 3 个标准误内一致。

## 2.4 实现中抓到并修复的两个真 bug

1. **Heston 特征函数缺对数分支项**：标准式需要 ln((1−g·e^{−dτ})/(1−g))，
   原实现漏了 (1−g)，与蒙特卡洛分布对不上（u=2.5 时偏差约 0.11）。已修复，
   并新增“CF vs 模拟分布”回归测试。
2. **全截断 Euler 的漂移偏差**：log S 的漂移/扩散使用了“同一步更新后的 v”
   与同一个 z，在 ρ≠0 时引入约 ρ·σ_v/2 的偏差（2M 路径终端均值差约 2%）。
   改为步初 v 的标准顺序后，均值回到理论值。

这两个 bug 也说明：只有解析/退化/分布级的多重验证才能发现这类数值问题。

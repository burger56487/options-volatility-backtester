# 项目总览与面试讲稿（求职用）

## 一句话

一个以真实 SPY 行情为底、合成期权链为研究对象、并逐步接入真实期权快照的
可复现期权定价/波动率/做市研究平台：从数据边界、严格样本外、账户与执行引擎、
定价与曲面，到归因与风险，每一层都有测试和可追溯元数据。

## 已验证的硬结果（讲的时候只挑 2–3 个）

1. 账户引擎滚动跨式回测：真实 SPY 上 17 笔交易**逐笔通过 PnL 桥对账**
   （最大差异 3.5e-11），总 PnL −8038.77 与 legacy 路径 −8038.86 等价。
2. 严格时间线 + 样本外：信号滞后一日、train/validation/test 三区段、
   测试集锁定 + Bonferroni 试参次数记录；样本不足自动 `insufficient_sample`。
3. 真实 SPY 期权链：Cboe 快照 → 清洗/欧式边界分级/远期与股息估计 →
   bid/mid/ask 三档 IV → SVI 校准 + 蝶式/日历套利检查 → 曲面插值定价。
4. 定价矩阵：BSM/CRR/FD/蒙特卡洛/Heston/Merton/LV 全部带退化与收敛测试
   （Heston σ_v→0 → BS，Merton λ→0 → BS，LV 恢复恒定波动率）。
5. 风险：Delta-Normal/Delta-Gamma/历史/过滤历史/MC VaR + Euler 贡献恒等式
   + Kupiec/Christoffersen 回测。

## 边界（主动说，加分）

- 历史策略回测使用合成期权报价，不是真实历史期权回测；
- 真实期权链是快照分析与曲面研究，不做策略回测数据源；
- 做市仿真与 RL 为教学研究，订单流假设不等于实盘；
- 保证金与成本模型为研究性简化，不代表交易所/券商规则。

## 3 分钟口播稿（中文）

我的硕士方向是高频做市里的连续时间强化学习，为了让论文里的模型能被严格验证，
我搭了一套从数据到风险的可复现期权研究平台。结构上分四层：第一层是数据和口径，
我明确区分真实标的历史行情、合成期权报价和真实期权快照，并给每次运行写入数据
类型、评估模式、Git 版本等元数据；第二层是研究可信度，我把回测信号严格滞后一日，
用训练/验证/测试三区段做样本外评估，测试集锁定，并做移动块自助置信区间和
多重检验修正；第三层是定价和建模，我实现了从 Black-Scholes、二叉树、有限差分到
蒙特卡洛的统一定价引擎，再到真实期权链的隐含波动率、SVI 曲面，以及 Heston、
跳跃扩散和局部波动率，每个模型都有退化测试和收敛测试；第四层是工程落地，我把
账户、执行引擎和交易前风控串进回测，17 笔滚动跨式逐笔通过 PnL 桥对账，与旧路径
结果等价。我最有成就感的是过程中抓到的几个真 bug，比如期权到期结算没有把权利金
损失计入已实现损益、PnL 桥重复计了结算、Delta-Gamma 里漏了坐标变换项——这些
经历让我相信，量化研究里"验证"和"建模"同样重要。

## English one-liner

Reproducible options research platform spanning data boundaries, strict
out-of-sample evaluation, account/execution/risk engine, multi-model pricing,
real-chain IV/SVI surfaces, PnL attribution and risk backtesting, with a
verified C++-adjacent pricing library.

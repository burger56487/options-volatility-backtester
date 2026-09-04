# 5 分钟项目演示脚本

> 原则：每个环节给出“讲什么 + 跑什么 + 预期看到什么”，演示前先跑一遍
> 确保输出与预期一致。

## 0. 准备（演示前）

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python scripts/run_validation_report.py     # 验证报告已生成
```

## 1. 开场（30 秒）——项目是什么

“我做了一个端到端的期权定价与风险研究平台：六种定价模型、真实 SPY 数据
的波动率曲面、严格时间线的回测，和一个事件驱动做市仿真。技术上用 C++
加速核心计算，FastAPI/PostgreSQL/Docker 做数据平台，全部有测试和 CI。
我最看重的是数值正确性——PnL 对账精度到 1e-11。”

## 2. 架构（1 分钟）——三层结构

打开 [docs/architecture.md](../docs/architecture.md)，按图讲：

- 计算层：定价/曲面/回测/对冲/做市，共享统一数据结构；
- C++ 内核：批量定价/IV/MC/情景重估（ctypes 绑定）；
- 服务层与数据层：FastAPI → SQLite/PostgreSQL → Streamlit。

## 3. 核心功能演示（2 分钟）——跑一个定价、看曲面、看回测对比

### 3.1 定价 + 验证报告（30 秒）

```bash
PYTHONPATH=. python scripts/run_validation_report.py
```

指向 `outputs/testing/validation_report.json`：`all_passed: true`，展开讲
“CRR 收敛误差每翻倍步数减半、Heston σ_v→0 精确回 BS”。

### 3.2 SVI 曲面（30 秒）

打开 `report/figures/svi_surface.png`：真实链 872 条活跃、734 条反解成功，
分到期 RMSE 3.9e-5~1.7e-4。

### 3.3 回测与负结果（1 分钟）

```bash
PYTHONPATH=. python scripts/run_account_rolling_demo.py
```

预期：17 笔、PnL −8038.77、reconciliation failures 0。
接着打开 `report/figures/volatility_filter_comparison.png` 讲负结果叙事：
“无过滤 Sharpe-like −1.74 是期权经济学的正确预测；过滤后 +0.32。”

## 4. 技术亮点（1 分钟）——挑两个深入

- C++：打开 `docs/high_performance.md`，讲两种加速口径（40.8x 独立基准 vs
  3.8x 批量），强调“不夸大、区分口径”；
- 事件驱动做市：讲固定事件顺序防前视、同订单流/多种子公平对照。

## 5. 诚实收尾（30 秒）——局限与收获

“样本只有 6–33 笔，统计功效有限；历史期权报价是合成的。正因如此我把
验证放在第一位——项目中抓到的 Heston CF 与 Euler 两个 bug，恰恰证明
多层验证的价值。”

## 被打断时的快速回应

- “策略亏钱？”→ 负结果叙事（3.3）；
- “17 笔太小？”→ 承认局限 + 置信区间/样本外做法；
- “为什么 ctypes 不是 pybind11？”→ 无 Python 头文件依赖、释放 GIL，
  一致性已用测试锁定；
- “怎么保证没前视？”→ 信号滞后一日 + 特征时间审计 + 测试集锁定。

## 追问深度参考

每个写进 CV 的技术点至少准备 3 层追问，素材见：

- 研究报告：[report/README.md](../report/README.md)；
- 模块文档：`docs/`；
- 数字来源：`outputs/` 与复现指南 `docs/reproduction.md`。

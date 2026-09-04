# 研究报告

本项目研究报告（Markdown，GitHub 可直接阅读）。所有数字均来自仓库内
`outputs/`、`tests/` 或 `scripts/` 可复现的运行结果；引用处标注了来源文件。

## 章节

1. [引言](sections/01_introduction.md)
2. [定价引擎](sections/02_pricing_engine.md)
3. [波动率曲面](sections/03_volatility_surface.md)
4. [回测框架](sections/04_backtesting.md)
5. [动态对冲](sections/05_dynamic_hedging.md)
6. [做市仿真](sections/06_market_making.md)
7. [高性能计算](sections/07_high_performance.md)
8. [验证体系](sections/08_validation.md)
9. [结果与讨论](sections/09_results.md)
10. [局限性](sections/10_limitations.md)
11. [结论与未来工作](sections/11_conclusion.md)

## 配套材料

- [系统架构图与数据流](../docs/architecture.md)
- [模型验证报告生成器](../scripts/run_validation_report.py)
- 复现指南与模块文档：见仓库根 `README.md` 的“文档”一节。

## 诚实性声明

本报告是研究项目报告，展示方法论严谨性与工程能力，不是学术论文，也不是
商业产品文档。历史回测使用真实 SPY 标的价格与**合成期权报价**；真实期权链
仅用于快照分析，不构成真实历史期权回测。

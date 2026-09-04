# 测试与 CI

## 测试分层

75+ 测试文件、300+ 用例：单元（定价/Greeks/仓储）、集成（账户+执行+风控、
API+DB）、数值验证、属性测试（Hypothesis）、可复现性、golden 回归、
模型验证套件、性能回归（perf）。

## 常用命令

```bash
PYTHONPATH=. python -m pytest -q -m "not perf"                 # 全量
PYTHONPATH=. python -m pytest -q -m "property or golden or model_validation"
PYTHONPATH=. python -m pytest -q -m perf                       # 本机性能
PYTHONPATH=. python scripts/run_validation_report.py           # 验证报告
```

PostgreSQL 测试在无 `DATABASE_URL` 时自动跳过；CI 通过服务容器真实执行。

## 覆盖率

门槛 80%（`pyproject.toml`），实测 ~86%。报告以 artifact 上传，不设
“刷行覆盖率”导向的更高门槛。

## CI（GitHub Actions）

- `lint`：ruff 错误级检查 + mypy（`src/validation`）；
- `test`：pytest+覆盖率、模型验证报告、C++ 编译/测试、PostgreSQL 服务
  容器、验证/覆盖率 artifact；
- `perf`：`workflow_dispatch` 手动触发。

## 已知边界

性能测试对硬件敏感，默认不进 CI；C++ 测试依赖编译产物（缺失时 skip，
构建后独立步骤执行）。

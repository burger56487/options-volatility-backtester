# 数据平台（Storage / API / Dashboard / Docker）

## 存储

- 默认 SQLite：`outputs/app.db`；
- 设置 `DATABASE_URL` 时使用 PostgreSQL（`src/storage/postgres_repository.py`，
  含迁移与 created_at/status 索引）；
- 工厂入口：`src/storage/repository.py::connect_run_repository()`；
- 运行目录导入：`src/storage/importer.py`。

## API（FastAPI）

```bash
uvicorn src.api.run_server:application --reload
```

端点：`GET /health`、`POST /pricing/vanilla`、`POST /pricing/surface`、
`POST /runs`（后台任务）、`GET /runs`、`GET /runs/{id}`。

## 看板

```bash
streamlit run scripts/dashboard.py
```

当前看板读取 SQLite 输出库；PostgreSQL 部署建议通过 API `/runs` 查询。

## Docker

```bash
docker compose up --build                    # SQLite + API(8000) + 看板(8501)
docker compose --profile postgres up --build # + postgres:16 + Postgres API
```

## 局限

无生产级认证/高并发/高可用；PostgreSQL 为可选路径（CI 服务容器验证），
看板尚未原生读 Postgres。

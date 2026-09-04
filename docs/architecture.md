# 系统架构与数据流

三张图对应任务十八的架构要求：系统分层、数据流、做市事件流。每张图后附
说明文字与诚实标注（研究仿真平台，非生产系统）。

## 1. 系统架构（分层）

```mermaid
graph TB
    subgraph "展示层"
        DASH[Streamlit 看板]
    end
    subgraph "服务层"
        API[FastAPI<br/>定价/曲面/实验任务]
    end
    subgraph "计算层"
        PRICE[定价引擎<br/>BS/CRR/FD/MC/Heston/Merton/LV]
        VOL[波动率曲面<br/>IV 反解 + SVI 校准 + 无套利检查]
        BT[回测框架<br/>账户/执行/时间线/样本外]
        HEDGE[动态对冲<br/>固定频率/阈值/成本感知/RL]
        MM[做市仿真<br/>多合约 Greeks + DP/RL]
        VALID[模型验证套件]
        CPP[C++ 加速核心<br/>批量 BS/IV/MC/情景/VaR]
    end
    subgraph "数据层"
        DB[(SQLite 默认 / PostgreSQL 可选)]
        DATA[(data/ 原始+处理<br/>outputs/ 运行结果)]
    end
    DASH --> API
    API --> PRICE & VOL & BT & HEDGE & MM
    PRICE --> CPP
    BT --> CPP
    VALID --> PRICE & VOL & BT & HEDGE & MM
    API --> DB
    BT --> DATA
    VOL --> DATA
```

说明：FastAPI 提供定价/曲面/实验任务接口；计算层各模块通过统一数据结构
协作（`src/domain`、`src/market_data/schemas`）；C++ 内核通过 ctypes 被
Python 调用（计算段释放 GIL）。结果默认落 SQLite，设置 `DATABASE_URL`
时切换到 PostgreSQL 仓储（CI 用服务容器验证）。

## 2. 数据流（SPY → 曲面 → 回测 → 落库 → 看板）

```mermaid
flowchart LR
    A[SPY 历史行情 yfinance] --> B[清洗/校验/血缘]
    C[真实期权快照 Cboe] --> D[报价清洗+无套利检查]
    B --> E[合成期权链<br/>透明 IV 曲面规则]
    D --> F[隐含波动率反解]
    F --> G[SVI 校准 + 残差/稳定性]
    E --> H[回测：时间线/账户/执行/风控]
    G --> H
    H --> I[结果 CSV/JSON + run_metadata]
    I --> J[(SQLite/PostgreSQL runs)]
    J --> K[FastAPI /runs 查询]
    J --> L[Streamlit 看板]
```

说明：真实历史期权链不可得，因此“合成期权链”与“真实期权快照分析”是
两条独立数据路径，报告与 README 严格区分二者，不做“真实历史期权回测”
的表述。

## 3. 做市仿真事件流（事件驱动）

```mermaid
sequenceDiagram
    participant O as 订单流(Point process)
    participant E as 事件循环
    participant S as 状态(现金/库存/Greeks)
    participant Q as 报价策略
    participant R as 风险检查
    participant A as 账户/执行引擎
    participant M as 盯市与对账
    O->>E: 客户订单到达(带序列号)
    E->>S: 更新市场/波动率状态
    S->>Q: 状态/剩余时间/限额
    Q->>E: 买/卖报价偏移
    E->>R: 交易前限额检查
    R-->>E: 通过/拒绝/降量
    E->>A: 成交 + 现金流/手续费
    A->>S: 更新库存/现金/风险桶
    loop 每日收盘
        E->>M: 盯市估值
        M->>A: PnL 桥对账
        A->>S: 日亏损/回撤熔断检查
    end
```

说明：同一时间戳下的事件按序列号确定顺序执行，避免“用成交后信息生成
成交前报价”的前视；RL/DP 与规则策略在同一订单流、同一 seed 下对照评估
（`src/market_making/study.py`）。

## 图例与配色约定

- 分层图：展示/服务/计算/数据自上而下；
- 数据流图：从左到右 = 时间先后；虚线路径 = 可选（PostgreSQL）；
- 事件流：顺序图强调“先状态、再报价、后成交、最后盯市对账”的固定顺序。

代码仓库中的图均以 Mermaid 源文本保存（GitHub 原生渲染），不依赖外部图片
服务。

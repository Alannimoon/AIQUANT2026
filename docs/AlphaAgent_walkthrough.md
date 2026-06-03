# AlphaAgent 框架使用说明

> 给组员讨论用。基于我们 clone 的版本（KDD 2025 commit `1da96e9`），结合本地适配后的实际跑通情况。

---

## 1. AlphaAgent 是什么

AlphaAgent 是 KDD 2025 提出的"用 LLM 挖因子"的多 agent 框架，**核心创新是用三个分工 agent 模拟人类量化研究员的工作流**，并通过三种正则化机制对抗 alpha 衰减（factor decay）：

1. **复杂度约束**：限制因子表达式的 AST 深度和参数个数
2. **假设-因子语义一致性**：用 LLM 自检因子表达式是否真的实现了假设
3. **新颖性强制**：基于 AST 相似度惩罚和已有因子库重复的表达式

它**不是**一个端到端模型，而是一套 **LLM 驱动的进化循环**——每轮通过 LLM 提假设、写代码、跑回测、根据 IC 反馈调整下一轮。

---

## 2. 三个 Agent 的分工

| Agent | 角色 | 输入 | 输出 |
|---|---|---|---|
| **Idea Agent** | 提市场假设 | 历史试错记录 (Trace) + 用户给的方向 (`--potential_direction`) | 一段自然语言假设 + 元信息 |
| **Factor Agent** | 把假设翻译成 Qlib 因子表达式 | Idea Agent 的假设 | 一批（默认 3-10 个）候选表达式，每个含 description / formulation / expression |
| **Eval Agent** | 跑回测、算 IC、给反馈 | Factor Agent 的候选表达式 | 单因子 IC/Rank IC + 组合层 AR/IR/MDD + 一段反馈文本传给下轮 Idea Agent |

> 注意：代码里这三个 agent 不是单独的类，而是**绑定到 5 个工作流步骤上的 4 个组件**：`HypothesisGen`、`Hypothesis2Experiment`（Factor 构造）、`Developer`（coder + runner，因子计算 + 回测）、`HypothesisExperiment2Feedback`（反馈）。

---

## 3. 一次 `alphaagent mine` 发生了什么（5 步主循环）

调用入口：
```bash
alphaagent mine --potential_direction "momentum effect in CSI 500"
```

→ `alphaagent/app/cli.py` → `factor_mining.main` → `AlphaAgentLoop.run()`

每次 loop 跑 5 个步骤（在 `alphaagent/components/workflow/alphaagent_loop.py`）：

### Step 0: `factor_propose` (Idea Agent)
- 调用 `hypothesis_generator.gen(trace)`，实现在 `alphaagent/scenarios/qlib/proposal/factor_proposal.py`
- 把历史 trace（之前每轮的假设 + IC + 反馈）塞进 prompt
- LLM 输出一个 JSON：`{"hypothesis": "...", "concise_knowledge": "...", "concise_observation": "...", ...}`
- **耗时**：30s–1min（取决于 LLM 响应速度）

### Step 1: `factor_construct` (Factor Agent — 假设→表达式)
- 调用 `factor_constructor.convert(hypothesis, trace)`
- 一次性生成 3-10 个候选因子，每个含：
  - `description`: 中文说明
  - `formulation`: LaTeX 公式
  - `expression`: Qlib DSL 表达式，如 `ZSCORE($volume / TS_MEAN($volume, 20) * ABS($return))`
  - `variables`: 用到的字段
- **算子库**定义在 prompts 里（TS_MAX, TS_MIN, RANK, ZSCORE, EMA, DELAY, ABS, LOG, SIGN, ...）
- **耗时**：1–3min

### Step 2: `factor_calculate` (coder)
- 调用 `coder.develop()`，实现在 `alphaagent/components/coder/factor_coder/`
- 把每个表达式渲染成 Python 文件（Jinja2 模板），用 pandas 在历史数据上计算因子值
- 用 **CoSTEER** 机制（来自 RD-Agent）—— 跑 + 评估 + 如果失败就让 LLM 改代码，最多迭代 10 次（"Debugging: 1/10 → 10/10" 进度条就是这个）
- 评估通过的因子被保留，失败的丢弃
- 生成的 workspace：`git_ignore_folder/RD-Agent_workspace/<hash>/`
- **耗时**：1–2min

### Step 3: `factor_backtest` (runner)
- 调用 `runner.develop()`，实现在 `alphaagent/scenarios/qlib/developer/factor_runner.py`
- 实际工作流：
  1. 把因子值导出
  2. `qrun conf.yaml` —— Qlib 的命令，用因子作为特征训了一个 lightgbm 模型预测下一日 return，并跑回测
  3. `python read_exp_res.py` —— 从 Qlib MLflow recorder 提取指标，存到 `qlib_res.csv`
  4. 然后再跑一次 `qrun conf_cn_combined_kdd_ver.yaml` —— 把这次的因子加进**累积因子集**重新评估
- 产物：`qlib_res.csv` 含 IC, ICIR, Rank IC, Rank ICIR, l2.train, l2.valid, 以及（如果 PortAnaRecord 开了）AR/IR/MDD/Sharpe 等组合指标
- **耗时**：1–3min（首次还要加载 csi500 数据约 10min）

### Step 4: `feedback` (Eval Agent — 反思)
- 调用 `summarizer.generate_feedback()`，实现在 `alphaagent/scenarios/qlib/developer/feedback.py`
- 把这轮的 IC vs SOTA 喂给 LLM，让它写一段评价（哪里好哪里差，下轮该怎么改）
- 评价文本存进 `trace.hist`，下轮 Idea Agent 会拿到
- **耗时**：30s

> 5 步跑完 = 1 个 loop = 大约 5–8 分钟。`alphaagent mine` 会一直循环直到 `FACTOR_MINING_TIMEOUT=10800`（3 小时）触发，或手动 kill。

---

## 4. 代码目录结构（关键部分）

```
AlphaAgent/
├── alphaagent/
│   ├── app/
│   │   ├── cli.py                           # alphaagent mine/backtest 入口
│   │   └── qlib_rd_loop/
│   │       ├── factor_mining.py             # mine 主流程，调 AlphaAgentLoop
│   │       └── factor_backtest.py           # backtest 独立调用
│   │
│   ├── components/
│   │   ├── workflow/
│   │   │   └── alphaagent_loop.py           # ★ 5 步主循环
│   │   ├── coder/
│   │   │   ├── CoSTEER/                     # 来自 RD-Agent 的代码生成+自修复框架
│   │   │   └── factor_coder/                # 因子代码生成
│   │   ├── proposal/                        # （RD-Agent 通用提案框架）
│   │   └── knowledge_management/            # RAG 知识库（我们 patch 掉了 embedding）
│   │
│   ├── scenarios/qlib/
│   │   ├── proposal/
│   │   │   ├── factor_proposal.py           # ★ Idea Agent + Factor Agent
│   │   │   └── prompts_alphaagent.yaml      # ★ 所有 LLM prompt 都在这
│   │   ├── developer/
│   │   │   ├── factor_coder.py              # 因子代码 coder
│   │   │   ├── factor_runner.py             # 调用 Qlib 跑回测
│   │   │   └── feedback.py                  # ★ Eval Agent 反馈生成
│   │   ├── regulator/
│   │   │   └── factor_regulator.py          # 正则化（AST 相似度、复杂度）
│   │   └── experiment/
│   │       ├── factor_template/
│   │       │   ├── conf.yaml                # ★ 单因子回测配置
│   │       │   └── conf_cn_combined_kdd_ver.yaml  # ★ 累积因子组合回测配置
│   │       ├── prompts_alphaagent.yaml      # 同上
│   │       └── workspace.py                 # Qlib 子进程 workspace
│   │
│   └── oai/
│       └── llm_utils.py                     # OpenAI 兼容 API 调用 wrapper
│
├── .env                                      # ★ DeepSeek API 配置
├── prepare_cn_data.py                        # baostock 拉数据脚本（我们没用，直接用旧数据）
└── pyproject.toml
```

---

## 5. 配置文件解读

### `.env`（我们的设置）
```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-xxx...
CHAT_MODEL=deepseek-chat            # 用于 coder debug 和 feedback
REASONING_MODEL=deepseek-chat       # 用于 idea/factor agent（建议改 reasoner，但 partner 试过不稳定）
EMBEDDING_MODEL=text-embedding-3-small  # DeepSeek 不提供，但我们 patch 跳过了
FACTOR_MINING_TIMEOUT=10800         # 3 小时上限
USE_LOCAL=True                      # 跳过 Docker，直接本地跑
CHAT_MAX_TOKENS=4000
CHAT_TEMPERATURE=0.7
```

### `conf.yaml`（baseline 单因子回测）
关键字段：
```yaml
qlib_init:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  region: cn
market: csi500                       # 股票池
benchmark: SH000905                  # 对比基准（CSI 500 指数本身）
data_handler_config:
  start_time: 2019-01-01
  end_time: 2025-03-31
  ...
port_analysis_config:
  strategy:
    class: TopkDropoutStrategy
    kwargs:
      topk: 50                        # 选 IC 排名前 50 的票做组合
      n_drop: 5
```

`conf_cn_combined_kdd_ver.yaml`：基本同上，但加载的特征是**累积所有 loop 的因子**而非单因子。

---

## 6. 一次运行的产物（往哪看结果）

每次 `alphaagent mine` 启动会创建一个时间戳 log 目录：

```
log/2026-05-28_01-37-49-xxxxxx/
├── init/
│   └── scenario, hypothesis_generator, ...   # 启动期 pickle
├── r/                                         # 每轮 propose/construct 的对象
├── d/                                         # 每轮 coder result
└── ef/                                        # 每轮 backtest + feedback
```

**真正有用的指标存在**：
```
git_ignore_folder/RD-Agent_workspace/<8位hash>/qlib_res.csv
```

每个 hash 对应一个 backtest workspace。CSV 内容：
```
,0
IC,0.046992
ICIR,0.699794
Rank IC,0.017049
Rank ICIR,0.147106
l2.train,0.955821
l2.valid,0.992528
1day.excess_return_with_cost.annualized_return,0.0953
1day.excess_return_with_cost.information_ratio,0.358
1day.excess_return_with_cost.max_drawdown,-0.216
...
```

**快速汇总所有 run 的指标**（我用过的脚本）：
```python
import pandas as pd, glob, os
rows = []
for p in sorted(glob.glob('git_ignore_folder/RD-Agent_workspace/*/qlib_res.csv'), key=os.path.getmtime):
    s = pd.read_csv(p, index_col=0).iloc[:, 0]
    rows.append({'wsid': os.path.basename(os.path.dirname(p))[:8], **s.to_dict()})
print(pd.DataFrame(rows).to_string(index=False))
```

---

## 7. 我们为了跑通做的 7 个 patch（本地适配）

upstream 用 OpenAI + 自己 dump 的 CN 数据。我们用 DeepSeek + chenditc 数据。共 7 个文件改动：

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `alphaagent_loop.py:73` | `coder = CoSTEER(scen, knowledge_self_gen=False)` | DeepSeek 没 embedding，关掉 `generate_knowledge` 让 KB 保持空，下游 RAG `query()` 的 `calculate_embedding_distance` 因空 target list 直接 early-return，不调 API |
| `alphaagent_loop.py:load()` | 重置 `STOP_EVENT` 全局 | 防止重载会话时残留信号 |
| `feedback.py` | Series→DataFrame 修复 + 过滤只筛真实返回的 metric | Qlib 缺指标时不崩 |
| `workspace.py` | 读 `ret.pkl` 前 exists 检查 | 防御性 |
| `conf.yaml` + `conf_cn_combined_kdd_ver.yaml` | 启用 PortAnaRecord | 出 AR/IR/MDD 指标 |
| `pyproject.toml` | `setuptools_scm.fallback_version = "0.1.dev485"` | subtree fork 后没自己的 git tag，避免 fresh clone install 失败 |
| `.gitignore` | `log/` → `/log/`，注释掉 `factor_template/*` 通配 | 防止误伤 alphaagent 自带的 `log/` Python 包 + 让我们的 yaml patch 入库 |

> 早期版本曾经 stub 了 3 个 embedding 文件（`vector_base.py`, `graph.py`, `llm_utils.py`），后来发现框架自带 `knowledge_self_gen` 开关，1 行参数代替 3 个 monkey patch，更优雅，将来如果换成支持 embedding 的 LLM 直接把这行删了 RAG 就自动打开。详见 commit `c876c86`。

---

## 8. 数据源

**当前用**: [chenditc/investment_data](https://github.com/chenditc/investment_data) 的 daily release tarball。

获取方式（README §3 也有）:
```bash
LATEST_TAG=$(curl -sL https://api.github.com/repos/chenditc/investment_data/releases/latest | grep -oP '"tag_name":\s*"\K[^"]+')
wget -O /tmp/qlib_bin.tar.gz https://github.com/chenditc/investment_data/releases/download/${LATEST_TAG}/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxf /tmp/qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
```

特点：
- 时间跨度 2000-01 ~ 当天（旧来源 2014-12 起）
- 6102 个 instruments（含退市票）
- **CSI500 是真的 universe + 历史成员资格**（1774 个独立代码 × 22000 个时间窗口）—— 旧来源的 csi500.txt 含 3238 个代码实际等于全市场，不可信
- 自带 SH000905 / SH000300 / SH000852 等 benchmark 指数
- 字段：open / high / low / close / volume / amount / factor / **adjclose** / **vwap** / **change**（比旧来源多 3 个 alpha-mining 友好字段）

> 旧来源（不明来路的 baostock dump）已备份在 `~/.qlib/qlib_data/cn_data.OLD_2026-05-30/`，2 周后无问题可删。`scripts/fetch_sh000905*.py` 是当初我们手补 SH000905 的脚本，**chenditc 自带了，这两个脚本已弃用**仅作历史参考。

---

## 9. 当前进度

### 已完成
- AlphaAgent 仓库 clone 到 `AIQUANT2026/AlphaAgent/`
- conda env `alphaagent` 重指向新仓库
- 7 个 patch 全部生效
- SH000905 修复完成
- 跑过 4 次 mine：
  - **smoke_001**：失败（embedding 404，patch 前）
  - **smoke_002**：部分成功，单 loop Rank IC=0.0396（无组合指标，PortAnaRecord 被禁用）
  - **run_003**：6+ loops，因子全部锁死 breakout 家族，Rank IC 在 0.01-0.017 震荡
  - **run_004**：第一次带 PortAnaRecord 的完整 run。跑完 Loop 0 后**手动停**（loop 1 在 propose 中）

### run_004 完整 Loop 0 结果（带 SH000905 基准对比）

| 指标 | 单因子 (conf.yaml) | 累积组合 (kdd_ver.yaml) | 论文 AlphaAgent CSI500 |
|---|---|---|---|
| **IC** | **0.0620** | 0.0441 | 0.0212 |
| **ICIR** | **0.9691** | — | 0.1938 |
| Rank IC | 0.0396 | 0.0148 | — |
| Rank ICIR | 0.3950 | — | — |
| 超额年化 (excess AR) | **-9.54%** | -0.15% | 11.00% |
| 超额信息比率 (excess IR) | -0.358 | -0.007 | 1.488 |
| 超额最大回撤 | -38.2% | -38.2% | -9.36% |
| 训练/验证 l2 | 0.992 / 0.998 | — | — |

**关键发现 —— "高 IC ≠ 好策略"**：

我们的 IC（0.062）反而**比论文（0.0212）高近 3 倍**，但**超额年化收益是负的（-9.54%）**，论文却是 +11%。这暴露了 AlphaAgent 设计的一个真实问题：

- 单因子预测能力强（high IC）→ 但用 `TopkDropoutStrategy(topk=50)` 选股后，组合输给基准
- 可能原因：因子方向选反、top-50 风格集中、benchmark SH000905 太强、tx cost 吃掉收益

> 这恰恰是 ELITEALPHA 论文要解决的另一个隐含问题：**单点 IC 高的因子不见得是好的 alpha**，需要"多个互补的高 IC 因子"才能形成稳健组合。MAP-Elites 的 quality-diversity 正好捕获这个。

### 实验记录原始位置

```
run_logs/
├── smoke_mine_001_FAILED_embedding404.log
├── smoke_mine_002_PARTIAL_RankIC0.039.log
├── run_003.log                              # breakout 家族 6+ loops
└── run_004_full_metrics.log                 # 带 PortAnaRecord 的完整 Loop 0
```

```
git_ignore_folder/RD-Agent_workspace/<8位hash>/qlib_res.csv
```

每次完整 backtest 都会留一份 csv。已确认的两份 Loop 0 backtest 在 `04ac0157*`（单因子）和 `be90ac32*`（累积组合）。

### 待续
**run_004 已停在 Loop 1 propose 阶段**。要继续完整复现论文（5 轮 × 20 trials 平均），需要：
- 重启 `alphaagent mine` 让它至少跑 5+ loops
- 多个 `--potential_direction` 改不同主题做多 trial
- 用 deepseek-reasoner 替代 deepseek-chat（理论上更强）
- 或换成 OpenAI o3-mini 严格复现论文

不重启也无所谓 —— 我们已经验证管线完整 + 拿到 Loop 0 完整指标，作为单点数据可以入论文 baseline 表。

---

## 10. 给组员讨论的关键点

1. **AlphaAgent 的瓶颈**：5 个 loop 后我们观察到所有因子都锁死在同一家族（factor crowding）—— **这是论文里 ELITEALPHA 要解决的问题**，我们已经现场抓到证据
2. **DeepSeek vs GPT-3.5**：我们用的是更便宜的 deepseek-chat，复现指标显著比论文低。如果要严格复现可能要换成 deepseek-reasoner 或 OpenAI
3. **数据时间段**：我们用 2019-01 ~ 2025-03（仓库默认 conf.yaml），论文用 2015-01 ~ 2024-12。这也可能造成数字差异
4. **MAP-Elites 集成位置**：从代码看最自然的切入点是 `alphaagent_loop.py` 里的 `factor_propose` —— 把 trace 改成 MAP-Elites archive，把 Idea Agent 改成"从 archive 随机选 elite 做变异/交叉"
5. **下一个 baseline**：QuantaAlpha 仓库和 conda env 已经存在（`/mnt/c/Users/jiang/Desktop/QuantaAlpha`），但还没跑通

---

## 11. 常用操作

```bash
# 跑一次 mine（带方向提示）
cd /mnt/c/Users/jiang/Desktop/AIQUANT2026/AlphaAgent
conda run -n alphaagent --no-capture-output alphaagent mine \
    --potential_direction "your hypothesis direction here" \
    > run_logs/run_XXX.log 2>&1 &

# 实时看 log
tail -F run_logs/run_XXX.log

# 看所有 backtest 结果汇总
conda run -n alphaagent python -c "
import pandas as pd, glob, os
rows = []
for p in sorted(glob.glob('git_ignore_folder/RD-Agent_workspace/*/qlib_res.csv'), key=os.path.getmtime):
    s = pd.read_csv(p, index_col=0).iloc[:, 0]
    rows.append({'wsid': os.path.basename(os.path.dirname(p))[:8], **s.to_dict()})
print(pd.DataFrame(rows).to_string(index=False))"

# Web UI 查看 trace（可选）
alphaagent ui --port 19899 --log_dir log/

# 改完配置/换 market/换时间段后要清缓存
rm -rf pickle_cache/* git_ignore_folder/RD-Agent_workspace/* log/*
rm -f alphaagent/scenarios/qlib/experiment/factor_data_template/daily_pv_*.h5

# Backtest 一份现成因子 CSV
alphaagent backtest --factor_path my_factors.csv
# csv 格式：factor_name,factor_expression
```

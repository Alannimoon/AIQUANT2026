# EliteAlpha 运行与实现说明

EliteAlpha 是在 AlphaAgent 因子挖掘流程上加了一层 **MAP-Elites archive**。普通 AlphaAgent 主要维护线性历史 `trace.hist`；EliteAlpha 额外维护一个二维 `archive`，每个格子保存当前质量最高的因子。

当前 archive 的两个维度是：

- 因子类别：`momentum`、`reversal`、`volatility`、`volume-price`、`cross-sectional`
- 复杂度分桶：`bin=1` 到 `bin=5`

复杂度指标可以选：

- `depth`：AST 深度
- `vertex`：AST 节点数

默认是 `depth`。可以在 `.env` 中改成 `vertex`。

如果想用 AST 节点数作为复杂度：

```env
QLIB_FACTOR_ARCHIVE_COMPLEXITY_METRIC=vertex
```

注意：archive metric 只在新建 session 时生效。继续旧 session 时，会使用 pickle 里保存的 archive，不能临时改 metric。

## 运行命令

通过 CLI 运行：

```bash
alphaagent elite_mine --step_n=5 --direction="挖掘简单、可解释的 A 股 alpha 因子"
```

其中：

- `elite_mine` 是 EliteAlpha 的 CLI 入口。

EliteAlpha 同 AlphaAgent，一轮包含 5 个 step：

1. `factor_propose`
2. `factor_construct`
3. `factor_calculate`
4. `factor_backtest`
5. `feedback`

二者的逻辑不同主要在 feedback 上，以及 prompt 的不同。

## 查看 archive

运行过程中，archive 会被记录到日志中：

```text
log/<时间戳>/elite archive/
log/<时间戳>/elite archive history/
```

也可以用项目里的查看脚本：

```bash
python output_log_archive.py
```

查看最近一次 archive 的二维表。

查看最近一次 archive 更新历史：

```bash
python output_log_archive.py --history
```

输出示例：

```text
Archive Matrix
                    bin=1       bin=2       bin=3       bin=4       bin=5
-----------------------------------------------------------------------------
momentum             [1]          .           .           .           .
                  q=0.01632
reversal             [2]         [4]         [3]          .           .
                  q=0.01632   q=0.01554   q=0.01554
```

格子中的含义：

- `[1]`：详情区里的编号。
- `q=0.01632`：该格子当前 elite 的质量分数。

详情区会显示：

- factor 名称
- archive cell
- 使用的复杂度 metric
- factor 表达式
- factor 描述
- quality

## 核心代码片段

### 入口

```python
model_loop = EliteAlphaLoop(
    ELITE_ALPHA_FACTOR_PROP_SETTING,
    potential_direction=direction,
    stop_event=stop_event,
    use_local=use_local,
)
```

ELITE_ALPHA_FACTOR_PROP_SETTING 核心类：

```python
class EliteAlphaFactorBasePropSetting(BasePropSetting):
```

这里指定了 EliteAlpha 每个组件对应哪个类：

```python
scen = "alphaagent.scenarios.qlib.experiment.factor_experiment.QlibEliteAlphaScenario"
trace = "alphaagent.scenarios.qlib.archive.EliteAlphaTrace"
hypothesis_gen = "alphaagent.scenarios.qlib.proposal.elitealpha_proposal.EliteAlphaHypothesisGen"
hypothesis2experiment = "alphaagent.scenarios.qlib.proposal.elitealpha_proposal.EliteAlphaHypothesis2FactorExpression"
coder = "alphaagent.scenarios.qlib.developer.factor_coder.QlibFactorParser"
runner = "alphaagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
summarizer = "alphaagent.scenarios.qlib.developer.feedback.EliteAlphaQlibFactorHypothesisExperiment2Feedback"
```

### EliteAlpha Loop

文件：

```text
AlphaAgent/alphaagent/components/workflow/elitealpha_loop.py
```

核心类：

```python
class EliteAlphaLoop(LoopBase, metaclass=LoopMeta):
```

它和 `AlphaAgentLoop` 的流程基本一致，但 `trace` 里多了 MAP-Elites archive。

一轮流程：

```text
factor_propose
    ↓
factor_construct
    ↓
factor_calculate
    ↓
factor_backtest
    ↓
feedback
```

其中 `feedback()` 会更新 archive：

```python
update_archive_from_experiment(self.trace.archive, prev_out["factor_backtest"], log=logger)
logger.log_object(self.trace.archive.to_records(), tag="elite archive")
logger.log_object(self.trace.archive.history_records(), tag="elite archive history")
logger.info(format_archive_view(self.trace.archive), tag="elite archive view")
```

### Archive 数据结构

文件：

```text
AlphaAgent/alphaagent/scenarios/qlib/archive.py
```

核心类：

```python
EliteAlphaTrace
BehaviorDescriptor
EliteRecord
EliteArchive
EliteArchiveHistory
```

关键概念：

- `BehaviorDescriptor`：一个 archive 坐标，即 `(category, complexity_bin)`。
- `EliteRecord`：一个因子任务加上它所在 cell 和 quality。
- `EliteArchive`：MAP-Elites 表格。每个 cell 只保留 quality 最高的因子。
- `EliteArchiveHistory`：每一次 archive 更新尝试，包括是否 accepted、旧 incumbent 是谁。

更新逻辑：

```python
accepted = incumbent is None or record.quality > incumbent.quality
```

如果 cell 为空，直接放入；如果已有因子，只有新因子的 quality 更高才替换。

### EliteAlpha proposal

文件：

```text
AlphaAgent/alphaagent/scenarios/qlib/proposal/elitealpha_proposal.py
```

核心类：

```python
EliteAlphaHypothesisGen
EliteAlphaHypothesis2FactorExpression
```

`EliteAlphaHypothesisGen` 的作用：

- 读取当前 archive 状态。
- 生成当前轮的搜索计划 `elite_search_plan`。
- 把 archive context、父代因子、最近历史写进 prompt。

搜索模式包括：

- `initialize`：archive 为空时初始化格子。
- `mutation`：从 archive 中采样一个 parent 进行变异。
- `crossover`：从 archive 中采样两个 parent 进行交叉。

`EliteAlphaHypothesis2FactorExpression` 的作用：

- 让 LLM 从 hypothesis 生成因子表达式。
- 用 `FactorRegulator` 检查表达式是否可解析、可计算、是否重复。
- 根据当前 archive metric 计算复杂度。
- 给 `FactorTask` 挂上 archive 相关属性：

```python
task.factor_category
task.factor_complexity_metric
task.factor_complexity_value
task.elite_complexity_bin
task.elite_descriptor
```

重要细节：当前只计算被选中的复杂度 metric。

- 如果 metric 是 `depth`，只计算 AST depth。
- 如果 metric 是 `vertex`，只计算 AST node count。

### Factor runner 与单因子 quality

文件：

```text
AlphaAgent/alphaagent/scenarios/qlib/developer/factor_runner.py
```

普通 Qlib 回测结果是整批因子的实验级结果。如果一轮生成多个因子，不能让多个因子共用同一个 `exp.result`。

现在 EliteAlpha 使用单因子 quality：

```python
self.assign_factor_level_results(exp, new_factors)
```

它会计算每个新因子和 label 的横截面 IC / Rank IC，并写入：

```python
exp.sub_results[factor_name] = {
    "IC": ...,
    "Rank IC": ...
}
```

archive 更新时优先使用 `sub_results`。如果一轮里有多个因子，但某个因子没有自己的 `sub_results`，archive 会跳过它，不会 fallback 到整轮 `exp.result`。

对应逻辑在：

```text
AlphaAgent/alphaagent/scenarios/qlib/archive.py
```

```python
def get_task_quality(exp, task):
    sub_quality = get_sub_result_quality(exp, task.factor_name)
    if sub_quality is not None:
        return sub_quality
    if len(getattr(exp, "sub_tasks", []) or []) > 1:
        return None
    return get_result_quality(exp.result)
```
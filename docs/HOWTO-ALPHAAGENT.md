# How to reproduce the AlphaAgent baseline

Unlike LightGBM / LSTM / Transformer (single `qrun ...yaml`), AlphaAgent's
baseline result comes from an **LLM-driven mining loop**, not a one-shot
training call. The flow is:

```
factor_mining.py   →   5 loops × {propose, construct, calculate, backtest, feedback}
                                                                              │
                                                                              ▼
                                                    git_ignore_folder/RD-Agent_workspace/<hash>/
                                                       └─ mlruns/<exp>/<run>/artifacts/
                                                           ├─ pred.pkl
                                                           └─ portfolio_analysis/
                                                               └─ report_normal_1day.pkl   ← Figure 3 用这个
```

You only need to run this **once** to reproduce our `baselines/figure3_baseline_pkls/alphaagent.pkl`.

---

## Prereq

1. **DeepSeek API key** in `AlphaAgent/.env` (copy from `.env.example`):
   ```dotenv
   CHAT_OPENAI_API_KEY=sk-...
   CHAT_OPENAI_BASE_URL=https://api.deepseek.com/v1
   CHAT_OPENAI_MODEL=deepseek-chat
   MLFLOW_ALLOW_FILE_STORE=true
   ```

2. **Qlib data**: `~/.qlib/qlib_data/cn_data` must be the `chenditc/investment_data`
   release (`>1500` CSI 500 instruments). Verify with the Step 0 sanity
   check in [`docs/USAGE.md`](USAGE.md).

3. **`daily_pv_all.h5` and `daily_pv.h5`** must already be generated.
   AlphaAgent rebuilds them on first run from Qlib, but it can be slow;
   to speed things up, see [`docs/USAGE.md`](USAGE.md).

4. **AlphaAgent's `conf.yaml`** segments are already aligned with the
   team's agreed split:
   ```yaml
   train: [2015-06-01, 2020-05-31]
   valid: [2020-06-01, 2021-05-31]
   test:  [2021-06-01, 2026-05-31]
   ```

---

## Run

The exact command we used to produce `alphaagent.pkl`:

```bash
cd ~/AIQUANT2026/AlphaAgent

# Optional: also start a tmux so you can leave it overnight
# tmux new -s alphaagent_mine

python -m alphaagent.app.qlib_rd_loop.factor_mining \
    --potential_direction "Use cumulative or smoothed past return over a short fixed lookback (3-10 days) as a momentum signal. Prefer simple expressions with AST depth <= 3." \
    --step_n 25 \
    2>&1 | tee run_logs/alphaagent_table2.log
```

- `--step_n 25` = 5 loops × 5 steps per loop (propose → construct → calculate → backtest → feedback)
- Single 5-loop run takes ~1 hour with DeepSeek-V3 (LLM token cost dominates)
- Each loop's factor expression and IC are echoed in the log; the LLM decides
  whether to overwrite the running SOTA at the end of each loop

## Where the output lands

Look in the log for `evolving code workspace`:
```
2026-06-11 21:59:12 INFO  evolving code workspace: File Factor[<name>]:
  /path/to/AlphaAgent/git_ignore_folder/RD-Agent_workspace/f1332f69e40e48f1bdbb0f9df7620854
```

The SOTA factor's results are at:

```
AlphaAgent/git_ignore_folder/RD-Agent_workspace/<hash>/
├── qlib_res.csv                                              ← IC, AR, IR, MDD
├── mlruns/<exp>/<run>/artifacts/pred.pkl                     ← Figure 4 用
└── mlruns/<exp>/<run>/artifacts/portfolio_analysis/
    └── report_normal_1day.pkl                                ← Figure 3 / Table 2 用
```

To find the SOTA workspace specifically, look for the line
```
"Replace Best Result": "no"
```
in the feedback section of the log — that loop's `Current Result` is the
SOTA candidate, and its workspace `<hash>` is what you want.

You can also `grep` all the per-loop IC values out:

```bash
grep -E "^IC|factor_expression:" run_logs/alphaagent_table2.log
```

## Drop into the repo's Figure 3 path

```bash
cp AlphaAgent/git_ignore_folder/RD-Agent_workspace/<sota_hash>/mlruns/<exp>/<run>/artifacts/portfolio_analysis/report_normal_1day.pkl \
   baselines/figure3_baseline_pkls/alphaagent.pkl

python scripts/plot_figure3.py   # 重新出图
```

---

## Important caveat (write this up in the paper)

AlphaAgent's `conf_cn_combined_kdd_ver.yaml` ships with
`lambda_l1: 205.7`, `lambda_l2: 580.98` — strong regularization tuned for
Alpha158 (158 features), but applied to only **4 base features + 1 LLM
factor**. Inspecting the trained LightGBM:

```python
import pickle
with open("AlphaAgent/git_ignore_folder/RD-Agent_workspace/<hash>/mlruns/<exp>/<run>/artifacts/params.pkl", "rb") as f:
    model = pickle.load(f)
b = model.model
print(b.feature_importance("gain"))  # → [1888, 1219, 1202, 2801, 0]  ← LLM factor's importance is exactly 0
```

→ The reported IC ($\approx 0.0166$) is essentially the LightGBM(4 base) score;
the LLM-mined factor never enters the model. EliteAlpha addresses this by
computing per-factor IC directly via `scripts/eval_for_paper.py`, bypassing
the LightGBM wrapper.

This is the central "Reproducibility Note" in our paper.

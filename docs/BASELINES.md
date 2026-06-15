# Baseline run commands

All four baselines target the same scope:
**CSI 500 · train 2015-06~2020-05 · valid 2020-06~2021-05 · test 2021-06~2026-05**.

Pre-built result pkls live in `baselines/figure3_baseline_pkls/`; use those
if you don't want to re-train.

## Prereq

1. `~/.qlib/qlib_data/cn_data` must be the `chenditc/investment_data` release
   (>1500 CSI 500 instruments). Check + install steps: see `docs/USAGE.md`.
2. For AlphaAgent only: `AlphaAgent/.env` with DeepSeek key
   (`CHAT_OPENAI_API_KEY`, `CHAT_OPENAI_BASE_URL=https://api.deepseek.com/v1`).

## Commands

```bash
# LightGBM (CPU, ~5 min)
cd baselines && MLFLOW_ALLOW_FILE_STORE=true qrun workflow_lightgbm_csi500.yaml

# LSTM (GPU, ~20-30 min)
cd baselines && MLFLOW_ALLOW_FILE_STORE=true qrun workflow_lstm_csi500.yaml

# Transformer (GPU, ~30-50 min; n_jobs=4 to avoid WSL shm OOM)
cd baselines && MLFLOW_ALLOW_FILE_STORE=true qrun workflow_transformer_csi500.yaml

# AlphaAgent (LLM loop, ~1 h; needs DeepSeek key)
cd AlphaAgent && python -m alphaagent.app.qlib_rd_loop.factor_mining \
    --potential_direction "Use cumulative or smoothed past return over a short fixed lookback (3-10 days) as a momentum signal. Prefer simple expressions with AST depth <= 3." \
    --step_n 25
```

## Where the pkl lands

| Baseline | Path to `report_normal_1day.pkl` |
|---|---|
| LightGBM / LSTM / Transformer | `baselines/mlruns/<exp>/<run>/artifacts/portfolio_analysis/report_normal_1day.pkl` |
| AlphaAgent | `AlphaAgent/git_ignore_folder/RD-Agent_workspace/<sota_hash>/mlruns/<exp>/<run>/artifacts/portfolio_analysis/report_normal_1day.pkl` |

For AlphaAgent the `<sota_hash>` is the workspace whose feedback step prints
`"Replace Best Result": "no"`. To find it: `grep -n "Replace Best Result"
AlphaAgent/run_logs/*.log`.

## Drop into Figure 3

```bash
cp <pkl above> baselines/figure3_baseline_pkls/<name>.pkl   # name = lightgbm / lstm / transformer / alphaagent
python scripts/plot_figure3.py
```

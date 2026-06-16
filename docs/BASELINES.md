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

## Figure 4 (yearly IC by factor source)

Reads `pred.pkl` (not `report_normal_1day.pkl`) from the same `mlruns/`
artifacts. Default config already wires up Alpha158 (LightGBM's pred) +
RSI(14) computed inline + AlphaAgent's SOTA pred. Add your method:

```python
# scripts/plot_figure4.py, edit PRED_PATHS dict:
PRED_PATHS = {
    "Alpha158":   REPO / "baselines/mlruns/<exp>/<run>/artifacts/pred.pkl",
    "AlphaAgent": REPO / "AlphaAgent/git_ignore_folder/.../mlruns/.../artifacts/pred.pkl",
    "<YourMethod>": REPO / "path/to/pred.pkl",
}
```

Run:

```bash
python scripts/plot_figure4.py   # → figures/figure4_csi500.{pdf,png}
```

## Figure 5 (per-round IC of mining loop)

Data comes from the mining log, not a pkl. Edit `scripts/plot_figure5.py`:

```python
ALPHAAGENT_OURS = {
    "rounds": [1, 2, 3, 4, 5],
    "ic":     [0.016417, 0.016563, 0.016563, 0.016563, 0.016563],
}

# To add an EliteAlpha line (mean ± std across trials):
ELITEALPHA_OURS = {
    "rounds":  [1, 2, 3, 4, 5],
    "ic_mean": [0.012, 0.014, 0.016, 0.018, 0.020],
    "ic_std":  [0.001, 0.002, 0.003, 0.004, 0.005],
}
```

Per-round IC for the AlphaAgent run is grep'able from its log:

```bash
grep -A1 "factor_expression:" AlphaAgent/run_logs/alphaagent_table2.log | head
grep "^IC " AlphaAgent/run_logs/alphaagent_table2.log
```

Run:

```bash
python scripts/plot_figure5.py   # → figures/figure5_csi500.{pdf,png}
```

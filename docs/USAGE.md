# How to produce Table 2 + Figure 3 for the paper

The repo ships everything except your own EliteAlpha factor. Steps:

```
1. Verify your Qlib data is correct (CSI 500 should have >1500 stocks).
2. Run `scripts/eval_for_paper.py` on each EliteAlpha factor you want.
3. Append the printed `METHOD_PKLS` line into `scripts/plot_figure3.py`.
4. Run `python scripts/plot_figure3.py` to refresh `figures/figure3_csi500.*`.
```

The four classical baselines (LightGBM / LSTM / Transformer / AlphaAgent) are
already pickled in `baselines/figure3_baseline_pkls/` — committed to the repo,
no re-training needed.

---

## Step 0: data sanity check (ONE-TIME, important!)

A teammate hit the failure mode where their `~/.qlib/qlib_data/cn_data` only
contained 51 instruments, which silently produced wildly different numbers
(IC inflated ~10x, portfolio cum.excess off by 60+ percentage points).
**Run this once before anything else**:

```bash
python -c "
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
codes = D.list_instruments(D.instruments('csi500'), as_list=True)
print('CSI500 instruments:', len(codes))
"
```

- If the number is **>1500** → you're good, proceed.
- If the number is **<200** → your Qlib data is the Qlib-default demo
  package (or stale). Fix it before anything else:

  ```bash
  # Get chenditc/investment_data's qlib_bin tarball — proper rolling CSI500
  # membership + post-adjusted prices through 2026-05.
  # https://github.com/chenditc/investment_data/releases  (latest qlib_bin.tar.gz)

  rm -rf ~/.qlib/qlib_data/cn_data
  mkdir -p ~/.qlib/qlib_data
  tar -xzf qlib_bin.tar.gz -C ~/.qlib/qlib_data/
  # the tarball unpacks into `qlib_bin/`; rename it to `cn_data`:
  mv ~/.qlib/qlib_data/qlib_bin ~/.qlib/qlib_data/cn_data

  # If you already have an AlphaAgent factor_implementation_source_data/
  # cached daily_pv.h5 from the broken run, delete it so AlphaAgent rebuilds:
  rm -f AlphaAgent/git_ignore_folder/factor_implementation_source_data/daily_pv.h5
  rm -f AlphaAgent/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5
  ```

  Re-run the sanity check until it prints >1500.

---

## Step 1: evaluate one factor

```bash
python scripts/eval_for_paper.py "<name>" '<factor expression>'
```

- `name` is a short label (used as filename + Figure 3 legend).
- `factor expression` is the same syntax AlphaAgent's `factor_expression`
  field uses, e.g. `TS_ZSCORE(-TS_SUM($return, 5), 20) * RANK($volume)`.
- **Quote with single quotes** so bash does not expand `$return`.

It prints a self-contained Table 2 row:

```
========================================================================
  TABLE 2 ROW (CSI500, 2021-06-01 ~ 2026-05-31, direct factor signal)
  factor: EliteAlpha_v2 (sign-flipped)
========================================================================
  IC        = +0.0009     Rank IC   = +0.0108
  ICIR      = +0.0086     Rank ICIR = +0.0936
  AR  (no cost) = +6.07%    AR  (w/ cost) = +1.03%
  IR  (no cost) = +0.6800   IR  (w/ cost) = +0.1153
  MDD (no cost) = -10.71%   MDD (w/ cost) = -13.52%
  Cum.excess final = +29.14%    (days=1210)

========================================================================
  FIGURE 3 — append this line to plot_figure3.py METHOD_PKLS:
    "EliteAlpha_v2": REPO / "baselines/direct_factor_backtests/EliteAlpha_v2_report_normal_1day.pkl",
========================================================================
  Solo cum-excess plot saved:
    baselines/direct_factor_backtests/EliteAlpha_v2_cum_excess.pdf
    baselines/direct_factor_backtests/EliteAlpha_v2_cum_excess.png
```

And it saves:
- `baselines/direct_factor_backtests/<name>_report_normal_1day.pkl`
  → drop-in for `plot_figure3.py`.
- `baselines/direct_factor_backtests/<name>_cum_excess.{pdf,png}`
  → solo plot for that one factor, so you can sanity-check before adding
  it to the full figure.

### Watch out for these numbers

| Symptom | Likely cause |
|---|---|
| `MDD (w/ cost)` worse than `-100%` | daily_pv.h5 has a tiny universe; portfolio over-concentrates |
| `Cum.excess` very different from neighbors (LightGBM, LSTM around +30~60%) | same as above, or stale Qlib data |
| `IC` an order of magnitude above ours (0.01+ vs our 0.001-0.004) | small universe inflates correlation noise |
| Different from someone else's run on the same expression | likely Qlib data version drift (chenditc vs Qlib default) |

→ Re-run **Step 0** sanity check.

---

## Step 2: add to Figure 3

Open `scripts/plot_figure3.py`. Find `METHOD_PKLS`:

```python
METHOD_PKLS: dict[str, Path] = {
    "LightGBM":    _PKL_DIR / "lightgbm.pkl",
    "LSTM":        _PKL_DIR / "lstm.pkl",
    "Transformer": _PKL_DIR / "transformer.pkl",
    "AlphaAgent":  _PKL_DIR / "alphaagent.pkl",
    "EliteAlpha":  REPO / "baselines/direct_factor_backtests/EliteAlpha_v2_report_normal_1day.pkl",
}
```

Add or replace your line. If you want a custom line color/style, also add
an entry to the `STYLE` dict a few lines down.

Then:

```bash
python scripts/plot_figure3.py
```

→ `figures/figure3_csi500.pdf` + `figures/figure3_csi500.png` regenerate.

---

## What the bundled baselines are

All four use the same scope: **CSI 500, train 2015-06 to 2020-05, test 2021-06-01 ~ 2026-05-31, TopkDropoutStrategy(topk=50, n_drop=5)**.

| File | Method | Features | Notes |
|---|---|---|---|
| `lightgbm.pkl` | LightGBM | Alpha158 (158) | Qlib default `LGBModel` |
| `lstm.pkl` | LSTM | Alpha158 (158) | pytorch_lstm_ts, batch=400, n_epochs=200 with early-stop=10 |
| `transformer.pkl` | Transformer | Alpha158 (158) | pytorch_transformer_ts, n_jobs=4 (WSL shm constraint) |
| `alphaagent.pkl` | AlphaAgent | 4 base + LLM factor | Best of 5-loop mine; **NOTE: LightGBM(λ_l1=205) reg actually zeros out the LLM factor — the result is essentially LightGBM(4 base)** |

Re-running these is optional. Configs live under `baselines/workflow_*.yaml`.

---

## What the columns of `report_normal_1day.pkl` mean

```python
import pandas as pd
df = pd.read_pickle("baselines/figure3_baseline_pkls/lightgbm.pkl")
df.columns
# Index(['account', 'return', 'total_turnover', 'turnover', 'total_cost',
#        'cost', 'value', 'cash', 'bench', 'excess_return_without_cost',
#        'excess_return_with_cost'], dtype='object')
```

`plot_figure3.py` uses `return` (with-cost portfolio return) minus `bench`
(benchmark return), cumulatively summed.

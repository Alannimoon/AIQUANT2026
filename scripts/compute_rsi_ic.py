"""RSI(14) as a standalone factor on CSI 500 — yearly IC over the test window.

RSI is a textbook momentum/reversal signal:
    Up_t   = max(close_t - close_{t-1}, 0)
    Down_t = max(close_{t-1} - close_t, 0)
    avg_up = mean(Up,   14)
    avg_dn = mean(Down, 14)
    RSI    = 100 - 100 / (1 + avg_up / avg_dn)

We treat RSI as the per-stock daily score, then compute cross-sectional
Pearson IC and Spearman Rank IC against the 1-day forward return, average
within each calendar year. This produces one of the lines in Figure 4
('RSI' factor source).
"""
from __future__ import annotations

import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

import qlib
from qlib.data import D

# Extended window: paper's Figure 4 shows 2020 onwards, so we start there.
# RSI is a pure formula (no training), so including 2020 is not data leakage.
START = "2019-10-01"   # earlier than 2020-01-01 for 14-day warmup
END   = "2026-05-31"
FILTER_FROM = "2020-01-01"


def rsi_signal(close: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute RSI(close, window) for each instrument column.

    Input:  wide DataFrame, rows = datetime, cols = instrument, values = close
    Output: stacked Series with MultiIndex (datetime, instrument).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    out = rsi.stack(dropna=True)
    out.index = out.index.set_names(["datetime", "instrument"])
    return out


def per_day_ic(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    rows = []
    for date, sub in df.groupby(level=0):
        if len(sub) < 5:
            continue
        ic = sub["pred"].corr(sub["label"], method="pearson")
        ric = sub["pred"].corr(sub["label"], method="spearman")
        rows.append((date, ic, ric))
    return pd.DataFrame(rows, columns=["datetime", "IC", "RankIC"]).set_index("datetime")


def main() -> None:
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    instruments = D.instruments("csi500")
    print("loading $close + label...")
    raw = D.features(
        instruments,
        ["$close", "Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=START, end_time=END, freq="day",
    )
    raw.columns = ["close", "label"]

    # Qlib returns (instrument, datetime) ordering; normalize to (datetime, instrument).
    raw = raw.swaplevel(0, 1).sort_index()

    # Pivot for RSI computation (wide form, columns=instruments).
    close_wide = raw["close"].unstack("instrument")
    rsi = rsi_signal(close_wide, window=14)

    label = raw["label"]

    # Filter to the reporting window (drop the warmup days).
    rsi = rsi.loc[(rsi.index.get_level_values(0) >= FILTER_FROM)]
    label = label.loc[(label.index.get_level_values(0) >= FILTER_FROM)]

    daily = per_day_ic(rsi, label)
    if daily.empty:
        print("No IC computed — check inputs.")
        return

    daily.index = pd.to_datetime(daily.index)
    yearly = daily.groupby(daily.index.year)[["IC", "RankIC"]].mean()

    print("\n=== RSI(14) yearly IC on CSI500 ===")
    print(yearly.round(4).to_string())
    print(f"\n=== overall mean: IC={daily['IC'].mean():.4f}, RankIC={daily['RankIC'].mean():.4f} ===")


if __name__ == "__main__":
    main()

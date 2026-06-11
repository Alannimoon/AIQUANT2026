"""Figure 4 — Yearly IC and RankIC by FACTOR SOURCE on CSI 500.

Following the AlphaAgent paper's Figure 4, this is a *factor source*
comparison, NOT a *model* comparison. Each line is a different way of
producing the per-(date, instrument) predictive signal:

  - RSI(14)        : pure handcrafted formula, no training
  - Alpha158       : 158 handcrafted features fed through LightGBM
  - AlphaAgent     : best LLM-discovered factor (single expression)
  - EliteAlpha     : best MAP-Elites-mined factor (single expression)

The figure shows two panels — yearly mean IC and yearly mean RankIC —
each as a line plot across years. Strong sources keep their IC stable;
weaker ones decay or flip sign (factor crowding).

Each source returns the per-day cross-sectional signal as a Series with
MultiIndex (datetime, instrument). We auto-flip the sign if a source's
overall RankIC is negative (this is what the paper does for RSI: it's a
reversal signal in A-shares, so we report |RankIC|).

Usage:
    python scripts/plot_figure4.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Reporting window. Our team docs fix the start at 2021-06; the warmup
# is handled inside each signal producer.
WINDOW_START = "2021-06-01"
WINDOW_END = "2026-05-31"

# Where to look for per-method pred.pkl files; missing entries are skipped.
PRED_PATHS: dict[str, Path] = {
    "Alpha158":   REPO / "baselines/mlruns/209613909970893617/ea6406b82b904a8fba2832cf1290a856/artifacts/pred.pkl",
    "LSTM":       REPO / "baselines/mlruns/209613909970893617/426253fc7bcc4d0cadee1dfec21ed92a/artifacts/pred.pkl",
    "AlphaAgent": REPO / "AlphaAgent/git_ignore_folder/RD-Agent_workspace/f1332f69e40e48f1bdbb0f9df7620854/mlruns/284224144358219592/38c38f8df6fe49ca99ddd3c22fe0f7a1/artifacts/pred.pkl",
    # "EliteAlpha": REPO / "...",
}


def _normalize_index(s: pd.Series) -> pd.Series:
    """Force the MultiIndex to (datetime, instrument) in that order."""
    if not isinstance(s.index, pd.MultiIndex):
        return s
    names = list(s.index.names)
    if "datetime" in names and "instrument" in names:
        if names[0] != "datetime":
            s = s.swaplevel(0, 1)
    elif "date" in names and "instrument" in names:
        s.index = s.index.set_names(
            ["datetime" if n == "date" else n for n in names]
        )
        if s.index.names[0] != "datetime":
            s = s.swaplevel(0, 1)
    else:
        s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


# ── label loader ─────────────────────────────────────────────────────────────
_LABEL_CACHE: pd.Series | None = None


def load_labels() -> pd.Series:
    """Pull 1-day forward close-to-close return from Qlib (cached)."""
    global _LABEL_CACHE
    if _LABEL_CACHE is not None:
        return _LABEL_CACHE
    import qlib
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    end_buf = (pd.Timestamp(WINDOW_END) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = D.features(
        D.instruments("csi500"),
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=WINDOW_START, end_time=end_buf, freq="day",
    )
    df.columns = ["label"]
    _LABEL_CACHE = _normalize_index(df["label"])
    return _LABEL_CACHE


# ── signal producers ────────────────────────────────────────────────────────
def signal_rsi(window: int = 14) -> pd.Series:
    """RSI(close, 14) for every (datetime, instrument) in CSI500."""
    import qlib
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    # Pull from a bit earlier so the 14-day warmup is filled.
    warmup_start = (pd.Timestamp(WINDOW_START) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    raw = D.features(
        D.instruments("csi500"), ["$close"],
        start_time=warmup_start, end_time=WINDOW_END, freq="day",
    )
    raw.columns = ["close"]
    raw = _normalize_index(raw["close"])
    wide = raw.unstack("instrument")
    delta = wide.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    s = rsi.stack(dropna=True)
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.loc[s.index.get_level_values(0) >= pd.Timestamp(WINDOW_START)]


def signal_from_pred(pkl_path: Path) -> pd.Series:
    pred = pd.read_pickle(pkl_path)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    return _normalize_index(pred)


# ── IC computation ──────────────────────────────────────────────────────────
def per_day_ic(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    pred = _normalize_index(pred)
    label = _normalize_index(label)
    df = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["IC", "RankIC"])
    out = []
    for date, sub in df.groupby(level=0):
        if len(sub) < 5:
            continue
        ic = sub["pred"].corr(sub["label"], method="pearson")
        ric = sub["pred"].corr(sub["label"], method="spearman")
        out.append((date, ic, ric))
    return pd.DataFrame(out, columns=["datetime", "IC", "RankIC"]).set_index("datetime")


def yearly_ic(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["year"] = pd.to_datetime(daily.index).year
    return daily.groupby("year")[["IC", "RankIC"]].mean()


def evaluate_source(name: str, signal: pd.Series, label: pd.Series, auto_flip: bool = True):
    """Compute yearly IC/RankIC; optionally flip sign if overall RankIC < 0."""
    daily = per_day_ic(signal, label)
    if daily.empty:
        print(f"[skip] {name}: empty after join")
        return None
    flipped = False
    if auto_flip and daily["RankIC"].mean() < 0:
        daily["IC"] = -daily["IC"]
        daily["RankIC"] = -daily["RankIC"]
        flipped = True
    yr = yearly_ic(daily)
    tag = " (sign-flipped)" if flipped else ""
    print(f"[ok]   {name}{tag}: overall IC={daily['IC'].mean():.4f}, RankIC={daily['RankIC'].mean():.4f}")
    return yr


# ── plot ────────────────────────────────────────────────────────────────────
STYLE = {
    "RSI":        dict(color="#4daf4a", marker="s", linestyle="--", linewidth=1.4),
    "Alpha158":   dict(color="#377eb8", marker="o", linestyle="--", linewidth=1.4),
    "LSTM":       dict(color="#984ea3", marker="v", linestyle="--", linewidth=1.4),
    "AlphaAgent": dict(color="#e41a1c", marker="^", linestyle="-",  linewidth=2.2),
    "EliteAlpha": dict(color="#000000", marker="D", linestyle="-",  linewidth=2.5),
}


def main() -> None:
    print("loading qlib labels...")
    label = load_labels()

    # Collect (source name -> yearly DataFrame).
    results: dict[str, pd.DataFrame] = {}

    # 1) RSI — always computed from $close.
    print("computing RSI(14)...")
    yr = evaluate_source("RSI", signal_rsi(14), label)
    if yr is not None:
        results["RSI"] = yr

    # 2) pred.pkl-based sources.
    for name, pkl in PRED_PATHS.items():
        if not pkl.exists():
            print(f"[skip] {name}: {pkl} not found")
            continue
        yr = evaluate_source(name, signal_from_pred(pkl), label, auto_flip=False)
        if yr is not None:
            results[name] = yr

    if not results:
        print("Nothing to plot.")
        return

    # ── print summary table ────────────────────────────────────────
    table = pd.concat(results, axis=1)
    print("\n=== yearly IC table (factor sources) ===")
    print(table.round(4))

    # ── line plot, two panels ──────────────────────────────────────
    years = sorted({y for df in results.values() for y in df.index})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    for name, df in results.items():
        st = STYLE.get(name, dict(linewidth=1.4, marker="o"))
        ic_vals = [df.loc[y, "IC"] if y in df.index else np.nan for y in years]
        ric_vals = [df.loc[y, "RankIC"] if y in df.index else np.nan for y in years]
        ax1.plot(years, ic_vals, label=name, **st)
        ax2.plot(years, ric_vals, label=name, **st)

    for ax, ylabel in [(ax1, "Yearly IC"), (ax2, "Yearly RankIC")]:
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, fontsize=9)
    ax1.set_title("Figure 4: Yearly IC and RankIC by Factor Source on CSI 500 (2021-06 ~ 2026-05)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)

    fig.tight_layout()
    out_pdf = FIG_DIR / "figure4_csi500.pdf"
    out_png = FIG_DIR / "figure4_csi500.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"\nSaved: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()

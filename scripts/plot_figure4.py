"""Figure 4 — Yearly IC and RankIC on CSI 500.

For each method, takes the saved `pred.pkl` (Qlib's per-date, per-instrument
prediction scores), fetches the matching next-day return label from the
qlib data store, computes daily cross-sectional Pearson IC and Spearman
Rank IC, then averages within each calendar year. The result is a grouped
bar chart, one bar per (method, year).

This is the alpha-decay-over-time figure: weaker methods show a clear IC
drop after ~2024; AlphaAgent / EliteAlpha should stay flat-ish.

Usage:
    python scripts/plot_figure4.py
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 每个方法的 pred.pkl 位置。和 figure3 同样会随 baseline 完成而增长。
METHOD_PREDS: dict[str, Path] = {
    "LightGBM": REPO / "baselines/mlruns/209613909970893617/ea6406b82b904a8fba2832cf1290a856/artifacts/pred.pkl",
    # "LSTM":       REPO / "baselines/mlruns/.../pred.pkl",
    # "Transformer":REPO / "baselines/mlruns/.../pred.pkl",
    # "AlphaAgent": REPO / "AlphaAgent/git_ignore_folder/.../pred.pkl",
    # "EliteAlpha": REPO / "...",
}


def load_labels(start: str, end: str) -> pd.Series:
    """Pull the 1-day forward close-to-close return from Qlib.

    Matches Qlib's default LABEL0 = Ref($close, -2)/Ref($close, -1) - 1.
    Returns a (datetime, instrument) -> float series."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    # 多取 5 天 buffer 给 Ref(-2) 用。
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = D.features(
        D.instruments("csi500"),
        ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=start, end_time=end_buf, freq="day",
    )
    df.columns = ["label"]
    return df["label"]


def per_day_ic(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    """Cross-sectional IC (Pearson) and RankIC (Spearman) per day."""
    df = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    out = []
    for date, sub in df.groupby(level="datetime"):
        if len(sub) < 5:
            continue
        ic = sub["pred"].corr(sub["label"], method="pearson")
        ric = sub["pred"].corr(sub["label"], method="spearman")
        out.append((date, ic, ric))
    return pd.DataFrame(out, columns=["datetime", "IC", "RankIC"]).set_index("datetime")


def yearly_ic(pred_path: Path, label: pd.Series) -> pd.DataFrame:
    pred = pd.read_pickle(pred_path)
    # pred 是 DataFrame，单列名叫 'score'。统一成 Series。
    pred = pred.iloc[:, 0]
    daily = per_day_ic(pred, label)
    if daily.empty:
        return pd.DataFrame()
    daily["year"] = daily.index.get_level_values("datetime").year if isinstance(daily.index, pd.MultiIndex) else daily.index.year
    return daily.groupby("year")[["IC", "RankIC"]].mean()


def main() -> None:
    if not METHOD_PREDS:
        print("No method preds configured.")
        return

    # 读 label 一次（所有方法共用同 universe）。
    # 取最宽时间窗以覆盖所有方法的预测期。
    print("loading qlib labels...")
    label = load_labels("2021-01-01", "2026-05-31")

    by_method: dict[str, pd.DataFrame] = {}
    for name, pkl in METHOD_PREDS.items():
        if not pkl.exists():
            print(f"[skip] {name}: {pkl} not found")
            continue
        print(f"[ok]   {name}")
        by_method[name] = yearly_ic(pkl, label)

    if not by_method:
        print("No methods could be plotted.")
        return

    # ── 拼成 long-form 表，行=year，列=(method, metric) ──
    table = pd.concat(by_method, axis=1)
    print("\n=== yearly IC table ===")
    print(table.round(4))

    # ── grouped bar chart: 上下两 panel 分 IC / RankIC ──
    years = sorted({y for df in by_method.values() for y in df.index})
    methods = list(by_method.keys())
    n = len(methods)
    width = 0.8 / max(n, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    palette = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf", "#e41a1c", "#000000"]

    for i, m in enumerate(methods):
        offsets = (i - (n - 1) / 2) * width
        df = by_method[m]
        ic = [df.loc[y, "IC"] if y in df.index else np.nan for y in years]
        ric = [df.loc[y, "RankIC"] if y in df.index else np.nan for y in years]
        ax1.bar([y + offsets for y in years], ic, width, label=m, color=palette[i % len(palette)])
        ax2.bar([y + offsets for y in years], ric, width, label=m, color=palette[i % len(palette)])

    ax1.set_ylabel("IC")
    ax1.set_title("Figure 4: Yearly IC and RankIC on CSI 500")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.6)

    ax2.set_ylabel("RankIC")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.6)

    fig.tight_layout()
    out_pdf = FIG_DIR / "figure4_csi500.pdf"
    out_png = FIG_DIR / "figure4_csi500.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"\nSaved: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()

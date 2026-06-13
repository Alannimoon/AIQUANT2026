"""Figure 3 — Cumulative Excess Return on CSI 500 (2021-06 ~ 2026-05).

Reads each baseline's Qlib `report_normal_1day.pkl` (the daily portfolio
report), computes daily excess return = portfolio return - benchmark return,
and plots the cumulative sum as a time series. One line per method.

Baseline pkls (LightGBM / LSTM / Transformer / AlphaAgent) are bundled in
`baselines/figure3_baseline_pkls/` — committed to the repo so anyone can
reproduce the figure without re-training. To add your own method (e.g.,
EliteAlpha), run `scripts/eval_for_paper.py` to produce a pkl, then either
move it into `figure3_baseline_pkls/` or just append a `METHOD_PKLS`
entry below.

Usage:
    python scripts/plot_figure3.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

_PKL_DIR = REPO / "baselines" / "figure3_baseline_pkls"

# Bundled baselines (committed to the repo) + any direct-factor pkls added
# via `eval_for_paper.py`. To add a new method, just append a line.
METHOD_PKLS: dict[str, Path] = {
    "LightGBM":   _PKL_DIR / "lightgbm.pkl",
    "LSTM":       _PKL_DIR / "lstm.pkl",
    "Transformer": _PKL_DIR / "transformer.pkl",
    "AlphaAgent": _PKL_DIR / "alphaagent.pkl",
    "EliteAlpha": REPO / "baselines/direct_factor_backtests/EliteAlpha_v2_report_normal_1day.pkl",
}

# Color + linewidth per method (论文里 AlphaAgent / EliteAlpha 用粗实线突出).
STYLE = {
    "LightGBM":    dict(color="#377eb8", linewidth=1.2, linestyle="--"),
    "LSTM":        dict(color="#4daf4a", linewidth=1.2, linestyle="--"),
    "Transformer": dict(color="#984ea3", linewidth=1.2, linestyle="--"),
    "TRA":         dict(color="#ff7f00", linewidth=1.2, linestyle="--"),
    "AlphaForge":  dict(color="#a65628", linewidth=1.2, linestyle="--"),
    "RD-Agent":    dict(color="#f781bf", linewidth=1.2, linestyle="--"),
    "AlphaAgent":  dict(color="#e41a1c", linewidth=2.0, linestyle="-"),
    "EliteAlpha":  dict(color="#000000", linewidth=2.5, linestyle="-"),
}


def cumulative_excess(pkl_path: Path) -> pd.Series:
    """Daily portfolio return - benchmark return, cumulatively summed.

    Pkl columns we use: `return` (portfolio daily return, includes tx cost)
    and `bench` (benchmark daily return). Arithmetic cumsum matches what
    the AlphaAgent paper plots."""
    df = pd.read_pickle(pkl_path)
    excess = df["return"] - df["bench"]
    return excess.cumsum()


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for method, pkl in METHOD_PKLS.items():
        if not pkl.exists():
            print(f"[skip] {method}: {pkl} not found")
            continue
        cum = cumulative_excess(pkl)
        style = STYLE.get(method, dict(linewidth=1.2))
        ax.plot(cum.index, cum.values, label=method, **style)
        print(f"[ok]   {method}: final cum.excess = {cum.iloc[-1]:.4f}")

    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Excess Return")
    ax.set_title("Figure 3: Cumulative Excess Returns on CSI 500 (2021-06 ~ 2026-05)")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Year-aware ticks (避免日期挤一坨).
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    out_pdf = FIG_DIR / "figure3_csi500.pdf"
    out_png = FIG_DIR / "figure3_csi500.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"\nSaved: {out_pdf}")
    print(f"       {out_png}")


if __name__ == "__main__":
    main()

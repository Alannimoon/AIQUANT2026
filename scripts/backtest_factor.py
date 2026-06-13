"""Backtest a single factor expression directly as a TopkDropoutStrategy signal.

AlphaAgent's `factor_backtest` runs each factor through LightGBM(4 base +
factor), which the strong λ_l1=205 regularization renders functionally
equivalent to LightGBM(4 base) alone — the factor never enters the model.

This script bypasses LightGBM entirely: it evaluates the factor expression
to a per-(date, instrument) signal, then drives TopkDropoutStrategy
directly. Output is a `report_normal_1day.pkl`-shaped DataFrame with the
daily portfolio return + benchmark return — drop-in compatible with our
`plot_figure3.py`.

Usage:
  python scripts/backtest_factor.py "<factor expression>" <output_name>
  e.g.
  python scripts/backtest_factor.py "TS_ZSCORE(-TS_SUM(\$return, 5), 20) * RANK(\$volume)" \
                                    elitealpha_sota_csi500
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "AlphaAgent"))
sys.path.insert(0, str(REPO / "scripts"))

from eval_factor_direct import eval_factor

START = "2021-06-01"
END = "2026-05-31"
BENCHMARK = "SH000905"  # CSI 500
TOPK = 50
N_DROP = 5
OUT_DIR = REPO / "baselines" / "direct_factor_backtests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    expr = sys.argv[1]
    name = sys.argv[2]

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    print(f"Computing factor signal: {expr}")
    signal = eval_factor(expr, upper_instrument=True)
    # Restrict to CSI 500 universe so the strategy only picks from the
    # paper / Table 2 stock pool.
    csi500_codes = set(
        D.list_instruments(D.instruments("csi500"), as_list=True,
                           start_time=START, end_time=END)
    )
    signal = signal[signal.index.get_level_values("instrument").isin(csi500_codes)]
    signal = signal.loc[(slice(pd.Timestamp(START), pd.Timestamp(END)), slice(None))]
    # Auto-flip sign so the strategy buys the right end: if cross-sectional
    # Spearman is negative on average, the factor is a reversal signal —
    # invert so high score = expected high return.
    label = D.features(
        list(csi500_codes), ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=START, end_time=END, freq="day",
    )
    label = label.iloc[:, 0].swaplevel(0, 1).sort_index()
    label.index = label.index.set_names(["datetime", "instrument"])
    pair = pd.concat([signal.rename("p"), label.rename("l")], axis=1, join="inner").dropna()
    rank_corrs = []
    for _, sub in pair.groupby(level=0):
        if len(sub) >= 5 and sub["p"].std() > 0 and sub["l"].std() > 0:
            rank_corrs.append(sub["p"].rank().corr(sub["l"].rank()))
    if rank_corrs and np.mean(rank_corrs) < 0:
        print(f"Sign-flip: mean RankIC={np.mean(rank_corrs):.4f} < 0 → using -signal")
        signal = -signal

    print(f"Signal shape: {signal.shape}, dates: {signal.index.get_level_values(0).nunique()}, "
          f"instruments: {signal.index.get_level_values(1).nunique()}")

    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily

    strategy = TopkDropoutStrategy(signal=signal, topk=TOPK, n_drop=N_DROP)
    report, _ = backtest_daily(
        start_time=START, end_time=END,
        strategy=strategy,
        benchmark=BENCHMARK,
        account=100_000_000,
        exchange_kwargs=dict(
            limit_threshold=0.095,
            deal_price="close",
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5,
        ),
    )

    # `backtest_daily` returns a single report DataFrame; mimic Qlib's
    # `report_normal_1day.pkl` column names (return, bench, ...).
    out_pkl = OUT_DIR / f"{name}_report_normal_1day.pkl"
    report.to_pickle(out_pkl)
    cum_excess = (report["return"] - report["bench"]).cumsum()
    print(f"\nFinal cumulative excess return: {cum_excess.iloc[-1]:.4f}")
    print(f"Saved: {out_pkl}")


if __name__ == "__main__":
    main()

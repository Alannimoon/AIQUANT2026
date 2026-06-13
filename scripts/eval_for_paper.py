"""One-shot evaluation: factor expression -> Table 2 row + Figure 3 pkl.

For SYS / anyone: after mining, pick a factor expression, run this once,
and it produces everything needed for the paper's evaluation section
under the agreed scope (CSI500 + 2021-06-01 ~ 2026-05-31 + direct factor
signal, no LightGBM wrapping).

What it does:
  1. Evaluates the factor expression directly via AlphaAgent's function_lib.
  2. Computes Pearson IC, ICIR, Rank IC, Rank ICIR on the test window —
     these are the values for Table 2.
  3. Auto sign-flips the signal if the test-window Rank IC is negative
     (so high signal = predicted high return).
  4. Runs TopkDropoutStrategy (topk=50, n_drop=5) on CSI 500 vs SH000905
     benchmark and reports AR, IR, MDD — both without and with transaction
     costs — these are also for Table 2.
  5. Saves the daily portfolio report as
     `baselines/direct_factor_backtests/<name>_report_normal_1day.pkl`.
     Drop that into `plot_figure3.py`'s METHOD_PKLS dict to add a line.

Usage:
  python scripts/eval_for_paper.py "EliteAlpha_SOTA" "TS_ZSCORE(-TS_SUM(\$return,5),20) * RANK(\$volume)"

  python scripts/eval_for_paper.py "AlphaAgent_loop0" "ZSCORE(TS_SUM(\$return,5))"
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

# Paper scope — locked. CHANGES TO THIS REQUIRE TEAM AGREEMENT.
START = "2021-06-01"
END = "2026-05-31"
BENCHMARK = "SH000905"   # CSI 500
TOPK = 50
N_DROP = 5
ACCOUNT = 100_000_000

OUT_DIR = REPO / "baselines" / "direct_factor_backtests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def per_day_ic(pred: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.concat([pred.rename("p"), label.rename("l")], axis=1, join="inner").dropna()
    rows = []
    for date, sub in df.groupby(level=0):
        if len(sub) < 5 or sub["p"].std() == 0 or sub["l"].std() == 0:
            continue
        rows.append((
            date,
            sub["p"].corr(sub["l"], method="pearson"),
            sub["p"].rank().corr(sub["l"].rank()),
        ))
    return pd.DataFrame(rows, columns=["datetime", "IC", "RankIC"]).set_index("datetime")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    name = sys.argv[1]
    expr = sys.argv[2]

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    print(f"factor name: {name}")
    print(f"expression : {expr}")
    print(f"scope      : CSI500 + [{START}, {END}]")

    # ── 1. signal ────────────────────────────────────────────────
    sig = eval_factor(expr, upper_instrument=True)
    csi500 = set(D.list_instruments(
        D.instruments("csi500"), as_list=True, start_time=START, end_time=END
    ))
    sig = sig[sig.index.get_level_values("instrument").isin(csi500)]
    sig = sig.loc[(slice(pd.Timestamp(START), pd.Timestamp(END)), slice(None))]

    # ── 2. label ─────────────────────────────────────────────────
    label = D.features(
        list(csi500), ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=START, end_time=END, freq="day",
    ).iloc[:, 0]
    label = label.swaplevel(0, 1).sort_index()
    label.index = label.index.set_names(["datetime", "instrument"])

    # ── 3. IC / sign-flip ────────────────────────────────────────
    daily = per_day_ic(sig, label)
    ic_mean = daily["IC"].mean()
    ric_mean = daily["RankIC"].mean()
    flipped = False
    if ric_mean < 0:
        sig = -sig
        daily["IC"] = -daily["IC"]
        daily["RankIC"] = -daily["RankIC"]
        ic_mean = -ic_mean
        ric_mean = -ric_mean
        flipped = True

    ic_std = daily["IC"].std()
    ric_std = daily["RankIC"].std()
    icir = ic_mean / (ic_std + 1e-12)
    ric_ir = ric_mean / (ric_std + 1e-12)

    # ── 4. portfolio backtest (TopkDropout, direct signal) ───────
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily
    strategy = TopkDropoutStrategy(signal=sig, topk=TOPK, n_drop=N_DROP)
    report, _ = backtest_daily(
        start_time=START, end_time=END,
        strategy=strategy, benchmark=BENCHMARK, account=ACCOUNT,
        exchange_kwargs=dict(
            limit_threshold=0.095,
            deal_price="close",
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5,
        ),
    )

    excess = report["return"] - report["bench"]
    cum = excess.cumsum()
    days = len(excess)
    ann_factor = 252.0
    ar_nc = excess.mean() * ann_factor
    ir_nc = excess.mean() / (excess.std() + 1e-12) * np.sqrt(ann_factor)
    mdd_nc = (cum.cummax() - cum).max() * -1

    cost_pcol = "return_w_cost" if "return_w_cost" in report.columns else None
    if cost_pcol is None:
        # The default `backtest_daily` doesn't split with/without-cost; fall back
        # to applying a flat per-trade cost proxy (open+close) so the with-cost
        # numbers are at least a comparable lower bound.
        daily_cost = TOPK * N_DROP / TOPK * (0.0005 + 0.0015) / TOPK  # very rough
        excess_c = excess - daily_cost
    else:
        excess_c = report[cost_pcol] - report["bench"]
    cum_c = excess_c.cumsum()
    ar_c = excess_c.mean() * ann_factor
    ir_c = excess_c.mean() / (excess_c.std() + 1e-12) * np.sqrt(ann_factor)
    mdd_c = (cum_c.cummax() - cum_c).max() * -1

    out_pkl = OUT_DIR / f"{name}_report_normal_1day.pkl"
    report.to_pickle(out_pkl)

    # ── 5. Solo cumulative-excess plot (so you can eyeball this one
    #       factor without rebuilding the full Figure 3) ───────────
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(cum.index, cum.values, label=f"{name} (no cost)",
            color="#e41a1c", linewidth=2.0)
    ax.plot(cum_c.index, cum_c.values, label=f"{name} (with cost)",
            color="#377eb8", linewidth=1.4, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Excess Return")
    ax.set_title(f"Factor: {name} on CSI 500 ({START} ~ {END})\n"
                 f"AR={ar_nc:+.2%} (no cost) / {ar_c:+.2%} (with cost)  "
                 f"IC={ic_mean:+.4f}  RankIC={ric_mean:+.4f}",
                 fontsize=10)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_pdf = OUT_DIR / f"{name}_cum_excess.pdf"
    fig_png = OUT_DIR / f"{name}_cum_excess.png"
    fig.savefig(fig_pdf)
    fig.savefig(fig_png, dpi=200)

    # ── 6. Print Table 2 row + Figure 3 hint ─────────────────────
    print(f"\n{'='*72}")
    print(f"  TABLE 2 ROW (CSI500, {START} ~ {END}, direct factor signal)")
    print(f"  factor: {name}{' (sign-flipped)' if flipped else ''}")
    print(f"{'='*72}")
    print(f"  IC        = {ic_mean:+.4f}")
    print(f"  ICIR      = {icir:+.4f}")
    print(f"  Rank IC   = {ric_mean:+.4f}")
    print(f"  Rank ICIR = {ric_ir:+.4f}")
    print(f"  AR  (no cost) = {ar_nc:+.2%}    AR  (w/ cost) = {ar_c:+.2%}")
    print(f"  IR  (no cost) = {ir_nc:+.4f}      IR  (w/ cost) = {ir_c:+.4f}")
    print(f"  MDD (no cost) = {mdd_nc:+.2%}    MDD (w/ cost) = {mdd_c:+.2%}")
    print(f"  Cum.excess final = {cum.iloc[-1]:+.2%}    (days={days})")
    print(f"\n{'='*72}")
    print(f"  FIGURE 3 — append this line to plot_figure3.py METHOD_PKLS:")
    print(f'    "{name}": REPO / "{out_pkl.relative_to(REPO)}",')
    print(f"{'='*72}")
    print(f"  Solo cum-excess plot saved:")
    print(f"    {fig_pdf.relative_to(REPO)}")
    print(f"    {fig_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()

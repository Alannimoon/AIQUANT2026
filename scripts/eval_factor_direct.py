"""Evaluate a factor expression DIRECTLY as the predictive signal.

AlphaAgent's `factor_backtest` pipeline feeds the factor expression as an
input feature to a LightGBM model and reports the *model's* IC — which
washes out differences between factor expressions (we observed two very
different factors both yielding IC=0.01656 via that path).

This script bypasses the model: it just evaluates the factor expression
to a (date, instrument) signal and computes the cross-sectional IC vs.
the 1-day forward return label. That matches how RSI is treated in
Figure 4 — a "factor source" line, not a "model on top of factor" line.

Usage:
  python scripts/eval_factor_direct.py "TS_ZSCORE(-TS_SUM(\$return, 5), 20) * RANK(\$volume)"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "AlphaAgent"))

# AlphaAgent's TS_*, RANK, ZSCORE, etc. live here.
from alphaagent.components.coder.factor_coder.function_lib import *  # noqa: F401,F403
from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol

# Re-use any of the daily_pv.h5 already produced by a previous mine —
# they all share the same OHLCV/return columns across CSI500.
_DEFAULT_PV = REPO / "AlphaAgent/git_ignore_folder/RD-Agent_workspace/a37761649d9c4f03b90300f349778dff/daily_pv.h5"


def eval_factor(expr_str: str, daily_pv_path: Path = _DEFAULT_PV,
                upper_instrument: bool = True) -> pd.Series:
    """Compute factor signal for every (datetime, instrument).

    Mirrors what `factor.py` in each workspace does: parse the expression,
    swap `$col` -> `df['$col']`, and `eval()` it in scope of function_lib.

    `upper_instrument`: daily_pv.h5 stores lowercase tickers; default True
    uppercases them to match Qlib's CSI500 universe. Set False when joining
    against labels pulled with lowercase instruments (SYS's full-market path).
    """
    df = pd.read_hdf(daily_pv_path, key="data")
    expr = parse_symbol(expr_str, df.columns)
    expr = parse_expression(expr)
    # `$close` -> `df['$close']`, etc. (function_lib operators stay as-is)
    for col in df.columns:
        expr = expr.replace(col[1:], f"df['{col}']")
    result = eval(expr)
    if isinstance(result, pd.DataFrame):
        # single-column DataFrame from groupby outputs — collapse.
        result = result.iloc[:, 0]
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result.index = result.index.set_names(["datetime", "instrument"])
    if upper_instrument:
        new_idx = pd.MultiIndex.from_arrays(
            [result.index.get_level_values(0),
             result.index.get_level_values(1).str.upper()],
            names=["datetime", "instrument"],
        )
        result.index = new_idx
    return result.astype(np.float64)


def per_day_ic(signal: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.concat([signal.rename("pred"), label.rename("label")], axis=1).dropna()
    out = []
    for date, sub in df.groupby(level=0):
        if len(sub) < 5:
            continue
        ic = sub["pred"].corr(sub["label"], method="pearson")
        ric = sub["pred"].corr(sub["label"], method="spearman")
        out.append((date, ic, ric))
    return pd.DataFrame(out, columns=["datetime", "IC", "RankIC"]).set_index("datetime")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    expr = sys.argv[1]

    # Scope overrides — match SYS's archive scoring with `all` + full window,
    # or our paper scope with `csi500` + 2021-06~2026-05.
    universe = os.environ.get("EVAL_UNIVERSE", "csi500").lower()  # "csi500" | "all"
    start = os.environ.get("EVAL_START", "2021-06-01")
    end = os.environ.get("EVAL_END", "2026-05-31")

    print(f"factor: {expr}")
    print(f"scope: universe={universe}, window=[{start}, {end}]")
    # For universe="all", keep lowercase tickers (daily_pv.h5 format) so the
    # join with Qlib label (also lowercase when fed lowercase instruments)
    # works without forcing an upper/lower mismatch.
    signal = eval_factor(expr, upper_instrument=(universe == "csi500"))
    print(f"signal shape: {signal.shape}, dates: {signal.index.get_level_values(0).nunique()}, "
          f"instruments: {signal.index.get_level_values(1).nunique()}")

    import qlib
    from qlib.data import D
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    if universe == "csi500":
        instruments = D.instruments("csi500")
    elif universe == "all":
        # Use the daily_pv.h5's own universe (3237 instruments incl. indices),
        # which is what SYS's `calculate_factor_level_ic` does.
        instruments = sorted({str(i) for i in signal.index.get_level_values(1)})
    else:
        raise ValueError(f"EVAL_UNIVERSE must be 'csi500' or 'all', got {universe!r}")
    raw = D.features(
        instruments, ["Ref($close, -2)/Ref($close, -1) - 1"],
        start_time=start, end_time=end, freq="day",
    )
    raw.columns = ["label"]
    label = raw["label"].swaplevel(0, 1).sort_index()
    label.index = label.index.set_names(["datetime", "instrument"])

    daily = per_day_ic(signal, label)
    daily.index = pd.to_datetime(daily.index)
    yearly = daily.groupby(daily.index.year)[["IC", "RankIC"]].mean()

    print("\n=== Yearly IC table ===")
    print(yearly.round(4).to_string())
    print(f"\n=== Overall: IC={daily['IC'].mean():.4f}, RankIC={daily['RankIC'].mean():.4f}, "
          f"ICIR={daily['IC'].mean()/(daily['IC'].std()+1e-12):.4f} ===")


if __name__ == "__main__":
    main()

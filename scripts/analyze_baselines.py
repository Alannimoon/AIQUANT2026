"""Per-direction summary of a baseline phase (A or C).

Reads ~/baseline_metrics/<phase>_*.csv (the qlib backtest result tables that
run_baselines.sh copies out of each workspace) and the matching
run_logs/baseline_<phase>_NN_<slug>.log files for factor expressions.

Outputs:
  1. Per-direction IC / RankIC / AR / IR / MDD summary (mean / max over the
     backtests in that direction).
  2. Per-direction depth histogram and (category x depth) breakdown from
     factor expressions extracted from the mine log.
  3. Top-5 backtests by IC, with their direction and short metric snapshot.

Usage:
  python scripts/analyze_baselines.py A
  python scripts/analyze_baselines.py C
  python scripts/analyze_baselines.py            # defaults to A
"""
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "AlphaAgent"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the canonical parser + classifier we already maintain.
from ast_depth import classify, extract_expressions_from_log, factor_depth  # noqa: E402

# Direction display order (matches scripts/run_baselines.sh).
ORDER = ["momentum", "reversal", "volatility", "volume_price", "cross_section"]


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------
def load_metrics(metrics_dir: Path, phase: str) -> pd.DataFrame:
    """Each wsid is copied once per direction's cleanup pass, so the same
    backtest can appear with several slug prefixes. Take the earliest mtime
    occurrence — that's the direction whose mine actually produced it."""
    by_wsid: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    pattern = str(metrics_dir / f"{phase}_*.csv")
    for f in glob.glob(pattern):
        stem = Path(f).stem  # e.g. "A_momentum_04ac0157"
        slug_part, _, wsid = stem.rpartition("_")
        if len(wsid) != 8:
            continue
        slug = slug_part[len(phase) + 1:]  # strip "A_" / "C_"
        by_wsid[wsid].append((os.path.getmtime(f), slug, f))

    rows = []
    for wsid, recs in by_wsid.items():
        recs.sort()
        _, slug, fpath = recs[0]
        s = pd.read_csv(fpath, index_col=0).iloc[:, 0]
        rows.append({
            "direction":  slug,
            "wsid":       wsid,
            "IC":         s.get("IC", float("nan")),
            "ICIR":       s.get("ICIR", float("nan")),
            "RankIC":     s.get("Rank IC", float("nan")),
            "RankICIR":   s.get("Rank ICIR", float("nan")),
            "AR_cost":    s.get("1day.excess_return_with_cost.annualized_return", float("nan")),
            "IR_noCost":  s.get("1day.excess_return_without_cost.information_ratio", float("nan")),
            "MDD_cost":   s.get("1day.excess_return_with_cost.max_drawdown", float("nan")),
        })
    return pd.DataFrame(rows)


def metric_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("(no metrics found)")
        return
    df = df.copy()
    df["_ord"] = df["direction"].apply(
        lambda d: ORDER.index(d) if d in ORDER else 999
    )
    grouped = df.groupby("direction").agg(
        n=("IC", "count"),
        IC_mean=("IC", "mean"),
        IC_max=("IC", "max"),
        RankIC_mean=("RankIC", "mean"),
        RankIC_max=("RankIC", "max"),
        AR_mean=("AR_cost", "mean"),
        AR_max=("AR_cost", "max"),
        IR_mean=("IR_noCost", "mean"),
        IR_max=("IR_noCost", "max"),
        MDD_worst=("MDD_cost", "min"),
    ).round(4)
    # Restore ORDER ordering.
    grouped["_ord"] = grouped.index.map(
        lambda d: ORDER.index(d) if d in ORDER else 999
    )
    grouped = grouped.sort_values("_ord").drop("_ord", axis=1)
    print(grouped.to_string())


# ---------------------------------------------------------------------------
# 2. Depth + category from mine logs
# ---------------------------------------------------------------------------
def analyze_logs(logs_dir: Path, phase: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pattern = str(logs_dir / f"baseline_{phase}_*_*.log")
    for log in sorted(glob.glob(pattern)):
        name = Path(log).stem  # e.g. "baseline_A_00_momentum"
        parts = name.split("_")
        if len(parts) < 4:
            continue
        direction = "_".join(parts[3:])
        exprs = extract_expressions_from_log(Path(log))
        depths = []
        categories: Counter[str] = Counter()
        depth_by_cat: dict[str, list[int]] = defaultdict(list)
        for e in exprs:
            try:
                d = factor_depth(e)
                c = classify(e)
                depths.append(d)
                categories[c] += 1
                depth_by_cat[c].append(d)
            except Exception:
                continue
        out[direction] = {
            "n_expressions": len(exprs),
            "depth_min": min(depths) if depths else None,
            "depth_median": sorted(depths)[len(depths) // 2] if depths else None,
            "depth_max": max(depths) if depths else None,
            "depth_hist": dict(Counter(depths).most_common()),
            "categories": dict(categories.most_common()),
        }
    return out


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------
def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "A"
    if phase not in {"A", "C"}:
        print(f"Phase must be 'A' or 'C', got {phase!r}", file=sys.stderr)
        return 2

    home = Path.home()
    metrics_dir = home / "baseline_metrics"
    logs_dir = home / "AIQUANT2026" / "AlphaAgent" / "run_logs"

    print(f"\n############ Phase {phase} ############\n")

    print("=== 1. Backtest metrics per direction ===")
    df = load_metrics(metrics_dir, phase)
    print(f"({len(df)} unique backtests)\n")
    metric_summary(df)

    print("\n=== 2. Factor expression analysis per direction ===")
    summary = analyze_logs(logs_dir, phase)
    for direction in ORDER:
        if direction not in summary:
            continue
        s = summary[direction]
        print(f"\n{direction}:")
        print(f"  expressions: {s['n_expressions']}")
        if s["depth_min"] is not None:
            print(f"  depth        : min={s['depth_min']}, median={s['depth_median']}, max={s['depth_max']}")
            print(f"  depth hist   : {s['depth_hist']}")
        print(f"  categories   : {s['categories']}")

    print("\n=== 3. Top 5 backtests by IC ===")
    if not df.empty:
        top = df.sort_values("IC", ascending=False).head(5)
        print(top[["direction", "wsid", "IC", "RankIC", "AR_cost",
                   "IR_noCost", "MDD_cost"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

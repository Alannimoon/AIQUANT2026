"""AST depth calculator + factor classifier for manual MAP-Elites analysis.

Built on top of AlphaAgent's pyparsing-based factor parser
(alphaagent.components.coder.factor_coder.factor_ast).

Depth convention (matches the ELITEALPHA proposal §3.2 complexity axis):
    leaf (variable / number)          -> depth 1
    function call                     -> 1 + max(depth(arg) for arg in args)
    binary op                         -> 1 + max(depth(left), depth(right))
    conditional (A?B:C)               -> 1 + max(depth(condition), depth(true), depth(false))

CLI usage:
    python scripts/ast_depth.py 'ZSCORE($volume / (TS_MEAN($volume, 20) + 1e-8))'
    python scripts/ast_depth.py --from-log AlphaAgent/run_logs/run_NNN.log
    python scripts/ast_depth.py --scan-workspaces AlphaAgent/git_ignore_folder/RD-Agent_workspace
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the AlphaAgent package importable when this script is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "AlphaAgent"))

from alphaagent.components.coder.factor_coder.factor_ast import (  # noqa: E402
    count_depth as factor_depth,
)


# ---------------------------------------------------------------------------
# Heuristic category classifier
# ---------------------------------------------------------------------------
# Maps each ELITEALPHA category to keyword signatures we look for in the
# expression text. Order matters: first match wins. This is a coarse stand-in
# for the LLM-based classifier the paper proposes; good enough to bucket the
# manual MAP-Elites runs we'll do.
# Order matters: more specific signatures come first.
_CATEGORY_RULES = [
    # volume_price wins whenever any volume/amount/vwap field is used.
    ("volume_price", [r"\$volume", r"\$amount", r"\$vwap"]),

    # volatility: explicit dispersion measures over returns/prices.
    ("volatility", [r"TS_STD\(\$return",
                    r"TS_VAR\(\$return",
                    r"TS_STD\(\$close",
                    r"TS_MAD",
                    r"BB_UPPER", r"BB_LOWER"]),

    # reversal: negated returns or distance-from-recent-high gating.
    ("reversal", [r"-\s*1\s*\*\s*TS_SUM\(\$return",
                  r"\(\s*-\s*TS_SUM\(\$return",
                  r"TS_SUM\(\$return[^)]*\)\s*<\s*0",
                  r"\$close\s*<\s*0\.\d+\s*\*\s*TS_MAX",
                  r"DELAY\(\$close,\s*\d+\)\s*-\s*\$open"]),

    # cross_section: RANK/ZSCORE as the outermost wrap of the factor.
    ("cross_section", [r"^\s*\(*\s*RANK\(",
                       r"^\s*\(*\s*ZSCORE\(",
                       r"^\s*\(*\s*PERCENTILE\(",
                       r"REGBETA", r"REGRESI"]),

    # momentum: any return-based aggregation that the rules above didn't claim.
    ("momentum", [r"TS_SUM\(\$return",
                  r"TS_MEAN\(\$return",
                  r"WMA\(\$return",
                  r"DECAYLINEAR\(\$return",
                  r"EMA\(\$return",
                  r"\$close\s*/\s*DELAY\(\$close",
                  r"DELTA\(\$close",
                  r"RSI", r"MACD"]),
]


def classify(expr: str) -> str:
    """Cheap regex-based category guess. Falls back to 'other'."""
    for name, patterns in _CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, expr):
                return name
    return "other"


# ---------------------------------------------------------------------------
# Log mining
# ---------------------------------------------------------------------------
_EXPR_LINE = re.compile(r"factor_expression:\s+(.+?)\s*$")
# Reject capture groups that look like log preludes (timestamp + log-level)
# or that don't contain at least one factor-shaped token ($var or func name(...).
_LOOKS_LIKE_LOGLINE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}")
_LOOKS_LIKE_EXPR = re.compile(r"[\$A-Z_][\w$]*\s*\(|\$[A-Za-z_]+")


def extract_expressions_from_log(log_path: Path) -> list[str]:
    """Pull every `factor_expression: ...` line out of a mine log, filtering
    captures that are clearly log noise (timestamps, summary lines) rather
    than real factor expressions."""
    out: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _EXPR_LINE.search(line)
            if not m:
                continue
            expr = m.group(1).strip()
            if _LOOKS_LIKE_LOGLINE.match(expr):
                continue
            if not _LOOKS_LIKE_EXPR.search(expr):
                continue
            out.append(expr)
    return out


def scan_workspaces(workspace_root: Path) -> list[tuple[str, str | None, dict]]:
    """For each RD-Agent workspace dir, return (wsid, factor_py_text, metrics).

    Each workspace contains the rendered factor.py and qlib_res.csv. We extract
    the actual expression by grepping the expression literal embedded in factor.py.
    """
    rows: list[tuple[str, str | None, dict]] = []
    for ws in sorted(workspace_root.glob("*"), key=lambda p: p.stat().st_mtime):
        if not ws.is_dir():
            continue
        factor_py = ws / "factor.py"
        qlib_csv = ws / "qlib_res.csv"
        expr = None
        if factor_py.exists():
            m = re.search(r'expression\s*=\s*"([^"]+)"', factor_py.read_text())
            if m:
                expr = m.group(1)
        metrics = {}
        if qlib_csv.exists():
            for line in qlib_csv.read_text().splitlines()[1:]:
                if "," in line:
                    k, _, v = line.partition(",")
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass
        rows.append((ws.name[:8], expr, metrics))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_distribution(items: list[tuple[str, int, str]]) -> None:
    """Print a category x depth occupancy table + per-cell expressions."""
    grid: dict[tuple[str, int], list[str]] = defaultdict(list)
    for expr, d, cat in items:
        grid[(cat, d)].append(expr)

    categories = ["momentum", "reversal", "volatility",
                  "volume_price", "cross_section", "other"]
    depths = sorted({d for _, d, _ in items}) or [1]

    # Header
    header = "category".ljust(15) + "".join(f"  d={d:<3}" for d in depths)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = cat.ljust(15)
        for d in depths:
            row += f"  {len(grid.get((cat, d), [])):<5}"
        print(row)

    print("\n=== Sample expression per non-empty cell ===")
    for cat in categories:
        for d in depths:
            cell = grid.get((cat, d), [])
            if cell:
                print(f"[{cat}, d={d}]  ({len(cell)} factors)")
                print(f"    {cell[0]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("expression", nargs="?",
                   help="A single factor expression to analyze.")
    p.add_argument("--from-log", type=Path,
                   help="Mine log; extracts every `factor_expression:` line.")
    p.add_argument("--scan-workspaces", type=Path,
                   help="Path to RD-Agent_workspace; reads each factor.py.")
    args = p.parse_args(argv)

    if args.expression:
        d = factor_depth(args.expression)
        cat = classify(args.expression)
        print(f"depth    : {d}")
        print(f"category : {cat}")
        return 0

    items: list[tuple[str, int, str]] = []
    seen: set[str] = set()

    if args.from_log:
        for expr in extract_expressions_from_log(args.from_log):
            if expr in seen:
                continue
            seen.add(expr)
            try:
                items.append((expr, factor_depth(expr), classify(expr)))
            except Exception as e:
                print(f"!! parse failed for: {expr[:80]}... -> {e}", file=sys.stderr)

    if args.scan_workspaces:
        for wsid, expr, _metrics in scan_workspaces(args.scan_workspaces):
            if not expr or expr in seen:
                continue
            seen.add(expr)
            try:
                items.append((expr, factor_depth(expr), classify(expr)))
            except Exception as e:
                print(f"!! ws {wsid} parse failed: {e}", file=sys.stderr)

    if not items:
        p.error("Pass an expression, --from-log, or --scan-workspaces.")

    print(f"Analyzed {len(items)} unique expressions.\n")
    report_distribution(items)
    print(f"\nDepth histogram: "
          f"{dict(Counter(d for _, d, _ in items).most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

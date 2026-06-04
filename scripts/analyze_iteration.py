"""Per-loop evolution analyzer for a baseline mine log.

For each baseline_<phase>_NN_<slug>.log file, walks the log linearly and
attributes every `factor_expression:` line to the most-recently-seen
`loop_index=N` marker. Reports, for each loop:

  - n_total          : how many `factor_expression:` lines occurred in the loop
  - n_unique         : deduped within the loop
  - n_novel          : not seen in any earlier loop in the same mine
  - novelty_ratio    : n_novel / n_unique (1.0 = fully fresh, 0.0 = pure repeat)
  - depth_median     : median AST depth in the loop
  - category_top1    : the dominant ELITEALPHA category

The output is meant to answer one question: when the Idea/Factor agents loop
on the same hypothesis with `trace.hist` feedback, are they exploring the
expression space or merely shuffling syntax around the same core formula?

Usage:
  python scripts/analyze_iteration.py [phase=A] [logs_dir=AlphaAgent/run_logs]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "AlphaAgent"))
sys.path.insert(0, str(REPO / "scripts"))

from ast_depth import classify, factor_depth  # noqa: E402

ORDER = ["momentum", "reversal", "volatility", "volume_price", "cross_section"]

_LOOP_MARKER = re.compile(r"loop_index=(\d+)")
_EXPR_LINE = re.compile(r"factor_expression:\s+(.+?)\s*$")
_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}")
_LOOKS_LIKE_EXPR = re.compile(r"[\$A-Z_][\w$]*\s*\(|\$[A-Za-z_]+")


def per_loop(log_path: Path) -> dict[int, list[str]]:
    """Group factor expressions by the loop_index they appeared under."""
    loop_exprs: dict[int, list[str]] = defaultdict(list)
    cur = -1
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lm = _LOOP_MARKER.search(line)
            if lm:
                cur = int(lm.group(1))
            em = _EXPR_LINE.search(line)
            if em and cur >= 0:
                expr = em.group(1).strip()
                if _TIMESTAMP_PREFIX.match(expr):
                    continue
                if not _LOOKS_LIKE_EXPR.search(expr):
                    continue
                loop_exprs[cur].append(expr)
    return loop_exprs


def median(xs: list[int]) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def analyze_one(log_path: Path, dump: bool = False, max_len: int = 110) -> None:
    loops = per_loop(log_path)
    if not loops:
        print(f"  (no factor_expression lines found)")
        return

    seen: set[str] = set()
    print(
        f"  {'loop':<5} "
        f"{'total':>6} {'unique':>7} {'novel':>6} {'novel%':>7} "
        f"{'d_med':>6} {'d_max':>6}  cat_top1"
    )
    loop_unique: dict[int, list[str]] = {}
    for loop_idx in sorted(loops):
        exprs = loops[loop_idx]
        unique = list(dict.fromkeys(exprs))  # preserve insertion order, dedupe
        unique_set = set(unique)
        novel = unique_set - seen
        depths = []
        cats: Counter[str] = Counter()
        for e in unique:
            try:
                depths.append(factor_depth(e))
            except Exception:
                pass
            cats[classify(e)] += 1
        seen |= unique_set

        novel_ratio = (len(novel) / len(unique)) if unique else 0.0
        top_cat = cats.most_common(1)[0][0] if cats else "-"
        d_med = median(depths)
        d_max = max(depths) if depths else None
        print(
            f"  {loop_idx:<5} "
            f"{len(exprs):>6} {len(unique):>7} {len(novel):>6} "
            f"{novel_ratio*100:>6.1f}% "
            f"{d_med if d_med is not None else '-':>6} "
            f"{d_max if d_max is not None else '-':>6}  {top_cat}"
        )
        loop_unique[loop_idx] = unique

    if dump:
        print()
        for loop_idx in sorted(loop_unique):
            print(f"  --- loop {loop_idx} unique expressions ---")
            for e in loop_unique[loop_idx]:
                try:
                    d = factor_depth(e)
                except Exception:
                    d = "?"
                c = classify(e)
                shown = e if len(e) <= max_len else e[:max_len] + "..."
                print(f"    [d={d}, {c}] {shown}")
            print()


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", nargs="?", default="A",
                   help="Phase tag (A or C). Default: A.")
    p.add_argument("logs_dir", nargs="?", default=None,
                   help="run_logs/ dir. Default: ~/AIQUANT2026/AlphaAgent/run_logs.")
    p.add_argument("--dump", "-v", action="store_true",
                   help="List every unique expression per loop, tagged with "
                        "[depth, category].")
    p.add_argument("--only", default=None,
                   help="Only analyze this direction slug (e.g. momentum).")
    p.add_argument("--max-len", type=int, default=110,
                   help="Truncate expressions longer than N chars (default 110).")
    args = p.parse_args()

    logs_dir = (
        Path(args.logs_dir) if args.logs_dir
        else Path.home() / "AIQUANT2026" / "AlphaAgent" / "run_logs"
    )
    pattern = f"baseline_{args.phase}_*_*.log"
    log_files = sorted(logs_dir.glob(pattern))
    if args.only:
        log_files = [f for f in log_files if f.stem.endswith(args.only)]
    if not log_files:
        print(f"No logs matching {pattern} in {logs_dir}", file=sys.stderr)
        return 1

    # Sort by ELITEALPHA category order for readability.
    def _order(p: Path) -> tuple[int, str]:
        name = p.stem
        for i, slug in enumerate(ORDER):
            if name.endswith(slug):
                return (i, name)
        return (999, name)

    log_files.sort(key=_order)

    print(f"\n############ Phase {args.phase}: per-loop evolution ############\n")
    for log in log_files:
        print(f"=== {log.stem} ===")
        analyze_one(log, dump=args.dump, max_len=args.max_len)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

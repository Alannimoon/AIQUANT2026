#!/usr/bin/env bash
set -euo pipefail

# Extract the accepted EliteAlpha archive factors from one round in
# elite_archive_progress.txt, run AlphaAgent's standard multi-factor backtest,
# and copy the generated Qlib feature matrix pkl to a requested path.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

ARCHIVE_LOG=""
ROUND=""
OUTPUT_PKL=""
FACTOR_CSV=""
BACKTEST_LOG_DIR=""
REPORT_PATH="${LIGHTGBM_REPORT_PATH:-lightgbm_report.txt}"
PAPER_PKL_PATH="${PAPER_PKL_PATH:-}"
FIGURE3_SCRIPT="${FIGURE3_SCRIPT:-../scripts/plot_figure3.py}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-git_ignore_folder/RD-Agent_workspace}"
STEP_N="${STEP_N:-4}"
FACTOR_TOP_K="${FACTOR_TOP_K:-0}"
FACTOR_PER_CATEGORY_L="${FACTOR_PER_CATEGORY_L:-0}"
FACTOR_CORR_DEDUPE_THRESHOLD="${FACTOR_CORR_DEDUPE_THRESHOLD:-0}"
QLIB_CONFIG_NAME="${QLIB_CONFIG_NAME:-}"
RANK_LABEL_CONFIG_NAME="${RANK_LABEL_CONFIG_NAME:-conf_cn_combined_elite_rank_label.yaml}"
ARCHIVE_BACKTEST_WARMUP_DAYS="${ARCHIVE_BACKTEST_WARMUP_DAYS:-180}"
KEEP_CSV=0
DRY_RUN=0
USE_LIGHT=0
USE_RANK_LABEL=0
FLIP_ALL=0
GENERATE_REPORT=1
UPDATE_FIGURE3=1

# Factor ranking formula. Higher score is selected first:
#   score = validation_Rank_IC
#
# If validation_Rank_IC is unavailable, fall back to:
#   score = QUALITY_WEIGHT * quality - DEPTH_REG_COEF * (depth ** DEPTH_REG_POWER)
#
# To discourage complex factors, increase DEPTH_REG_COEF. Set it to 0.0
# to rank fallback records by quality only.
QUALITY_WEIGHT="${QUALITY_WEIGHT:-1.0}"
DEPTH_REG_COEF="${DEPTH_REG_COEF:-0.0}"
DEPTH_REG_POWER="${DEPTH_REG_POWER:-1.0}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python python3 python.exe py.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: no Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./backtest_elite_archive.sh \
    --archive-log log/.../elite_archive_progress.txt \
    --round 12 \
    --top-k 20 \
    --output-pkl git_ignore_folder/elite_round12_combined_factors_df.pkl

Options:
  --archive-log PATH   Path to elite_archive_progress.txt.
  --round N           Round number to extract from the archive progress file.
  --top-k N           Select only the top N factors by score. Default 0 means all.
  --per-category-l N  Select at most N factors from each EliteAlpha category after scoring.
                      Default 0 means no per-category limit.
  --corr-dedupe-threshold X
                      Drop highly correlated factors before --top-k. When abs(corr) > X,
                      each correlated group keeps only the factor with highest validation Rank IC.
                      Default 0 disables this filter. A typical value is 0.9.
  --output-pkl PATH   Where to copy the generated combined_factors_df.pkl.
  --factor-csv PATH   Optional path for the temporary factor CSV.
  --log-dir PATH      Optional AlphaAgent log trace dir for this backtest.
  --report-path PATH  Write a LightGBM report after backtest. Default: lightgbm_report.txt.
  --paper-pkl PATH    Destination report pkl for Figure 3. Default:
                      ../baselines/direct_factor_backtests/EliteAlpha_round${ROUND}_LGBM_report_normal_1day.pkl
  --step-n N          Workflow steps to run; default 4 reaches factor_backtest.
  --qlib-config NAME  Optional qlib config template to use, e.g. conf_cn_combined_elite_only.yaml.
  --light             Use QLIB_FACTOR_USE_LIGHTWEIGHT_QLIB_TEST=true.
  --rank-label        Use the rank-normalized-label LightGBM template.
  --flip-all          Export every selected expression as -1 * (expr). Useful for old archives whose quality used sign-flip.
  --no-report         Do not write the LightGBM report.
  --no-plot           Do not copy ret.pkl or run scripts/plot_figure3.py.
  --keep-csv          Keep the generated factor CSV when --factor-csv is omitted.
  --dry-run           Only extract factors and write the CSV; do not backtest.
  -h, --help          Show this help.

Environment:
  ALPHAAGENT_BIN      AlphaAgent command to run, default: alphaagent
  PYTHON_BIN          Python command used by this script
  FACTOR_TOP_K        Default --top-k value, default: 0
  FACTOR_PER_CATEGORY_L
                      Default --per-category-l value, default: 0
  FACTOR_CORR_DEDUPE_THRESHOLD
                      Default --corr-dedupe-threshold value, default: 0
  QUALITY_WEIGHT      Ranking coefficient for quality, default: 1.0
  DEPTH_REG_COEF      Ranking penalty coefficient for depth, default: 0.005
  DEPTH_REG_POWER     Depth penalty exponent, default: 1.0
  WORKSPACE_ROOT      Workspace search root, default: git_ignore_folder/RD-Agent_workspace
  STEP_N              Default --step-n value, default: 4
  QLIB_CONFIG_NAME    Optional qlib config template override
  RANK_LABEL_CONFIG_NAME
                      Config used by --rank-label, default: conf_cn_combined_elite_rank_label.yaml
  LIGHTGBM_REPORT_PATH
                      Default --report-path value, default: lightgbm_report.txt
  PAPER_PKL_PATH      Default --paper-pkl value
  FIGURE3_SCRIPT      Plot script path, default: ../scripts/plot_figure3.py
  ARCHIVE_BACKTEST_WARMUP_DAYS
                      Extra calendar days before the qlib start date used only while computing factors, default: 180
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-log)
      ARCHIVE_LOG="${2:-}"
      shift 2
      ;;
    --round)
      ROUND="${2:-}"
      shift 2
      ;;
    --top-k)
      FACTOR_TOP_K="${2:-}"
      shift 2
      ;;
    --per-category-l)
      FACTOR_PER_CATEGORY_L="${2:-}"
      shift 2
      ;;
    --corr-dedupe-threshold)
      FACTOR_CORR_DEDUPE_THRESHOLD="${2:-}"
      shift 2
      ;;
    --output-pkl)
      OUTPUT_PKL="${2:-}"
      shift 2
      ;;
    --factor-csv)
      FACTOR_CSV="${2:-}"
      KEEP_CSV=1
      shift 2
      ;;
    --log-dir)
      BACKTEST_LOG_DIR="${2:-}"
      shift 2
      ;;
    --report-path)
      REPORT_PATH="${2:-}"
      shift 2
      ;;
    --paper-pkl)
      PAPER_PKL_PATH="${2:-}"
      shift 2
      ;;
    --step-n)
      STEP_N="${2:-}"
      shift 2
      ;;
    --qlib-config)
      QLIB_CONFIG_NAME="${2:-}"
      shift 2
      ;;
    --light)
      USE_LIGHT=1
      shift
      ;;
    --rank-label)
      USE_RANK_LABEL=1
      if [[ -z "${QLIB_CONFIG_NAME}" ]]; then
        QLIB_CONFIG_NAME="${RANK_LABEL_CONFIG_NAME}"
      fi
      shift
      ;;
    --flip-all)
      FLIP_ALL=1
      shift
      ;;
    --no-report)
      GENERATE_REPORT=0
      shift
      ;;
    --no-plot)
      UPDATE_FIGURE3=0
      shift
      ;;
    --keep-csv)
      KEEP_CSV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      KEEP_CSV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${ARCHIVE_LOG}" || -z "${ROUND}" ]]; then
  echo "ERROR: --archive-log and --round are required." >&2
  usage >&2
  exit 2
fi

if [[ "${DRY_RUN}" -eq 0 && -z "${OUTPUT_PKL}" ]]; then
  echo "ERROR: --output-pkl is required unless --dry-run is used." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "${ARCHIVE_LOG}" ]]; then
  echo "ERROR: archive log not found: ${ARCHIVE_LOG}" >&2
  exit 1
fi

case "${ROUND}" in
  ''|*[!0-9]*)
    echo "ERROR: --round must be a positive integer, got: ${ROUND}" >&2
    exit 2
    ;;
esac

case "${FACTOR_TOP_K}" in
  ''|*[!0-9]*)
    echo "ERROR: --top-k must be a non-negative integer, got: ${FACTOR_TOP_K}" >&2
    exit 2
    ;;
esac

case "${FACTOR_PER_CATEGORY_L}" in
  ''|*[!0-9]*)
    echo "ERROR: --per-category-l must be a non-negative integer, got: ${FACTOR_PER_CATEGORY_L}" >&2
    exit 2
    ;;
esac

if ! FACTOR_CORR_DEDUPE_THRESHOLD="${FACTOR_CORR_DEDUPE_THRESHOLD}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    value = float(os.environ["FACTOR_CORR_DEDUPE_THRESHOLD"])
except ValueError:
    sys.exit(1)
if value < 0 or value >= 1:
    sys.exit(1)
PY
then
  echo "ERROR: --corr-dedupe-threshold must be a float in [0, 1), got: ${FACTOR_CORR_DEDUPE_THRESHOLD}" >&2
  exit 2
fi

RUN_STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
TEMP_ROOT="git_ignore_folder/elite_archive_round_${ROUND}_${RUN_STAMP}_$$"
mkdir -p "${TEMP_ROOT}"

if [[ -z "${FACTOR_CSV}" ]]; then
  FACTOR_CSV="${TEMP_ROOT}/elite_round_${ROUND}_factors.csv"
fi
mkdir -p "$(dirname "${FACTOR_CSV}")"

cleanup() {
  if [[ "${KEEP_CSV}" -eq 0 ]]; then
    rm -rf "${TEMP_ROOT}"
  fi
}
trap cleanup EXIT

echo "Extracting EliteAlpha archive factors:"
echo "  archive_log=${ARCHIVE_LOG}"
echo "  round=${ROUND}"
echo "  factor_csv=${FACTOR_CSV}"
echo "  top_k=${FACTOR_TOP_K} (0 means all)"
echo "  per_category_l=${FACTOR_PER_CATEGORY_L} (0 means no category limit)"
echo "  corr_dedupe_threshold=${FACTOR_CORR_DEDUPE_THRESHOLD} (0 means disabled)"
echo "  score=validation_Rank_IC, fallback=(${QUALITY_WEIGHT})*quality - (${DEPTH_REG_COEF})*(depth**${DEPTH_REG_POWER})"
echo "  flip_all=${FLIP_ALL}"
echo "  rank_label=${USE_RANK_LABEL}"
echo "  qlib_config=${QLIB_CONFIG_NAME:-<runner default>}"
echo "  archive_backtest_warmup_days=${ARCHIVE_BACKTEST_WARMUP_DAYS}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ "${GENERATE_REPORT}" -eq 1 ]]; then
    echo "  lightgbm_report=${REPORT_PATH}"
  else
    echo "  lightgbm_report=<disabled>"
  fi
  echo "  figure3_update=${UPDATE_FIGURE3}"
fi

ARCHIVE_LOG="${ARCHIVE_LOG}" ROUND="${ROUND}" FACTOR_CSV="${FACTOR_CSV}" \
FACTOR_TOP_K="${FACTOR_TOP_K}" FACTOR_PER_CATEGORY_L="${FACTOR_PER_CATEGORY_L}" \
FACTOR_CORR_DEDUPE_THRESHOLD="${FACTOR_CORR_DEDUPE_THRESHOLD}" \
QUALITY_WEIGHT="${QUALITY_WEIGHT}" \
DEPTH_REG_COEF="${DEPTH_REG_COEF}" DEPTH_REG_POWER="${DEPTH_REG_POWER}" \
FLIP_ALL="${FLIP_ALL}" QLIB_CONFIG_NAME="${QLIB_CONFIG_NAME}" \
RANK_LABEL_CONFIG_NAME="${RANK_LABEL_CONFIG_NAME}" \
ARCHIVE_BACKTEST_WARMUP_DAYS="${ARCHIVE_BACKTEST_WARMUP_DAYS}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import hashlib
import math
import os
import pickle
import re
import sys
from pathlib import Path

archive_log = Path(os.environ["ARCHIVE_LOG"])
round_id = int(os.environ["ROUND"])
factor_csv = Path(os.environ["FACTOR_CSV"])
top_k = int(os.environ["FACTOR_TOP_K"])
per_category_l = int(os.environ["FACTOR_PER_CATEGORY_L"])
corr_dedupe_threshold = float(os.environ["FACTOR_CORR_DEDUPE_THRESHOLD"])
quality_weight = float(os.environ["QUALITY_WEIGHT"])
depth_reg_coef = float(os.environ["DEPTH_REG_COEF"])
depth_reg_power = float(os.environ["DEPTH_REG_POWER"])
flip_all = os.environ.get("FLIP_ALL", "0") == "1"
qlib_config_name = (os.environ.get("QLIB_CONFIG_NAME") or os.environ.get("RANK_LABEL_CONFIG_NAME") or "").strip()
warmup_days = max(int(os.environ.get("ARCHIVE_BACKTEST_WARMUP_DAYS", "180")), 0)

text = archive_log.read_text(encoding="utf-8", errors="replace")
block_pattern = re.compile(
    r"(?ms)^={20,}\s*\n"
    r"round=(?P<round>\d+)\b(?P<body>.*?)(?=^={20,}\s*\nround=|\Z)"
)

matches = [m for m in block_pattern.finditer(text) if int(m.group("round")) == round_id]
if not matches:
    print(f"ERROR: round={round_id} not found in {archive_log}", file=sys.stderr)
    sys.exit(1)

block = matches[-1].group(0)
records_match = re.search(r"^records:\s*(\d+)\s*$", block, flags=re.MULTILINE)
expected_records = int(records_match.group(1)) if records_match else None

rows = []
seen_names = set()

def clean_name(name: str, idx: str) -> str:
    name = name.strip()
    sanitized = re.sub(r"\W+", "_", name)
    sanitized = sanitized.strip("_")
    if not sanitized or re.match(r"^\d", sanitized):
        sanitized = f"elite_factor_{idx}"
    candidate = sanitized
    counter = 2
    while candidate in seen_names:
        candidate = f"{sanitized}_{counter}"
        counter += 1
    seen_names.add(candidate)
    return candidate

def normalize_category(category) -> str:
    if category is None:
        return "UNKNOWN"
    if isinstance(category, (tuple, list)) and category:
        category = category[0]
    text = str(category).strip()
    return text or "UNKNOWN"

def parse_float(raw: str | None, default: float = math.nan) -> float:
    if raw is None:
        return default
    text = str(raw).strip()
    if text.lower() in {"", "none", "nan", "n/a"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default

def first_float(patterns: list[str], text: str, default: float = math.nan) -> float:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = parse_float(match.group(1), default=math.nan)
            if not math.isnan(value):
                return value
    return default

def fallback_score(quality: float, depth: float) -> float:
    if math.isnan(quality):
        return float("-inf")
    depth_value = 0.0 if math.isnan(depth) else max(depth, 0.0)
    return quality_weight * quality - depth_reg_coef * (depth_value ** depth_reg_power)

def ranking_score(validation_rank_ic: float, quality: float, depth: float) -> float:
    if not math.isnan(validation_rank_ic):
        return validation_rank_ic
    return fallback_score(quality, depth)

def selection_key(row: dict) -> tuple[float, float, float, float]:
    validation_rank_ic = row.get("validation_rank_ic", math.nan)
    quality = row.get("quality", math.nan)
    depth = row.get("depth", math.nan)
    return (
        float("-inf") if math.isnan(validation_rank_ic) else validation_rank_ic,
        row.get("score", float("-inf")),
        float("-inf") if math.isnan(quality) else quality,
        0.0 if math.isnan(depth) else -depth,
    )

def resolve_source_path(raw_path: str) -> Path:
    source = Path(raw_path.strip())
    if source.is_absolute():
        return source

    candidates = [
        Path.cwd() / source,
        archive_log.parent / source,
        archive_log.resolve().parent / source,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / source

def resolve_existing_path(raw_path: str) -> Path | None:
    path = Path(raw_path.strip())
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        archive_log.parent / path,
        archive_log.resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def record_to_dict(record) -> dict:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())

    data = {}
    for key in [
        "factor_name",
        "factor_expression",
        "factor_description",
        "category",
        "depth_bin",
        "factor_ast_depth",
        "factor_ast_node_count",
        "factor_complexity_metric",
        "factor_complexity_value",
        "quality",
        "train_rank_ic",
        "validation_rank_ic",
        "archive_factor_values_path",
        "factor_values_path",
    ]:
        if hasattr(record, key):
            data[key] = getattr(record, key)
    return data

def records_from_loaded_object(loaded) -> list:
    if isinstance(loaded, dict):
        for key in ("records", "archive"):
            value = loaded.get(key)
            if value is not None:
                try:
                    return list(value)
                except TypeError:
                    pass
    archive = getattr(loaded, "archive", None)
    if archive is not None:
        if hasattr(archive, "records"):
            try:
                return list(archive.records())
            except TypeError:
                pass
        cells = getattr(archive, "cells", None)
        if cells is not None:
            try:
                return list(cells.values())
            except TypeError:
                pass
    try:
        return list(loaded)
    except TypeError:
        return []

def add_row(
    raw_name: str,
    expr: str,
    quality: float,
    depth: float,
    idx: str,
    validation_rank_ic: float = math.nan,
    category: str = "UNKNOWN",
    factor_values_path: str | None = None,
) -> None:
    expr = expr.strip()
    if not expr:
        return
    raw_expr = expr
    if flip_all:
        expr = f"-1 * ({expr})"
    score = ranking_score(validation_rank_ic, quality, depth)
    rows.append(
        {
            "raw_factor_name": raw_name,
            "raw_factor_expression": raw_expr,
            "factor_name": clean_name(raw_name, idx),
            "factor_expression": expr,
            "quality": quality,
            "validation_rank_ic": validation_rank_ic,
            "depth": depth,
            "score": score,
            "category": normalize_category(category),
            "factor_values_path": str(factor_values_path) if factor_values_path else "",
        }
    )

def load_rows_from_source_pickle() -> Path | None:
    source_match = re.search(r"^source:\s*(?P<source>.+?)\s*$", block, flags=re.MULTILINE)
    if not source_match:
        return None

    source = resolve_source_path(source_match.group("source"))
    if not source.exists():
        parent = source.parent
        alternatives = sorted(parent.glob("*.pkl"), key=lambda path: (path.stat().st_mtime, str(path)))
        if alternatives:
            source = alternatives[-1]
        else:
            print(f"WARNING: archive source pickle not found: {source}", file=sys.stderr)
            return None

    return load_rows_from_pickle_path(source, "archive source")

def load_rows_from_session_checkpoint() -> Path | None:
    session_match = re.search(r"^session[=:]\s*(?P<session>.+?)\s*$", block, flags=re.MULTILINE)
    if not session_match:
        return None

    session_path = resolve_existing_path(session_match.group("session"))
    if session_path is None:
        print(
            f"WARNING: session checkpoint not found: {session_match.group('session')}",
            file=sys.stderr,
        )
        return None

    return load_rows_from_pickle_path(session_path, "session checkpoint")

def load_rows_from_pickle_path(source: Path, label: str) -> Path | None:
    try:
        with source.open("rb") as f:
            loaded = pickle.load(f)
    except Exception as exc:
        print(f"WARNING: failed to load {label} {source}: {exc}", file=sys.stderr)
        return None

    iterator = records_from_loaded_object(loaded)
    if not iterator:
        print(f"WARNING: {label} does not contain archive records: {source}", file=sys.stderr)
        return None

    for idx, record in enumerate(iterator, start=1):
        data = record_to_dict(record)
        raw_name = str(data.get("factor_name") or f"elite_factor_{idx}")
        expr = str(data.get("factor_expression") or "")
        quality = parse_float(
            data.get("quality", data.get("train_Rank IC", data.get("train_rank_ic"))),
            default=math.nan,
        )
        validation_rank_ic = parse_float(
            data.get("validation_Rank IC", data.get("validation_rank_ic")),
            default=math.nan,
        )
        category = data.get("category", data.get("cell", "UNKNOWN"))
        depth = parse_float(
            data.get(
                "factor_ast_depth",
                data.get("factor_complexity_value", data.get("depth_bin")),
            ),
            default=math.nan,
        )
        factor_values_path = data.get("archive_factor_values_path") or data.get("factor_values_path")
        add_row(raw_name, expr, quality, depth, str(idx), validation_rank_ic, category, factor_values_path)

    return source

def parse_detail_entries(text_block: str) -> list[dict]:
    entry_pattern = re.compile(r"(?m)^\[(?P<idx>\d+)\]\s+(?P<header>.+?)\n\s+expr:\s*(?P<expr>.*?)\s*$")
    parsed = []
    for match in entry_pattern.finditer(text_block):
        header = match.group("header").strip()
        cell_match = re.search(r"\bcell=\((?P<category>[^,]+),\s*(?P<depth_bin>[0-9]+)\)", header)
        if not cell_match:
            continue
        raw_name = header.split("|", 1)[0].strip()
        expr = match.group("expr").strip()
        quality = first_float([r"\bquality=([^\s|]+)"], header)
        validation_rank_ic = first_float([r"\bvalidation_Rank IC=([^\s|]+)"], header)
        depth = first_float(
            [
                r"\bast_depth=([^\s|]+)",
                r"\bmetric_value=([^\s|]+)",
                r"\bdepth_bin=([^\s|]+)",
                r"\bcell=\([^,]+,\s*([0-9.]+)\)",
            ],
            header,
        )
        parsed.append(
            {
                "idx": match.group("idx"),
                "cell": (cell_match.group("category").strip(), int(cell_match.group("depth_bin"))),
                "category": cell_match.group("category").strip(),
                "raw_name": raw_name,
                "expr": expr,
                "quality": quality,
                "validation_rank_ic": validation_rank_ic,
                "depth": depth,
            }
        )
    return parsed

def extract_full_details_section(round_block: str) -> str | None:
    full_details_marker = "\nDetails\n"
    if full_details_marker not in round_block:
        return None
    return full_details_marker + round_block.rsplit(full_details_marker, 1)[1]

def extract_changed_details_section(round_block: str) -> str | None:
    changed_marker = "\nChanged Details\n"
    if changed_marker not in round_block:
        return None
    section = changed_marker + round_block.rsplit(changed_marker, 1)[1]
    full_snapshot_marker = "\nFull archive snapshot"
    if full_snapshot_marker in section:
        section = section.split(full_snapshot_marker, 1)[0]
    return section

def load_rows_from_text_history() -> None:
    state_by_cell = {}
    for round_match in block_pattern.finditer(text):
        current_round = int(round_match.group("round"))
        if current_round > round_id:
            break

        round_block = round_match.group(0)
        full_section = extract_full_details_section(round_block)
        if full_section is not None:
            full_entries = parse_detail_entries(full_section)
            if full_entries:
                state_by_cell = {entry["cell"]: entry for entry in full_entries}
                continue

        changed_section = extract_changed_details_section(round_block)
        if changed_section is None:
            continue
        for entry in parse_detail_entries(changed_section):
            state_by_cell[entry["cell"]] = entry

    for idx, entry in enumerate(state_by_cell.values(), start=1):
        add_row(
            entry["raw_name"],
            entry["expr"],
            entry["quality"],
            entry["depth"],
            str(idx),
            entry.get("validation_rank_ic", math.nan),
            entry.get("category", "UNKNOWN"),
        )

def load_train_corr_scope() -> tuple[object, object, object, bool]:
    import pandas as pd
    import yaml

    config_name = qlib_config_name or "conf_cn_combined_elite_rank_label.yaml"
    config_path = Path("alphaagent/scenarios/qlib/experiment/factor_template") / config_name
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(
            f"WARNING: failed to read qlib config for corr dedupe {config_path}: {exc}; "
            "using default train segment",
            file=sys.stderr,
        )
        config = {}

    segments = config.get("task", {}).get("dataset", {}).get("kwargs", {}).get("segments", {})
    train = segments.get("train")
    if isinstance(train, (list, tuple)) and len(train) >= 2:
        train_start = pd.Timestamp(train[0])
        train_end = pd.Timestamp(train[1])
    else:
        train_start = pd.Timestamp("2015-06-01")
        train_end = pd.Timestamp("2020-05-31")

    evidence_end = train_end
    evidence_start = max(train_start, evidence_end - pd.Timedelta(days=365))
    source_start = evidence_start - pd.Timedelta(days=warmup_days)
    market = (
        config.get("market")
        or (config.get("data_handler_config", {}) or {}).get("instruments")
        or "csi500"
    )
    uppercase_instruments = str(market).lower() not in {"all", "csi300_ext"}
    return source_start, evidence_start, evidence_end, uppercase_instruments

def filter_frame_by_date(df, start_time, end_time):
    import pandas as pd

    if not isinstance(df.index, pd.MultiIndex):
        return df
    date_level = None
    for candidate in ("datetime", "date", "time"):
        if candidate in df.index.names:
            date_level = candidate
            break
    if date_level is None:
        date_level = 0
    dates = pd.to_datetime(df.index.get_level_values(date_level))
    mask = (dates >= start_time) & (dates <= end_time)
    return df.loc[mask]

def filter_series_by_date(series, start_time, end_time):
    import pandas as pd

    if not isinstance(series.index, pd.MultiIndex):
        return series
    date_level = "datetime" if "datetime" in series.index.names else 0
    dates = pd.to_datetime(series.index.get_level_values(date_level))
    mask = (dates >= start_time) & (dates <= end_time)
    return series.loc[mask]

def safe_corr_cache_name(raw_name: str, raw_expression: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_name)).strip("._") or "factor"
    digest = hashlib.md5(f"{raw_name}\0{raw_expression}".encode("utf-8")).hexdigest()[:12]
    return f"{safe_name}_{digest}.pkl"

def archive_corr_values_dir_from_log() -> Path | None:
    candidates: list[Path] = []
    session_match = re.search(r"^session[=:]\s*(?P<session>.+?)\s*$", block, flags=re.MULTILINE)
    if session_match:
        session_path = resolve_existing_path(session_match.group("session"))
        if session_path is not None:
            parts = list(session_path.parts)
            if "__session__" in parts:
                root = Path(*parts[:parts.index("__session__")])
                candidates.append(root / "archive_corr_values")
            candidates.append(session_path / "archive_corr_values")

    candidates.extend(sorted(archive_log.parent.glob("*/archive_corr_values")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None

def resolve_cached_corr_values_path(row: dict, corr_dir: Path | None) -> Path | None:
    raw_path = str(row.get("factor_values_path") or "").strip()
    if raw_path:
        resolved = resolve_existing_path(raw_path)
        if resolved is not None and resolved.is_file():
            return resolved
    if corr_dir is None:
        return None

    expected = corr_dir / safe_corr_cache_name(
        str(row.get("raw_factor_name") or row["factor_name"]),
        str(row.get("raw_factor_expression") or row["factor_expression"]),
    )
    if expected.exists():
        return expected

    safe_prefix = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(row.get("raw_factor_name") or row["factor_name"]),
    ).strip("._")
    if not safe_prefix:
        return None
    matches = sorted(corr_dir.glob(f"{safe_prefix}_*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None

def factor_value_corr(left, right) -> float | None:
    import numpy as np
    import pandas as pd

    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    if left.empty or right.empty:
        return None

    if left.index.equals(right.index):
        arr_left = left.to_numpy(dtype=np.float64, copy=False)
        arr_right = right.to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(arr_left) & np.isfinite(arr_right)
        if int(mask.sum()) < 2:
            return None
        arr_left = arr_left[mask]
        arr_right = arr_right[mask]
    else:
        pair = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner")
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 2:
            return None
        arr_left = pair["left"].to_numpy(dtype=np.float64, copy=False)
        arr_right = pair["right"].to_numpy(dtype=np.float64, copy=False)

    if float(np.std(arr_left)) == 0.0 or float(np.std(arr_right)) == 0.0:
        return None
    return float(np.corrcoef(arr_left, arr_right)[0, 1])

def corr_dedupe_rows(candidate_rows: list[dict], threshold: float) -> list[dict]:
    if threshold <= 0 or len(candidate_rows) <= 1:
        return candidate_rows

    import numpy as np
    import pandas as pd

    source_start, evidence_start, evidence_end, uppercase_instruments = load_train_corr_scope()
    print(
        "Computing factor corr dedupe on train evidence year: "
        f"{evidence_start.date()} to {evidence_end.date()}, threshold={threshold:g}"
    )

    corr_dir = archive_corr_values_dir_from_log()
    series_by_idx = {}
    missing_cache_indices = []
    for idx, row in enumerate(candidate_rows):
        cache_path = resolve_cached_corr_values_path(row, corr_dir)
        if cache_path is not None:
            try:
                values = pd.read_pickle(cache_path)
                values = filter_series_by_date(values, evidence_start, evidence_end)
                values = values.astype(np.float32, copy=False)
                if len(values) == 0:
                    raise ValueError("empty cached factor values after evidence-year filter")
                series_by_idx[idx] = values
                continue
            except Exception as exc:
                print(
                    f"WARNING: failed to read corr cache for {row['factor_name']} at {cache_path}: {exc}",
                    file=sys.stderr,
                )
        missing_cache_indices.append(idx)

    if series_by_idx:
        print(
            f"Loaded cached corr values for {len(series_by_idx)} / {len(candidate_rows)} factors"
            + (f" from {corr_dir}" if corr_dir is not None else "")
        )

    if missing_cache_indices:
        from alphaagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

        daily_pv_path = Path("alphaagent/scenarios/qlib/experiment/factor_data_template/daily_pv_all.h5")
        if not daily_pv_path.exists():
            print(
                f"WARNING: corr dedupe has {len(missing_cache_indices)} missing cached factors "
                f"and cannot load {daily_pv_path}; keeping those factors without corr edges",
                file=sys.stderr,
            )
            missing_cache_indices = []
        else:
            try:
                daily_pv = pd.read_hdf(daily_pv_path, key="data")
                daily_pv = filter_frame_by_date(daily_pv, source_start, evidence_end)
            except Exception as exc:
                print(
                    f"WARNING: failed to load {daily_pv_path}: {exc}; "
                    "keeping missing-cache factors without corr edges",
                    file=sys.stderr,
                )
                missing_cache_indices = []

    for idx in missing_cache_indices:
        row = candidate_rows[idx]
        try:
            values = QlibFactorRunner.evaluate_factor_expression_direct(
                None,
                row["factor_expression"],
                daily_pv,
                upper_instrument=uppercase_instruments,
            )
            values = filter_series_by_date(values, evidence_start, evidence_end)
            values = values.astype(np.float32, copy=False)
            if len(values) == 0:
                raise ValueError("empty factor values after evidence-year filter")
            series_by_idx[idx] = values
        except Exception as exc:
            print(
                f"WARNING: corr unavailable for {row['factor_name']}; keeping it. reason={exc}",
                file=sys.stderr,
            )

    if len(series_by_idx) <= 1:
        print("Corr dedupe kept all factors because fewer than two factor series were available.")
        return candidate_rows

    parent = list(range(len(candidate_rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    high_corr_edges = []
    available = sorted(series_by_idx)
    for offset, left_idx in enumerate(available):
        for right_idx in available[offset + 1:]:
            corr = factor_value_corr(series_by_idx[left_idx], series_by_idx[right_idx])
            if corr is None or pd.isna(corr):
                continue
            if abs(corr) > threshold:
                union(left_idx, right_idx)
                high_corr_edges.append((abs(corr), corr, left_idx, right_idx))

    components: dict[int, list[int]] = {}
    for idx in range(len(candidate_rows)):
        components.setdefault(find(idx), []).append(idx)

    keep_indices = set()
    dropped = []
    for member_indices in components.values():
        if len(member_indices) == 1:
            keep_indices.add(member_indices[0])
            continue
        best_idx = max(member_indices, key=lambda i: selection_key(candidate_rows[i]))
        keep_indices.add(best_idx)
        for idx in member_indices:
            if idx != best_idx:
                dropped.append((idx, best_idx))

    if not dropped:
        print("Corr dedupe found no abs(corr) above threshold.")
        return candidate_rows

    edge_lookup = {}
    for abs_corr, corr, left_idx, right_idx in high_corr_edges:
        edge_lookup.setdefault(frozenset((left_idx, right_idx)), (abs_corr, corr))

    print(f"Applied corr dedupe: kept {len(keep_indices)} / {len(candidate_rows)} factors")
    for dropped_idx, kept_idx in sorted(dropped, key=lambda item: selection_key(candidate_rows[item[0]]), reverse=True):
        edge = edge_lookup.get(frozenset((dropped_idx, kept_idx)))
        corr_text = "connected"
        if edge is not None:
            corr_text = f"corr={edge[1]:.4f}"
        dropped_row = candidate_rows[dropped_idx]
        kept_row = candidate_rows[kept_idx]
        print(
            f"  - drop {dropped_row['factor_name']} "
            f"(validation_Rank_IC={dropped_row['validation_rank_ic']:.8g}) "
            f"because {corr_text} with kept {kept_row['factor_name']} "
            f"(validation_Rank_IC={kept_row['validation_rank_ic']:.8g})"
        )

    return [row for idx, row in enumerate(candidate_rows) if idx in keep_indices]

loaded_source = load_rows_from_source_pickle()
if loaded_source is None:
    loaded_source = load_rows_from_session_checkpoint()

if loaded_source is not None:
    print(f"Loaded complete archive records from: {loaded_source}")
else:
    print("WARNING: falling back to text archive reconstruction", file=sys.stderr)
    load_rows_from_text_history()

if not rows:
    print(f"ERROR: no factor expressions found for round={round_id}", file=sys.stderr)
    sys.exit(1)

if expected_records is not None and len(rows) != expected_records:
    print(
        f"WARNING: records={expected_records}, parsed expressions={len(rows)}",
        file=sys.stderr,
    )

rows.sort(
    key=lambda row: (
        row["score"],
        float("-inf") if math.isnan(row["validation_rank_ic"]) else row["validation_rank_ic"],
        float("-inf") if math.isnan(row["quality"]) else row["quality"],
        0.0 if math.isnan(row["depth"]) else -row["depth"],
    ),
    reverse=True,
)
total_rows = len(rows)
if per_category_l > 0:
    before_category_limit = len(rows)
    category_counts: dict[str, int] = {}
    limited_rows = []
    for row in rows:
        category = normalize_category(row.get("category"))
        used = category_counts.get(category, 0)
        if used >= per_category_l:
            continue
        category_counts[category] = used + 1
        limited_rows.append(row)
    rows = limited_rows
    print(
        f"Applied per-category limit: kept {len(rows)} / {before_category_limit} "
        f"factors with at most {per_category_l} per category"
    )

rows = corr_dedupe_rows(rows, corr_dedupe_threshold)

if top_k > 0:
    rows = rows[:top_k]

factor_csv.parent.mkdir(parents=True, exist_ok=True)
with factor_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["factor_name", "factor_expression"])
    writer.writeheader()
    writer.writerows(
        {
            "factor_name": row["factor_name"],
            "factor_expression": row["factor_expression"],
        }
        for row in rows
    )

print(f"Selected {len(rows)} / {total_rows} factors by score and wrote them to {factor_csv}")
for row in rows:
    quality = "nan" if math.isnan(row["quality"]) else f"{row['quality']:.8g}"
    validation_rank_ic = "nan" if math.isnan(row["validation_rank_ic"]) else f"{row['validation_rank_ic']:.8g}"
    depth = "nan" if math.isnan(row["depth"]) else f"{row['depth']:.8g}"
    score = "-inf" if row["score"] == float("-inf") else f"{row['score']:.8g}"
    category = row.get("category", "UNKNOWN")
    print(
        f"  - score={score}, validation_Rank_IC={validation_rank_ic}, quality={quality}, "
        f"depth={depth}, category={category}, {row['factor_name']}: {row['factor_expression']}"
    )
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete. Factor CSV kept at: ${FACTOR_CSV}"
  exit 0
fi

if [[ -z "${BACKTEST_LOG_DIR}" ]]; then
  BACKTEST_LOG_DIR="log/archive_round_backtest_${ROUND}_${RUN_STAMP}"
fi
mkdir -p "${BACKTEST_LOG_DIR}" "$(dirname "${OUTPUT_PKL}")"

export LOG_TRACE_PATH="${BACKTEST_LOG_DIR}"
export log_trace_path="${BACKTEST_LOG_DIR}"
export CACHE_WITH_PICKLE=false
export cache_with_pickle=false
export QLIB_FACTOR_ARCHIVE_BACKTEST_STREAMING=true
export qlib_factor_archive_backtest_streaming=true
export QLIB_FACTOR_ARCHIVE_BACKTEST_WARMUP_DAYS="${ARCHIVE_BACKTEST_WARMUP_DAYS}"
export qlib_factor_archive_backtest_warmup_days="${ARCHIVE_BACKTEST_WARMUP_DAYS}"
if [[ -n "${QLIB_CONFIG_NAME}" ]]; then
  export QLIB_FACTOR_QLIB_CONFIG_NAME="${QLIB_CONFIG_NAME}"
  export qlib_factor_qlib_config_name="${QLIB_CONFIG_NAME}"
fi

if [[ "${USE_LIGHT}" -eq 1 ]]; then
  export QLIB_FACTOR_USE_LIGHTWEIGHT_QLIB_TEST=true
  export qlib_factor_use_lightweight_qlib_test=true
  if [[ -n "${QLIB_CONFIG_NAME}" ]]; then
    export QLIB_FACTOR_LIGHTWEIGHT_QLIB_CONFIG_NAME="${QLIB_CONFIG_NAME}"
    export qlib_factor_lightweight_qlib_config_name="${QLIB_CONFIG_NAME}"
  fi
fi

ALPHAAGENT_BIN="${ALPHAAGENT_BIN:-alphaagent}"
if ! command -v "${ALPHAAGENT_BIN}" >/dev/null 2>&1 && [[ ! -x "${ALPHAAGENT_BIN}" ]]; then
  echo "ERROR: AlphaAgent command not found: ${ALPHAAGENT_BIN}" >&2
  echo "Set ALPHAAGENT_BIN=/path/to/alphaagent if it is not on PATH." >&2
  exit 1
fi

echo "Running AlphaAgent multi-factor backtest:"
echo "  log_dir=${BACKTEST_LOG_DIR}"
echo "  output_pkl=${OUTPUT_PKL}"
echo "  workspace_root=${WORKSPACE_ROOT}"
echo "  step_n=${STEP_N}"
echo "  qlib_config=${QLIB_CONFIG_NAME:-<runner default>}"
echo "  lightweight=${QLIB_FACTOR_USE_LIGHTWEIGHT_QLIB_TEST:-false}"
echo "  archive_backtest_streaming=${QLIB_FACTOR_ARCHIVE_BACKTEST_STREAMING}"
BACKTEST_START_TS="$(date +%s)"
"${ALPHAAGENT_BIN}" backtest --factor_path "${FACTOR_CSV}" --step_n "${STEP_N}"

if ! GENERATED_PKL="$(
  BACKTEST_LOG_DIR="${BACKTEST_LOG_DIR}" WORKSPACE_ROOT="${WORKSPACE_ROOT}" BACKTEST_START_TS="${BACKTEST_START_TS}" QLIB_CONFIG_NAME="${QLIB_CONFIG_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
import re
import sys
from datetime import datetime
from pathlib import Path

roots = [Path(os.environ["BACKTEST_LOG_DIR"]), Path(os.environ["WORKSPACE_ROOT"])]
paths = []
for root in roots:
    if root.exists():
        paths.extend(root.rglob("combined_factors_df.pkl"))
if not paths:
    raise SystemExit(1)
start_ts = float(os.environ.get("BACKTEST_START_TS", "0"))
fresh_paths = [p for p in paths if p.stat().st_mtime >= start_ts - 5]
candidate_paths = fresh_paths or paths

log_text_parts = []
log_root = Path(os.environ["BACKTEST_LOG_DIR"])
if log_root.exists():
    for log_path in sorted(log_root.rglob("common_logs.log")):
        try:
            log_text_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
log_text = "\n".join(log_text_parts)

expected_config = os.environ.get("QLIB_CONFIG_NAME", "").strip()
execute_ts = None
execute_re = re.compile(
    r"(?m)^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*?"
    r"Execute Local Backtest: qrun (?P<config>\S+)"
)
execute_matches = list(execute_re.finditer(log_text))
if execute_matches:
    match = execute_matches[-1]
    expected_config = match.group("config")
    try:
        execute_ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        execute_ts = None

def workspace_for_pkl(path: Path) -> Path:
    return path.parent

def qrun_command_for_pkl(path: Path) -> str:
    workspace = workspace_for_pkl(path)
    params = sorted(
        workspace.glob("mlruns/*/*/params/cmd-sys.argv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for param_path in params:
        try:
            return param_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
    return ""

cmd_cache = {}
def qrun_command(path: Path) -> str:
    if path not in cmd_cache:
        cmd_cache[path] = qrun_command_for_pkl(path)
    return cmd_cache[path]

if expected_config:
    matched = [p for p in candidate_paths if expected_config in qrun_command(p)]
    if matched:
        candidate_paths = matched
    else:
        print(
            f"WARNING: no combined_factors_df.pkl workspace matched qrun config {expected_config}; using time match only",
            file=sys.stderr,
        )

target_ts = execute_ts if execute_ts is not None else start_ts
def score(path: Path):
    mtime = path.stat().st_mtime
    return (abs(mtime - target_ts), -mtime)

best = min(candidate_paths, key=score) if execute_ts is not None else max(candidate_paths, key=lambda p: p.stat().st_mtime)
if expected_config and expected_config not in qrun_command(best):
    print(
        f"WARNING: selected workspace qrun command does not contain expected config {expected_config}: {qrun_command(best)}",
        file=sys.stderr,
    )
print(best)
PY
)"; then
  echo "ERROR: backtest finished but no combined_factors_df.pkl was found under ${BACKTEST_LOG_DIR} or ${WORKSPACE_ROOT}" >&2
  exit 1
fi

GENERATED_WORKSPACE="$(dirname "${GENERATED_PKL}")"
cp "${GENERATED_PKL}" "${OUTPUT_PKL}"

echo "Copied generated factor matrix:"
echo "  from=${GENERATED_PKL}"
echo "  to=${OUTPUT_PKL}"

if [[ "${GENERATE_REPORT}" -eq 1 ]]; then
  mkdir -p "$(dirname "${REPORT_PATH}")"
  echo "Writing LightGBM report:"
  echo "  report=${REPORT_PATH}"
  echo "  workspace=${GENERATED_WORKSPACE}"
  if ! GENERATED_PKL="${GENERATED_PKL}" OUTPUT_PKL="${OUTPUT_PKL}" REPORT_PATH="${REPORT_PATH}" \
      GENERATED_WORKSPACE="${GENERATED_WORKSPACE}" FACTOR_CSV="${FACTOR_CSV}" \
      ARCHIVE_LOG="${ARCHIVE_LOG}" ROUND="${ROUND}" BACKTEST_LOG_DIR="${BACKTEST_LOG_DIR}" \
      "${PYTHON_BIN}" - <<'PY'
import ast
import csv
import datetime as dt
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - report should still be written.
    pd = None
    pandas_error = str(exc)
else:
    pandas_error = ""

generated_pkl = Path(os.environ["GENERATED_PKL"])
output_pkl = Path(os.environ["OUTPUT_PKL"])
workspace = Path(os.environ["GENERATED_WORKSPACE"])
factor_csv = Path(os.environ["FACTOR_CSV"])
archive_log = Path(os.environ["ARCHIVE_LOG"])
backtest_log_dir = Path(os.environ["BACKTEST_LOG_DIR"])
report_path = Path(os.environ["REPORT_PATH"])
round_id = os.environ["ROUND"]


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def fmt_num(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    try:
        if isinstance(value, str):
            return value
        if pd is not None and pd.isna(value):
            return "nan"
        number = float(value)
        abs_number = abs(number)
        if abs_number != 0 and (abs_number < 1e-4 or abs_number >= 1e6):
            return f"{number:.6e}"
        return f"{number:.10g}"
    except Exception:
        return str(value)


def section(lines, title):
    lines.append("")
    lines.append(title)
    lines.append("-" * 80)


def read_factor_csv(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("factor_name") or "").strip()
            expr = (row.get("factor_expression") or "").strip()
            if name:
                rows.append({"factor_name": name, "factor_expression": expr})
    return rows


def read_csv_kv(path):
    data = {}
    if not path.exists():
        return data
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                try:
                    data[row[0]] = float(row[1])
                except Exception:
                    data[row[0]] = row[1]
    return data


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def find_mlflow_run(workspace_path):
    params_pkls = sorted(
        workspace_path.glob("mlruns/*/*/artifacts/params.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not params_pkls:
        return None, None
    return params_pkls[0].parents[1], params_pkls[0]


def read_param(run_path, name):
    if run_path is None:
        return None
    path = run_path / "params" / name
    if not path.exists():
        return None
    return read_text(path).strip()


def read_metrics(run_path):
    metrics = {}
    if run_path is None:
        return metrics
    metrics_dir = run_path / "metrics"
    if not metrics_dir.exists():
        return metrics
    for path in sorted(metrics_dir.iterdir()):
        if not path.is_file():
            continue
        rows = [line.strip() for line in read_text(path).splitlines() if line.strip()]
        if not rows:
            continue
        parts = rows[-1].split()
        if len(parts) >= 2:
            try:
                metrics[path.name] = float(parts[1])
            except Exception:
                metrics[path.name] = rows[-1]
    return metrics


def collect_log_text(log_dir):
    pieces = []
    if log_dir.exists():
        for path in sorted(log_dir.rglob("common_logs.log")):
            pieces.append(read_text(path))
    return "\n".join(pieces)


def parse_factor_level_log(log_text, factor_names):
    info = {name: {} for name in factor_names}
    sign_re = re.compile(
        r"Sign-adjusted factor for (?P<name>\S+) because direct Rank IC was negative: "
        r"expression=(?P<expr>.*?), ast_depth=(?P<depth>\d+), complexity_value=(?P<complexity>[^\n]+)"
    )
    for match in sign_re.finditer(log_text):
        name = match.group("name")
        if name in info:
            info[name]["expr_used"] = match.group("expr").strip()
            info[name]["ast_depth"] = match.group("depth")
            info[name]["complexity_value"] = match.group("complexity").strip()
            info[name]["sign_adjusted_log"] = True

    quality_blocks = re.findall(
        r"Factor-level archive quality:\s*\n(?P<body>.*?)(?=\n\d{4}-\d{2}-\d{2} |\Z)",
        log_text,
        flags=re.S,
    )
    for block in quality_blocks:
        for line in block.splitlines():
            stripped = line.strip()
            for name in factor_names:
                if not stripped.startswith(name + " "):
                    continue
                parts = stripped.split()
                if len(parts) < 8:
                    continue
                try:
                    info[name].update(
                        {
                            "raw_IC": float(parts[1]),
                            "raw_Rank_IC": float(parts[2]),
                            "IC_after_sign_adjust": float(parts[3]),
                            "Rank_IC_after_sign_adjust": float(parts[4]),
                            "ICIR": float(parts[5]),
                            "Rank_ICIR": float(parts[6]),
                            "sign_flipped": parts[7] == "True",
                        }
                    )
                except Exception:
                    pass

    config_lines = [line for line in log_text.splitlines() if "Use qlib factor config" in line]
    qrun_lines = [line for line in log_text.splitlines() if "Execute Local Backtest: qrun" in line]
    return info, (config_lines[-1] if config_lines else ""), (qrun_lines[-1] if qrun_lines else "")


def extract_booster_text(path):
    if path is None or not path.exists():
        return ""
    data = path.read_bytes()
    start = data.find(b"tree\nversion")
    if start < 0:
        start = data.find(b"tree\r\nversion")
    if start < 0:
        return ""
    marker = b"\npandas_categorical:"
    end = data.find(marker, start)
    if end >= 0:
        next_newline = data.find(b"\n", end + 1)
        end = len(data) if next_newline < 0 else next_newline
    else:
        end = len(data)
    return data[start:end].decode("utf-8", errors="replace")


def parse_booster(booster_text, feature_mapping):
    if not booster_text:
        return {}
    trees = re.findall(r"^Tree=(\d+)", booster_text, flags=re.M)
    leaves = [int(x) for x in re.findall(r"^num_leaves=(\d+)", booster_text, flags=re.M)]
    split_counter = Counter()
    gain_counter = defaultdict(float)
    split_lines = re.findall(r"^split_feature=([^\n]+)", booster_text, flags=re.M)
    gain_lines = re.findall(r"^split_gain=([^\n]+)", booster_text, flags=re.M)
    for i, split_line in enumerate(split_lines):
        features = []
        for token in split_line.split():
            try:
                features.append(int(token))
            except Exception:
                pass
        gains = []
        if i < len(gain_lines):
            for token in gain_lines[i].split():
                try:
                    gains.append(float(token))
                except Exception:
                    gains.append(0.0)
        for j, feature in enumerate(features):
            split_counter[feature] += 1
            if j < len(gains):
                gain_counter[feature] += gains[j]
    rows = []
    for feature in sorted(set(split_counter) | set(gain_counter)):
        rows.append(
            (
                feature,
                feature_mapping.get(feature, f"Column_{feature}"),
                split_counter[feature],
                gain_counter[feature],
            )
        )
    rows.sort(key=lambda row: (row[3], row[2]), reverse=True)
    return {
        "tree_count": len(trees),
        "num_leaves_min": min(leaves) if leaves else None,
        "num_leaves_max": max(leaves) if leaves else None,
        "num_leaves_mean": sum(leaves) / len(leaves) if leaves else None,
        "importance_rows": rows,
    }


def flatten_col(col):
    if isinstance(col, tuple):
        return str(col[-1])
    return str(col)


factor_rows = read_factor_csv(factor_csv)
factor_names = [row["factor_name"] for row in factor_rows]
factor_expr_by_name = {row["factor_name"]: row["factor_expression"] for row in factor_rows}
log_text = collect_log_text(backtest_log_dir)
factor_log_info, config_line, qrun_line = parse_factor_level_log(log_text, factor_names)

mlflow_run, params_pkl = find_mlflow_run(workspace)
data_loader_raw = read_param(mlflow_run, "dataset.kwargs.handler.kwargs.data_loader.kwargs.dataloader_l")
learn_processors_raw = read_param(mlflow_run, "dataset.kwargs.handler.kwargs.learn_processors")
infer_processors_raw = read_param(mlflow_run, "dataset.kwargs.handler.kwargs.infer_processors")
base_features = []
if data_loader_raw:
    try:
        data_loader = ast.literal_eval(data_loader_raw)
        for item in data_loader:
            if isinstance(item, dict) and item.get("class", "").endswith("QlibDataLoader"):
                base_features = list(item.get("kwargs", {}).get("config", {}).get("feature", []))
                break
    except Exception:
        pass

matrix_info = {
    "loaded": False,
    "error": "",
    "shape": "",
    "memory_mb": "",
    "index_names": "",
    "columns": factor_names,
    "dtypes": {},
    "stats": [],
    "corr": "",
}
if pd is None:
    matrix_info["error"] = f"pandas import failed: {pandas_error}"
else:
    try:
        factor_df = pd.read_pickle(generated_pkl)
        factor_cols = [flatten_col(col) for col in factor_df.columns]
        factor_df_named = factor_df.copy()
        factor_df_named.columns = factor_cols
        matrix_info["loaded"] = True
        matrix_info["shape"] = str(factor_df.shape)
        matrix_info["memory_mb"] = factor_df.memory_usage(index=True, deep=True).sum() / 1024 / 1024
        matrix_info["index_names"] = str(list(factor_df.index.names))
        matrix_info["columns"] = factor_cols
        matrix_info["dtypes"] = dict(zip(factor_cols, [str(dtype) for dtype in factor_df.dtypes]))
        for col in factor_cols:
            series = factor_df_named[col]
            matrix_info["stats"].append(
                {
                    "name": col,
                    "count": int(series.count()),
                    "non_null_ratio": float(series.count() / len(series)) if len(series) else math.nan,
                    "nan_count": int(series.isna().sum()),
                    "mean": float(series.mean(skipna=True)),
                    "std": float(series.std(skipna=True)),
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                }
            )
        if len(factor_cols) <= 50:
            matrix_info["corr"] = factor_df_named.corr().to_string()
    except Exception as exc:
        matrix_info["error"] = str(exc)

feature_mapping = {}
for idx, feature in enumerate(base_features):
    feature_mapping[idx] = f"[base_qlib_feature] {feature}"
for offset, name in enumerate(matrix_info["columns"], start=len(base_features)):
    expr = factor_log_info.get(name, {}).get("expr_used") or factor_expr_by_name.get(name, "UNKNOWN_EXPR")
    feature_mapping[offset] = f"[archive_factor] {name}: {expr}"

qlib_res = read_csv_kv(workspace / "qlib_res.csv")
ic_debug_rows = read_csv_rows(workspace / "ic_debug_summary.csv")
metrics = read_metrics(mlflow_run)
booster_text = extract_booster_text(params_pkl)
booster_summary = parse_booster(booster_text, feature_mapping)

artifact_rows = []
if mlflow_run is not None:
    for path in sorted((mlflow_run / "artifacts").rglob("*")):
        if path.is_file():
            artifact_rows.append(
                (
                    path.relative_to(mlflow_run).as_posix(),
                    path.stat().st_size,
                    dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                )
            )

lines = []
lines.append("EliteAlpha Archive LightGBM Report")
lines.append("=" * 80)
lines.append(f"Generated at: {dt.datetime.now().isoformat(timespec='seconds')}")
section(lines, "Scope")
lines.append(f"round: {round_id}")
lines.append(f"workspace: {workspace}")
lines.append(f"generated_pkl: {generated_pkl}")
lines.append(f"copied_output_pkl: {output_pkl}")
lines.append(f"factor_csv: {factor_csv}")
lines.append(f"archive_log: {archive_log}")
lines.append(f"backtest_log_dir: {backtest_log_dir}")
lines.append(f"mlflow_run: {mlflow_run}")
lines.append(f"qrun command from mlflow: {read_param(mlflow_run, 'cmd-sys.argv') or '<missing>'}")
if config_line:
    lines.append(f"factor_runner config log: {config_line}")
if qrun_line:
    lines.append(f"execute log: {qrun_line}")

section(lines, "Input Factor Matrix")
if matrix_info["loaded"]:
    lines.append(f"shape: {matrix_info['shape']}")
    lines.append(f"memory_usage_MB: {matrix_info['memory_mb']:.3f}")
    lines.append(f"index_names: {matrix_info['index_names']}")
    lines.append("dtypes:")
    for name, dtype in matrix_info["dtypes"].items():
        lines.append(f"  - {name}: {dtype}")
else:
    lines.append(f"matrix load failed: {matrix_info['error']}")
lines.append("archive/static factor columns:")
for name in matrix_info["columns"]:
    info = factor_log_info.get(name, {})
    expr = info.get("expr_used") or factor_expr_by_name.get(name, "UNKNOWN_EXPR")
    lines.append(f"  - {name}")
    lines.append(f"    expr_used: {expr}")
    raw_expr = factor_expr_by_name.get(name)
    if raw_expr and raw_expr != expr:
        lines.append(f"    raw_expr_before_sign_adjust: {raw_expr}")
    for key in [
        "raw_IC",
        "raw_Rank_IC",
        "IC_after_sign_adjust",
        "Rank_IC_after_sign_adjust",
        "ICIR",
        "Rank_ICIR",
        "sign_flipped",
        "ast_depth",
        "complexity_value",
    ]:
        if key in info:
            lines.append(f"    {key}: {fmt_num(info[key])}")
if matrix_info["stats"]:
    lines.append("")
    lines.append("factor column stats:")
    for row in matrix_info["stats"]:
        lines.append(
            f"  - {row['name']}: count={row['count']}, "
            f"non_null_ratio={fmt_num(row['non_null_ratio'])}, nan_count={row['nan_count']}, "
            f"mean={fmt_num(row['mean'])}, std={fmt_num(row['std'])}, "
            f"min={fmt_num(row['min'])}, max={fmt_num(row['max'])}"
        )
if matrix_info["corr"]:
    lines.append("factor correlation:")
    lines.append(matrix_info["corr"])

section(lines, "Actual Training Feature Mapping")
lines.append("LightGBM dump uses Column_0..Column_N.")
for idx in sorted(feature_mapping):
    lines.append(f"  Column_{idx}: {feature_mapping[idx]}")
lines.append("")
lines.append("Raw data_loader param:")
lines.append(data_loader_raw or "<missing>")

section(lines, "Data Processors")
lines.append("learn_processors:")
lines.append(learn_processors_raw or "<missing>")
lines.append("infer_processors:")
lines.append(infer_processors_raw or "<missing>")

section(lines, "Model Parameters")
if mlflow_run is not None and (mlflow_run / "params").exists():
    for path in sorted((mlflow_run / "params").iterdir(), key=lambda p: p.name):
        if path.is_file() and (path.name.startswith("model.") or path.name == "cmd-sys.argv"):
            lines.append(f"{path.name}: {read_text(path).strip()}")
else:
    lines.append("<missing>")

section(lines, "Qlib Results")
if qlib_res:
    for key, value in qlib_res.items():
        lines.append(f"{key}: {fmt_num(value)}")
else:
    lines.append("<missing>")

section(lines, "IC Debug Summary")
if ic_debug_rows:
    for row in ic_debug_rows:
        for key, value in row.items():
            lines.append(f"{key}: {value}")
else:
    lines.append("<missing>")

section(lines, "MLflow Metrics")
if metrics:
    for key in sorted(metrics):
        lines.append(f"{key}: {fmt_num(metrics[key])}")
else:
    lines.append("<no mlflow metric files found>")

section(lines, "Artifacts")
if artifact_rows:
    for rel, size, mtime in artifact_rows:
        lines.append(f"{rel}: {size} bytes, modified={mtime}")
else:
    lines.append("<missing>")

section(lines, "LightGBM Booster Summary")
if booster_text:
    lines.append(f"params.pkl: {params_pkl}")
    lines.append(f"tree_count: {booster_summary.get('tree_count')}")
    lines.append(f"num_leaves_min: {fmt_num(booster_summary.get('num_leaves_min'))}")
    lines.append(f"num_leaves_max: {fmt_num(booster_summary.get('num_leaves_max'))}")
    lines.append(f"num_leaves_mean: {fmt_num(booster_summary.get('num_leaves_mean'))}")
    lines.append("split/gain importance reconstructed from booster dump:")
    for idx, mapping, splits, gain in booster_summary.get("importance_rows", []):
        lines.append(f"  Column_{idx}: splits={splits}, gain_sum={fmt_num(gain)}, {mapping}")
else:
    lines.append("<failed to extract booster text from params.pkl>")

section(lines, "Full LightGBM Booster Dump")
lines.append(booster_text or "<missing>")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote LightGBM report to {report_path}")
PY
  then
    echo "WARNING: failed to write LightGBM report: ${REPORT_PATH}" >&2
  fi
fi

if [[ "${UPDATE_FIGURE3}" -eq 1 ]]; then
  if [[ -z "${PAPER_PKL_PATH}" ]]; then
    PAPER_PKL_PATH="../baselines/direct_factor_backtests/EliteAlpha_round${ROUND}_LGBM_report_normal_1day.pkl"
  fi
  RET_PKL="${GENERATED_WORKSPACE}/ret.pkl"
  if [[ ! -f "${RET_PKL}" ]]; then
    echo "WARNING: ret.pkl not found, skip Figure 3 update: ${RET_PKL}" >&2
  else
    mkdir -p "$(dirname "${PAPER_PKL_PATH}")"
    cp "${RET_PKL}" "${PAPER_PKL_PATH}"
    echo "Copied Figure 3 report pkl:"
    echo "  from=${RET_PKL}"
    echo "  to=${PAPER_PKL_PATH}"
    if [[ -f "${FIGURE3_SCRIPT}" ]]; then
      echo "Running Figure 3 plot script:"
      echo "  script=${FIGURE3_SCRIPT}"
      "${PYTHON_BIN}" "${FIGURE3_SCRIPT}"
    else
      echo "WARNING: plot script not found, skip plotting: ${FIGURE3_SCRIPT}" >&2
    fi
  fi
fi

echo "Done."

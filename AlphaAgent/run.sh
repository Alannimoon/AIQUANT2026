#!/usr/bin/env bash
set -euo pipefail

# Run from the AlphaAgent project root so .env, log/, and output_log_archive.py
# are resolved consistently.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# One EliteAlpha loop has five workflow steps:
# factor_propose -> factor_construct -> factor_calculate -> factor_backtest -> feedback
STEPS_PER_ROUND="${STEPS_PER_ROUND:-5}"

# Conservative inner parallelism for candidate factor coding/calculation.
# Keep outer EliteAlpha rounds serial so archive updates remain deterministic.
MULTI_PROC_N="${MULTI_PROC_N:-12}"
export MULTI_PROC_N
export multi_proc_n="${MULTI_PROC_N}"

# Start a fresh EliteAlpha chain for this run. Rounds inside this script resume
# from the previous feedback checkpoint so the archive and trace keep growing.
SESSION_PATH=""

latest_feedback_session() {
  local root_arg="$1"
  LOG_SCAN_ROOT="${root_arg}" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["LOG_SCAN_ROOT"])
paths = [p for p in root.glob("__session__/*/*_feedback") if p.is_file()]
if paths:
    print(max(paths, key=lambda p: p.stat().st_mtime))
PY
}

default_log_dir() {
  date '+log/elite_run_%Y-%m-%d_%H-%M-%S'
}

# Put runner-level stdout/stderr and archive snapshots under the run's log
# directory instead of the AlphaAgent project root.
RUN_LOG_DIR="${RUN_LOG_DIR:-$(default_log_dir)}"
export RUN_LOG_DIR
mkdir -p "${RUN_LOG_DIR}"

# Keep one AlphaAgent trace folder for the whole run. The outer RUN_LOG_DIR
# remains the place for runner stdout/stderr and archive progress snapshots.
SESSION_LOG_DIR="${RUN_LOG_DIR}/elite_session_$(date '+%H-%M-%S')_$$"
mkdir -p "${SESSION_LOG_DIR}"

# Point AlphaAgent's internal logger at this fresh session folder. Export both
# casings because pydantic-settings is case-insensitive by default, while shell
# users tend to inspect the uppercase name.
export LOG_TRACE_PATH="${SESSION_LOG_DIR}"
export log_trace_path="${SESSION_LOG_DIR}"

log_file_path() {
  local configured="$1"
  local fallback_name="$2"

  if [[ -z "${configured}" ]]; then
    printf '%s/%s\n' "${RUN_LOG_DIR}" "${fallback_name}"
  elif [[ "${configured}" == /* || "${configured}" == */* ]]; then
    printf '%s\n' "${configured}"
  else
    printf '%s/%s\n' "${RUN_LOG_DIR}" "${configured}"
  fi
}

# Archive snapshots after each completed round are appended here. Use .txt so
# FileStorage does not parse runner output as AlphaAgent structured logs.
ARCHIVE_LOG="$(log_file_path "${ARCHIVE_LOG:-}" "elite_archive_progress.txt")"

# Full stdout/stderr stream for this runner is appended here.
RUN_LOG="$(log_file_path "${RUN_LOG:-}" "elite_run_stdout_stderr.txt")"
mkdir -p "$(dirname "${ARCHIVE_LOG}")" "$(dirname "${RUN_LOG}")"

# Keep printing to terminal while also saving both stdout and stderr.
exec > >(tee -a "${RUN_LOG}") 2>&1

DIRECTIONS=(
  "Use cumulative or smoothed past return over a short fixed lookback (3-10 days) as a momentum signal. Keep AST depth <= 5 and do not add a leading negative sign only to choose direction."
  "Short-term reversal: stocks that dropped sharply in the last 1-5 days tend to bounce. Use lagged returns or distance-from-recent-high. Keep AST depth <= 5 and do not add a leading negative sign only to choose direction."
  "Idiosyncratic volatility regime: factor should rise when recent realized vol (e.g. TS_STD of return over 10-20 days) increases. Keep AST depth <= 5 and do not add a leading negative sign only to choose direction."
  "Abnormal volume coinciding with price moves: e.g. current volume vs its rolling mean, possibly interacted with return sign. Keep AST depth <= 5 and do not add a leading negative sign only to choose direction."
  "Cross-sectional rank or z-score of a simple price/volume statistic (rank of recent return, rank of dollar volume). Use RANK/ZSCORE as the outer operator. Keep AST depth <= 5 and do not add a leading negative sign only to choose direction."
)

append_archive_log() {
  local round="$1"
  local direction_idx="$2"
  local direction="$3"
  local archive_root="${4:-${RUN_LOG_DIR}}"

  {
    echo
    echo "================================================================================"
    echo "round=${round} direction_idx=${direction_idx} time=$(date '+%F %T')"
    echo "session=${SESSION_PATH}"
    echo "direction=${direction}"
    echo "--------------------------------------------------------------------------------"
    echo "Light archive update"
    python output_log_archive.py --log-dir "${archive_root}" --light
    if (( round % 10 == 0 )); then
      echo
      echo "--------------------------------------------------------------------------------"
      echo "Full archive snapshot (every 10 rounds)"
      python output_log_archive.py --log-dir "${archive_root}"
    fi
  } | tee -a "${ARCHIVE_LOG}"
}

echo "EliteAlpha runner started."
echo "steps_per_round=${STEPS_PER_ROUND}"
echo "multi_proc_n=${MULTI_PROC_N}"
echo "run_log_dir=${RUN_LOG_DIR}"
echo "session_log_dir=${SESSION_LOG_DIR}"
echo "log_trace_path=${LOG_TRACE_PATH}"
echo "archive_log=${ARCHIVE_LOG}"
echo "run_log=${RUN_LOG}"
python - <<'PY'
from pathlib import Path
import inspect

try:
    import alphaagent
    import alphaagent.scenarios.qlib.archive as archive
    import alphaagent.scenarios.qlib.developer.factor_runner as factor_runner

    alphaagent_location = getattr(alphaagent, "__file__", None) or list(getattr(alphaagent, "__path__", []))
    print(f"alphaagent_package={alphaagent_location}")
    print(f"factor_runner_module={Path(inspect.getfile(factor_runner)).resolve()}")
    print(f"archive_module={Path(inspect.getfile(archive)).resolve()}")
    print(
        "factor_quality_cache_version="
        f"{getattr(factor_runner, 'FACTOR_LEVEL_QUALITY_CACHE_VERSION', '(missing)')}"
    )
except Exception as exc:
    print(f"WARNING: failed to inspect alphaagent import path: {exc}")
PY
echo "resume_session=enabled within this run; script start is fresh"

round=0
while true; do
  for idx in "${!DIRECTIONS[@]}"; do
    direction="${DIRECTIONS[$idx]}"
    round=$((round + 1))

    echo
    if [[ -z "${SESSION_PATH}" ]]; then
      echo ">>> Start fresh EliteAlpha round=${round}, direction_idx=${idx}"
      echo "fresh_log_trace_path=${SESSION_LOG_DIR}"
      if ! LOG_TRACE_PATH="${SESSION_LOG_DIR}" log_trace_path="${SESSION_LOG_DIR}" \
        alphaagent elite_mine --step_n="${STEPS_PER_ROUND}" --direction="${direction}"; then
        echo "WARNING: EliteAlpha round=${round}, direction_idx=${idx} failed. Continue with next direction."
        continue
      fi
    else
      echo ">>> Continue EliteAlpha round=${round}, direction_idx=${idx}"
      echo "resume_from=${SESSION_PATH}"
      if ! LOG_TRACE_PATH="${SESSION_LOG_DIR}" log_trace_path="${SESSION_LOG_DIR}" \
        alphaagent elite_mine --path="${SESSION_PATH}" --step_n="${STEPS_PER_ROUND}" --direction="${direction}"; then
        echo "WARNING: EliteAlpha round=${round}, direction_idx=${idx} failed. Continue with next direction."
        continue
      fi
    fi

    SESSION_PATH="$(latest_feedback_session "${SESSION_LOG_DIR}")"
    if [[ -z "${SESSION_PATH}" ]]; then
      echo "WARNING: no feedback checkpoint was produced in ${SESSION_LOG_DIR}; archive snapshot skipped."
      continue
    fi

    append_archive_log "${round}" "${idx}" "${direction}" "${RUN_LOG_DIR}"
  done
done

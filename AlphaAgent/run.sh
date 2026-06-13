#!/usr/bin/env bash
set -euo pipefail

# Run from the AlphaAgent project root so .env, log/, and output_log_archive.py
# are resolved consistently.
cd "$(dirname "$0")"

# One EliteAlpha loop has five workflow steps:
# factor_propose -> factor_construct -> factor_calculate -> factor_backtest -> feedback
STEPS_PER_ROUND="${STEPS_PER_ROUND:-5}"

# Conservative inner parallelism for candidate factor coding/calculation.
# Keep outer EliteAlpha rounds serial so archive updates remain deterministic.
MULTI_PROC_N="${MULTI_PROC_N:-4}"
export MULTI_PROC_N
export multi_proc_n="${MULTI_PROC_N}"

# Archive snapshots after each completed round are appended here.
ARCHIVE_LOG="${ARCHIVE_LOG:-elite_archive_progress.log}"

# Full stdout/stderr stream for this runner is appended here.
RUN_LOG="${RUN_LOG:-elite_run_stdout_stderr.log}"

# Keep printing to terminal while also saving both stdout and stderr.
exec > >(tee -a "${RUN_LOG}") 2>&1

# Optional: resume a known session path, for example:
#   ELITE_SESSION_PATH="log/2026-06-03_xxx/__session__/4/4_feedback" bash run.sh
SESSION_PATH="${ELITE_SESSION_PATH:-}"

# Optional: if set to 1, resume the latest finished feedback snapshot under log/.
# Keep default 0 to avoid accidentally continuing an old experiment.
RESUME_LATEST="${RESUME_LATEST:-0}"

DIRECTIONS=(
  "Use cumulative or smoothed past return over a short fixed lookback (3-10 days) as a momentum signal. Keep AST depth <= 5."
  "Short-term reversal: stocks that dropped sharply in the last 1-5 days tend to bounce. Use lagged negative returns or distance-from-recent-high. Keep AST depth <= 5."
  "Idiosyncratic volatility regime: factor should rise when recent realized vol (e.g. TS_STD of return over 10-20 days) increases. Keep AST depth <= 5."
  "Abnormal volume coinciding with price moves: e.g. current volume vs its rolling mean, possibly interacted with return sign. Keep AST depth <= 5."
  "Cross-sectional rank or z-score of a simple price/volume statistic (rank of recent return, rank of dollar volume). Use RANK/ZSCORE as the outer operator. Keep AST depth <= 5."
)

latest_feedback_session() {
  python - <<'PY'
from pathlib import Path

paths = list(Path("log").glob("*/__session__/*/4_feedback"))
if paths:
    print(max(paths, key=lambda p: p.stat().st_mtime))
PY
}

append_archive_log() {
  local round="$1"
  local direction_idx="$2"
  local direction="$3"

  {
    echo
    echo "================================================================================"
    echo "round=${round} direction_idx=${direction_idx} time=$(date '+%F %T')"
    echo "session=${SESSION_PATH}"
    echo "direction=${direction}"
    echo "--------------------------------------------------------------------------------"
    python output_log_archive.py
  } | tee -a "${ARCHIVE_LOG}"
}

if [[ -z "${SESSION_PATH}" && "${RESUME_LATEST}" == "1" ]]; then
  SESSION_PATH="$(latest_feedback_session)"
fi

echo "EliteAlpha runner started."
echo "steps_per_round=${STEPS_PER_ROUND}"
echo "multi_proc_n=${MULTI_PROC_N}"
echo "archive_log=${ARCHIVE_LOG}"
echo "run_log=${RUN_LOG}"
if [[ -n "${SESSION_PATH}" ]]; then
  echo "resume_session=${SESSION_PATH}"
else
  echo "resume_session=(new session will be created on first round)"
fi

round=0
while true; do
  for idx in "${!DIRECTIONS[@]}"; do
    direction="${DIRECTIONS[$idx]}"
    round=$((round + 1))

    echo
    echo ">>> Start EliteAlpha round=${round}, direction_idx=${idx}"

    if [[ -n "${SESSION_PATH}" ]]; then
      if ! alphaagent elite_mine "${SESSION_PATH}" --step_n="${STEPS_PER_ROUND}" --direction="${direction}"; then
        echo "WARNING: EliteAlpha round=${round}, direction_idx=${idx} failed. Continue with next direction."
        continue
      fi
    else
      if ! alphaagent elite_mine --step_n="${STEPS_PER_ROUND}" --direction="${direction}"; then
        echo "WARNING: EliteAlpha round=${round}, direction_idx=${idx} failed. Continue with next direction."
        continue
      fi
    fi

    SESSION_PATH="$(latest_feedback_session)"
    if [[ -z "${SESSION_PATH}" ]]; then
      echo "ERROR: cannot find latest feedback session under log/*/__session__/*/4_feedback" >&2
      exit 1
    fi

    append_archive_log "${round}" "${idx}" "${direction}"
  done
done

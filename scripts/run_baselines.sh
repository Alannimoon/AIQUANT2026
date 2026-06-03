#!/usr/bin/env bash
#
# Baseline runner for manual MAP-Elites exploration.
#
# Runs `alphaagent mine` sequentially across five directions intended to
# steer the Idea Agent into each of the ELITEALPHA proposal's five factor
# categories (momentum / reversal / volatility / volume-price / cross-section).
#
# Per direction:
#   - --step_n 25  -> at most 5 full loops (5 steps per loop)
#   - logs to run_logs/baseline_<NN>_<category>.log
#   - sequential (no parallelism) to stay under DeepSeek rate limits
#
# Expected wall time: ~2-3 hours total. Designed to be run inside tmux so an
# SSH disconnect doesn't kill the job.
#
# Usage (from AlphaAgent/):
#   tmux new -s mine
#   bash ../scripts/run_baselines.sh
#   # ctrl-b d to detach; `tmux attach -t mine` to come back

set -uo pipefail

# Resolve repo root so this script works regardless of cwd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
ALPHAAGENT_DIR="$REPO_ROOT/AlphaAgent"

cd "$ALPHAAGENT_DIR"
mkdir -p run_logs

# DeepSeek API is domestic; the proxy must not leak into the mine process.
unset http_proxy https_proxy all_proxy

# (category_slug, direction_prompt) pairs.
# Direction text deliberately includes a "keep expression simple" hint to push
# back on the LLM's bias toward deep AST factors.
DIRECTIONS=(
  "momentum"
  "Use cumulative or smoothed past return over a short fixed lookback (3-10 days) as a momentum signal. Prefer simple expressions with AST depth <= 3."

  "reversal"
  "Short-term reversal: stocks that dropped sharply in the last 1-5 days tend to bounce. Use lagged negative returns or distance-from-recent-high. Keep AST depth <= 3."

  "volatility"
  "Idiosyncratic volatility regime: factor should rise when recent realized vol (e.g. TS_STD of return over 10-20 days) increases. Keep AST depth <= 4."

  "volume_price"
  "Abnormal volume coinciding with price moves: e.g. current volume vs its rolling mean, possibly interacted with return sign. Keep AST depth <= 4."

  "cross_section"
  "Cross-sectional rank or z-score of a simple price/volume statistic (rank of recent return, rank of dollar volume). Use RANK/ZSCORE as the outer operator. Keep AST depth <= 3."
)

# --step_n 25 ~= 5 loops (5 steps each). Adjust if you want longer/shorter mines.
STEP_N=25

# Where to stash qlib_res.csv files so per-mine cleanup doesn't lose IC data.
METRICS_DIR="$HOME/baseline_metrics"
mkdir -p "$METRICS_DIR"

# After each mine: pull every qlib_res.csv into METRICS_DIR (tagged by the
# direction slug + workspace id), then nuke the big files inside each
# workspace. We keep factor.py (~few KB) and the small mlruns dir so the
# pickle session in AlphaAgent/log/ stays self-contained for Streamlit UI.
cleanup_after_mine() {
  local slug="$1"
  local saved=0
  for d in git_ignore_folder/RD-Agent_workspace/*/; do
    [ -d "$d" ] || continue
    if [ -f "$d/qlib_res.csv" ]; then
      wsid=$(basename "$d" | head -c 8)
      cp "$d/qlib_res.csv" "$METRICS_DIR/${slug}_${wsid}.csv" \
        && saved=$((saved+1))
    fi
  done
  # Reclaim the bulk of the per-workspace footprint.
  find git_ignore_folder/RD-Agent_workspace/ -name "daily_pv.h5" -delete 2>/dev/null
  find git_ignore_folder/RD-Agent_workspace/ -name "result.h5"   -delete 2>/dev/null
  # pickle_cache is RD-Agent's LLM response cache; we don't replay so nuke it.
  rm -rf pickle_cache/* 2>/dev/null
  local free
  free=$(df -h / | awk 'NR==2 {print $4}')
  echo "    ...saved $saved metric file(s), freed disk; ${free} now free"
}

run_one_phase() {
  # Args:
  #   $1 = phase tag (A or C). Used as a prefix in log filenames and metric
  #         filenames so the two phases never overwrite each other.
  #   $2 = depth cap, an integer or empty. When set, exported as
  #         ALPHAAGENT_DEPTH_CAP so the FactorRegulator rejects expressions
  #         deeper than this. When empty, no cap.
  local phase="$1"
  local cap="$2"

  if [ -n "$cap" ]; then
    export ALPHAAGENT_DEPTH_CAP="$cap"
    local phase_label="phase ${phase} (depth_cap=${cap})"
  else
    unset ALPHAAGENT_DEPTH_CAP
    local phase_label="phase ${phase} (no cap)"
  fi

  echo "" | tee -a run_logs/baseline_summary.log
  echo "############################################" | tee -a run_logs/baseline_summary.log
  echo "### START ${phase_label} at $(date)" | tee -a run_logs/baseline_summary.log
  echo "############################################" | tee -a run_logs/baseline_summary.log

  local phase_start=$(date +%s)
  local total=$((${#DIRECTIONS[@]}/2))

  for ((i=0; i<${#DIRECTIONS[@]}; i+=2)); do
    local idx=$((i/2))
    local slug="${DIRECTIONS[$i]}"
    local prompt="${DIRECTIONS[$i+1]}"
    local log="run_logs/baseline_${phase}_$(printf '%02d' "$idx")_${slug}.log"

    printf '\n--- %s [%d/%d] %-15s | starting %s\n' \
      "$phase" "$((idx+1))" "$total" "$slug" "$(date '+%H:%M:%S')" \
      | tee -a run_logs/baseline_summary.log

    local start=$(date +%s)
    local status="OK"
    if ! alphaagent mine --potential_direction "$prompt" --step_n "$STEP_N" \
         > "$log" 2>&1; then
      status="FAIL(exit=$?)"
    fi
    local end=$(date +%s)
    printf -- '--- %s [%d/%d] %-15s | %s after %4d s | %s\n' \
      "$phase" "$((idx+1))" "$total" "$slug" "$status" "$((end-start))" "$log" \
      | tee -a run_logs/baseline_summary.log

    cleanup_after_mine "${phase}_${slug}" | tee -a run_logs/baseline_summary.log
  done

  local phase_end=$(date +%s)
  local elapsed=$((phase_end-phase_start))
  printf -- '\n### END %s at %s, total %dm%02ds\n' \
    "$phase_label" "$(date)" $((elapsed/60)) $((elapsed%60)) \
    | tee -a run_logs/baseline_summary.log
}

start_global=$(date +%s)
echo "=== baseline runner started at $(date) ===" | tee -a run_logs/baseline_summary.log

# Phase A: prompt-only hint, no hard depth cap.
run_one_phase "A" ""

# Phase C: hard depth cap = 5 (matches ELITEALPHA §3.2 default L).
# FactorRegulator now rejects anything deeper than this, and the LLM gets
# fed back the rejection feedback inside CoSTEER's debug loop.
run_one_phase "C" "5"

end_global=$(date +%s)
total=$((end_global-start_global))
printf -- '\n=== baseline runner done at %s, total %dm%02ds ===\n' \
  "$(date)" $((total/60)) $((total%60)) | tee -a run_logs/baseline_summary.log

# Roll up depth/category by direction using the (now improved) ast_depth.py.
echo "" | tee -a run_logs/baseline_summary.log
echo "=== category x depth per direction ===" | tee -a run_logs/baseline_summary.log
for f in run_logs/baseline_[AC]_*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f" .log)" | tee -a run_logs/baseline_summary.log
  python "$REPO_ROOT/scripts/ast_depth.py" --from-log "$f" 2>/dev/null \
    | grep -E "^(Analyzed|category|momentum|reversal|volatility|volume_price|cross_section|other|Depth histogram|---)" \
    | tee -a run_logs/baseline_summary.log || true
done

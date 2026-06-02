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

start_global=$(date +%s)
echo "=== baseline runner started at $(date) ===" | tee -a run_logs/baseline_summary.log

for ((i=0; i<${#DIRECTIONS[@]}; i+=2)); do
  idx=$((i/2))
  slug="${DIRECTIONS[$i]}"
  prompt="${DIRECTIONS[$i+1]}"
  log="run_logs/baseline_$(printf '%02d' "$idx")_${slug}.log"

  printf '\n--- [%d/%d] %-15s | starting %s\n' \
    "$((idx+1))" "$((${#DIRECTIONS[@]}/2))" "$slug" "$(date '+%H:%M:%S')" \
    | tee -a run_logs/baseline_summary.log

  start=$(date +%s)
  if alphaagent mine --potential_direction "$prompt" --step_n "$STEP_N" \
       > "$log" 2>&1
  then
    status="OK"
  else
    status="FAIL(exit=$?)"
  fi
  end=$(date +%s)
  printf '--- [%d/%d] %-15s | %s after %4d s | %s\n' \
    "$((idx+1))" "$((${#DIRECTIONS[@]}/2))" "$slug" "$status" "$((end-start))" "$log" \
    | tee -a run_logs/baseline_summary.log
done

end_global=$(date +%s)
total=$((end_global-start_global))
printf '\n=== baseline runner done at %s, total %dm%02ds ===\n' \
  "$(date)" $((total/60)) $((total%60)) | tee -a run_logs/baseline_summary.log

# Quick summary of every backtest in this batch.
echo "" | tee -a run_logs/baseline_summary.log
echo "=== per-workspace metrics (latest first) ===" | tee -a run_logs/baseline_summary.log
python "$REPO_ROOT/scripts/ast_depth.py" \
  --scan-workspaces "$ALPHAAGENT_DIR/git_ignore_folder/RD-Agent_workspace" \
  2>/dev/null | tee -a run_logs/baseline_summary.log || true

#!/usr/bin/env bash
# Single-instance AgentTorch COVID-LLM benchmark sweep.
#
# Topology (2 nodes):
#   VLLM_NODE : vLLM server (Qwen3-32B, TP=4, 4 GPUs)
#   SIM_NODE  : one ChemGraph-style sim process (1 GPU) -> the instrumented runner
#
# For every sweep point this script:
#   1. ensures vLLM is up (via manage_vllm.sh) and reachable from SIM_NODE
#   2. starts data collectors:
#        - CPU sampler on VLLM_NODE and SIM_NODE
#        - GPU sampler on VLLM_NODE and SIM_NODE
#        - vLLM /metrics sampler (+ raw dumps) hitting VLLM_NODE:PORT
#        - vLLM backend log incremental capture + /metrics before/after snapshots
#   3. runs ONE run_sim_instrumented.py config on SIM_NODE (records per-step timing)
#   4. stops collectors, captures after-snapshots, writes per-run dir
#
# Sweeps (single instance, one at a time):
#   POPULATION sweep : --num-agents over sub-sampled astoria populations
#   GROUP sweep      : --llm-grouping-mode x --llm-history-k
#
# Designed to be driven by a PBS script (sets VLLM_NODE/SIM_NODE from PBS_NODEFILE)
# or run by hand on a debug allocation with explicit node names.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
ROOT="${AGENTTORCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCH="$ROOT/bench"
TOOLS="$BENCH/tools"

VLLM_NODE="${VLLM_NODE:-}"
SIM_NODE="${SIM_NODE:-}"
PORT="${VLLM_PORT:-8000}"
MODEL="${COVID_LLM_MODEL:-agenttorch-qwen3-32b}"
CONDA_ENV="${COVID_LLM_CONDA_ENV:-AgentTorch}"

# Sweep definitions
STEPS="${BENCH_STEPS:-21}"
EPISODES="${BENCH_EPISODES:-1}"
REPEATS="${BENCH_REPEATS:-3}"
MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-8}"
MAX_TOKENS="${BENCH_MAX_TOKENS:-64}"
SAMPLE_INTERVAL="${BENCH_SAMPLE_INTERVAL:-0.5}"
CUDA_SYNC="${BENCH_CUDA_SYNC:-1}"   # 1 -> accurate per-step GPU timing

# Population sweep: sizes (must have matching sub-population dirs; see make_subpopulation)
POP_SIZES="${BENCH_POP_SIZES:-1000 2000 5000 10000 20000 37518}"
SUBPOP_ROOT="${BENCH_SUBPOP_ROOT:-$ROOT/agent_torch/populations/_bench_subpops}"
FULL_POP_DIR="$ROOT/agent_torch/populations/astoria"

# Group sweep: grouping modes and history-k values
GROUP_MODES="${BENCH_GROUP_MODES:-age_week age_memory state_memory}"
HISTORY_KS="${BENCH_HISTORY_KS:-0 4}"
# For the group sweep, hold population fixed at this size:
GROUP_SWEEP_AGENTS="${BENCH_GROUP_SWEEP_AGENTS:-37518}"
# For the population sweep, hold grouping fixed at this mode:
POP_SWEEP_GROUP="${BENCH_POP_SWEEP_GROUP:-age_memory}"

WHICH="${BENCH_WHICH:-all}"   # all | population | group | smoke
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${BENCH_OUT_ROOT:-$ROOT/bench/results/single_instance_${STAMP}}"

START_VLLM="${BENCH_START_VLLM:-1}"
WAIT_ATTEMPTS="${BENCH_VLLM_WAIT_ATTEMPTS:-120}"
WAIT_SECONDS="${BENCH_VLLM_WAIT_SECONDS:-10}"

usage() {
  cat <<EOF
Usage: VLLM_NODE=<n> SIM_NODE=<n> $(basename "$0") [all|population|group|smoke]

Key env overrides:
  VLLM_NODE, SIM_NODE        (required) compute node short names
  VLLM_PORT=$PORT
  BENCH_STEPS=$STEPS BENCH_EPISODES=$EPISODES BENCH_REPEATS=$REPEATS
  BENCH_POP_SIZES="$POP_SIZES"
  BENCH_GROUP_MODES="$GROUP_MODES"  BENCH_HISTORY_KS="$HISTORY_KS"
  BENCH_SAMPLE_INTERVAL=$SAMPLE_INTERVAL
  BENCH_MAX_CONCURRENCY=$MAX_CONCURRENCY BENCH_MAX_TOKENS=$MAX_TOKENS
  BENCH_CUDA_SYNC=$CUDA_SYNC
  BENCH_OUT_ROOT=<dir>
  BENCH_START_VLLM=$START_VLLM
EOF
}

[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage; exit 0; }
[[ -n "${1:-}" ]] && WHICH="$1"
[[ -n "$VLLM_NODE" && -n "$SIM_NODE" ]] || { echo "Set VLLM_NODE and SIM_NODE" >&2; usage; exit 2; }

BASE_URL="http://$VLLM_NODE:$PORT/v1"
METRICS_URL="http://127.0.0.1:$PORT/metrics"   # sampled ON the vllm node
mkdir -p "$OUT_ROOT"

echo "=================================================================="
echo " AgentTorch single-instance sweep"
echo "   VLLM_NODE = $VLLM_NODE   SIM_NODE = $SIM_NODE   port = $PORT"
echo "   which     = $WHICH"
echo "   out       = $OUT_ROOT"
echo "   steps=$STEPS episodes=$EPISODES repeats=$REPEATS sample=${SAMPLE_INTERVAL}s"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Remote helper: run a command on a node inside the conda env
# ---------------------------------------------------------------------------
ssh_node() {  # ssh_node <node> <command-string>
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"
}

py_env_prefix() {
  cat <<EOF
module use /soft/modulefiles >/dev/null 2>&1
module load conda >/dev/null 2>&1
conda activate $(printf '%q' "$CONDA_ENV") >/dev/null 2>&1
export PYTHONUNBUFFERED=1
export NO_PROXY="\${NO_PROXY:-},$VLLM_NODE,127.0.0.1,localhost"
export no_proxy="\${no_proxy:-},$VLLM_NODE,127.0.0.1,localhost"
cd $(printf '%q' "$ROOT")
EOF
}

# ---------------------------------------------------------------------------
# vLLM lifecycle (reuse manage_vllm.sh) + readiness
# ---------------------------------------------------------------------------
start_vllm() {
  [[ "$START_VLLM" == 1 ]] || { echo "[vllm] START_VLLM=0, skipping start"; return 0; }
  echo "[vllm] starting on $VLLM_NODE via manage_vllm.sh"
  VLLM_NODE="$VLLM_NODE" VLLM_PORT="$PORT" VLLM_SERVED_MODEL_NAME="$MODEL" \
    "$ROOT/manage_vllm.sh" start >"$OUT_ROOT/vllm_start.log" 2>&1 || true
}

wait_vllm() {
  echo "[vllm] waiting for API from $SIM_NODE -> $BASE_URL"
  for attempt in $(seq 1 "$WAIT_ATTEMPTS"); do
    if ssh_node "$SIM_NODE" "curl --noproxy '*' -fsS -m 8 $BASE_URL/models >/dev/null 2>&1"; then
      echo "[vllm] ready (attempt $attempt)"
      return 0
    fi
    sleep "$WAIT_SECONDS"
  done
  echo "[vllm] NOT ready after $((WAIT_ATTEMPTS*WAIT_SECONDS))s" >&2
  return 1
}

resolve_vllm_log() {
  VLLM_PID="$(ssh_node "$VLLM_NODE" "pgrep -f 'vllm serve.*--port $PORT' | head -1" | awk '/^[0-9]+$/{print;exit}')"
  if [[ -n "${VLLM_PID:-}" ]]; then
    VLLM_LOG="$(ssh_node "$VLLM_NODE" "readlink /proc/$VLLM_PID/fd/1 2>/dev/null" | awk '/^\//{print;exit}')"
  fi
  echo "[vllm] pid=${VLLM_PID:-?} log=${VLLM_LOG:-?}"
}

# ---------------------------------------------------------------------------
# Per-run samplers
# ---------------------------------------------------------------------------
SAMPLER_PIDS=()

start_samplers() {  # start_samplers <run_dir>
  local run_dir="$1"
  SAMPLER_PIDS=()

  # CPU + GPU on both nodes
  for spec in "vllm:$VLLM_NODE" "sim:$SIM_NODE"; do
    local role="${spec%%:*}" node="${spec##*:}"
    ssh_node "$node" "$(py_env_prefix); python3 $(printf '%q' "$TOOLS/cpu_sampler.py") --interval $SAMPLE_INTERVAL --host $node" \
      >"$run_dir/cpu_${role}_${node}.csv" 2>"$run_dir/cpu_${role}_${node}.err" &
    SAMPLER_PIDS+=("$!")
    ssh_node "$node" "$(py_env_prefix); python3 $(printf '%q' "$TOOLS/gpu_sampler.py") --interval $SAMPLE_INTERVAL --host $node" \
      >"$run_dir/gpu_${role}_${node}.csv" 2>"$run_dir/gpu_${role}_${node}.err" &
    SAMPLER_PIDS+=("$!")
  done

  # vLLM /metrics sampled ON the vllm node (direct loopback, no proxy)
  mkdir -p "$run_dir/vllm_metrics_raw"
  ssh_node "$VLLM_NODE" "$(py_env_prefix); python3 $(printf '%q' "$TOOLS/vllm_metrics_sampler.py") --url $METRICS_URL --interval $SAMPLE_INTERVAL --host $VLLM_NODE" \
    >"$run_dir/vllm_metrics.csv" 2>"$run_dir/vllm_metrics.err" &
  SAMPLER_PIDS+=("$!")
}

stop_samplers() {
  for pid in "${SAMPLER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${SAMPLER_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  SAMPLER_PIDS=()
  # best-effort: kill orphaned remote samplers
  for node in "$VLLM_NODE" "$SIM_NODE"; do
    ssh_node "$node" "pkill -u $USER -f 'cpu_sampler.py|gpu_sampler.py|vllm_metrics_sampler.py' 2>/dev/null || true" || true
  done
}

snapshot_metrics() {  # snapshot_metrics <run_dir> <when>
  ssh_node "$VLLM_NODE" "curl --noproxy '*' -fsS -m 8 $METRICS_URL" \
    >"$1/vllm_metrics_$2.prom" 2>/dev/null || true
}

capture_backend_log_tail() {  # capture_backend_log_tail <run_dir> <start_size>
  local run_dir="$1" start="$2"
  [[ -n "${VLLM_LOG:-}" ]] || return 0
  ssh_node "$VLLM_NODE" "tail -c +$((start + 1)) $(printf '%q' "$VLLM_LOG")" \
    >"$run_dir/vllm_backend_incremental.log" 2>/dev/null || true
}

backend_log_size() {
  [[ -n "${VLLM_LOG:-}" ]] || { echo 0; return; }
  ssh_node "$VLLM_NODE" "stat -c %s $(printf '%q' "$VLLM_LOG") 2>/dev/null" | awk '/^[0-9]+$/{print;exit}' || echo 0
}

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
run_one() {  # run_one <run_label> <num_agents> <pop_dir> <group_mode> <history_k> <rep>
  local label="$1" agents="$2" pop_dir="$3" group="$4" hk="$5" rep="$6"
  local run_dir="$OUT_ROOT/$label"
  mkdir -p "$run_dir"

  echo
  echo "------------------------------------------------------------------"
  echo "[run] $label  agents=$agents group=$group history_k=$hk rep=$rep"
  echo "[run] pop_dir=$pop_dir"
  echo "------------------------------------------------------------------"

  local log_start; log_start="$(backend_log_size)"
  snapshot_metrics "$run_dir" "before"
  start_samplers "$run_dir"
  # let samplers establish a baseline
  sleep 2

  local sync_flag=""
  [[ "$CUDA_SYNC" == 1 ]] && sync_flag="--cuda-sync"

  ssh_node "$SIM_NODE" "$(py_env_prefix); python3 $(printf '%q' "$BENCH/run_sim_instrumented.py") \
    --base-url $(printf '%q' "$BASE_URL") \
    --model $(printf '%q' "$MODEL") \
    --repo-root $(printf '%q' "$ROOT") \
    --population-dir $(printf '%q' "$pop_dir") \
    --run-dir $(printf '%q' "$run_dir") \
    --run-label $(printf '%q' "$label") \
    --num-agents $agents --steps $STEPS --episodes $EPISODES \
    --llm-grouping-mode $group --llm-history-k $hk \
    --llm-max-concurrency $MAX_CONCURRENCY --max-tokens $MAX_TOKENS \
    --output-config $(printf '%q' "/tmp/${USER}_bench_${label}.yaml") \
    $sync_flag" \
    >"$run_dir/runner_stdout.log" 2>"$run_dir/runner_stderr.log" || {
      echo "[run] FAILED: $label (see $run_dir/runner_stderr.log)" >&2
    }

  sleep 2
  stop_samplers
  snapshot_metrics "$run_dir" "after"
  capture_backend_log_tail "$run_dir" "$log_start"

  # quick status line
  if [[ -f "$run_dir/results.json" ]]; then
    echo "[run] OK -> $run_dir/results.json"
  else
    echo "[run] NO results.json for $label" >&2
  fi
}

# ---------------------------------------------------------------------------
# Population sub-sampling (build dirs once)
# ---------------------------------------------------------------------------
ensure_subpops() {
  echo "[pop] ensuring sub-populations under $SUBPOP_ROOT for sizes: $POP_SIZES"
  local need=()
  for n in $POP_SIZES; do
    if [[ "$n" == "37518" ]]; then continue; fi   # use full astoria directly
    [[ -f "$SUBPOP_ROOT/astoria_n${n}/subpopulation_info.json" ]] || need+=("$n")
  done
  if [[ ${#need[@]} -gt 0 ]]; then
    ssh_node "$SIM_NODE" "$(py_env_prefix); python3 $(printf '%q' "$TOOLS/make_subpopulation.py") \
      --source $(printf '%q' "$FULL_POP_DIR") --out-root $(printf '%q' "$SUBPOP_ROOT") \
      --prefix astoria --sizes ${need[*]}" 2>&1 | sed 's/^/[pop] /'
  fi
}

pop_dir_for() {  # echo population dir for size N
  local n="$1"
  if [[ "$n" == "37518" ]]; then echo "$FULL_POP_DIR"; else echo "$SUBPOP_ROOT/astoria_n${n}"; fi
}

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
sweep_population() {
  ensure_subpops
  for n in $POP_SIZES; do
    local pop_dir; pop_dir="$(pop_dir_for "$n")"
    for rep in $(seq 1 "$REPEATS"); do
      run_one "pop_n${n}_${POP_SWEEP_GROUP}_hk0_rep${rep}" \
        "$n" "$pop_dir" "$POP_SWEEP_GROUP" 0 "$rep"
    done
  done
}

sweep_group() {
  local pop_dir; pop_dir="$(pop_dir_for "$GROUP_SWEEP_AGENTS")"
  for group in $GROUP_MODES; do
    for hk in $HISTORY_KS; do
      for rep in $(seq 1 "$REPEATS"); do
        run_one "group_${group}_hk${hk}_n${GROUP_SWEEP_AGENTS}_rep${rep}" \
          "$GROUP_SWEEP_AGENTS" "$pop_dir" "$group" "$hk" "$rep"
      done
    done
  done
}

sweep_smoke() {
  # tiny: one small pop, one group, one rep, few steps
  local pop_dir; pop_dir="$(pop_dir_for 2000)"
  STEPS=3 run_one "smoke_n2000_age_memory_hk0_rep1" 2000 "$pop_dir" age_memory 0 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
trap 'stop_samplers' EXIT

start_vllm
wait_vllm || { echo "Aborting: vLLM not reachable" >&2; exit 1; }
resolve_vllm_log

# record run-level metadata
cat >"$OUT_ROOT/sweep_meta.json" <<EOF
{
  "stamp": "$STAMP",
  "vllm_node": "$VLLM_NODE",
  "sim_node": "$SIM_NODE",
  "port": $PORT,
  "model": "$MODEL",
  "which": "$WHICH",
  "steps": $STEPS, "episodes": $EPISODES, "repeats": $REPEATS,
  "max_concurrency": $MAX_CONCURRENCY, "max_tokens": $MAX_TOKENS,
  "sample_interval": $SAMPLE_INTERVAL, "cuda_sync": $CUDA_SYNC,
  "pop_sizes": "$POP_SIZES", "pop_sweep_group": "$POP_SWEEP_GROUP",
  "group_modes": "$GROUP_MODES", "history_ks": "$HISTORY_KS",
  "group_sweep_agents": $GROUP_SWEEP_AGENTS,
  "base_url": "$BASE_URL", "vllm_pid": "${VLLM_PID:-}", "vllm_log": "${VLLM_LOG:-}"
}
EOF

case "$WHICH" in
  smoke) ensure_subpops; sweep_smoke ;;
  population) sweep_population ;;
  group) sweep_group ;;
  all) sweep_population; sweep_group ;;
  *) echo "Unknown WHICH=$WHICH" >&2; exit 2 ;;
esac

echo
echo "=================================================================="
echo " Sweep complete. Results: $OUT_ROOT"
echo "=================================================================="
ls -1 "$OUT_ROOT"

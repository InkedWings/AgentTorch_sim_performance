#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_covid_llm_remote.sh RUN_NODE VLLM_NODE [-- run_covid_llm.py args...]

Example:
  scripts/run_covid_llm_remote.sh x3005c0s13b0n0 x3005c0s37b1n0 -- --steps 21 --episodes 1

Optional env vars:
  COVID_LLM_PORT=8000
  COVID_LLM_MODEL=agenttorch-qwen3-32b
  COVID_LLM_CONDA_ENV=AgentTorch
  COVID_LLM_LOG_DIR=logs/covid_llm_runs
  COVID_LLM_FOREGROUND=1   # stream output instead of starting with nohup
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  usage
  exit 0
fi

RUN_NODE="$1"
VLLM_NODE="$2"
shift 2
[[ "${1:-}" == "--" ]] && shift

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"

PORT="${COVID_LLM_PORT:-8000}"
MODEL="${COVID_LLM_MODEL:-agenttorch-qwen3-32b}"
CONDA_ENV="${COVID_LLM_CONDA_ENV:-AgentTorch}"
LOG_DIR="${COVID_LLM_LOG_DIR:-$ROOT/logs/covid_llm_runs}"
FOREGROUND="${COVID_LLM_FOREGROUND:-0}"
BASE_URL="http://$VLLM_NODE:$PORT/v1"

quote_args() {
  local out="" arg
  for arg in "$@"; do
    out+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$out"
}

EXTRA_ARGS="$(quote_args "$@")"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/covid_llm_${RUN_NODE}_to_${VLLM_NODE}_${STAMP}.log"
PIDFILE="$LOG_DIR/covid_llm_${RUN_NODE}_to_${VLLM_NODE}.pid"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
module use /soft/modulefiles
module load conda
conda activate $(printf '%q' "$CONDA_ENV")
cd $(printf '%q' "$ROOT")
export NO_PROXY="\${NO_PROXY:-},$VLLM_NODE,$VLLM_NODE.hsn.cm.polaris.alcf.anl.gov"
export no_proxy="\${no_proxy:-},$VLLM_NODE,$VLLM_NODE.hsn.cm.polaris.alcf.anl.gov"
export PYTHONUNBUFFERED=1
python scripts/run_covid_llm.py \
  --base-url $(printf '%q' "$BASE_URL") \
  --model $(printf '%q' "$MODEL") \
  --repo-root $(printf '%q' "$ROOT") \
  --output-config $(printf '%q' "/tmp/${USER}_agenttorch_covid_llm.yaml") \
  $EXTRA_ARGS
EOF
)

mkdir -p "$LOG_DIR"

echo "Run node:  $RUN_NODE"
echo "vLLM URL:  $BASE_URL"
echo "Model:     $MODEL"

if [[ "$FOREGROUND" == "1" ]]; then
  ssh "$RUN_NODE" "bash -lc $(printf '%q' "$REMOTE_CMD")"
else
  ssh "$RUN_NODE" "mkdir -p $(printf '%q' "$LOG_DIR"); nohup bash -lc $(printf '%q' "$REMOTE_CMD") > $(printf '%q' "$LOG") 2>&1 < /dev/null & echo \$! > $(printf '%q' "$PIDFILE")"
  echo "Started detached."
  echo "PID file:  $PIDFILE"
  echo "Log:       $LOG"
  echo "Tail with: tail -f $LOG"
fi

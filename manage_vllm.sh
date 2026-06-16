#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${VLLM_SCRIPT:-$(readlink -f "${BASH_SOURCE[0]}")}"
ROOT="${VLLM_ROOT:-$(cd "$(dirname "$SCRIPT")" && pwd)}"
JOBID="${VLLM_PBS_JOBID:-${PBS_JOBID:-7140549}}"
NODE="${VLLM_NODE:-auto}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-32B}"
SERVED="${VLLM_SERVED_MODEL_NAME:-agenttorch-qwen3-32b}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
CUDA_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"

AGENTIC_ROOT="${AGENTIC_ROOT:-$(cd "$ROOT/../Agentic" && pwd)}"
SIF="${VLLM_CONTAINER:-$AGENTIC_ROOT/containers/vllm-openai-v0.19.1.sif}"
HF_HOME_DIR="${VLLM_HF_HOME:-$AGENTIC_ROOT/hf_cache}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT/vllm_cache}"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT/logs/vllm}"
SERVICE="${VLLM_SERVICE_NAME:-vllm_agenttorch_qwen3_32b_8000}"
PIDFILE="$LOG_DIR/$SERVICE.pid"
LATEST_LOG="$LOG_DIR/$SERVICE.latest.log"

PROXY="${PROXY_URL:-http://proxy.alcf.anl.gov:3128}"
HTTP_PROXY="${HTTP_PROXY:-$PROXY}"
HTTPS_PROXY="${HTTPS_PROXY:-$PROXY}"
http_proxy="${http_proxy:-$HTTP_PROXY}"
https_proxy="${https_proxy:-$HTTPS_PROXY}"
NO_PROXY="${NO_PROXY:-${no_proxy:-127.0.0.1,localhost,::1}}"
no_proxy="${no_proxy:-$NO_PROXY}"
VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"

APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/local/scratch/$USER-vllm-apptainer-tmpdir}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/local/scratch/$USER-vllm-apptainer-cachedir}"

usage() {
  cat <<EOF
Usage: $(basename "$0") start|status|stop|restart|logs [N|-f]|test|config|endpoint

Common overrides:
  VLLM_NODE=x3005c0s37b1n0
  VLLM_PBS_JOBID=$JOBID
  VLLM_MODEL=Qwen/Qwen3-32B
  VLLM_SERVED_MODEL_NAME=agenttorch-qwen3-32b
  VLLM_TENSOR_PARALLEL_SIZE=1|2|4
  VLLM_CUDA_VISIBLE_DEVICES=0,1,2,3
  VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
EOF
}

discover_node() {
  [[ "$NODE" != "auto" ]] && { echo "$NODE"; return; }
  qstat -n "$JOBID" 2>/dev/null \
    | tr '+[:space:]' '\n' \
    | sed -n 's#^\(x[0-9][^/]*\)/.*#\1#p' \
    | head -n 1
}

resolve_node() {
  NODE="$(discover_node)"
  [[ -n "$NODE" ]] || { echo "Could not resolve VLLM node. Set VLLM_NODE." >&2; exit 1; }
}

on_node() {
  local short fqdn
  short="$(hostname -s 2>/dev/null || hostname)"
  fqdn="$(hostname -f 2>/dev/null || hostname)"
  [[ "$NODE" == local || "$short" == "$NODE" || "$fqdn" == "$NODE" || "$fqdn" == "$NODE."* ]]
}

remote_env() {
  local name
  for name in \
    VLLM_ROOT VLLM_SCRIPT VLLM_PBS_JOBID VLLM_NODE VLLM_MODEL \
    VLLM_SERVED_MODEL_NAME VLLM_PORT VLLM_TENSOR_PARALLEL_SIZE \
    VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_CUDA_VISIBLE_DEVICES \
    VLLM_CONTAINER VLLM_HF_HOME VLLM_CACHE_ROOT VLLM_LOG_DIR \
    VLLM_SERVICE_NAME VLLM_LOG_STATS_INTERVAL VLLM_ENABLE_AUTO_TOOL_CHOICE \
    VLLM_TOOL_CALL_PARSER VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS VLLM_ENFORCE_EAGER \
    VLLM_ENABLE_PREFIX_CACHING AGENTIC_ROOT PROXY_URL HTTP_PROXY HTTPS_PROXY \
    http_proxy https_proxy NO_PROXY no_proxy HF_TOKEN APPTAINER_TMPDIR \
    APPTAINER_CACHEDIR; do
    [[ ${!name+x} ]] && printf ' %s=%q' "$name" "${!name}"
  done
}

maybe_ssh() {
  [[ "${LOCAL:-0}" == 1 ]] && return
  resolve_node
  on_node && return
  local args="" arg
  for arg in "$@"; do
    args+=" $(printf '%q' "$arg")"
  done
  exec ssh "$NODE" "env$(remote_env) VLLM_NODE=$(printf '%q' "$NODE") bash $(printf '%q' "$SCRIPT") --local$args"
}

load_env() {
  ml use /soft/modulefiles
  ml spack-pe-base
  ml apptainer
  export HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy
  export APPTAINER_TMPDIR APPTAINER_CACHEDIR CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$HF_HOME_DIR" "$VLLM_CACHE_ROOT" "$LOG_DIR"
}

vllm_pids() {
  ps -u "$USER" -o pid=,args= \
    | awk -v model="$MODEL" -v served="$SERVED" -v port="$PORT" '
        index($0, "vllm serve") &&
        (index($0, model) || index($0, served)) &&
        index($0, "--port " port) { print $1 }'
}

managed_pids() {
  local p pp
  for p in $(vllm_pids); do
    echo "$p"
    pp="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d "[:space:]" || true)"
    [[ "$pp" =~ ^[0-9]+$ && "$pp" != 1 ]] && echo "$pp"
  done | sort -n -u
}

port_open() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$PORT$"
}

latest_log() {
  [[ -e "$LATEST_LOG" ]] && { echo "$LATEST_LOG"; return; }
  ls -t "$LOG_DIR/${SERVICE}"_*.log 2>/dev/null | head -n 1
}

run_server() {
  load_env
  [[ -f "$SIF" ]] || { echo "Missing container: $SIF" >&2; exit 2; }
  local serve_args=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "$SERVED"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --trust-remote-code
  )
  [[ "$ENABLE_AUTO_TOOL_CHOICE" == 1 ]] && serve_args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
  [[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]] && serve_args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ "$ENFORCE_EAGER" == 1 ]] && serve_args+=(--enforce-eager)
  [[ "$ENABLE_PREFIX_CACHING" == 1 ]] && serve_args+=(--enable-prefix-caching)
  [[ "$ENABLE_PREFIX_CACHING" == 0 ]] && serve_args+=(--no-enable-prefix-caching)
  exec apptainer exec --nv \
    --bind /lus/eagle:/lus/eagle,/local/scratch:/local/scratch \
    --env HF_HOME="$HF_HOME_DIR" \
    --env HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR" \
    --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    --env VLLM_LOG_STATS_INTERVAL="$VLLM_LOG_STATS_INTERVAL" \
    --env HTTP_PROXY="$HTTP_PROXY" --env HTTPS_PROXY="$HTTPS_PROXY" \
    --env http_proxy="$http_proxy" --env https_proxy="$https_proxy" \
    --env NO_PROXY="$NO_PROXY" --env no_proxy="$no_proxy" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$SIF" "${serve_args[@]}"
}

start() {
  mkdir -p "$LOG_DIR"
  local pids log pid
  pids="$(vllm_pids | xargs echo)"
  [[ -n "$pids" ]] && { echo "Already running: $pids"; status; return; }
  port_open && { echo "Port $PORT already listening; refusing to start." >&2; exit 3; }
  log="$LOG_DIR/${SERVICE}_$(date +%Y%m%d_%H%M%S).log"
  nohup setsid bash -lc "exec $(printf '%q' "$SCRIPT") --local run" > "$log" 2>&1 &
  pid="$!"
  echo "$pid" > "$PIDFILE"
  ln -sfn "$log" "$LATEST_LOG"
  echo "Started vLLM on $NODE"
  echo "PID: $pid"
  echo "Log: $log"
}

stop() {
  local pids
  pids="$(managed_pids | xargs echo)"
  [[ -z "$pids" ]] && { echo "Not running"; rm -f "$PIDFILE"; return; }
  echo "Stopping: $pids"
  kill -TERM $pids 2>/dev/null || true
  sleep 5
  pids="$(managed_pids | xargs echo)"
  [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Stopped"
}

endpoint() {
  resolve_node
  echo "http://$NODE:$PORT/v1"
}

status() {
  local pids log
  pids="$(vllm_pids | xargs echo)"
  log="$(latest_log || true)"
  echo "Node:  $(hostname -f 2>/dev/null || hostname)"
  echo "Model: $MODEL as $SERVED"
  echo "URL:   http://$NODE:$PORT/v1"
  echo "TP:    $TP"
  echo "Stats interval for new starts: ${VLLM_LOG_STATS_INTERVAL}s"
  echo "Tools: auto=${ENABLE_AUTO_TOOL_CHOICE}, parser=${TOOL_CALL_PARSER}"
  echo "Chat template kwargs: ${DEFAULT_CHAT_TEMPLATE_KWARGS:-none}"
  echo "Eager: enforce=${ENFORCE_EAGER}"
  echo "Prefix cache: enable=${ENABLE_PREFIX_CACHING}"
  [[ -n "$pids" ]] && { echo "State: running"; ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -p "$(echo "$pids" | tr ' ' ',')" || true; } || echo "State: stopped"
  [[ -n "$log" ]] && echo "Log:   $log"
  port_open && echo "Port:  $PORT listening" || echo "Port:  $PORT not listening"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
}

logs() {
  local n="${1:-120}" log
  log="$(latest_log)" || { echo "No log found" >&2; exit 1; }
  [[ "$n" == "-f" || "$n" == follow ]] && tail -f "$log" || tail -n "$n" "$log"
}

test_api() {
  port_open || { echo "Port $PORT is not listening" >&2; exit 1; }
  curl --noproxy '*' -sS "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool
  local body
  body='{"model":"'"$SERVED"'","messages":[{"role":"user","content":"Reply with OK only. /no_think"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
  curl --noproxy '*' -sS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$body" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

config() {
  resolve_node
  cat <<EOF
VLLM_NODE=$NODE
VLLM_PBS_JOBID=$JOBID
VLLM_MODEL=$MODEL
VLLM_SERVED_MODEL_NAME=$SERVED
VLLM_PORT=$PORT
VLLM_TENSOR_PARALLEL_SIZE=$TP
VLLM_MAX_MODEL_LEN=$MAX_LEN
VLLM_GPU_MEMORY_UTILIZATION=$GPU_UTIL
VLLM_CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
VLLM_LOG_STATS_INTERVAL=$VLLM_LOG_STATS_INTERVAL
VLLM_ENABLE_AUTO_TOOL_CHOICE=$ENABLE_AUTO_TOOL_CHOICE
VLLM_TOOL_CALL_PARSER=$TOOL_CALL_PARSER
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS=$DEFAULT_CHAT_TEMPLATE_KWARGS
VLLM_ENFORCE_EAGER=$ENFORCE_EAGER
VLLM_ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING
VLLM_CONTAINER=$SIF
VLLM_HF_HOME=$HF_HOME_DIR
VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT
VLLM_LOG_DIR=$LOG_DIR
EOF
}

LOCAL=0
[[ "${1:-}" == "--local" ]] && { LOCAL=1; shift; }
cmd="${1:-status}"; shift || true
maybe_ssh "$cmd" "$@"

case "$cmd" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs "${1:-120}" ;;
  test) test_api ;;
  config) config ;;
  endpoint) endpoint ;;
  run) run_server ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac

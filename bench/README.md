# AgentTorch Single-Instance Benchmark Harness

Measures end-to-end + per-step performance of one LLM-driven COVID simulation
against a shared vLLM backend, sweeping **population size** and **grouping
mode**, while capturing the full resource picture (latency, CPU, GPU
util/power/mem, and the vLLM Prometheus surface).

Topology: 2 nodes — `VLLM_NODE` (Qwen3-32B, TP=4, 4×A100) and `SIM_NODE`
(one sim process, 1 GPU). Validated on a debug allocation 2026-06-17.

## What gets collected (per sweep run dir)

| File | Content |
|------|---------|
| `results.json` | config + init/total timing + **per-step & per-substep wall times** + episode summary (infected, isolation rate, …) + sim-GPU peak mem + LLM request count |
| `steps.csv` | per-step time + substep breakdown |
| `cpu_{vllm,sim}_*.csv` | node CPU util / freq / package power (RAPL), both nodes |
| `gpu_{vllm,sim}_*.csv` | per-GPU util, mem util, mem used, power, temp, SM/mem clocks, pstate, both nodes |
| `vllm_metrics.csv` | **full vLLM Prometheus surface** sampled over time (98 series: running/waiting queues, kv-cache usage, prompt/generation tokens, TTFT/ITL/queue/prefill/decode histograms, …) |
| `vllm_metrics_{before,after}.prom` | counter snapshots → exact per-run token/request deltas |
| `vllm_metrics_raw/` | full raw `/metrics` text per poll (nothing lost) |
| `vllm_backend_incremental.log` | vLLM server log slice for this run |
| `llm_trace.jsonl` | per-group prompt/response/decision trace |

Sampling interval default 0.5s. Per-step GPU timing uses `torch.cuda.synchronize`
(`--cuda-sync`, on by default in the sweep) so timings reflect real compute.

## Components

- `run_sim_instrumented.py` — runs ONE sim config, emits `results.json`/`steps.csv`.
  Reads per-step timing from the runner patch (`runner.py` exposes
  `bench_step_times`/`bench_substep_times` when `BENCH_CUDA_SYNC` is set).
- `tools/make_subpopulation.py` — builds clean sub-sampled population dirs
  (required: `--num-agents` alone can't subsample due to a hard shape assert).
- `tools/{cpu,gpu}_sampler.py` — node CPU / GPU time series.
- `tools/vllm_metrics_sampler.py` — vLLM `/metrics` → long-format CSV (+ raw).
- `tools/aggregate_results.py` — joins everything into `_analysis/runs_summary.csv`
  (one row/run) and `_analysis/steps_long.csv` (one row/step), time-aligning
  samplers to each run window.
- `sweep_single_instance.sh` — orchestrator (starts vLLM + samplers, runs sweeps).
- `submit_single_instance.pbs` — PBS wrapper (2 nodes).

## Sweeps

- **population**: `BENCH_POP_SIZES="1000 2000 5000 10000 20000 37518"`, grouping
  held at `age_memory`. Sub-populations are **density-preserving**: the network
  keeps the induced subgraph plus random in-subset top-up edges so the
  edges-per-agent ratio matches full astoria (3.635 edges/agent on net 0) at
  every size. This holds network density constant across the sweep ("vary
  scale, hold density") so transmission and agent-state differentiation behave
  consistently. Use `make_subpopulation.py --no-densify` for the plain induced
  subgraph instead.
- **group**: `BENCH_GROUP_MODES="age_week age_memory state_memory"` ×
  `BENCH_HISTORY_KS="0 4"`, population held at 37518.
- Each point repeated `BENCH_REPEATS=3` (take median offline).

## Run it

**On a live allocation (manual):**
```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/AgentTorch
VLLM_NODE=<node0> SIM_NODE=<node1> bash bench/sweep_single_instance.sh smoke      # quick check
VLLM_NODE=<node0> SIM_NODE=<node1> bash bench/sweep_single_instance.sh population
VLLM_NODE=<node0> SIM_NODE=<node1> bash bench/sweep_single_instance.sh group
VLLM_NODE=<node0> SIM_NODE=<node1> bash bench/sweep_single_instance.sh all
```

**Submit to queue:**
```bash
qsub bench/submit_single_instance.pbs                                  # all sweeps, 2 nodes
qsub -v BENCH_WHICH=population bench/submit_single_instance.pbs
qsub -A SDR -q debug -l walltime=01:00:00 -v BENCH_WHICH=smoke bench/submit_single_instance.pbs
```

**Analyze:**
```bash
python3 bench/tools/aggregate_results.py bench/results/single_instance_<jobid>
# -> bench/results/.../_analysis/runs_summary.csv  + steps_long.csv
```

## Notes / knobs

- `BENCH_START_VLLM=0` to reuse an already-running vLLM.
- vLLM defaults: TP=4, max-model-len 24576, gpu-mem-util 0.90 (via `manage_vllm.sh`).
- The PBS account/queue in `submit_single_instance.pbs` default to `SDR`/`preemptable`
  (lc-mpi balance is negative; SDR and ChemGraph have credit). Override with
  `qsub -A <proj> -q <queue>`.
- Generated subpopulations live in `agent_torch/populations/_bench_subpops/`
  (gitignored, reusable across runs).

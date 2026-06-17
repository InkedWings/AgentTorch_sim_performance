#!/usr/bin/env python
"""Instrumented single-instance AgentTorch COVID-LLM runner for benchmarking.

Extends scripts/run_covid_llm.py with rich, machine-readable instrumentation:
  * per-step and per-substep wall-clock timing (from runner.bench_step_times,
    exposed by the runner.py patch; BENCH_CUDA_SYNC=true makes GPU timings
    reflect real compute, not just kernel-launch),
  * full run metadata + summary metrics emitted as a single results.json,
  * per-step timing also written as steps.csv,
  * CUDA peak-memory and device info,
  * markers (start/end epoch) so external samplers (CPU/GPU/vLLM) can be
    time-aligned to this run.

It does NOT manage vLLM or system samplers — the sweep orchestrator does that
around this process. This script only runs ONE simulation configuration and
records everything about it.

Output (under --run-dir):
  results.json   — one JSON object: config, env, timing, summary, markers
  steps.csv      — per-step wall time (+ optional substep breakdown)
  run.env.json   — resolved arguments
"""
import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import torch
import yaml

from agent_torch.core.helpers import read_config
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.models import covid
from agent_torch.models.covid.llm import CovidIsolationTemplate
from agent_torch.models.covid.substeps.new_transmission.action import (
    MakeIsolationDecision,
)


def load_config(config_path: Path, population_dir: Path, args: argparse.Namespace):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata = config["simulation_metadata"]
    metadata["population_dir"] = str(population_dir)
    metadata["num_agents"] = args.num_agents
    metadata["num_episodes"] = args.episodes
    metadata["num_steps_per_episode"] = args.steps
    metadata["EXECUTION_MODE"] = "llm"
    metadata["device"] = args.device
    metadata["memory_dir"] = str(args.memory_dir)
    metadata["LLM_HISTORY_K"] = args.llm_history_k
    metadata["LLM_MAX_ARCHETYPE_MEMORY_GROUPS"] = args.llm_max_archetype_memory_groups
    metadata["LLM_GROUPING_MODE"] = args.llm_grouping_mode
    metadata["BENCH_CUDA_SYNC"] = bool(args.cuda_sync)
    if args.llm_trace_path:
        metadata["LLM_TRACE_PATH"] = str(args.llm_trace_path)

    tmp_config = args.output_config
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    with tmp_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return read_config(str(tmp_config))


def build_llm(args: argparse.Namespace):
    parsed = urlparse(args.base_url)
    if parsed.hostname:
        for name in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(name, "")
            entries = [item for item in current.split(",") if item]
            if parsed.hostname not in entries:
                entries.append(parsed.hostname)
            os.environ[name] = ",".join(entries)

    return LangchainLLM(
        openai_api_key=args.api_key,
        agent_profile="",
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_concurrency=args.llm_max_concurrency,
    )


def summarize_state(state):
    daily_infected = state["environment"]["daily_infected"].detach().cpu()
    daily_deaths = state["environment"]["daily_deaths"].detach().cpu()
    stages = (
        state["agents"]["citizens"]["disease_stage"]
        .detach()
        .flatten()
        .to(torch.int64)
        .cpu()
    )
    vals, counts = torch.unique(stages, return_counts=True)
    summary = {
        "infected_sum": float(daily_infected.sum().item()),
        "deaths_sum": float(daily_deaths.sum().item()),
        "stage_counts": {int(v): int(c) for v, c in zip(vals, counts)},
    }
    citizens = state["agents"]["citizens"]
    for key, reducer in (
        ("last_isolation_decision", "mean"),
        ("isolation_streak_days", "mean"),
        ("num_isolation_days", "sum"),
        ("awaiting_test_result", "mean"),
        ("positive_test_result", "sum"),
        ("num_tests_taken", "sum"),
        ("num_positive_tests", "sum"),
        ("is_quarantined", "mean"),
        ("num_quarantine_days", "sum"),
    ):
        if key in citizens:
            t = citizens[key].detach().float()
            summary[f"{key}_{reducer}"] = float(
                t.mean().item() if reducer == "mean" else t.sum().item()
            )
    return summary


def count_trace_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="agenttorch-qwen3-32b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--llm-history-k", type=int, default=0)
    parser.add_argument("--llm-max-archetype-memory-groups", type=int, default=4096)
    parser.add_argument("--llm-max-concurrency", type=int, default=8)
    parser.add_argument(
        "--llm-grouping-mode",
        default="age_memory",
        choices=["age_week", "age_memory", "state_memory"],
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--num-agents", type=int, default=37518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cuda-sync",
        action="store_true",
        help="torch.cuda.synchronize() around each step/substep for accurate "
        "GPU timing (small overhead).",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--population-dir",
        type=Path,
        default=None,
        help="Explicit population dir; default repo_root/agent_torch/populations/astoria",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory for this single run's artifacts.",
    )
    parser.add_argument("--run-label", default="", help="Free-form label for this run.")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--output-config", type=Path, default=None)
    parser.add_argument("--llm-trace-path", type=Path, default=None)
    parser.add_argument("--append-trace", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    population_dir = (
        args.population_dir.resolve()
        if args.population_dir
        else repo_root / "agent_torch/populations/astoria"
    )
    config_path = (
        args.config_path.resolve()
        if args.config_path
        else repo_root / "agent_torch/models/covid/yamls/config.yaml"
    )
    if args.memory_dir is None:
        args.memory_dir = run_dir / "memory"
    if args.output_config is None:
        args.output_config = run_dir / "resolved_config.yaml"
    if args.llm_trace_path is None:
        args.llm_trace_path = run_dir / "llm_trace.jsonl"
    for p in (args.memory_dir, args.output_config, args.llm_trace_path):
        if not Path(p).is_absolute():
            setattr(args, "_tmp", repo_root / p)
    args.llm_trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append_trace:
        args.llm_trace_path.write_text("", encoding="utf-8")

    # Persist resolved args early so a crash still leaves a record.
    resolved = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (run_dir / "run.env.json").write_text(json.dumps(resolved, indent=2), "utf-8")

    config = load_config(config_path, population_dir, args)
    llm = build_llm(args)
    template = CovidIsolationTemplate()
    template.memory_size = args.llm_max_archetype_memory_groups
    isolation_archetype = Archetype(prompt=template, llm=llm, n_arch=1)
    isolation_archetype.broadcast(str(population_dir))
    MakeIsolationDecision.set_behavior(isolation_archetype._behavior)

    device_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }
    if torch.cuda.is_available():
        device_info["gpu_name"] = torch.cuda.get_device_name(0)

    print(f"[bench] run_label={args.run_label}")
    print(f"[bench] base_url={args.base_url} model={args.model}")
    print(
        f"[bench] agents={args.num_agents} steps={args.steps} episodes={args.episodes} "
        f"grouping={args.llm_grouping_mode} history_k={args.llm_history_k} "
        f"max_groups={args.llm_max_archetype_memory_groups} "
        f"concurrency={args.llm_max_concurrency} cuda_sync={args.cuda_sync}"
    )
    print(f"[bench] population_dir={population_dir}")
    print(f"[bench] device={device_info}")

    runner = covid.get_runner(config, covid.registry)

    wall_start_epoch = time.time()
    total_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runner.init()
    init_elapsed = time.perf_counter() - total_start
    print(f"[bench] init_ok elapsed={init_elapsed:.3f}")

    episodes = []
    for episode in range(args.episodes):
        ep_wall_start = time.time()
        ep_start = time.perf_counter()
        runner.step(args.steps)
        ep_elapsed = time.perf_counter() - ep_start
        ep_wall_end = time.time()

        summary = summarize_state(runner.state)
        step_times = list(getattr(runner, "bench_step_times", []) or [])
        substep_times = [list(s) for s in getattr(runner, "bench_substep_times", []) or []]
        substep_names = list(getattr(runner, "bench_substep_names", []) or [])

        ep_record = {
            "episode": episode,
            "wall_start_epoch": ep_wall_start,
            "wall_end_epoch": ep_wall_end,
            "elapsed_s": ep_elapsed,
            "summary": summary,
            "step_times_s": step_times,
            "substep_times_s": substep_times,
            "substep_names": substep_names,
        }
        episodes.append(ep_record)
        print(
            f"[bench] episode_ok {episode + 1}/{args.episodes} elapsed={ep_elapsed:.3f} "
            f"infected_sum={summary['infected_sum']:.3f} "
            f"isolation_rate={summary.get('last_isolation_decision_mean', 0.0):.4f}"
        )
        if episode + 1 < args.episodes:
            runner.reset()

    total_elapsed = time.perf_counter() - total_start
    wall_end_epoch = time.time()

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    trace_lines = count_trace_lines(args.llm_trace_path)

    results = {
        "run_label": args.run_label,
        "markers": {
            "wall_start_epoch": wall_start_epoch,
            "wall_end_epoch": wall_end_epoch,
        },
        "config": {
            "num_agents": args.num_agents,
            "steps": args.steps,
            "episodes": args.episodes,
            "llm_grouping_mode": args.llm_grouping_mode,
            "llm_history_k": args.llm_history_k,
            "llm_max_archetype_memory_groups": args.llm_max_archetype_memory_groups,
            "llm_max_concurrency": args.llm_max_concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "model": args.model,
            "base_url": args.base_url,
            "population_dir": str(population_dir),
            "cuda_sync": bool(args.cuda_sync),
        },
        "device": device_info,
        "timing": {
            "init_elapsed_s": init_elapsed,
            "total_elapsed_s": total_elapsed,
        },
        "sim_gpu_peak_mem_mb": peak_mem_mb,
        "llm_trace_lines_total": trace_lines,
        "episodes": episodes,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), "utf-8")

    # Flat per-step CSV (across episodes) for quick plotting.
    import csv

    with (run_dir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["episode", "step", "step_time_s"]
        max_sub = max(
            (len(ss) for ep in episodes for ss in ep["substep_times_s"]),
            default=0,
        )
        names = episodes[0]["substep_names"] if episodes else []
        for i in range(max_sub):
            header.append(f"substep_{i}_{names[i] if i < len(names) else i}_s")
        w.writerow(header)
        for ep in episodes:
            for si, st in enumerate(ep["step_times_s"]):
                row = [ep["episode"], si, f"{st:.6f}"]
                subs = ep["substep_times_s"][si] if si < len(ep["substep_times_s"]) else []
                for i in range(max_sub):
                    row.append(f"{subs[i]:.6f}" if i < len(subs) else "")
                w.writerow(row)

    print(
        f"[bench] full_ok total_elapsed={total_elapsed:.3f} "
        f"sim_gpu_peak_mem_mb={peak_mem_mb if peak_mem_mb is None else round(peak_mem_mb, 1)} "
        f"trace_lines={trace_lines}"
    )
    print(f"[bench] results -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
